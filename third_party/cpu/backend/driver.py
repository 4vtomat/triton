import os
import hashlib
import importlib
import importlib.resources
import tempfile
import time
import subprocess

import triton
import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from pathlib import Path
from triton._C.libtriton import llvm

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")

# QEMU execution mode for cross-compilation
_qemu_binary = os.getenv("TRITON_CPU_QEMU_EXEC")
_qemu_mode = _qemu_binary is not None



# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

include_dirs = []
library_dirs = [_triton_C_dir]
libraries = ["stdc++"]

# Skip non-existent paths
sys_include_dir = os.path.join(_dirname, "include")
if os.path.exists(sys_include_dir):
    include_dirs.append(sys_include_dir)

sys_lib_dir = os.path.join(_dirname, "lib")
if os.path.exists(sys_lib_dir):
    library_dirs.append(sys_lib_dir)


def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        if _qemu_mode:
            # QEMU mode: cache .so and return path (kernel_ptr will be the path)
            key = hashlib.md5(kernel).hexdigest()
            cache = get_cache_manager(key)
            print("kernel name:", name)
            so_path = cache.get_file(f"{name}.so")
            if so_path is None:
                so_path = cache.put(kernel, f"{name}.so", binary=True)
            return (None, so_path, 0, 0)
        else:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
                f.write(kernel)
                f.flush()
                import ctypes
                lib = ctypes.cdll.LoadLibrary(f.name)
                fn_ptr = getattr(lib, name)
                fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
                return (lib, fn_ptr_as_void_p, 0, 0)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr"):
            return "O"
        return {
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = "iiiOKOOOO" + args_format

    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}

    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")

    arg_ptrs_list = ', '.join(f"&arg{i}" for i in signature.keys())
    kernel_fn_args = [i for i, ty in signature.items() if i not in constants and ty != "constexpr"]
    signature_without_constexprs = {i: ty for i, ty in signature.items() if ty != "constexpr"}
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args)
    kernel_fn_arg_types = ', '.join([f"{ty_to_cpp(signature[i])}" for i in kernel_fn_args] + ["uint32_t"] * 6)

    # generate glue code
    src = f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <optional>
#include <stdio.h>
#include <string>
#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

inline bool getBoolEnv(const std::string &env) {{
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) {{ return std::tolower(c); }});
  return str == "on" || str == "true" || str == "1";
}}

inline std::optional<int64_t> getIntEnv(const std::string &env) {{
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return std::nullopt;

  char *endptr;
  long int result = std::strtol(cstr, &endptr, 10);
  if (endptr == cstr)
    assert(false && "invalid integer");
  return result;
}}

using kernel_ptr_t = void(*)({kernel_fn_arg_types});

typedef struct _DevicePtrInfo {{
  void* dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}

static std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY, uint32_t gridZ) {{
  std::unique_ptr<uint32_t[][3]> grids(new uint32_t[gridX * gridY * gridZ][3]);
  // TODO: which order would be more effective for cache locality?
  for (uint32_t z = 0; z < gridZ; ++z) {{
    for (uint32_t y = 0; y < gridY; ++y) {{
      for (uint32_t x = 0; x < gridX; ++x) {{
        grids[z * gridY * gridX + y * gridX + x][0] = x;
        grids[z * gridY * gridX + y * gridX + x][1] = y;
        grids[z * gridY * gridX + y * gridX + x][2] = z;
      }}
    }}
  }}
  return grids;
}}

static void run_omp_kernels(uint32_t gridX, uint32_t gridY, uint32_t gridZ, int num_threads, kernel_ptr_t kernel_ptr {(', ' + arg_decls) if len(arg_decls) > 0 else ''}) {{
  // TODO: Consider using omp collapse(3) clause for simplicity?
  size_t N = gridX * gridY * gridZ;
  if (N == 1) {{
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} 0, 0, 0, 1, 1, 1);
      return;
  }}

  auto all_grids = get_all_grids(gridX, gridY, gridZ);
  int omp_max_threads = 1;
  #ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
  #endif // _OPENMP
  int max_threads = (num_threads > 0) ? num_threads : omp_max_threads;

  // Don't pay OMP overhead price when a single thread is used.
  if (max_threads == 1) {{
    for (size_t i = 0; i < N; ++i) {{
      const auto [x, y, z] = all_grids[i];
      (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
    }}
    return;
  }}

  // For now, use the default chunk size, total iterations / max_threads.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(max_threads)
#endif // _OPENMP
  for (size_t i = 0; i < N; ++i) {{
    const auto [x, y, z] = all_grids[i];
    (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
  }}
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *py_obj_stream;
  void* pKrnl;

  {' '.join([f"{_extracted_type(ty)} arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &py_obj_stream, &pKrnl,
                                       &kernel_metadata, &launch_metadata,
                                       &launch_enter_hook, &launch_exit_hook {', ' + arg_ptrs_list if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  void *pStream = PyLong_AsVoidPtr(py_obj_stream);
  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(pKrnl);

  // Extract num_threads metadata.
  int num_threads = 0;
  PyObject *num_threads_attr = PyObject_GetAttrString(kernel_metadata, "num_threads");
  if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature_without_constexprs.items()])};
  run_omp_kernels(gridX, gridY, gridZ, num_threads, kernel_ptr {(', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"arg{i}" for i, ty in signature_without_constexprs.items())) if len(signature_without_constexprs) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_cpu_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_cpu_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


class QEMULauncher(object):
    """Launcher that executes kernels via QEMU user-mode emulation."""

    def _get_or_compile_launcher(self, launcher_src):
        key = hashlib.md5(launcher_src.encode("utf-8")).hexdigest()
        cache = get_cache_manager(key)
        launcher_path = cache.get_file("launcher")
        if launcher_path and os.path.exists(launcher_path):
            return launcher_path

        # Compile launcher
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "launcher.c")
            with open(src_path, "w") as f:
                f.write(launcher_src)

            cc = os.environ.get("CC", "gcc")
            target_cpu_env = os.environ.get("TRITON_CPU_TARGET_CPU", "")
            launcher_bin = os.path.join(tmpdir, "launcher")
            cc_cmd = [cc, src_path, "-o", launcher_bin, "-ldl", "-Wl,--export-dynamic"]
            if ":" in target_cpu_env:
                cc_cmd += [f"-mcpu={target_cpu_env.split(':', 1)[1]}"]
            subprocess.check_call(cc_cmd)

            with open(launcher_bin, "rb") as f:
                launcher_path = cache.put(f.read(), "launcher", binary=True)

        os.chmod(launcher_path, 0o755)
        return launcher_path

    def _generate_launcher_source(self, constants, signature):
        """Generate C source for a launcher that calls kernel with exact signature."""

        def _serialize_signature(sig):
            if isinstance(sig, tuple):
                return ','.join(map(_serialize_signature, sig))
            return sig
        signature = ','.join(map(_serialize_signature, signature.values()))
        signature = list(filter(bool, signature.split(',')))
        arg_types = [ty_to_cpp(ty) for ty in signature if ty and ty not in constants and ty != 'constexpr']
        kernel_args = ', '.join(arg_types + ['uint32_t'] * 6)

        # convert qemu inputs to kernel inputs
        arg_defs = []
        for i, arg_type in enumerate(arg_types):
            # get argv value
            arg_defs.append(f"char* val{i} = argv[6 + {i}];")
            if '*' in arg_type:  # Pointer type - uses shared memory
                arg_defs.append(f"""
    {arg_type} arg{i};
    if (strchr(val{i}, ':')) {{
        // Shared memory format: "size:shm_name"
        size_t size{i};
        char shm_name{i}[256];
        sscanf(val{i}, "%zu:%s", &size{i}, shm_name{i});

        char shm_path{i}[300];
        snprintf(shm_path{i}, sizeof(shm_path{i}), "{self.shared_mem_dir}%s", shm_name{i});

        int fd{i} = open(shm_path{i}, O_RDWR);
        if (fd{i} < 0) {{
            fprintf(stderr, "Failed to open shared memory %s\\n", shm_path{i});
            return 1;
        }}

        void* ptr{i} = mmap(NULL, size{i}, PROT_READ | PROT_WRITE, MAP_SHARED, fd{i}, 0);
        close(fd{i});
        if (ptr{i} == MAP_FAILED) {{
            fprintf(stderr, "Failed to mmap shared memory\\n");
            return 1;
        }}
        arg{i} = ({arg_type})ptr{i};
    }} else {{
        arg{i} = ({arg_type})(uintptr_t)strtoull(val{i}, NULL, 10);
    }}""")
            elif arg_type == 'float':
                arg_defs.append(f"""
    // Scalar float argument {i}
    {arg_type} arg{i} = strtof(val{i}, NULL);""")
            elif arg_type == 'double':
                arg_defs.append(f"""
    // Scalar double argument {i}
    {arg_type} arg{i} = strtod(val{i}, NULL);""")
            elif 'int64_t' in arg_type:
                arg_defs.append(f"""
    // int64 argument {i}
    {arg_type} arg{i} = ({arg_type})strtoll(val{i}, NULL, 10);""")
            elif 'uint64_t' in arg_type:
                arg_defs.append(f"""
    // uint64 argument {i}
    {arg_type} arg{i} = ({arg_type})strtoull(val{i}, NULL, 10);""")
            else:
                arg_defs.append(f"""
    // Integer argument {i}
    {arg_type} arg{i} = ({arg_type})atoi(val{i});""")

        arg_list = ', '.join([f"arg{i}" for i in range(len(arg_types))] +
                            ['x', 'y', 'z', 'gridX', 'gridY', 'gridZ'])

        return f"""#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef void (*kernel_fn_t)({kernel_args});

int main(int argc, char* argv[]) {{
    if (argc < 6) {{
        fprintf(stderr, "Usage: %s <kernel.so> <kernel_name> <gridX> <gridY> <gridZ> [args...]\\n", argv[0]);
        return 1;
    }}

    const char* so_path = argv[1];
    const char* kernel_name = argv[2];
    uint32_t gridX = (uint32_t)atoi(argv[3]);
    uint32_t gridY = (uint32_t)atoi(argv[4]);
    uint32_t gridZ = (uint32_t)atoi(argv[5]);

    // Preload TritonCPURuntime stub to provide triton_assert, etc.
    void* runtime_handle = dlopen("/tmp/libTritonCPURuntime_riscv.so", RTLD_NOW | RTLD_GLOBAL);
    if (!runtime_handle) {{
        fprintf(stderr, "Warning: Failed to preload TritonCPURuntime: %s\\n", dlerror());
    }}

    // Load kernel with RTLD_GLOBAL to resolve dependencies
    void* handle = dlopen(so_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {{
        fprintf(stderr, "Failed to load %s: %s\\n", so_path, dlerror());
        return 1;
    }}

    kernel_fn_t kernel_ptr = (kernel_fn_t)dlsym(handle, kernel_name);
    if (!kernel_ptr) {{
        fprintf(stderr, "Failed to find %s: %s\\n", kernel_name, dlerror());
        dlclose(handle);
        return 1;
    }}

    // Parse arguments with correct types and handle shared memory
{''.join(arg_defs)}

    // Execute kernel for each grid point
    for (uint32_t gz = 0; gz < gridZ; gz++) {{
        for (uint32_t gy = 0; gy < gridY; gy++) {{
            for (uint32_t gx = 0; gx < gridX; gx++) {{
                uint32_t x = gx;
                uint32_t y = gy;
                uint32_t z = gz;
                kernel_ptr({arg_list});
            }}
        }}
    }}

    dlclose(handle);
    return 0;
}}
"""

    def __init__(self, src, metadata):
        self.shared_mem_dir = f"/dev/shm/"
        self.metadata = metadata
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = src.constants if hasattr(src, "constants") else dict()
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        launcher_src = self._generate_launcher_source(constants, signature)
        self.launcher_path = self._get_or_compile_launcher(launcher_src)

    def launch(self, gridX, gridY, gridZ, stream, kernel_ptr, kernel_metadata,
               launch_metadata, launch_enter_hook, launch_exit_hook, *args):
        import ctypes

        cmd = [_qemu_binary]
        target_cpu_env = os.environ.get("TRITON_CPU_TARGET_CPU", "")
        if ":" in target_cpu_env:
            cmd += ["-cpu", target_cpu_env.split(":", 1)[1]]
        qemu_lib_path = os.environ.get("TRITON_CPU_QEMU_LD_PATH")
        if qemu_lib_path:
            cmd += ["-L", qemu_lib_path]

        cmd += [self.launcher_path, kernel_ptr, getattr(self.metadata, 'name', 'kernel'),
                str(gridX), str(gridY), str(gridZ)]

        tensor_info = []
        arg_idx = 0
        for arg in args:
            # we use shared memory file to pass tensor data so that we can share
            # between qemu device and host via mmap
            if hasattr(arg, 'data_ptr'):
                storage = arg.untyped_storage()
                storage_offset_bytes = arg.storage_offset() * arg.element_size()
                nbytes = len(storage) - storage_offset_bytes
                start_ptr = storage.data_ptr() + storage_offset_bytes

                shm_name = f"triton_{os.getpid()}_{arg_idx}_{id(arg)}"
                shm_path = self.shared_mem_dir + shm_name
                with open(shm_path, "wb") as shm_f:
                    data = bytes(ctypes.cast(start_ptr, ctypes.POINTER(ctypes.c_char * nbytes)).contents)
                    shm_f.write(data)
                cmd.append(f"{nbytes}:{shm_name}")
                tensor_info.append((arg, nbytes, shm_path))
            else:
                cmd.append(str(arg) if isinstance(arg, (int, float)) else str(int(arg)))
            arg_idx += 1

        if launch_enter_hook:
            launch_enter_hook(launch_metadata)
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if launch_exit_hook:
            launch_exit_hook(launch_metadata)

        if result.returncode != 0:
            raise RuntimeError(f"QEMU execution failed: {result.stderr.decode('utf-8', errors='replace')}")

        # Read results from shared memory
        for tensor, nbytes, shm_path in tensor_info:
            with open(shm_path, "rb") as shm_f:
                data = shm_f.read(nbytes)
                storage = tensor.untyped_storage()
                storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
                start_ptr = storage.data_ptr() + storage_offset_bytes
                ctypes.memmove(start_ptr, data, nbytes)
            os.unlink(shm_path)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPULauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids)
        mod = compile_module_from_src(src, "__triton_cpu_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = QEMULauncher if _qemu_mode else CPULauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu", self.get_current_device())

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        target_cpu_env = os.getenv("TRITON_CPU_TARGET_CPU", "")
        if ":" in target_cpu_env:
            cpu_arch = target_cpu_env.split(":")[0]
        return GPUTarget("cpu", cpu_arch, 0)

    def get_device_interface(self):
        return CPUDeviceInterface()

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        def do_bench_cpu(*args, **kwargs):
            if not 'measure_time_with_hooks' in kwargs:
                kwargs['measure_time_with_hooks'] = True
            return do_bench(*args, **kwargs)

        return do_bench_cpu

    def get_empty_cache_for_benchmark(self):
        import torch

        # A typical LLC size for high-end server CPUs are ~400MB.
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    # TODO maybe CPU should do anything here
    def clear_cache(self, cache):
        cache.zero_()
