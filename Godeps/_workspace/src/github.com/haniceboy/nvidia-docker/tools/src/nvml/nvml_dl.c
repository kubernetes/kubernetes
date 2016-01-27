// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

#include <stddef.h>
#include <dlfcn.h>

#include "nvml_dl.h"

#define DLSYM(x, sym)                         \
do {                                          \
    x = dlsym(handle, #sym);                  \
    if (dlerror() != NULL) {                  \
        return NVML_ERROR_FUNCTION_NOT_FOUND; \
    }                                         \
} while (0)

typedef nvmlReturn_t (*nvmlSym_t)();

static void *handle;

char *NVML_DL(nvmlInit)(void)
{
    handle = dlopen(NULL, RTLD_NOW);
    return (handle ? NULL : dlerror());
}

void NVML_DL(nvmlShutdown)(void)
{
    dlclose(handle);
}

nvmlReturn_t NVML_DL(nvmlDeviceGetTopologyCommonAncestor)(
  nvmlDevice_t dev1, nvmlDevice_t dev2, nvmlGpuTopologyLevel_t *info)
{
    nvmlSym_t sym;

    DLSYM(sym, nvmlDeviceGetTopologyCommonAncestor);
    return (*sym)(dev1, dev2, info);
}
