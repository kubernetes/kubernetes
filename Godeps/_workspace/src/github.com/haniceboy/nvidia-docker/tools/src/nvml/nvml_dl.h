// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

#ifndef _NVML_DL_H_
#define _NVML_DL_H_

#include <nvml.h>

#define NVML_DL(x) x##_dl

extern char *NVML_DL(nvmlInit)(void);
extern void NVML_DL(nvmlShutdown)(void);
extern nvmlReturn_t NVML_DL(nvmlDeviceGetTopologyCommonAncestor)(
  nvmlDevice_t, nvmlDevice_t, nvmlGpuTopologyLevel_t *);

#endif // _NVML_DL_H_
