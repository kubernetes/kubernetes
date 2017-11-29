// +build ignore

/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package nvme

// Commented out because otherwise bazel regenerates, which requires linux headers
//
//// #include "linux/nvme_ioctl.h"
//import "C"
//
//type nvmeAdminCmd C.struct_nvme_passthru_cmd
//
//const sizeof_nvmeAdminCmd = C.sizeof_struct_nvme_admin_cmd
