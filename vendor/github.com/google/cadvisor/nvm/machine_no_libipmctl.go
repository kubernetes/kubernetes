// +build !libipmctl !cgo

// Copyright 2020 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nvm

import (
	info "github.com/google/cadvisor/info/v1"
	"k8s.io/klog/v2"
)

// GetInfo returns information specific for non-volatile memory modules.
// When libipmctl is not available zero value is returned.
func GetInfo() (info.NVMInfo, error) {
	return info.NVMInfo{}, nil
}

// Finalize un-initializes libipmctl. See https://github.com/google/cadvisor/issues/2457.
// When libipmctl is not available it just logs that it's being called.
func Finalize() {
	klog.V(4).Info("libimpctl not available, doing nothing.")
}
