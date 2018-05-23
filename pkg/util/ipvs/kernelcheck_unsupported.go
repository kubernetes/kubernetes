// +build !linux

/*
Copyright 2018 The Kubernetes Authors.

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

package ipvs

import (
	utilsexec "k8s.io/utils/exec"
)

// RequiredIPVSKernelModulesAvailableCheck tests IPVS required kernel modules.
type RequiredIPVSKernelModulesAvailableCheck struct {
	Executor utilsexec.Interface
}

// Name returns label for RequiredIPVSKernelModulesAvailableCheck
func (r RequiredIPVSKernelModulesAvailableCheck) Name() string {
	return "RequiredIPVSKernelModulesAvailable"
}

// Check try to validates IPVS required kernel modules exists or not.
func (r RequiredIPVSKernelModulesAvailableCheck) Check() (warnings, errors []error) {

	return nil, nil
}
