// +build linux

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
	"fmt"
	"regexp"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	utilsexec "k8s.io/utils/exec"

	"github.com/golang/glog"
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
// The name of function can not be changed.
func (r RequiredIPVSKernelModulesAvailableCheck) Check() (warnings, errors []error) {
	glog.V(1).Infoln("validating the kernel module IPVS required exists in machine or not")

	// Find out loaded kernel modules
	out, err := r.Executor.Command("cut", "-f1", "-d", " ", "/proc/modules").CombinedOutput()
	if err != nil {
		errors = append(errors, fmt.Errorf("error getting installed ipvs required kernel modules: %v(%s)", err, out))
		return nil, errors
	}
	mods := strings.Split(string(out), "\n")

	wantModules := sets.NewString()
	loadModules := sets.NewString()
	wantModules.Insert(ipvsModules...)
	loadModules.Insert(mods...)
	modules := wantModules.Difference(loadModules).UnsortedList()

	// Check builtin modules exist or not
	if len(modules) != 0 {
		kernelVersionFile := "/proc/sys/kernel/osrelease"
		b, err := r.Executor.Command("cut", "-f1", "-d", " ", kernelVersionFile).CombinedOutput()
		if err != nil {
			errors = append(errors, fmt.Errorf("error getting os release kernel version: %v(%s)", err, out))
			return nil, errors
		}

		kernelVersion := strings.TrimSpace(string(b))
		builtinModsFilePath := fmt.Sprintf("/lib/modules/%s/modules.builtin", kernelVersion)
		out, err := r.Executor.Command("cut", "-f1", "-d", " ", builtinModsFilePath).CombinedOutput()
		if err != nil {
			errors = append(errors, fmt.Errorf("error getting required builtin kernel modules: %v(%s)", err, out))
			return nil, errors
		}

		builtInModules := sets.NewString()
		for _, builtInMode := range ipvsModules {
			match, _ := regexp.Match(builtInMode+".ko", out)
			if !match {
				builtInModules.Insert(string(builtInMode))
			}
		}
		if len(builtInModules) != 0 {
			warnings = append(warnings, fmt.Errorf(
				"the IPVS proxier will not be used, because the following required kernel modules are not loaded: %v or no builtin kernel ipvs support: %v\n"+
					"you can solve this problem with following methods:\n 1. Run 'modprobe -- ' to load missing kernel modules;\n"+
					"2. Provide the missing builtin kernel ipvs support\n", modules, builtInModules))
		}
	}

	return warnings, errors
}
