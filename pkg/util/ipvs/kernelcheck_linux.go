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

	"github.com/lithammer/dedent"
	"k8s.io/klog"
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
	klog.V(1).Infoln("validating the kernel module IPVS required exists in machine or not")

	kernelVersion, ipvsModules, err := GetKernelVersionAndIPVSMods(r.Executor)
	if err != nil {
		errors = append(errors, err)
	}

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
			warnings = append(warnings, fmt.Errorf(dedent.Dedent(`

				The IPVS proxier may not be used because the following required kernel modules are not loaded: %v
				or no builtin kernel IPVS support was found: %v.
				However, these modules may be loaded automatically by kube-proxy if they are available on your system.
				To verify IPVS support:

				   Run "lsmod | grep 'ip_vs|nf_conntrack'" and verify each of the above modules are listed.

				If they are not listed, you can use the following methods to load them:

				1. For each missing module run 'modprobe $modulename' (e.g., 'modprobe ip_vs', 'modprobe ip_vs_rr', ...)
				2. If 'modprobe $modulename' returns an error, you will need to install the missing module support for your kernel.
				`), modules, builtInModules))
		}
	}

	return warnings, errors
}
