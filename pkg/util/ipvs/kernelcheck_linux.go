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

	pkgerrors "github.com/pkg/errors"
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
		return nil, errors
	}

	// Find out loaded kernel modules
	out, err := r.Executor.Command("cut", "-f1", "-d", " ", "/proc/modules").CombinedOutput()
	if err != nil {
		errors = append(errors, pkgerrors.Wrapf(err, "error getting installed ipvs required kernel modules: %s", out))
		return nil, errors
	}
	mods := strings.Split(string(out), "\n")

	wantModules := sets.NewString()
	loadModules := sets.NewString()
	wantModules.Insert(ipvsModules...)
	loadModules.Insert(mods...)
	modules := wantModules.Difference(loadModules).UnsortedList()

	if len(modules) == 0 {
		return nil, nil
	}

	// Check if loadable modules are installed
	var missingModules []string
	for _, module := range modules {
		out, err = r.Executor.Command("modinfo", module).CombinedOutput()
		if err != nil {
			warnings = append(warnings, pkgerrors.Wrapf(err, "error getting module info for %q; output: %s", module, strings.TrimSpace(string(out))))
			missingModules = append(missingModules, module)
		}
	}

	if len(missingModules) == 0 {
		return warnings, nil
	}

	// Check if builtin modules exist
	builtinModsFilePath := fmt.Sprintf("/lib/modules/%s/modules.builtin", kernelVersion)
	out, err = r.Executor.Command("cut", "-f1", "-d", " ", builtinModsFilePath).CombinedOutput()
	if err != nil {
		errors = append(errors, pkgerrors.Wrapf(err, "error getting required builtin kernel modules: %s", out))
		return warnings, errors
	}

	var missingBuiltins []string
	for _, module := range missingModules {
		match, _ := regexp.Match(module+".ko", out)
		if !match {
			missingBuiltins = append(missingBuiltins, module)
		}
	}

	if len(missingBuiltins) != 0 {
		warnings = append(warnings, pkgerrors.Errorf(
			"the IPVS proxier will not be used, because the following required kernel modules are not loaded or found: %v\n"+
				"You can solve this problem by providing builtin or loadable kernel IPVS support\n", missingBuiltins))
		return warnings, nil
	}

	return warnings, nil
}
