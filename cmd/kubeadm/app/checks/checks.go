/*
Copyright 2016 The Kubernetes Authors.

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

package checks

import (
	"fmt"
	"os"
)

// PreFlightCheckers validate the state of the system to ensure kubeadm will be
// successful as often as possilble.
type PreFlightChecker interface {
	Check() (warnings []string, errors []string)
}

// KubeletServiceChecker verifies that if this is a systemd system, the
// kubelet service exists, is enabled, and running.  We do not want to directly
// assume systemd is being used, so if this does not appear to be the case we
// just silently skip these checks and return no warnings or errors.
type KubeletServiceChecker struct {
	initSystem InitSystem
}

func (ksc KubeletServiceChecker) Check() (warnings []string, errors []string) {

	warnings = []string{}

	// If we weren't able to detect a supported init system, warn and exit.
	if ksc.initSystem == nil {
		warnings = append(warnings, "no kubeadm supported init system detected, skipping service checks")
		return warnings, nil
	}

	if !ksc.initSystem.ServiceExists("kubelet") {
		// TODO: Should this be a hard error?
		warnings = append(warnings, "kubelet service does not exist")
		return warnings, nil
	}

	if !ksc.initSystem.ServiceIsEnabled("kubelet") {
		// TODO: Should we enable it?
		warnings = append(warnings, "kubelet service is not enabled, please run 'systemctl enable kubelet'")
	}

	if !ksc.initSystem.ServiceIsActive("kubelet") {
		// TODO: Should we start it? This might count as a hard error, we know service exists here.
		errors = append(errors, "kubelet service is not active, please run 'systemctl start kubelet'")
	}

	return warnings, nil
}

func RunMasterChecks() {
	checks := []PreFlightChecker{
		KubeletServiceChecker{initSystem: getInitSystem()},
	}
	foundErrors := false
	for _, check := range checks {
		fmt.Printf("Running check %s\n", check)

		warnings, errors := check.Check()
		for _, warnMsg := range warnings {
			fmt.Printf("WARNING: %s\n", warnMsg)
		}
		for _, errMsg := range errors {
			foundErrors = true
			fmt.Printf("ERROR: %s\n", errMsg)
		}
	}
	if foundErrors {
		os.Exit(1)
	}
}
