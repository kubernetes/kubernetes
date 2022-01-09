/*/*
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

package apparmor

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/opencontainers/runc/libcontainer/apparmor"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	utilpath "k8s.io/utils/path"
)

// Whether AppArmor should be disabled by default.
// Set to true if the wrong build tags are set (see validate_disabled.go).
var isDisabledBuild bool

// Validator is a interface for validating that a pod with an AppArmor profile can be run by a Node.
type Validator interface {
	Validate(pod *v1.Pod) error
	ValidateHost() error
}

// NewValidator is in order to find AppArmor FS
func NewValidator() Validator {
	if err := validateHost(); err != nil {
		return &validator{validateHostErr: err}
	}
	appArmorFS, err := getAppArmorFS()
	if err != nil {
		return &validator{
			validateHostErr: fmt.Errorf("error finding AppArmor FS: %v", err),
		}
	}
	return &validator{
		appArmorFS: appArmorFS,
	}
}

type validator struct {
	validateHostErr error
	appArmorFS      string
}

func (v *validator) Validate(pod *v1.Pod) error {
	if !isRequired(pod) {
		return nil
	}

	if v.ValidateHost() != nil {
		return v.validateHostErr
	}

	var retErr error
	podutil.VisitContainers(&pod.Spec, podutil.AllContainers, func(container *v1.Container, containerType podutil.ContainerType) bool {
		retErr = ValidateProfileFormat(GetProfileName(pod, container.Name))
		if retErr != nil {
			return false
		}
		return true
	})

	return retErr
}

func (v *validator) ValidateHost() error {
	return v.validateHostErr
}

// Verify that the host and runtime is capable of enforcing AppArmor profiles.
func validateHost() error {
	// Check feature-gates
	if !utilfeature.DefaultFeatureGate.Enabled(features.AppArmor) {
		return errors.New("AppArmor disabled by feature-gate")
	}

	// Check build support.
	if isDisabledBuild {
		return errors.New("binary not compiled for linux")
	}

	// Check kernel support.
	if !apparmor.IsEnabled() {
		return errors.New("AppArmor is not enabled on the host")
	}

	return nil
}

// ValidateProfileFormat checks the format of the profile.
func ValidateProfileFormat(profile string) error {
	if profile == "" || profile == v1.AppArmorBetaProfileRuntimeDefault || profile == v1.AppArmorBetaProfileNameUnconfined {
		return nil
	}
	if !strings.HasPrefix(profile, v1.AppArmorBetaProfileNamePrefix) {
		return fmt.Errorf("invalid AppArmor profile name: %q", profile)
	}
	return nil
}

func getAppArmorFS() (string, error) {
	mountsFile, err := os.Open("/proc/mounts")
	if err != nil {
		return "", fmt.Errorf("could not open /proc/mounts: %v", err)
	}
	defer mountsFile.Close()

	scanner := bufio.NewScanner(mountsFile)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 3 {
			// Unknown line format; skip it.
			continue
		}
		if fields[2] == "securityfs" {
			appArmorFS := path.Join(fields[1], "apparmor")
			if ok, err := utilpath.Exists(utilpath.CheckFollowSymlink, appArmorFS); !ok {
				msg := fmt.Sprintf("path %s does not exist", appArmorFS)
				if err != nil {
					return "", fmt.Errorf("%s: %v", msg, err)
				}
				return "", errors.New(msg)
			}
			return appArmorFS, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error scanning mounts: %v", err)
	}

	return "", errors.New("securityfs not found")
}
