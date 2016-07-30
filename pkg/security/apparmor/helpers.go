/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

import "k8s.io/kubernetes/pkg/api"

// Values used by function stubs for testing.
var (
	isRequiredStub     = false
	getProfileNameStub = ""
)

// Checks whether app armor is required for pod to be run.
func isRequired(pod *api.Pod) bool {
	// TODO: Replace this stub once an AppArmor API is merged.
	return isRequiredStub
}

// Returns the name of the profile to use with the container.
func getProfileName(container *api.Container) string {
	// TODO: Replace this stub once an AppArmor API is merged.
	return getProfileNameStub
}
