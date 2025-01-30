/*
Copyright 2020 The Kubernetes Authors.

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

package v1

// APILifecycleIntroduced returns the release in which the API struct was introduced as int versions of major and minor for comparison.
func (in *ComponentStatus) APILifecycleIntroduced() (major, minor int) {
	return 1, 0
}

// APILifecycleDeprecated returns the release in which the API struct was or will be deprecated as int versions of major and minor for comparison.
func (in *ComponentStatus) APILifecycleDeprecated() (major, minor int) {
	return 1, 19
}

// APILifecycleIntroduced returns the release in which the API struct was introduced as int versions of major and minor for comparison.
func (in *ComponentStatusList) APILifecycleIntroduced() (major, minor int) {
	return 1, 0
}

// APILifecycleDeprecated returns the release in which the API struct was or will be deprecated as int versions of major and minor for comparison.
func (in *ComponentStatusList) APILifecycleDeprecated() (major, minor int) {
	return 1, 19
}
