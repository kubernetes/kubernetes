/*
Copyright 2024 The Kubernetes Authors.

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

package resourceupdates

// Update is a struct that represents an update to a pod when
// the resource changes it's status.
// Later we may need to add fields like container name, resource name, and a new status.
type Update struct {
	// PodUID is the UID of the pod which status needs to be updated.
	PodUIDs []string
}
