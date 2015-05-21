/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package volume

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func GetAccessModesAsString(modes []api.PersistentVolumeAccessMode) string {
	modesAsString := ""

	if contains(modes, api.ReadWriteOnce) {
		appendAccessMode(&modesAsString, "RWO")
	}
	if contains(modes, api.ReadOnlyMany) {
		appendAccessMode(&modesAsString, "ROX")
	}
	if contains(modes, api.ReadWriteMany) {
		appendAccessMode(&modesAsString, "RWX")
	}

	return modesAsString
}

func appendAccessMode(modes *string, mode string) {
	if *modes != "" {
		*modes += ","
	}
	*modes += mode
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}
