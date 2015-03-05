/*
Copyright 2014 Google Inc. All rights reserved.

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

func GetAccessModesAsString(modes []api.AccessModeType) string {

	modesAsString := ""

	if contains(modes, api.ReadWriteOnce) {
		modesAsString += "RWO"
	}
	if contains(modes, api.ReadOnlyMany) {
		modesAsString += "ROX"
	}
	if contains(modes, api.ReadWriteMany) {
		modesAsString += "RWX"
	}

	return modesAsString
}

func contains(modes []api.AccessModeType, mode api.AccessModeType) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func GetAccessModeType(source api.PersistentVolumeSource) []api.AccessModeType {

	if source.AWSElasticBlockStore != nil || source.HostPath != nil {
		return []api.AccessModeType{api.ReadWriteOnce}
	}

	if source.GCEPersistentDisk != nil {
		return []api.AccessModeType{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
		}
	}

	if source.NFSMount != nil {
		return []api.AccessModeType{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
			api.ReadWriteMany,
		}
	}

	return []api.AccessModeType{}
}
