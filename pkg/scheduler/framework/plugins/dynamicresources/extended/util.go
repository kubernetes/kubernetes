/*
Copyright 2025 The Kubernetes Authors.

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

package extended

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/api/resource/v1beta1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

func DeviceClassMapping(draManager framework.SharedDRAManager) (map[v1.ResourceName]string, error) {
	classes, err := draManager.DeviceClasses().List()
	extendedResources := make(map[v1.ResourceName]string, len(classes))
	if err != nil {
		return nil, err
	}
	for _, c := range classes {
		if c.Spec.ExtendedResourceName == nil {
			extendedResources[v1.ResourceName(v1beta1.ResourceDeviceClassPrefix+c.Name)] = c.Name
		} else {
			extendedResources[v1.ResourceName(*c.Spec.ExtendedResourceName)] = c.Name
		}
	}
	return extendedResources, nil
}
