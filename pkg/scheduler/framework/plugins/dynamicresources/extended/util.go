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
	resourceapi "k8s.io/api/resource/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// deviceClassCmp returns true when classi is created later than classj, or
// when they are created at the same time, classi's Name is lexicographically sorted before classj's Name.
// This helps the transition of extended resource name from one device class to another.
func deviceClassCmp(classi, classj *resourceapi.DeviceClass) bool {
	if classi.CreationTimestamp.Equal(&classj.CreationTimestamp) {
		return classi.Name < classj.Name
	}
	return classj.CreationTimestamp.Before(&classi.CreationTimestamp)
}

func preferableClass(newClass *resourceapi.DeviceClass, mapping map[v1.ResourceName]*resourceapi.DeviceClass, name v1.ResourceName) *resourceapi.DeviceClass {
	if class, ok := mapping[name]; ok {
		if deviceClassCmp(newClass, class) {
			return newClass
		}
		return class
	}
	return newClass
}

// DeviceClassMapping creates the mapping of extended resource name to device class name.
// It always includes the implicit extended resource name for each device class.
// The device class MUST NOT BE modified.
func DeviceClassMapping(draManager fwk.SharedDRAManager) (map[v1.ResourceName]*resourceapi.DeviceClass, error) {
	classes, err := draManager.DeviceClasses().List()
	if err != nil {
		return nil, err
	}
	mapping := make(map[v1.ResourceName]*resourceapi.DeviceClass, len(classes))
	for _, c := range classes {
		// implicit extended resource name
		name := v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + c.Name)
		mapping[name] = c

		// explicit extended resource name, if any
		if c.Spec.ExtendedResourceName != nil {
			name = v1.ResourceName(*c.Spec.ExtendedResourceName)
			mapping[name] = preferableClass(c, mapping, name)
		}
	}
	return mapping, nil
}
