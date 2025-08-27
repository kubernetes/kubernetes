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
	"context"
	"sort"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

func DeviceClassMapping(draManager framework.SharedDRAManager) (map[v1.ResourceName]string, error) {
	classes, err := draManager.DeviceClasses().List()
	if err != nil {
		return nil, err
	}
	resClasses := make(map[v1.ResourceName][]*resourceapi.DeviceClass, len(classes))
	for _, c := range classes {
		name := v1.ResourceName(resourceapi.ResourceDeviceClassPrefix + c.Name)
		if c.Spec.ExtendedResourceName != nil {
			name = v1.ResourceName(*c.Spec.ExtendedResourceName)
		}
		cls := resClasses[name]
		cls = append(cls, c)
		resClasses[v1.ResourceName(name)] = cls
	}
	extendedResources := make(map[v1.ResourceName]string, len(resClasses))
	for name, cls := range resClasses {
		// Primary sort by creation timestamp, newest first
		// Secondary sort by class name, ascending order
		sort.Slice(cls, func(i, j int) bool {
			if cls[i].CreationTimestamp.UnixNano() == cls[j].CreationTimestamp.UnixNano() {
				return cls[i].Name < cls[j].Name
			}
			return cls[i].CreationTimestamp.UnixNano() > cls[j].CreationTimestamp.UnixNano()
		})
		if len(cls) > 1 {
			klog.FromContext(context.Background()).V(5).Info("Device classes found", "total", len(cls), "name", cls[0].Name)
		}
		extendedResources[name] = cls[0].Name
	}
	return extendedResources, nil
}
