/*
Copyright 2022 The Kubernetes Authors.

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

package install

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/quota/v1/evaluator/core"
	"k8s.io/utils/clock"
)

// DefaultUpdateFilter returns the default update filter for resource update events for consideration for quota.
func DefaultUpdateFilter() func(resource schema.GroupVersionResource, oldObj, newObj interface{}) bool {
	return func(resource schema.GroupVersionResource, oldObj, newObj interface{}) bool {
		switch resource.GroupResource() {
		case schema.GroupResource{Resource: "pods"}:
			oldPod := oldObj.(*v1.Pod)
			newPod := newObj.(*v1.Pod)
			return core.QuotaV1Pod(oldPod, clock.RealClock{}) && !core.QuotaV1Pod(newPod, clock.RealClock{})
		case schema.GroupResource{Resource: "services"}:
			oldService := oldObj.(*v1.Service)
			newService := newObj.(*v1.Service)
			return core.GetQuotaServiceType(oldService) != core.GetQuotaServiceType(newService)
		case schema.GroupResource{Resource: "persistentvolumeclaims"}:
			oldPVC := oldObj.(*v1.PersistentVolumeClaim)
			newPVC := newObj.(*v1.PersistentVolumeClaim)
			return core.RequiresQuotaReplenish(newPVC, oldPVC)
		}

		return false
	}
}
