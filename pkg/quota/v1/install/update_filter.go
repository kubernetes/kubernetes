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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
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
			// when Resources changed
			if feature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) && hasResourcesChanged(oldPod, newPod) {
				return true
			}

			// when scope changed
			if core.IsTerminating(oldPod) != core.IsTerminating(newPod) {
				return true
			}

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

// hasResourcesChanged function to compare resources in container statuses.
// It iterates over newPod statuses to detect both changes to existing resources
// and the initial population of resources (when oldPod has no container statuses yet).
func hasResourcesChanged(oldPod *v1.Pod, newPod *v1.Pod) bool {
	if containerStatusResourcesChanged(oldPod.Status.ContainerStatuses, newPod.Status.ContainerStatuses) {
		return true
	}
	if containerStatusResourcesChanged(oldPod.Status.InitContainerStatuses, newPod.Status.InitContainerStatuses) {
		return true
	}
	return false
}

// containerStatusResourcesChanged returns true if any container's Resources
// field changed between oldStatuses and newStatuses. It iterates over newStatuses
// to detect both modifications and the initial population of Resources
// (e.g., when a newly-created pod receives its first Kubelet status update).
func containerStatusResourcesChanged(oldStatuses, newStatuses []v1.ContainerStatus) bool {
	oldStatusMap := make(map[string]*v1.ContainerStatus, len(oldStatuses))
	for i := range oldStatuses {
		oldStatusMap[oldStatuses[i].Name] = &oldStatuses[i]
	}
	for i := range newStatuses {
		newStatus := &newStatuses[i]
		oldStatus, found := oldStatusMap[newStatus.Name]
		if !found {
			// Container status is new; if it has Resources populated, resources changed.
			if newStatus.Resources != nil {
				return true
			}
			continue
		}
		if !apiequality.Semantic.DeepEqual(oldStatus.Resources, newStatus.Resources) {
			return true
		}
	}
	return false
}
