/*
Copyright 2016 The Kubernetes Authors.

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

package core

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/clock"
)

// legacyObjectCountAliases are what we used to do simple object counting quota with mapped to alias
var legacyObjectCountAliases = map[schema.GroupVersionResource]corev1.ResourceName{
	corev1.SchemeGroupVersion.WithResource("configmaps"):             corev1.ResourceConfigMaps,
	corev1.SchemeGroupVersion.WithResource("resourcequotas"):         corev1.ResourceQuotas,
	corev1.SchemeGroupVersion.WithResource("replicationcontrollers"): corev1.ResourceReplicationControllers,
	corev1.SchemeGroupVersion.WithResource("secrets"):                corev1.ResourceSecrets,
}

// NewEvaluators returns the list of static evaluators that manage more than counts
func NewEvaluators(f quota.ListerForResourceFunc, i informers.SharedInformerFactory) ([]quota.Evaluator, error) {
	// these evaluators have special logic
	result := []quota.Evaluator{
		NewPodEvaluator(f, clock.RealClock{}),
		NewServiceEvaluator(f),
		NewPersistentVolumeClaimEvaluator(f),
	}
	var claimGetter resourceClaimPodOwnerGetter
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
		var podLister corev1listers.PodLister
		var deviceClassMapping *extendedresourcecache.ExtendedResourceCache
		if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
			podLister = i.Core().V1().Pods().Lister()
			logger := klog.FromContext(context.Background())
			deviceClassMapping = extendedresourcecache.NewExtendedResourceCache(logger)
			if _, err := i.Resource().V1().DeviceClasses().Informer().AddEventHandler(deviceClassMapping); err != nil {
				return nil, fmt.Errorf("failed to add device class informer event handler: %w", err)
			}
			var err error
			claimGetter, err = makeResourceClaimPodOwnerGetter(i.Resource().V1().ResourceClaims())
			if err != nil {
				return nil, err
			}
		}
		result = append(result, NewResourceClaimEvaluator(f, deviceClassMapping, podLister, claimGetter))
	}

	// these evaluators require an alias for backwards compatibility
	for gvr, alias := range legacyObjectCountAliases {
		result = append(result,
			generic.NewObjectCountEvaluator(gvr.GroupResource(), generic.ListResourceUsingListerFunc(f, gvr), alias))
	}
	return result, nil
}
