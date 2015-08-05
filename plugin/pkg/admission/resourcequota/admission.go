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

package resourcequota

import (
	"fmt"
	"io"
	"math/rand"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/resourcequota"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

func init() {
	admission.RegisterPlugin("ResourceQuota", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewResourceQuota(client), nil
	})
}

type quota struct {
	*admission.Handler
	client  client.Interface
	indexer cache.Indexer
}

// NewResourceQuota creates a new resource quota admission control handler
func NewResourceQuota(client client.Interface) admission.Interface {
	lw := &cache.ListWatch{
		ListFunc: func() (runtime.Object, error) {
			return client.ResourceQuotas(api.NamespaceAll).List(labels.Everything())
		},
		WatchFunc: func(resourceVersion string) (watch.Interface, error) {
			return client.ResourceQuotas(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), resourceVersion)
		},
	}
	indexer, reflector := cache.NewNamespaceKeyedIndexerAndReflector(lw, &api.ResourceQuota{}, 0)
	reflector.Run()
	return createResourceQuota(client, indexer)
}

func createResourceQuota(client client.Interface, indexer cache.Indexer) admission.Interface {
	return &quota{
		Handler: admission.NewHandler(admission.Create, admission.Update),
		client:  client,
		indexer: indexer,
	}
}

var resourceToResourceName = map[string]api.ResourceName{
	"pods":                   api.ResourcePods,
	"services":               api.ResourceServices,
	"replicationcontrollers": api.ResourceReplicationControllers,
	"resourcequotas":         api.ResourceQuotas,
	"secrets":                api.ResourceSecrets,
	"persistentvolumeclaims": api.ResourcePersistentVolumeClaims,
}

func (q *quota) Admit(a admission.Attributes) (err error) {
	if a.GetSubresource() != "" {
		return nil
	}

	if a.GetOperation() == "DELETE" {
		return nil
	}

	key := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Namespace: a.GetNamespace(),
			Name:      "",
		},
	}

	// concurrent operations that modify quota tracked resources can cause a conflict when incrementing usage
	// as a result, we will attempt to increment quota usage per request up to numRetries limit
	// we fuzz each retry with an interval period to attempt to improve end-user experience during concurrent operations
	numRetries := 10
	interval := time.Duration(rand.Int63n(90)+int64(10)) * time.Millisecond

	items, err := q.indexer.Index("namespace", key)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("Unable to %s %s at this time because there was an error enforcing quota", a.GetOperation(), a.GetResource()))
	}
	if len(items) == 0 {
		return nil
	}

	for i := range items {

		quota := items[i].(*api.ResourceQuota)

		for retry := 1; retry <= numRetries; retry++ {

			// we cannot modify the value directly in the cache, so we copy
			status := &api.ResourceQuotaStatus{
				Hard: api.ResourceList{},
				Used: api.ResourceList{},
			}
			for k, v := range quota.Status.Hard {
				status.Hard[k] = *v.Copy()
			}
			for k, v := range quota.Status.Used {
				status.Used[k] = *v.Copy()
			}

			dirty, err := IncrementUsage(a, status, q.client)
			if err != nil {
				return admission.NewForbidden(a, err)
			}

			if dirty {
				// construct a usage record
				usage := api.ResourceQuota{
					ObjectMeta: api.ObjectMeta{
						Name:            quota.Name,
						Namespace:       quota.Namespace,
						ResourceVersion: quota.ResourceVersion,
						Labels:          quota.Labels,
						Annotations:     quota.Annotations},
				}
				usage.Status = *status
				_, err = q.client.ResourceQuotas(usage.Namespace).UpdateStatus(&usage)
				if err == nil {
					break
				}

				// we have concurrent requests to update quota, so look to retry if needed
				if retry == numRetries {
					return admission.NewForbidden(a, fmt.Errorf("Unable to %s %s at this time because there are too many concurrent requests to increment quota", a.GetOperation(), a.GetResource()))
				}
				time.Sleep(interval)
				// manually get the latest quota
				quota, err = q.client.ResourceQuotas(usage.Namespace).Get(quota.Name)
				if err != nil {
					return admission.NewForbidden(a, err)
				}
			}
		}
	}
	return nil
}

// IncrementUsage updates the supplied ResourceQuotaStatus object based on the incoming operation
// Return true if the usage must be recorded prior to admitting the new resource
// Return an error if the operation should not pass admission control
func IncrementUsage(a admission.Attributes, status *api.ResourceQuotaStatus, client client.Interface) (bool, error) {
	dirty := false
	set := map[api.ResourceName]bool{}
	for k := range status.Hard {
		set[k] = true
	}
	obj := a.GetObject()
	// handle max counts for each kind of resource (pods, services, replicationControllers, etc.)
	if a.GetOperation() == admission.Create {
		resourceName := resourceToResourceName[a.GetResource()]
		hard, hardFound := status.Hard[resourceName]
		if hardFound {
			used, usedFound := status.Used[resourceName]
			if !usedFound {
				return false, fmt.Errorf("Quota usage stats are not yet known, unable to admit resource until an accurate count is completed.")
			}
			if used.Value() >= hard.Value() {
				return false, fmt.Errorf("Limited to %s %s", hard.String(), resourceName)
			} else {
				status.Used[resourceName] = *resource.NewQuantity(used.Value()+int64(1), resource.DecimalSI)
				dirty = true
			}
		}
	}
	// handle memory/cpu constraints, and any diff of usage based on memory/cpu on updates
	if a.GetResource() == "pods" && (set[api.ResourceMemory] || set[api.ResourceCPU]) {
		pod := obj.(*api.Pod)
		deltaCPU := resourcequotacontroller.PodCPU(pod)
		deltaMemory := resourcequotacontroller.PodMemory(pod)
		// if this is an update, we need to find the delta cpu/memory usage from previous state
		if a.GetOperation() == admission.Update {
			oldPod, err := client.Pods(a.GetNamespace()).Get(pod.Name)
			if err != nil {
				return false, err
			}
			oldCPU := resourcequotacontroller.PodCPU(oldPod)
			oldMemory := resourcequotacontroller.PodMemory(oldPod)
			deltaCPU = resource.NewMilliQuantity(deltaCPU.MilliValue()-oldCPU.MilliValue(), resource.DecimalSI)
			deltaMemory = resource.NewQuantity(deltaMemory.Value()-oldMemory.Value(), resource.DecimalSI)
		}

		hardMem, hardMemFound := status.Hard[api.ResourceMemory]
		if hardMemFound {
			if set[api.ResourceMemory] && resourcequotacontroller.IsPodMemoryUnbounded(pod) {
				return false, fmt.Errorf("Limited to %s memory, but pod has no specified memory limit", hardMem.String())
			}
			used, usedFound := status.Used[api.ResourceMemory]
			if !usedFound {
				return false, fmt.Errorf("Quota usage stats are not yet known, unable to admit resource until an accurate count is completed.")
			}
			if used.Value()+deltaMemory.Value() > hardMem.Value() {
				return false, fmt.Errorf("Limited to %s memory", hardMem.String())
			} else {
				status.Used[api.ResourceMemory] = *resource.NewQuantity(used.Value()+deltaMemory.Value(), resource.DecimalSI)
				dirty = true
			}
		}
		hardCPU, hardCPUFound := status.Hard[api.ResourceCPU]
		if hardCPUFound {
			if set[api.ResourceCPU] && resourcequotacontroller.IsPodCPUUnbounded(pod) {
				return false, fmt.Errorf("Limited to %s CPU, but pod has no specified cpu limit", hardCPU.String())
			}
			used, usedFound := status.Used[api.ResourceCPU]
			if !usedFound {
				return false, fmt.Errorf("Quota usage stats are not yet known, unable to admit resource until an accurate count is completed.")
			}
			if used.MilliValue()+deltaCPU.MilliValue() > hardCPU.MilliValue() {
				return false, fmt.Errorf("Limited to %s CPU", hardCPU.String())
			} else {
				status.Used[api.ResourceCPU] = *resource.NewMilliQuantity(used.MilliValue()+deltaCPU.MilliValue(), resource.DecimalSI)
				dirty = true
			}
		}
	}
	return dirty, nil
}
