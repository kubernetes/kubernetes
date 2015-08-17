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

package limitranger

import (
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

func init() {
	admission.RegisterPlugin("LimitRanger", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewLimitRanger(client, Limit), nil
	})
}

// limitRanger enforces usage limits on a per resource basis in the namespace
type limitRanger struct {
	*admission.Handler
	client    client.Interface
	limitFunc LimitFunc
	indexer   cache.Indexer
}

// Admit admits resources into cluster that do not violate any defined LimitRange in the namespace
func (l *limitRanger) Admit(a admission.Attributes) (err error) {

	// Ignore all calls to subresources
	if a.GetSubresource() != "" {
		return nil
	}

	obj := a.GetObject()
	resource := a.GetResource()
	name := "Unknown"
	if obj != nil {
		name, _ = meta.NewAccessor().Name(obj)
		if len(name) == 0 {
			name, _ = meta.NewAccessor().GenerateName(obj)
		}
	}

	key := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Namespace: a.GetNamespace(),
			Name:      "",
		},
	}
	items, err := l.indexer.Index("namespace", key)
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("Unable to %s %s at this time because there was an error enforcing limit ranges", a.GetOperation(), resource))
	}
	if len(items) == 0 {
		return nil
	}

	// ensure it meets each prescribed min/max
	for i := range items {
		limitRange := items[i].(*api.LimitRange)
		err = l.limitFunc(limitRange, a.GetResource(), a.GetObject())
		if err != nil {
			return admission.NewForbidden(a, err)
		}
	}
	return nil
}

// NewLimitRanger returns an object that enforces limits based on the supplied limit function
func NewLimitRanger(client client.Interface, limitFunc LimitFunc) admission.Interface {
	lw := &cache.ListWatch{
		ListFunc: func() (runtime.Object, error) {
			return client.LimitRanges(api.NamespaceAll).List(labels.Everything())
		},
		WatchFunc: func(resourceVersion string) (watch.Interface, error) {
			return client.LimitRanges(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), resourceVersion)
		},
	}
	indexer, reflector := cache.NewNamespaceKeyedIndexerAndReflector(lw, &api.LimitRange{}, 0)
	reflector.Run()
	return &limitRanger{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		client:    client,
		limitFunc: limitFunc,
		indexer:   indexer,
	}
}

// Min returns the lesser of its 2 arguments
func Min(a int64, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// Max returns the greater of its 2 arguments
func Max(a int64, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// Limit enforces resource requirements of incoming resources against enumerated constraints
// on the LimitRange.  It may modify the incoming object to apply default resource requirements
// if not specified, and enumerated on the LimitRange
func Limit(limitRange *api.LimitRange, resourceName string, obj runtime.Object) error {
	switch resourceName {
	case "pods":
		return PodLimitFunc(limitRange, obj.(*api.Pod))
	}
	return nil
}

// defaultContainerResourceRequirements returns the default requirements for a container
// the requirement.Limits are taken from the LimitRange defaults (if specified)
// the requirement.Requests are taken from the LimitRange min (if specified)
func defaultContainerResourceRequirements(limitRange *api.LimitRange) api.ResourceRequirements {
	requirements := api.ResourceRequirements{}
	requirements.Limits = api.ResourceList{}
	requirements.Requests = api.ResourceList{}

	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		if limit.Type == api.LimitTypeContainer {
			for k, v := range limit.Default {
				value := v.Copy()
				requirements.Limits[k] = *value
			}
		}
	}
	return requirements
}

// mergePodResourceRequirements merges enumerated requirements with default requirements
func mergePodResourceRequirements(pod *api.Pod, defaultRequirements *api.ResourceRequirements) {
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		if container.Resources.Limits == nil {
			container.Resources.Limits = api.ResourceList{}
		}
		if container.Resources.Requests == nil {
			container.Resources.Requests = api.ResourceList{}
		}
		for k, v := range defaultRequirements.Limits {
			_, found := container.Resources.Limits[k]
			if !found {
				container.Resources.Limits[k] = *v.Copy()
			}
		}
		for k, v := range defaultRequirements.Requests {
			_, found := container.Resources.Requests[k]
			if !found {
				container.Resources.Requests[k] = *v.Copy()
			}
		}
	}
}

// PodLimitFunc enforces resource requirements enumerated by the pod against
// the specified LimitRange.  The pod may be modified to apply default resource
// requirements if not specified, and enumerated on the LimitRange
func PodLimitFunc(limitRange *api.LimitRange, pod *api.Pod) error {

	defaultResources := defaultContainerResourceRequirements(limitRange)
	mergePodResourceRequirements(pod, &defaultResources)

	podCPU := int64(0)
	podMem := int64(0)

	minContainerCPU := int64(0)
	minContainerMem := int64(0)
	maxContainerCPU := int64(0)
	maxContainerMem := int64(0)

	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		containerCPU := container.Resources.Limits.Cpu().MilliValue()
		containerMem := container.Resources.Limits.Memory().Value()

		if i == 0 {
			minContainerCPU = containerCPU
			minContainerMem = containerMem
			maxContainerCPU = containerCPU
			maxContainerMem = containerMem
		}

		podCPU = podCPU + container.Resources.Limits.Cpu().MilliValue()
		podMem = podMem + container.Resources.Limits.Memory().Value()

		minContainerCPU = Min(containerCPU, minContainerCPU)
		minContainerMem = Min(containerMem, minContainerMem)
		maxContainerCPU = Max(containerCPU, maxContainerCPU)
		maxContainerMem = Max(containerMem, maxContainerMem)
	}

	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		for _, minOrMax := range []string{"Min", "Max"} {
			var rl api.ResourceList
			switch minOrMax {
			case "Min":
				rl = limit.Min
			case "Max":
				rl = limit.Max
			}
			for k, v := range rl {
				observed := int64(0)
				enforced := int64(0)
				var err error
				switch k {
				case api.ResourceMemory:
					enforced = v.Value()
					switch limit.Type {
					case api.LimitTypePod:
						observed = podMem
						err = fmt.Errorf("%simum memory usage per pod is %s", minOrMax, v.String())
					case api.LimitTypeContainer:
						observed = maxContainerMem
						err = fmt.Errorf("%simum memory usage per container is %s", minOrMax, v.String())
					}
				case api.ResourceCPU:
					enforced = v.MilliValue()
					switch limit.Type {
					case api.LimitTypePod:
						observed = podCPU
						err = fmt.Errorf("%simum CPU usage per pod is %s, but requested %s", minOrMax, v.String(), resource.NewMilliQuantity(observed, resource.DecimalSI))
					case api.LimitTypeContainer:
						observed = maxContainerCPU
						err = fmt.Errorf("%simum CPU usage per container is %s", minOrMax, v.String())
					}
				}
				switch minOrMax {
				case "Min":
					if observed < enforced {
						return err
					}
				case "Max":
					if observed > enforced {
						return err
					}
				}
			}
		}
	}
	return nil
}
