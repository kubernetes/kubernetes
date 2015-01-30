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

package limitranger

import (
	"fmt"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func init() {
	admission.RegisterPlugin("LimitRanger", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewLimitRanger(client, PodLimitFunc), nil
	})
}

// limitRanger enforces usage limits on a per resource basis in the namespace
type limitRanger struct {
	client    client.Interface
	limitFunc LimitFunc
}

// Admit admits resources into cluster that do not violate any defined LimitRange in the namespace
func (l *limitRanger) Admit(a admission.Attributes) (err error) {
	// ignore deletes
	if a.GetOperation() == "DELETE" {
		return nil
	}

	// look for a limit range in current namespace that requires enforcement
	// TODO: Move to cache when issue is resolved: https://github.com/GoogleCloudPlatform/kubernetes/issues/2294
	items, err := l.client.LimitRanges(a.GetNamespace()).List(labels.Everything())
	if err != nil {
		return err
	}

	// ensure it meets each prescribed min/max
	for i := range items.Items {
		limitRange := &items.Items[i]
		err = l.limitFunc(limitRange, a.GetResource(), a.GetObject())
		if err != nil {
			return err
		}
	}
	return nil
}

// NewLimitRanger returns an object that enforces limits based on the supplied limit function
func NewLimitRanger(client client.Interface, limitFunc LimitFunc) admission.Interface {
	return &limitRanger{client: client, limitFunc: limitFunc}
}

func Min(a int64, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func Max(a int64, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// PodLimitFunc enforces that a pod spec does not exceed any limits specified on the supplied limit range
func PodLimitFunc(limitRange *api.LimitRange, resourceName string, obj runtime.Object) error {
	if resourceName != "pods" {
		return nil
	}

	pod := obj.(*api.Pod)

	podCPU := int64(0)
	podMem := int64(0)

	minContainerCPU := int64(0)
	minContainerMem := int64(0)
	maxContainerCPU := int64(0)
	maxContainerMem := int64(0)

	for i := range pod.Spec.Containers {
		container := pod.Spec.Containers[i]
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
						return apierrors.NewForbidden(resourceName, pod.Name, err)
					}
				case "Max":
					if observed > enforced {
						return apierrors.NewForbidden(resourceName, pod.Name, err)
					}
				}
			}
		}
	}
	return nil
}
