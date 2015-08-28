/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package initialresources

import (
	"flag"
	"io"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

var (
	source     = flag.String("ir-data-source", "influxdb", "Data source used by InitialResources. Supported options: influxdb.")
	percentile = flag.Int64("ir-percentile", 90, "Which percentile of samples should InitialResources use when estimating resources. For experiment purposes.")
)

const (
	samplesThreshold = 60
	week             = 7 * 24 * time.Hour
	month            = 30 * 24 * time.Hour
)

// WARNING: this feature is experimental and will definitely change.
func init() {
	admission.RegisterPlugin("InitialResources", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		s, err := newDataSource(*source)
		if err != nil {
			return nil, err
		}
		return newInitialResources(s), nil
	})
}

type initialResources struct {
	*admission.Handler
	source dataSource
}

func newInitialResources(source dataSource) admission.Interface {
	return &initialResources{
		Handler: admission.NewHandler(admission.Create),
		source:  source,
	}
}

func (ir initialResources) Admit(a admission.Attributes) (err error) {
	// Ignore all calls to subresources or resources other than pods.
	if a.GetSubresource() != "" || a.GetResource() != string(api.ResourcePods) {
		return nil
	}
	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	ir.estimateAndFillResourcesIfNotSet(pod)
	return nil
}

// The method veryfies whether resources should be set for the given pod and
// if there is estimation available the method fills Request field.
func (ir initialResources) estimateAndFillResourcesIfNotSet(pod *api.Pod) {
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		req := c.Resources.Requests
		lim := c.Resources.Limits
		var cpu, mem *resource.Quantity
		var err error
		if _, ok := req[api.ResourceCPU]; !ok {
			if _, ok2 := lim[api.ResourceCPU]; !ok2 {
				cpu, err = ir.getEstimation(api.ResourceCPU, c)
				if err != nil {
					glog.Errorf("Error while trying to estimate resources: %v", err)
				}
			}
		}
		if _, ok := req[api.ResourceMemory]; !ok {
			if _, ok2 := lim[api.ResourceMemory]; !ok2 {
				mem, err = ir.getEstimation(api.ResourceMemory, c)
				if err != nil {
					glog.Errorf("Error while trying to estimate resources: %v", err)
				}
			}
		}

		// If Requests doesn't exits and an estimation was made, create Requests.
		if req == nil && (cpu != nil || mem != nil) {
			c.Resources.Requests = api.ResourceList{}
			req = c.Resources.Requests
		}
		if cpu != nil {
			glog.Infof("CPU estimation for container %v in pod %v/%v is %v", c.Name, pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, cpu.String())
			req[api.ResourceCPU] = *cpu
		}
		if mem != nil {
			glog.Infof("Memory estimation for container %v in pod  %v/%v is %v", c.Name, pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, mem.String())
			req[api.ResourceMemory] = *mem
		}
	}
	// TODO(piosz): verify the estimates fits in LimitRanger
}

func (ir initialResources) getEstimation(kind api.ResourceName, c *api.Container) (*resource.Quantity, error) {
	end := time.Now()
	start := end.Add(-week)
	var usage, samples int64
	var err error

	// Historical data from last 7 days for the same image:tag.
	if usage, samples, err = ir.source.GetUsagePercentile(kind, *percentile, c.Image, true, start, end); err != nil {
		return nil, err
	}
	if samples < samplesThreshold {
		// Historical data from last 30 days for the same image:tag.
		start := end.Add(-month)
		if usage, samples, err = ir.source.GetUsagePercentile(kind, *percentile, c.Image, true, start, end); err != nil {
			return nil, err
		}
	}
	if samples < samplesThreshold {
		// Historical data from last 30 days for the same image.
		start := end.Add(-month)
		image := strings.Split(c.Image, ":")[0]
		if usage, samples, err = ir.source.GetUsagePercentile(kind, *percentile, image, false, start, end); err != nil {
			return nil, err
		}
	}

	if samples > 0 && kind == api.ResourceCPU {
		return resource.NewMilliQuantity(usage, resource.DecimalSI), nil
	}
	if samples > 0 && kind == api.ResourceMemory {
		return resource.NewQuantity(usage, resource.DecimalSI), nil
	}
	return nil, nil
}
