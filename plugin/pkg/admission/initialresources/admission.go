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
	"sort"
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
	source     = flag.String("ir-data-source", "influxdb", "Data source used by InitialResources. Supported options: influxdb, gcm.")
	percentile = flag.Int64("ir-percentile", 90, "Which percentile of samples should InitialResources use when estimating resources. For experiment purposes.")
	nsOnly     = flag.Bool("ir-namespace-only", false, "Whether the estimation should be made only based on data from the same namespace.")
)

const (
	initialResourcesAnnotation = "kubernetes.io/initial-resources"
	samplesThreshold           = 30
	week                       = 7 * 24 * time.Hour
	month                      = 30 * 24 * time.Hour
)

// WARNING: this feature is experimental and will definitely change.
func init() {
	admission.RegisterPlugin("InitialResources", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		s, err := newDataSource(*source)
		if err != nil {
			return nil, err
		}
		return newInitialResources(s, *percentile, *nsOnly), nil
	})
}

type initialResources struct {
	*admission.Handler
	source     dataSource
	percentile int64
	nsOnly     bool
}

func newInitialResources(source dataSource, percentile int64, nsOnly bool) admission.Interface {
	return &initialResources{
		Handler:    admission.NewHandler(admission.Create),
		source:     source,
		percentile: percentile,
		nsOnly:     nsOnly,
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
	annotations := []string{}
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		req := c.Resources.Requests
		lim := c.Resources.Limits
		var cpu, mem *resource.Quantity
		var err error
		if _, ok := req[api.ResourceCPU]; !ok {
			if _, ok2 := lim[api.ResourceCPU]; !ok2 {
				cpu, err = ir.getEstimation(api.ResourceCPU, c, pod.ObjectMeta.Namespace)
				if err != nil {
					glog.Errorf("Error while trying to estimate resources: %v", err)
				}
			}
		}
		if _, ok := req[api.ResourceMemory]; !ok {
			if _, ok2 := lim[api.ResourceMemory]; !ok2 {
				mem, err = ir.getEstimation(api.ResourceMemory, c, pod.ObjectMeta.Namespace)
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
		setRes := []string{}
		if cpu != nil {
			glog.Infof("CPU estimation for container %v in pod %v/%v is %v", c.Name, pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, cpu.String())
			setRes = append(setRes, string(api.ResourceCPU))
			req[api.ResourceCPU] = *cpu
		}
		if mem != nil {
			glog.Infof("Memory estimation for container %v in pod  %v/%v is %v", c.Name, pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, mem.String())
			setRes = append(setRes, string(api.ResourceMemory))
			req[api.ResourceMemory] = *mem
		}
		if len(setRes) > 0 {
			sort.Strings(setRes)
			a := strings.Join(setRes, ", ") + " request for container " + c.Name
			annotations = append(annotations, a)
		}
	}
	if len(annotations) > 0 {
		if pod.ObjectMeta.Annotations == nil {
			pod.ObjectMeta.Annotations = make(map[string]string)
		}
		val := "Initial Resources plugin set: " + strings.Join(annotations, "; ")
		pod.ObjectMeta.Annotations[initialResourcesAnnotation] = val
	}
}

func (ir initialResources) getEstimation(kind api.ResourceName, c *api.Container, ns string) (*resource.Quantity, error) {
	end := time.Now()
	start := end.Add(-week)
	var usage, samples int64
	var err error

	// Historical data from last 7 days for the same image:tag within the same namespace.
	if usage, samples, err = ir.source.GetUsagePercentile(kind, ir.percentile, c.Image, ns, true, start, end); err != nil {
		return nil, err
	}
	if samples < samplesThreshold {
		// Historical data from last 30 days for the same image:tag within the same namespace.
		start := end.Add(-month)
		if usage, samples, err = ir.source.GetUsagePercentile(kind, ir.percentile, c.Image, ns, true, start, end); err != nil {
			return nil, err
		}
	}

	// If we are allowed to estimate only based on data from the same namespace.
	if ir.nsOnly {
		if samples < samplesThreshold {
			// Historical data from last 30 days for the same image within the same namespace.
			start := end.Add(-month)
			image := strings.Split(c.Image, ":")[0]
			if usage, samples, err = ir.source.GetUsagePercentile(kind, ir.percentile, image, ns, false, start, end); err != nil {
				return nil, err
			}
		}
	} else {
		if samples < samplesThreshold {
			// Historical data from last 7 days for the same image:tag within all namespaces.
			start := end.Add(-week)
			if usage, samples, err = ir.source.GetUsagePercentile(kind, ir.percentile, c.Image, "", true, start, end); err != nil {
				return nil, err
			}
		}
		if samples < samplesThreshold {
			// Historical data from last 30 days for the same image:tag within all namespaces.
			start := end.Add(-month)
			if usage, samples, err = ir.source.GetUsagePercentile(kind, ir.percentile, c.Image, "", true, start, end); err != nil {
				return nil, err
			}
		}
		if samples < samplesThreshold {
			// Historical data from last 30 days for the same image within all namespaces.
			start := end.Add(-month)
			image := strings.Split(c.Image, ":")[0]
			if usage, samples, err = ir.source.GetUsagePercentile(kind, ir.percentile, image, "", false, start, end); err != nil {
				return nil, err
			}
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
