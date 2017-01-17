/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/golang/glog"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
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
	admission.RegisterPlugin("InitialResources", func(config io.Reader) (admission.Interface, error) {
		// TODO: remove the usage of flags in favor of reading versioned configuration
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
	if a.GetSubresource() != "" || a.GetResource().GroupResource() != api.Resource("pods") {
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
	var annotations []string
	for i := range pod.Spec.InitContainers {
		annotations = append(annotations, ir.estimateContainer(pod, &pod.Spec.InitContainers[i], "init container")...)
	}
	for i := range pod.Spec.Containers {
		annotations = append(annotations, ir.estimateContainer(pod, &pod.Spec.Containers[i], "container")...)
	}
	if len(annotations) > 0 {
		if pod.ObjectMeta.Annotations == nil {
			pod.ObjectMeta.Annotations = make(map[string]string)
		}
		val := "Initial Resources plugin set: " + strings.Join(annotations, "; ")
		pod.ObjectMeta.Annotations[initialResourcesAnnotation] = val
	}
}

func (ir initialResources) estimateContainer(pod *api.Pod, c *api.Container, message string) []string {
	var annotations []string
	req := c.Resources.Requests
	cpu := ir.getEstimationIfNeeded(api.ResourceCPU, c, pod.ObjectMeta.Namespace)
	mem := ir.getEstimationIfNeeded(api.ResourceMemory, c, pod.ObjectMeta.Namespace)
	// If Requests doesn't exits and an estimation was made, create Requests.
	if req == nil && (cpu != nil || mem != nil) {
		c.Resources.Requests = api.ResourceList{}
		req = c.Resources.Requests
	}
	setRes := []string{}
	if cpu != nil {
		glog.Infof("CPU estimation for %s %v in pod %v/%v is %v", message, c.Name, pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, cpu.String())
		setRes = append(setRes, string(api.ResourceCPU))
		req[api.ResourceCPU] = *cpu
	}
	if mem != nil {
		glog.Infof("Memory estimation for %s %v in pod %v/%v is %v", message, c.Name, pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, mem.String())
		setRes = append(setRes, string(api.ResourceMemory))
		req[api.ResourceMemory] = *mem
	}
	if len(setRes) > 0 {
		sort.Strings(setRes)
		a := strings.Join(setRes, ", ") + fmt.Sprintf(" request for %s %s", message, c.Name)
		annotations = append(annotations, a)
	}
	return annotations
}

// getEstimationIfNeeded estimates compute resource for container if its corresponding
// Request(min amount) and Limit(max amount) both are not specified.
func (ir initialResources) getEstimationIfNeeded(kind api.ResourceName, c *api.Container, ns string) *resource.Quantity {
	requests := c.Resources.Requests
	limits := c.Resources.Limits
	var quantity *resource.Quantity
	var err error
	if _, requestFound := requests[kind]; !requestFound {
		if _, limitFound := limits[kind]; !limitFound {
			quantity, err = ir.getEstimation(kind, c, ns)
			if err != nil {
				glog.Errorf("Error while trying to estimate resources: %v", err)
			}
		}
	}
	return quantity
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
