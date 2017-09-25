/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import (
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

type wrappedLoadBalancer struct {
	gce *GCECloud
}

func newWrappedLoadBalancer(gce *GCECloud) cloudprovider.LoadBalancer {
	return &wrappedLoadBalancer{gce: gce}
}

func (w *wrappedLoadBalancer) GetLoadBalancer(clusterName string, service *v1.Service) (status *v1.LoadBalancerStatus, exists bool, err error) {
	svc := w.convertServiceLBAnnotations(service)
	return w.gce.GetLoadBalancer(clusterName, svc)
}

func (w *wrappedLoadBalancer) EnsureLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	svc := w.convertServiceLBAnnotations(service)
	if err := validateServiceLBAnnotations(svc); err != nil {
		return nil, err
	}
	return w.gce.EnsureLoadBalancer(clusterName, svc, nodes)
}

func (w *wrappedLoadBalancer) UpdateLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) error {
	svc := w.convertServiceLBAnnotations(service)
	return w.gce.UpdateLoadBalancer(clusterName, svc, nodes)
}

func (w *wrappedLoadBalancer) EnsureLoadBalancerDeleted(clusterName string, service *v1.Service) error {
	svc := w.convertServiceLBAnnotations(service)
	return w.gce.EnsureLoadBalancerDeleted(clusterName, svc)
}

func (w *wrappedLoadBalancer) convertServiceLBAnnotations(svc *v1.Service) *v1.Service {
	results := map[string]string{}
	for k, v := range svc.Annotations {
		// Convert to the new key if the current key has been deprecated.
		if newKey, ok := deprecatedServiceLBAnnotations[k]; ok {
			// Do we need to log a warning every time?
			glog.Warning("key %s has been deprecated. Please use %q in the future", k, newKey)
			k = newKey
		}

		// Trim annotations owned by gated Alpha features.
		if r, ok := allServiceLBAnnotations[k]; ok {
			if r.alphaFeature != "" && !w.gce.AlphaFeatureGate.Enabled(r.alphaFeature) {
				continue
			}
			// Consider doing defaulting here? This would need to be
			// LB-type-aware (ILB vs ELB).
		}

		results[k] = v
	}

	// Make a deep copy of the Service and set the annotations
	newSvc := svc.DeepCopy()
	newSvc.Annotations = results
	return newSvc
}
