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

package ingress

import (
	"encoding/json"
	"sort"
	"strings"

	"k8s.io/api/core/v1"
	fedapi "k8s.io/kubernetes/federation/apis/federation"
)

// Compile time check for interface adherence
var _ sort.Interface = &FederatedServiceIngress{}

const (
	FederatedServiceIngressAnnotation = "federation.kubernetes.io/service-ingresses"
)

// FederatedServiceIngress implements sort.Interface.
type FederatedServiceIngress struct {
	fedapi.FederatedServiceIngress
}

func NewFederatedServiceIngress() *FederatedServiceIngress {
	return &FederatedServiceIngress{}
}

func (ingress *FederatedServiceIngress) String() string {
	annotationBytes, _ := json.Marshal(ingress)
	return string(annotationBytes[:])
}

// Len is to satisfy of sort.Interface.
func (ingress *FederatedServiceIngress) Len() int {
	return len(ingress.Items)
}

// Less is to satisfy of sort.Interface.
func (ingress *FederatedServiceIngress) Less(i, j int) bool {
	return (strings.Compare(ingress.Items[i].Cluster, ingress.Items[j].Cluster) < 0)
}

// Swap is to satisfy of sort.Interface.
func (ingress *FederatedServiceIngress) Swap(i, j int) {
	ingress.Items[i].Cluster, ingress.Items[j].Cluster = ingress.Items[j].Cluster, ingress.Items[i].Cluster
	ingress.Items[i].Items, ingress.Items[j].Items = ingress.Items[j].Items, ingress.Items[i].Items
}

// GetClusterLoadBalancerIngresses returns loadbalancer ingresses for given cluster if exist otherwise returns an empty slice
func (ingress *FederatedServiceIngress) GetClusterLoadBalancerIngresses(cluster string) []v1.LoadBalancerIngress {
	for _, clusterIngress := range ingress.Items {
		if cluster == clusterIngress.Cluster {
			return clusterIngress.Items
		}
	}
	return []v1.LoadBalancerIngress{}
}

// AddClusterLoadBalancerIngresses adds the ladbalancer ingresses for a given cluster to federated service ingress
func (ingress *FederatedServiceIngress) AddClusterLoadBalancerIngresses(cluster string, loadbalancerIngresses []v1.LoadBalancerIngress) {
	for i, clusterIngress := range ingress.Items {
		if cluster == clusterIngress.Cluster {
			ingress.Items[i].Items = append(ingress.Items[i].Items, loadbalancerIngresses...)
			return
		}
	}
	clusterNewIngress := fedapi.ClusterServiceIngress{Cluster: cluster, Items: loadbalancerIngresses}
	ingress.Items = append(ingress.Items, clusterNewIngress)
	sort.Sort(ingress)
}

// AddEndpoints add one or more endpoints to federated service ingress.
// endpoints are federated cluster's loadbalancer ip/hostname for the service
func (ingress *FederatedServiceIngress) AddEndpoints(cluster string, endpoints []string) *FederatedServiceIngress {
	lbIngress := []v1.LoadBalancerIngress{}
	for _, endpoint := range endpoints {
		lbIngress = append(lbIngress, v1.LoadBalancerIngress{IP: endpoint})
	}
	ingress.AddClusterLoadBalancerIngresses(cluster, lbIngress)
	return ingress
}

// RemoveEndpoint removes a single endpoint (ip/hostname) from the federated service ingress
func (ingress *FederatedServiceIngress) RemoveEndpoint(cluster string, endpoint string) *FederatedServiceIngress {
	for i, clusterIngress := range ingress.Items {
		if cluster == clusterIngress.Cluster {
			for j, lbIngress := range clusterIngress.Items {
				if lbIngress.IP == endpoint {
					ingress.Items[i].Items = append(ingress.Items[i].Items[:j], ingress.Items[i].Items[j+1:]...)
				}
			}
		}
	}
	return ingress
}

// ParseFederatedServiceIngress extracts federated service ingresses from a federated service
func ParseFederatedServiceIngress(service *v1.Service) (*FederatedServiceIngress, error) {
	ingress := FederatedServiceIngress{}
	if service.Annotations == nil {
		return &ingress, nil
	}
	federatedServiceIngressString, found := service.Annotations[FederatedServiceIngressAnnotation]
	if !found {
		return &ingress, nil
	}
	if err := json.Unmarshal([]byte(federatedServiceIngressString), &ingress); err != nil {
		return &ingress, err
	}
	return &ingress, nil
}

// UpdateIngressAnnotation updates the federated service with service ingress annotation
func UpdateIngressAnnotation(service *v1.Service, ingress *FederatedServiceIngress) *v1.Service {
	if service.Annotations == nil {
		service.Annotations = make(map[string]string)
	}
	service.Annotations[FederatedServiceIngressAnnotation] = ingress.String()
	return service
}
