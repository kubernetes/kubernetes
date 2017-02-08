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

package service

import (
	"encoding/json"
	"sort"
	"strings"

	"k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/pkg/api/v1"
)

const (
	FederatedServiceIngressAnnotation = "federation.kubernetes.io/service-ingresses"
)

type FederatedServiceIngress struct {
	v1beta1.FederatedServiceIngress
}

func NewFederatedServiceIngress() *FederatedServiceIngress {
	return &FederatedServiceIngress{}
}

func (ingress *FederatedServiceIngress) String() string {
	annotationBytes, _ := json.Marshal(ingress)
	return string(annotationBytes[:])
}

func (ingress *FederatedServiceIngress) Len() int {
	return len(ingress.Items)
}

func (ingress *FederatedServiceIngress) Less(i, j int) bool {
	regionComparison := strings.Compare(ingress.Items[i].Region, ingress.Items[j].Region)
	zoneComparison := strings.Compare(ingress.Items[i].Zones[0], ingress.Items[j].Zones[0])
	clusterComparison := strings.Compare(ingress.Items[i].Cluster, ingress.Items[j].Cluster)
	if regionComparison < 0 ||
		(regionComparison == 0 && zoneComparison < 0) ||
		(regionComparison == 0 && zoneComparison == 0 && clusterComparison < 0) {
		return true
	}
	return false
}

func (ingress *FederatedServiceIngress) Swap(i, j int) {
	ingress.Items[i].Region, ingress.Items[j].Region = ingress.Items[j].Region, ingress.Items[i].Region
	ingress.Items[i].Zones, ingress.Items[j].Zones = ingress.Items[j].Zones, ingress.Items[i].Zones
	ingress.Items[i].Cluster, ingress.Items[j].Cluster = ingress.Items[j].Cluster, ingress.Items[i].Cluster
	ingress.Items[i].Items, ingress.Items[j].Items = ingress.Items[j].Items, ingress.Items[i].Items
}

// GetClusterLoadBalancerIngress returns loadbalancer ingresses for given region, zone and cluster if exist otherwise returns nil
func (ingress *FederatedServiceIngress) GetClusterLoadBalancerIngress(region, zone, cluster string) []v1.LoadBalancerIngress {
	for _, clusterIngress := range ingress.Items {
		if region == clusterIngress.Region && zone == clusterIngress.Zones[0] && cluster == clusterIngress.Cluster {
			return clusterIngress.Items
		}
	}
	return []v1.LoadBalancerIngress{}
}

// SetClusterLoadBalancerIngress sets the cluster service ingress for a given region, zone and cluster
func (ingress *FederatedServiceIngress) SetClusterLoadBalancerIngress(region, zone, cluster string, loadbalancerIngress []v1.LoadBalancerIngress) {
	for i, clusterIngress := range ingress.Items {
		if region == clusterIngress.Region && zone == clusterIngress.Zones[0] && cluster == clusterIngress.Cluster {
			ingress.Items[i].Items = loadbalancerIngress
			return
		}
	}
	clusterNewIngress := v1beta1.ClusterServiceIngress{Region: region, Zones: []string{zone}, Cluster: cluster, Items: loadbalancerIngress}
	ingress.Items = append(ingress.Items, clusterNewIngress)
	sort.Sort(ingress)
}

// AddEndpoints add one or more endpoints to federated service ingress.
// endpoints are federated cluster's loadbalancer ip/hostname for the service
func (ingress *FederatedServiceIngress) AddEndpoints(region, zone, cluster string, endpoints []string) *FederatedServiceIngress {
	lbIngress := []v1.LoadBalancerIngress{}
	for _, endpoint := range endpoints {
		lbIngress = append(lbIngress, v1.LoadBalancerIngress{IP: endpoint})
	}
	ingress.SetClusterLoadBalancerIngress(region, zone, cluster, lbIngress)
	return ingress
}

// RemoveEndpoint removes a single endpoint (ip/hostname) from the federated service ingress
func (ingress *FederatedServiceIngress) RemoveEndpoint(region, zone, cluster string, endpoint string) *FederatedServiceIngress {
	for i, clusterIngress := range ingress.Items {
		if region == clusterIngress.Region && zone == clusterIngress.Zones[0] && cluster == clusterIngress.Cluster {
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
