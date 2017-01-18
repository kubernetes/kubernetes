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

package ingress

import (
	"encoding/json"
	"strings"

	fed "k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api/v1"
)

const (
	FederationServiceIngressAnnotation = "federation.kubernetes.io/service-ingress-endpoints"
)

type serviceIngress fed.FederatedServiceIngress

func NewServiceIngress() *serviceIngress {
	return &serviceIngress{}
}

func (si *serviceIngress) AddEndpoints(region, zone, cluster string, endpoints []string, healthy bool) *serviceIngress {
	if si.Endpoints == nil {
		si.Endpoints = make(map[string]map[string]map[string]*fed.ClusterServiceIngress)
	}
	if si.Endpoints[region] == nil {
		si.Endpoints[region] = make(map[string]map[string]*fed.ClusterServiceIngress)
	}
	if si.Endpoints[region][zone] == nil {
		si.Endpoints[region][zone] = make(map[string]*fed.ClusterServiceIngress)
	}
	clusterIngEps, ok := si.Endpoints[region][zone][cluster]
	if !ok {
		clusterIngEps = &fed.ClusterServiceIngress{}
	}
	clusterIngEps.Endpoints = append(clusterIngEps.Endpoints, endpoints...)
	clusterIngEps.Healthy = healthy
	si.Endpoints[region][zone][cluster] = clusterIngEps
	return si
}

func (si *serviceIngress) RemoveEndpoint(region, zone, cluster string, endpoint string) *serviceIngress {
	clusterIngress := si.Endpoints[region][zone][cluster]
	for i, ep := range clusterIngress.Endpoints {
		if ep == endpoint {
			clusterIngress.Endpoints = append(clusterIngress.Endpoints[:i], clusterIngress.Endpoints[i+1:]...)
		}
	}
	if len(clusterIngress.Endpoints) < 1 {
		clusterIngress.Healthy = false
	} else {
		clusterIngress.Healthy = true
	}
	si.Endpoints[region][zone][cluster] = clusterIngress
	return si
}

// GetOrCreateEmptyClusterServiceIngresses returns cluster service ingresses for given region, zone and cluster name if
// exist otherwise creates one and returns
func (si *serviceIngress) GetOrCreateEmptyClusterServiceIngresses(region, zone, cluster string) *fed.ClusterServiceIngress {
	if si.Endpoints == nil {
		si.Endpoints = make(map[string]map[string]map[string]*fed.ClusterServiceIngress)
	}
	if si.Endpoints[region] == nil {
		si.Endpoints[region] = make(map[string]map[string]*fed.ClusterServiceIngress)
	}
	if si.Endpoints[region][zone] == nil {
		si.Endpoints[region][zone] = make(map[string]*fed.ClusterServiceIngress)
	}
	if si.Endpoints[region][zone][cluster] == nil {
		return &fed.ClusterServiceIngress{}
	}

	return si.Endpoints[region][zone][cluster]
}

func (si *serviceIngress) GetJSONMarshalledBytes() []byte {
	byteArray, _ := json.Marshal(*si)
	return byteArray
}

func NewServiceIngressAnnotation(byteArray []byte) map[string]string {
	epAnnotation := make(map[string]string)
	epAnnotation[FederationServiceIngressAnnotation] = string(byteArray[:])
	return epAnnotation
}

// ParseFederationServiceIngresses extracts federation service ingresses from federation service
func ParseFederationServiceIngresses(fs *v1.Service) (*fed.FederatedServiceIngress, error) {
	fsIngress := fed.FederatedServiceIngress{}
	if fs.Annotations == nil {
		return &fsIngress, nil
	}
	fsIngressString, found := fs.Annotations[FederationServiceIngressAnnotation]
	if !found {
		return &fsIngress, nil
	}
	if err := json.Unmarshal([]byte(fsIngressString), &fsIngress); err != nil {
		return &fsIngress, err
	}
	return &fsIngress, nil
}

// UpdateFederationServiceIngresses updates the federation service with service ingress annotation
func UpdateFederationServiceIngresses(fs *v1.Service, si *serviceIngress) *v1.Service {
	annotationBytes, _ := json.Marshal(si)
	annotations := string(annotationBytes[:])
	if fs.Annotations == nil {
		fs.Annotations = make(map[string]string)
	}
	fs.Annotations[FederationServiceIngressAnnotation] = annotations
	return fs
}

type loadbalancerIngress []v1.LoadBalancerIngress

func NewLoadbalancerIngress() loadbalancerIngress {
	return loadbalancerIngress{}
}

func (lbi loadbalancerIngress) Len() int {
	return len(lbi)
}

func (lbi loadbalancerIngress) Less(i, j int) bool {
	ipComparison := strings.Compare(lbi[i].IP, lbi[j].IP)
	hostnameComparison := strings.Compare(lbi[i].Hostname, lbi[j].Hostname)
	if ipComparison < 0 || (ipComparison == 0 && hostnameComparison < 0) {
		return true
	}
	return false
}

func (lbi loadbalancerIngress) Swap(i, j int) {
	lbi[i].IP, lbi[j].IP = lbi[j].IP, lbi[i].IP
	lbi[i].Hostname, lbi[j].Hostname = lbi[j].Hostname, lbi[i].Hostname
}
