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

package hpe

import (
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
)

type Lbaas struct {
	LoadBalancer
}

var (
	nCfgMap *ncsConfigMaps
)

var HOSTNAME_TO_VRID_CFGMAP_NAME string
var VRID_TO_EXTIP_CFGMAP_NAME string
var SVC_TO_EXTIP_CFGMAP_NAME string
var CFGMAP_NAME_SPACE string

func getNCSKubeAPIClient(namespace string) (*ncsConfigMaps, error) {
	var err error
	HOSTNAME_TO_VRID_CFGMAP_NAME = "hostname-to-vrid-map"
	VRID_TO_EXTIP_CFGMAP_NAME = "vrid-to-ip-map"
	SVC_TO_EXTIP_CFGMAP_NAME = "svc-to-ip-map"
	CFGMAP_NAME_SPACE = "kube-system"

	if nCfgMap == nil {
		nCfgMap = &ncsConfigMaps{
			ns: namespace,
		}
		err = nCfgMap.init()
	}
	return nCfgMap, err
}

func (lbaas *Lbaas) GetLoadBalancer(clusterName string, apiService *v1.Service) (*v1.LoadBalancerStatus, bool, error) {
	status := &v1.LoadBalancerStatus{}
	return status, true, nil
}

func (lbaas *Lbaas) EnsureLoadBalancer(clusterName string, apiService *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v)", clusterName, apiService.Namespace, apiService.Name, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, apiService.Annotations)
	if len(apiService.Status.LoadBalancer.Ingress) != 0 {
		return &(apiService.Status.LoadBalancer), nil
	}
	status := &v1.LoadBalancerStatus{}

	apiClient, err := getNCSKubeAPIClient(CFGMAP_NAME_SPACE)
	if err != nil {
		glog.Errorf("kube-apiserver connection initialization failed")
		return status, nil
	}
	vridToHostCfg, err := apiClient.getCfgMap(HOSTNAME_TO_VRID_CFGMAP_NAME)
	if err != nil {
		glog.Errorf("Failed to get %s ConfigMap", HOSTNAME_TO_VRID_CFGMAP_NAME)
		return status, nil
	}
	externalIP := allocateIp()
	glog.V(1).Infof("Generated externalIP %s for service: %s", externalIP, apiService.Name)
	if externalIP == "" {
		glog.Errorf("Error getting External IP for service failed: %s", apiService.Name)
		return status, nil
	}
	vrids := apiClient.getVrids(vridToHostCfg.Data)
	vridToIPCfg, err := apiClient.getCfgMap(VRID_TO_EXTIP_CFGMAP_NAME)
	if err != nil {
		cfg := apiClient.createVridsExIPCfgMap(vrids, VRID_TO_EXTIP_CFGMAP_NAME, externalIP)
		apiClient.createNewCfgMap(cfg)
	} else {
		apiClient.updateVridsExIPCfgMap(vridToIPCfg, vrids, externalIP)
		apiClient.updateCfgMap(vridToIPCfg)
	}
	status.Ingress = []v1.LoadBalancerIngress{{IP: externalIP}}
	SvcToExIPCfg, err := apiClient.getCfgMap(SVC_TO_EXTIP_CFGMAP_NAME)
	if err != nil {
		cfg := apiClient.createSvcExIPCfgMap(apiService.Name, SVC_TO_EXTIP_CFGMAP_NAME, externalIP)
		apiClient.createNewCfgMap(cfg)
	} else {
		apiClient.updateSvcExIPCfgMap(SvcToExIPCfg, apiService.Name, externalIP)
		apiClient.updateCfgMap(SvcToExIPCfg)
	}
	glog.V(1).Infof("Assigned externalIP %s for service: %s", externalIP, apiService.Name)
	return status, nil
}

func (lbaas *Lbaas) UpdateLoadBalancer(clusterName string, apiService *v1.Service, nodes []*v1.Node) error {
	return nil
}

func (lbaas *Lbaas) EnsureLoadBalancerDeleted(clusterName string, service *v1.Service) error {
	extIP := ""
	apiClient, err := getNCSKubeAPIClient(CFGMAP_NAME_SPACE)
	SvcToExIPCfg, err := apiClient.getCfgMap(SVC_TO_EXTIP_CFGMAP_NAME)
	cfgDel := true
	if err != nil {
		glog.V(1).Infof("[EnsureLoadBalancerDeleted] Failed to get svc-to-ip-map")
		cfgDel = false
	}
	if len(service.Status.LoadBalancer.Ingress) != 0 {
		extIP = service.Status.LoadBalancer.Ingress[0].IP
	} else {
		if cfgDel {
			extIP = SvcToExIPCfg.Data[service.Name]
		}
	}

	if cfgDel {
		delete(SvcToExIPCfg.Data, service.Name)
		apiClient.updateCfgMap(SvcToExIPCfg)
	}
	if len(extIP) <= 0 {
		glog.V(1).Infof("[EnsureLoadBalancerDeleted] No external IP ")
		return nil
	}
	releaseIp(extIP)
	vridToIPCfg, err := apiClient.getCfgMap(VRID_TO_EXTIP_CFGMAP_NAME)
	if err != nil {
		glog.V(1).Infof("[EnsureLoadBalancerDeleted] Failed to get vrid-to-ip-map")
		return nil
	}
	extIP += ","
	data := vridToIPCfg.Data
	ok := false

	for k, v := range data {
		glog.V(1).Infof("[EnsureLoadBalancerDeleted] Removing association vrid to extIP")
		stripped := strings.Replace(v, extIP, "", -1)
		data[k] = stripped
		ok = true
	}
	if ok {
		vridToIPCfg.Data = data
		apiClient.updateCfgMap(vridToIPCfg)
		glog.V(1).Infof("[EnsureLoadBalancerDeleted] updated vrid-to-ip-map")
	}
	glog.V(1).Infof("Deleted service: %s", service.Name)
	return nil
}