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

package cluster

import (
	"fmt"
	"strings"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
	federation_v1beta1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

const (
	UserAgentName = "Cluster-Controller"
)

type ClusterClient struct {
	discoveryClient *discovery.DiscoveryClient
	kubeClient      *clientset.Clientset
}

func NewClusterClientSet(c *federation_v1beta1.Cluster) (*ClusterClient, error) {
	clusterConfig, err := util.BuildClusterConfig(c)
	if err != nil {
		return nil, err
	}
	var clusterClientSet = ClusterClient{}
	if clusterConfig != nil {
		clusterClientSet.discoveryClient = discovery.NewDiscoveryClientForConfigOrDie((restclient.AddUserAgent(clusterConfig, UserAgentName)))
		if clusterClientSet.discoveryClient == nil {
			return nil, nil
		}
		clusterClientSet.kubeClient = clientset.NewForConfigOrDie((restclient.AddUserAgent(clusterConfig, UserAgentName)))
		if clusterClientSet.kubeClient == nil {
			return nil, nil
		}
	}
	return &clusterClientSet, nil
}

// GetClusterHealthStatus gets the kubernetes cluster health status by requesting "/healthz"
func (self *ClusterClient) GetClusterHealthStatus() *federation_v1beta1.ClusterStatus {
	clusterStatus := federation_v1beta1.ClusterStatus{}
	currentTime := metav1.Now()
	newClusterReadyCondition := federation_v1beta1.ClusterCondition{
		Type:               federation_v1beta1.ClusterReady,
		Status:             v1.ConditionTrue,
		Reason:             "ClusterReady",
		Message:            "/healthz responded with ok",
		LastProbeTime:      currentTime,
		LastTransitionTime: currentTime,
	}
	newClusterNotReadyCondition := federation_v1beta1.ClusterCondition{
		Type:               federation_v1beta1.ClusterReady,
		Status:             v1.ConditionFalse,
		Reason:             "ClusterNotReady",
		Message:            "/healthz responded without ok",
		LastProbeTime:      currentTime,
		LastTransitionTime: currentTime,
	}
	newNodeOfflineCondition := federation_v1beta1.ClusterCondition{
		Type:               federation_v1beta1.ClusterOffline,
		Status:             v1.ConditionTrue,
		Reason:             "ClusterNotReachable",
		Message:            "cluster is not reachable",
		LastProbeTime:      currentTime,
		LastTransitionTime: currentTime,
	}
	newNodeNotOfflineCondition := federation_v1beta1.ClusterCondition{
		Type:               federation_v1beta1.ClusterOffline,
		Status:             v1.ConditionFalse,
		Reason:             "ClusterReachable",
		Message:            "cluster is reachable",
		LastProbeTime:      currentTime,
		LastTransitionTime: currentTime,
	}
	body, err := self.discoveryClient.RESTClient().Get().AbsPath("/healthz").Do().Raw()
	if err != nil {
		clusterStatus.Conditions = append(clusterStatus.Conditions, newNodeOfflineCondition)
	} else {
		if !strings.EqualFold(string(body), "ok") {
			clusterStatus.Conditions = append(clusterStatus.Conditions, newClusterNotReadyCondition, newNodeNotOfflineCondition)
		} else {
			clusterStatus.Conditions = append(clusterStatus.Conditions, newClusterReadyCondition)
		}
	}
	return &clusterStatus
}

// GetClusterZones gets the kubernetes cluster zones and region by inspecting labels on nodes in the cluster.
func (self *ClusterClient) GetClusterZones() (zones []string, region string, err error) {
	return getZoneNames(self.kubeClient)
}

// Find the name of the zone in which a Node is running
func getZoneNameForNode(node api.Node) (string, error) {
	for key, value := range node.Labels {
		if key == kubeletapis.LabelZoneFailureDomain {
			return value, nil
		}
	}
	return "", fmt.Errorf("Zone name for node %s not found. No label with key %s",
		node.Name, kubeletapis.LabelZoneFailureDomain)
}

// Find the name of the region in which a Node is running
func getRegionNameForNode(node api.Node) (string, error) {
	for key, value := range node.Labels {
		if key == kubeletapis.LabelZoneRegion {
			return value, nil
		}
	}
	return "", fmt.Errorf("Region name for node %s not found. No label with key %s",
		node.Name, kubeletapis.LabelZoneRegion)
}

// Find the names of all zones and the region in which we have nodes in this cluster.
func getZoneNames(client *clientset.Clientset) (zones []string, region string, err error) {
	zoneNames := sets.NewString()
	nodes, err := client.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		glog.Errorf("Failed to list nodes while getting zone names: %v", err)
		return nil, "", err
	}
	for i, node := range nodes.Items {
		// TODO: quinton-hoole make this more efficient.
		//       For non-multi-zone clusters the zone will
		//       be identical for all nodes, so we only need to look at one node
		//       For multi-zone clusters we know at build time
		//       which zones are included.  Rather get this info from there, because it's cheaper.
		zoneName, err := getZoneNameForNode(node)
		if err != nil {
			return nil, "", err
		}
		zoneNames.Insert(zoneName)
		if i == 0 {
			region, err = getRegionNameForNode(node)
			if err != nil {
				return nil, "", err
			}
		}
	}
	return zoneNames.List(), region, nil
}
