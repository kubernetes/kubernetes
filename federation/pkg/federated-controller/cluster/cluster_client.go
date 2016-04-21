/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/apis/extensions"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"strings"
)

const (
	UserAgentName = "Cluster-Controller"
	KubeAPIQPS    = 20.0
	KubeAPIBurst  = 30
)

type ClusterClient struct {
	clientSet       clientset.Interface
	discoveryClient *discovery.DiscoveryClient
}

func NewClusterClientSet(c *federation.Cluster) (*ClusterClient, error) {
	//TODO:How to get cluster IP(huangyuqi)
	var clusterClientSet = ClusterClient{}
	clusterConfig, err := clientcmd.BuildConfigFromFlags(c.Spec.ServerAddressByClientCIDRs[0].ServerAddress, "")
	if err != nil {
		return nil, err
	}
	//	clusterConfig.ContentConfig.GroupVersion.Version = "extensions"
	clusterConfig.QPS = KubeAPIQPS
	clusterConfig.Burst = KubeAPIBurst
	clusterClientSet.clientSet = clientset.NewForConfigOrDie(restclient.AddUserAgent(clusterConfig, UserAgentName))
	clusterClientSet.discoveryClient = discovery.NewDiscoveryClientForConfigOrDie((restclient.AddUserAgent(clusterConfig, UserAgentName)))
	return &clusterClientSet, err
}

// GetReplicaSetFromCluster get the replicaset from the kubernetes cluster
func (self *ClusterClient) GetReplicaSetFromCluster(subRsName string, subRsNameSpace string) (*extensions.ReplicaSet, error) {
	return self.clientSet.Extensions().ReplicaSets(subRsNameSpace).Get(subRsName)
}

// CreateReplicaSetToCluster create replicaset to the kubernetes cluster
func (self *ClusterClient) CreateReplicaSetToCluster(subRs *extensions.ReplicaSet) (*extensions.ReplicaSet, error) {
	return self.clientSet.Extensions().ReplicaSets(subRs.Namespace).Create(subRs)
}

// UpdateReplicaSetToCluster update replicaset to the kubernetes cluster
func (self *ClusterClient) UpdateReplicaSetToCluster(subRs *extensions.ReplicaSet) (*extensions.ReplicaSet, error) {
	return self.clientSet.Extensions().ReplicaSets(subRs.Namespace).Update(subRs)
}

// DeleteReplicasetFromCluster delete the replicaset from the kubernetes cluster
func (self *ClusterClient) DeleteReplicasetFromCluster(subRs *extensions.ReplicaSet) error {
	return self.clientSet.Extensions().ReplicaSets(subRs.Namespace).Delete(subRs.Name, &api.DeleteOptions{})
}

// GetClusterHealthStatus get the kubernetes cluster health status
func (self *ClusterClient) GetClusterHealthStatus() federation.ClusterPhase {
	body, err := self.discoveryClient.Get().AbsPath("/healthz").Do().Raw()
	if err != nil {
		return federation.ClusterOffline
	}
	if !strings.EqualFold(string(body), "ok") {
		return federation.ClusterPending
	}
	return federation.ClusterRunning
}
