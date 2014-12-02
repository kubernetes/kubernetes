/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

func joinCluster(hostname string, client client.Interface, cloud cloudprovider.Interface, masterName string) error {
	instances, ok := cloud.Instances()
	if !ok {
		return fmt.Errorf("Cloud provider: %v doesn't support instance discovery")
	}
	nodeResources, err := instances.GetNodeResources(hostname)
	if err != nil {
		return err
	}
	ip, err := instances.IPAddress(hostname)
	if err != nil {
		return err
	}
	minion := &api.Minion{
		ObjectMeta:    api.ObjectMeta{Name: hostname},
		NodeResources: *nodeResources,
		HostIP:        ip.String(),
	}
	_, err = client.Minions().Create(minion)
	return err
}

func registerKubelet(hostname string, client client.Interface, cloud cloudprovider.Interface, clusterName string) error {
	clusters, ok := cloud.Clusters()
	if !ok {
		return fmt.Errorf("Cloud provider: %v doesn't support cluster discovery", cloud)
	}
	clusterNames, err := clusters.ListClusters()
	if err != nil {
		return err
	}
	var cluster string
	if len(clusterName) == 0 && len(clusterNames) > 0 {
		// If no name is specified, just join the first cluster.
		cluster = clusterNames[0]
	}
	for _, name := range clusterNames {
		if name == clusterName {
			cluster = clusterName
		}
	}
	if len(cluster) == 0 {
		return fmt.Errorf("No such cluster: %s", clusterName)
	}
	master, err := clusters.Master(cluster)
	if err != nil {
		return err
	}
	return joinCluster(hostname, client, cloud, master)
}
