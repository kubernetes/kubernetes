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

func (gce *GCECloud) ListClusters() ([]string, error) {
	allClusters := []string{}

	for _, zone := range gce.managedZones {
		clusters, err := gce.listClustersInZone(zone)
		if err != nil {
			return nil, err
		}
		// TODO: Scoping?  Do we need to qualify the cluster name?
		allClusters = append(allClusters, clusters...)
	}

	return allClusters, nil
}

func (gce *GCECloud) Master(clusterName string) (string, error) {
	return "k8s-" + clusterName + "-master.internal", nil
}

func (gce *GCECloud) listClustersInZone(zone string) ([]string, error) {
	// TODO: use PageToken to list all not just the first 500
	list, err := gce.containerService.Projects.Zones.Clusters.List(gce.projectID, zone).Do()
	if err != nil {
		return nil, err
	}
	result := []string{}
	for _, cluster := range list.Clusters {
		result = append(result, cluster.Name)
	}
	return result, nil
}
