/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// getClusterZoneName returns the name of the zone where the specified cluster exists (e.g. "us-east1-c" on GCE, or "us-east-1b" on AWS)
func getClusterZoneName(clusterName string) string {
	// TODO: quinton: Get this from the cluster API object - from the annotation on a node in the cluster - it doesn't contain this yet.
	return "zone-of-cluster-" + clusterName
}

// getClusterRegionName returns the name of the region where the specified cluster exists (e.g. us-east1 on GCE, or "us-east-1" on AWS)
func getClusterRegionName(clusterName string) string {
	// TODO: quinton: Get this from the cluster API object - from the annotation on a node in the cluster - it doesn't contain this yet.
	return "region-of-cluster-" + clusterName
}

// getFederationDNSZoneName returns the name of the managed DNS Zone configured for this federation
func getFederationDNSZoneName() string {
	return "mydomain.com" // TODO: quinton: Get this from the federation configuration.
}

func ensureDNSRecords(clusterName string, cachedService *cachedService) error {
	// Quinton: Pseudocode....

	return nil
}
