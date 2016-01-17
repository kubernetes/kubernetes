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

package capability

var watchCapability map[string]int

func init() {
	watchCapability = make(map[string]int)
	watchCapability["pods"] = 1000
	watchCapability["nodes"] = 1000
	watchCapability["services"] = 100
	watchCapability["endpoints"] = 1000
	watchCapability["namespaces"] = 100
	watchCapability["controllers"] = 100
	watchCapability["podtemplates"] = 100
	watchCapability["limitranges"] = 100
	watchCapability["resourcequotas"] = 100
	watchCapability["secrets"] = 100
	watchCapability["serviceaccounts"] = 100
	watchCapability["persistentvolumes"] = 100
	watchCapability["persistentvolumeclaims"] = 100
	watchCapability["horizontalpodautoscalers"] = 100
	watchCapability["daemonsets"] = 100
	watchCapability["deployments"] = 100
	watchCapability["jobs"] = 100
	watchCapability["ingress"] = 100
}

func SetSizeByKey(key string, size int) {
	watchCapability[key] = size
}

func GetPodsSize() int {
	return watchCapability["pods"]
}

func GetNodesSize() int {
	return watchCapability["nodes"]
}

func GetServicesSize() int {
	return watchCapability["services"]
}

func GetEndpointsSize() int {
	return watchCapability["endpoints"]
}

func GetNamespacesSize() int {
	return watchCapability["namespaces"]
}

func GetControllersSize() int {
	return watchCapability["controllers"]
}

func GetPodTemplatesSize() int {
	return watchCapability["podtemplates"]
}

func GetLimitRangesSize() int {
	return watchCapability["limitranges"]
}

func GetResourceQuotasSize() int {
	return watchCapability["resourcequotas"]
}

func GetSecretsSize() int {
	return watchCapability["secrets"]
}

func GetServiceAccountsSize() int {
	return watchCapability["serviceaccounts"]
}

func GetPersistentVolumesSize() int {
	return watchCapability["persistentvolumes"]
}

func GetPersistentVolumeClaimsSize() int {
	return watchCapability["persistentvolumeclaims"]
}

func GetHorizontalPodAutoscalersSize() int {
	return watchCapability["horizontalpodautoscalers"]
}

func GetDaemonSetsSize() int {
	return watchCapability["daemonsets"]
}

func GetDeploymentsSize() int {
	return watchCapability["deployments"]
}

func GetJobsSize() int {
	return watchCapability["jobs"]
}

func GetIngressSize() int {
	return watchCapability["ingress"]
}
