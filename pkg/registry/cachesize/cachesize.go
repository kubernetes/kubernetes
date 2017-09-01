/*
Copyright 2014 The Kubernetes Authors.

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

//use for --watch-cache-sizes param of kube-apiserver
//make watch cache size of resources configurable
package cachesize

import (
	"strconv"
	"strings"

	"github.com/golang/glog"
)

type Resource string

const (
	APIServices                Resource = "apiservices"
	CertificateSigningRequests Resource = "certificatesigningrequests"
	ClusterRoles               Resource = "clusterroles"
	ClusterRoleBindings        Resource = "clusterrolebindings"
	ConfigMaps                 Resource = "configmaps"
	Controllers                Resource = "controllers"
	Daemonsets                 Resource = "daemonsets"
	Deployments                Resource = "deployments"
	Endpoints                  Resource = "endpoints"
	HorizontalPodAutoscalers   Resource = "horizontalpodautoscalers"
	Ingress                    Resource = "ingress"
	PodDisruptionBudget        Resource = "poddisruptionbudgets"
	StatefulSet                Resource = "statefulset"
	Jobs                       Resource = "jobs"
	LimitRanges                Resource = "limitranges"
	Namespaces                 Resource = "namespaces"
	NetworkPolicys             Resource = "networkpolicies"
	Nodes                      Resource = "nodes"
	PersistentVolumes          Resource = "persistentvolumes"
	PersistentVolumeClaims     Resource = "persistentvolumeclaims"
	Pods                       Resource = "pods"
	PodSecurityPolicies        Resource = "podsecuritypolicies"
	PodTemplates               Resource = "podtemplates"
	PriorityClasses            Resource = "priorityclasses"
	Replicasets                Resource = "replicasets"
	ResourceQuotas             Resource = "resourcequotas"
	CronJobs                   Resource = "cronjobs"
	Roles                      Resource = "roles"
	RoleBindings               Resource = "rolebindings"
	Secrets                    Resource = "secrets"
	ServiceAccounts            Resource = "serviceaccounts"
	Services                   Resource = "services"
	StorageClasses             Resource = "storageclasses"
)

// TODO: This shouldn't be a global variable.
var watchCacheSizes map[Resource]int

func init() {
	watchCacheSizes = make(map[Resource]int)
}

func InitializeWatchCacheSizes(expectedRAMCapacityMB int) {
	// This is the heuristics that from memory capacity is trying to infer
	// the maximum number of nodes in the cluster and set cache sizes based
	// on that value.
	// From our documentation, we officially recommend 120GB machines for
	// 2000 nodes, and we scale from that point. Thus we assume ~60MB of
	// capacity per node.
	// TODO: Revisit this heuristics
	clusterSize := expectedRAMCapacityMB / 60

	// We should specify cache size for a given resource only if it
	// is supposed to have non-default value.
	//
	// TODO: Figure out which resource we should have non-default value.
	watchCacheSizes[Controllers] = maxInt(5*clusterSize, 100)
	watchCacheSizes[Endpoints] = maxInt(10*clusterSize, 1000)
	watchCacheSizes[Nodes] = maxInt(5*clusterSize, 1000)
	watchCacheSizes[Pods] = maxInt(50*clusterSize, 1000)
	watchCacheSizes[Services] = maxInt(5*clusterSize, 1000)
	watchCacheSizes[APIServices] = maxInt(5*clusterSize, 1000)
}

func SetWatchCacheSizes(cacheSizes []string) {
	for _, c := range cacheSizes {
		tokens := strings.Split(c, "#")
		if len(tokens) != 2 {
			glog.Errorf("invalid value of watch cache capabilities: %s", c)
			continue
		}

		size, err := strconv.Atoi(tokens[1])
		if err != nil {
			glog.Errorf("invalid size of watch cache capabilities: %s", c)
			continue
		}
		if size < 0 {
			glog.Errorf("watch cache size cannot be negative: %s", c)
			continue
		}

		watchCacheSizes[Resource(strings.ToLower(tokens[0]))] = size
	}
}

// GetWatchCacheSizeByResource returns the configured watch cache size for the given resource.
// A nil value means to use a default size, zero means to disable caching.
func GetWatchCacheSizeByResource(resource string) (ret *int) { // TODO this should use schema.GroupResource for lookups
	if value, found := watchCacheSizes[Resource(resource)]; found {
		return &value
	}
	return nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
