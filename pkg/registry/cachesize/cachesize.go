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
	Controllers              Resource = "controllers"
	Daemonsets               Resource = "daemonsets"
	Deployments              Resource = "deployments"
	Endpoints                Resource = "endpoints"
	HorizontalPodAutoscalers Resource = "horizontalpodautoscalers"
	Ingress                  Resource = "ingress"
	Jobs                     Resource = "jobs"
	LimitRanges              Resource = "limitranges"
	Namespaces               Resource = "namespaces"
	Nodes                    Resource = "nodes"
	PersistentVolumes        Resource = "persistentvolumes"
	PersistentVolumeClaims   Resource = "persistentvolumeclaims"
	Pods                     Resource = "pods"
	PodTemplates             Resource = "podtemplates"
	Replicasets              Resource = "replicasets"
	ResourceQuotas           Resource = "resourcequotas"
	Secrets                  Resource = "secrets"
	ServiceAccounts          Resource = "serviceaccounts"
	Services                 Resource = "services"
)

var watchCacheSizes map[Resource]int

func init() {
	watchCacheSizes = make(map[Resource]int)
	watchCacheSizes[Controllers] = 100
	watchCacheSizes[Daemonsets] = 100
	watchCacheSizes[Deployments] = 100
	watchCacheSizes[Endpoints] = 1000
	watchCacheSizes[HorizontalPodAutoscalers] = 100
	watchCacheSizes[Ingress] = 100
	watchCacheSizes[Jobs] = 100
	watchCacheSizes[LimitRanges] = 100
	watchCacheSizes[Namespaces] = 100
	watchCacheSizes[Nodes] = 1000
	watchCacheSizes[PersistentVolumes] = 100
	watchCacheSizes[PersistentVolumeClaims] = 100
	watchCacheSizes[Pods] = 1000
	watchCacheSizes[PodTemplates] = 100
	watchCacheSizes[Replicasets] = 100
	watchCacheSizes[ResourceQuotas] = 100
	watchCacheSizes[Secrets] = 100
	watchCacheSizes[ServiceAccounts] = 100
	watchCacheSizes[Services] = 100
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

		watchCacheSizes[Resource(strings.ToLower(tokens[0]))] = size
	}
}

func GetWatchCacheSizeByResource(resource Resource) int {
	return watchCacheSizes[resource]
}
