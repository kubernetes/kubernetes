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

package predicates

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/schedulercache"
)

type ClusterInfo interface {
	GetClusterInfo(clusterName string) (*federation.Cluster, error)
}


type StaticClusterInfo struct {
	*federation.ClusterList
}

func (clusters StaticClusterInfo) GetClusterInfo(clusterName string) (*federation.Cluster, error) {
	for ix := range clusters.Items {
		if clusters.Items[ix].Name == clusterName {
			return &clusters.Items[ix], nil
		}
	}
	return nil, fmt.Errorf("failed to find cluster: %s, %#v", clusterName, clusters)
}

type ClientClusterInfo struct {
	*client.Client
}

func (clusters ClientClusterInfo) GetClusterInfo(clusterName string) (*federation.Cluster, error) {
	return clusters.Clusters().Get(clusterName)
}

type CachedClusterInfo struct {
	*cache.StoreToClusterLister
}
// GetClusterInfo returns cached data for the cluster 'id'.
func (c *CachedClusterInfo) GetClusterInfo(id string) (*federation.Cluster, error) {
	cluster, exists, err := c.Get(&federation.Cluster{ObjectMeta: api.ObjectMeta{Name: id}})

	if err != nil {
		return nil, fmt.Errorf("error retrieving cluster '%v' from cache: %v", id, err)
	}

	if !exists {
		return nil, fmt.Errorf("cluster '%v' is not in cache", id)
	}

	return cluster.(*federation.Cluster), nil
}

type ResourceFit struct {
	info ClusterInfo
}

type resourceRequest struct {
	milliCPU int64
	memory   int64
}

func getResourceRequest(rs *extensions.ReplicaSet) resourceRequest {
	result := getResourceRequestPerReplica(rs)
	replicas := int64(rs.Spec.Replicas)
	result.memory *= replicas
	result.milliCPU *= replicas
	return result
}

func getResourceRequestPerReplica(rs *extensions.ReplicaSet) resourceRequest{
	result := resourceRequest{}
	for _, container := range rs.Spec.Template.Spec.Containers {
		requests := container.Resources.Requests
		result.memory += requests.Memory().Value()
		result.milliCPU += requests.Cpu().MilliValue()
	}
	return result

}

// ReplicaSetFitsResources calculates fit based on requested, rather than used resources
func (r *ResourceFit) ReplicaSetFitsResources(rs *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	info, err := r.info.GetClusterInfo(clusterName)
	if err != nil {
		return false, err
	}

	allocatable := info.Status.Allocatable
	//reRequest = replicas * request per replica
	rsRequest := getResourceRequest(rs)
	if rsRequest.milliCPU == 0 && rsRequest.memory == 0 {
		return true, nil
	}
	//we do not need split the ReplicaSet in Phase 1, so check check allocatable resource for the whole ReplicaSet
	totalMilliCPU := allocatable.Cpu().MilliValue()
	totalMemory := allocatable.Memory().Value()
	if totalMilliCPU < rsRequest.milliCPU + clusterInfo.RequestedResource().MilliCPU {
		return false,
			newInsufficientResourceError(cpuResourceName, rsRequest.milliCPU, clusterInfo.RequestedResource().MilliCPU, totalMilliCPU)
	}
	if totalMemory < rsRequest.memory + clusterInfo.RequestedResource().Memory {
		return false,
			newInsufficientResourceError(memoryResoureceName, rsRequest.memory, clusterInfo.RequestedResource().Memory, totalMemory)
	}
	return true, nil
}

func NewResourceFitPredicate(info ClusterInfo) algorithm.FitPredicate {
	fit := &ResourceFit{
		info: info,
	}
	return fit.ReplicaSetFitsResources
}

func NewSelectorMatchPredicate(info ClusterInfo) algorithm.FitPredicate {
	selector := &ClusterSelector{
		info: info,
	}
	return selector.RSAnnotationMatches
}

func RCMatchesClusterLabels(rs *extensions.ReplicaSet, cluster *federation.Cluster) bool {
	// Get the current annotations from the object.
	annotations := rs.GetAnnotations()
	if annotations == nil {
		annotations = map[string]string{}
	}
	clusterSelection := annotations[federation.ClusterSelectorKey]
	selectedClusters := parseClusterSelectorAnnotation(clusterSelection)
	for _, selectedCluster := range selectedClusters {
		if selectedCluster == cluster.Name {
			return true
		}
	}
	return false
}

type ClusterSelector struct {
	info ClusterInfo
}

//Example of ReplicaSet with cluster selector annotation
//	apiVersion: v1
//	kind: ReplicaSet
//	metadata:
//		name: nginx-controller
//	annotations:
//		ubernetes.kubernetes.io/cluster-name: Foo, Bar
func parseClusterSelectorAnnotation(selectorAnnotationString string) (cluster []string){
	//assume ube-apiserver covers the validation, and the value should be a string of "Foo, Bar"
	return strings.Split(selectorAnnotationString, ",")
}

func (n *ClusterSelector) RSAnnotationMatches(rs *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	cluster, err := n.info.GetClusterInfo(clusterName)
	if err != nil {
		return false, err
	}
	return RCMatchesClusterLabels(rs, cluster), nil
}