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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/controlplane"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/schedulercache"

	"github.com/golang/glog"
)

type ClusterInfo interface {
	GetClusterInfo(clusterName string) (*controlplane.Cluster, error)
}


type StaticClusterInfo struct {
	*api.ClusterList
}

func (clusters StaticClusterInfo) GetClusterInfo(clusterName string) (*controlplane.Cluster, error) {
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

func (clusters ClientClusterInfo) GetClusterInfo(clusterName string) (*controlplane.Cluster, error) {
	return clusters.Clusters().Get(clusterName)
}

type CachedClusterInfo struct {
	*cache.StoreToClusterLister
}

type ResourceFit struct {
	info ClusterInfo
}

type resourceRequest struct {
	milliCPU int64
	memory   int64
}

func getResourceRequest(rc *api.ReplicationController) resourceRequest {
	result := resourceRequest{}
	for _, container := range rc.Spec.Template.Spec.Containers {
		requests := container.Resources.Requests
		result.memory += requests.Memory().Value()
		result.milliCPU += requests.Cpu().MilliValue()
	}
	return result
}

// PodFitsResources calculates fit based on requested, rather than used resources
func (r *ResourceFit) ReplicationControllerFitsResources(rc *api.ReplicationController, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	info, err := r.info.GetClusterInfo(clusterName)
	if err != nil {
		return false, err
	}
	//TODO: assume Capacity is the free resources of the clusters.
	allocatable := info.Status.Capacity

	podRequest := getResourceRequest(rc)
	if podRequest.milliCPU == 0 && podRequest.memory == 0 {
		return true, nil
	}
	//TODO: we do not need split the Federation RC in Phase 1, so check check allocatable resource for the whole RC works
	//If we decide to split the Federation RC to multiple clusters, we need split first and check allocatable resource for sub-rc here
	totalMilliCPU := allocatable.Cpu().MilliValue()
	totalMemory := allocatable.Memory().Value()
	if totalMilliCPU < podRequest.milliCPU+ clusterInfo.RequestedResource().MilliCPU {
		return false,
			newInsufficientResourceError(cpuResourceName, podRequest.milliCPU, clusterInfo.RequestedResource().MilliCPU, totalMilliCPU)
	}
	if totalMemory < podRequest.memory+ clusterInfo.RequestedResource().Memory {
		return false,
			newInsufficientResourceError(memoryResoureceName, podRequest.memory, clusterInfo.RequestedResource().Memory, totalMemory)
	}
	return true, nil
}

func NewResourceFitPredicate(info ClusterInfo) algorithm.FitPredicate {
	fit := &ResourceFit{
		info: info,
	}
	return fit.ReplicationControllerFitsResources
}

func NewSelectorMatchPredicate(info ClusterInfo) algorithm.FitPredicate {
	selector := &ClusterSelector{
		info: info,
	}
	return selector.RCSelectorMatches
}

// nodeMatchesNodeSelectorTerms checks if a node's labels satisfy a list of node selector terms,
// terms are ORed, and an emtpy a list of terms will match nothing.
func clusterMatchesClusterSelectorTerms(cluster *controlplane.Cluster, clusterSelectorTerms []api.ClusterSelectorTerm) bool {
	for _, req := range clusterSelectorTerms {
		clusterSelector, err := api.ClusterSelectorRequirementsAsSelector(req.MatchExpressions)
		if err != nil {
			glog.V(10).Infof("Failed to parse MatchExpressions: %+v, regarding as not match.", req.MatchExpressions)
			return false
		}
		if clusterSelector.Matches(labels.Set(cluster.Labels)) {
			return true
		}
	}
	return false
}

func RCMatchesClusterLabels(rc *api.ReplicationController, cluster *controlplane.Cluster) bool {
	// Check if cluster.Labels match rc.Spec.Template.Spec.ClusterSelector.
	if len(rc.Spec.Template.Spec.ClusterSelector) > 0 {
		selector := labels.SelectorFromSet(rc.Spec.Template.Spec.ClusterSelector)
		if !selector.Matches(labels.Set(cluster.Labels)) {
			return false
		}
	}
	return true
}

type ClusterSelector struct {
	info ClusterInfo
}

func (n *ClusterSelector) RCSelectorMatches(rc *api.ReplicationController, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	node, err := n.info.GetClusterInfo(clusterName)
	if err != nil {
		return false, err
	}
	return RCMatchesClusterLabels(rc, node), nil
}