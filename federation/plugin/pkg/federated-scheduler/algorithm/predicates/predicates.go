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

	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
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
	*release_1_3.Clientset
}

func (clusters ClientClusterInfo) GetClusterInfo(clusterName string) (*federation.Cluster, error) {
	return clusters.Clusters().Get(clusterName)
}

type CachedClusterInfo struct {
	*cache.StoreToClusterLister
}

// GetClusterInfo returns cached data for the cluster 'id'.
func (c *CachedClusterInfo) GetClusterInfo(id string) (*federation.Cluster, error) {
	cluster, exists, err := c.Get(&federation.Cluster{ObjectMeta: v1.ObjectMeta{Name: id}})

	if err != nil {
		return nil, fmt.Errorf("error retrieving cluster '%v' from cache: %v", id, err)
	}

	if !exists {
		return nil, fmt.Errorf("cluster '%v' is not in cache", id)
	}

	return cluster.(*federation.Cluster), nil
}

func NewSelectorMatchPredicate(info ClusterInfo) algorithm.FitPredicate {
	selector := &ClusterSelector{
		info: info,
	}
	return selector.RSAnnotationMatches
}

func rsMatchesClusterName(rs *extensions.ReplicaSet, cluster *federation.Cluster) bool {
	clusterSelection, ok := rs.Annotations[unversioned.ClusterSelectorKey]
	if !ok || clusterSelection == "" {
		glog.V(4).Infof("no target cluster is specified, any cluster can be scheduling candidate, return true.")
		return true
	}
	selectedClusters := parseClusterSelectorAnnotation(clusterSelection)
	for i := range selectedClusters {
		if selectedClusters[i] == cluster.Name {
			glog.V(4).Infof("target cluster is specified and found, return true.")
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
func parseClusterSelectorAnnotation(selectorAnnotationString string) []string {
	//assume ube-apiserver covers the validation, and the value should be a string of "Foo, Bar"
	targets := strings.Split(selectorAnnotationString, ",")
	results := []string{}
	for i := range targets {
		results = append(results, strings.TrimSpace(targets[i]))
	}
	return results
}

func (c *ClusterSelector) RSAnnotationMatches(rs *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	cluster, err := c.info.GetClusterInfo(clusterName)
	if err != nil {
		return false, err
	}
	return rsMatchesClusterName(rs, cluster), nil
}
