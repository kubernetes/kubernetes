/*
Copyright 2016 The Kubernetes Authors.

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

package cache

import (
	kubecache "k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/federation/apis/federation/v1beta1"
)

// StoreToClusterLister makes a Store have the List method of the metav1.ClusterInterface
// The Store must contain (only) clusters.
type StoreToClusterLister struct {
	kubecache.Store
}

func (s *StoreToClusterLister) List() (clusters v1beta1.ClusterList, err error) {
	for _, m := range s.Store.List() {
		clusters.Items = append(clusters.Items, *(m.(*v1beta1.Cluster)))
	}
	return clusters, nil
}
