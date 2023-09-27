/*
Copyright 2022 The KCP Authors.

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

package kcp

import (
	"context"

	"github.com/kcp-dev/logicalcluster/v3"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// ClusterAwareCRDClusterLister knows how to scope down to a ClusterAwareCRDLister for one cluster.
type ClusterAwareCRDClusterLister interface {
	Cluster(logicalcluster.Name) ClusterAwareCRDLister
}

// ClusterAwareCRDLister is a CRD lister that is kcp-specific.
type ClusterAwareCRDLister interface {
	// List lists all CRDs matching selector.
	List(ctx context.Context, selector labels.Selector) ([]*v1.CustomResourceDefinition, error)
	// Get gets a CRD by name.
	Get(ctx context.Context, name string) (*v1.CustomResourceDefinition, error)
	// Refresh gets the current/latest copy of the CRD from the cache. This is necessary to ensure the identity
	// annotation is present when called by crdHandler.getOrCreateServingInfoFor
	Refresh(crd *v1.CustomResourceDefinition) (*v1.CustomResourceDefinition, error)
}
