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

package metadata

import (
	"context"

	"github.com/kcp-dev/logicalcluster/v3"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/metadata"
)

type ClusterInterface interface {
	Cluster(logicalcluster.Path) metadata.Interface
	Resource(resource schema.GroupVersionResource) ResourceClusterInterface
}

type ResourceClusterInterface interface {
	Cluster(logicalcluster.Path) metadata.Getter
	List(ctx context.Context, opts metav1.ListOptions) (*metav1.PartialObjectMetadataList, error)
	Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error)
}
