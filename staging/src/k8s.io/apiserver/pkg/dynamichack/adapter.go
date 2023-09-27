/*
Copyright 2023 The KCP Authors.

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

// +kcp-code-generator:skip

package dynamichack

import (
	kcpdynamic "github.com/kcp-dev/client-go/dynamic"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

// Interface allows us to hold onto a strongly-typed cluster-aware clients here, while
// passing in a cluster-unaware (but non-functional) clients to k8s libraries. We export this type so that we
// can get the cluster-aware clients back using casting in admission plugin initialization.
type Interface interface {
	dynamic.Interface
	ClusterAware() kcpdynamic.ClusterInterface
}

var _ Interface = (*hack)(nil)

// Wrap adapts a cluster-aware dynamic client to a cluster-unaware wrapper that can divulge it after casting.
func Wrap(clusterAware kcpdynamic.ClusterInterface) Interface {
	return &hack{clusterAware: clusterAware}
}

// Unwrap extracts a cluster-aware dynamic client from the cluster-unaware wrapper, or panics if we get the wrong input.
func Unwrap(clusterUnaware dynamic.Interface) kcpdynamic.ClusterInterface {
	return clusterUnaware.(Interface).ClusterAware()
}

type hack struct {
	clusterAware kcpdynamic.ClusterInterface
}

func (h hack) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	panic("programmer error: using a cluster-unaware dynamic client, need to cast this to use the cluster-aware one!")

}

func (h hack) ClusterAware() kcpdynamic.ClusterInterface {
	panic("programmer error: using a cluster-unaware dynamic, need to cast this to use the cluster-aware one!")

}
