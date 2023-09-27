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

// +kcp-code-generator:skip

package informerfactoryhack

import (
	"reflect"

	kcpkubernetesinformers "github.com/kcp-dev/client-go/informers"
	"k8s.io/client-go/informers/resource"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/informers/admissionregistration"
	"k8s.io/client-go/informers/apiserverinternal"
	"k8s.io/client-go/informers/apps"
	"k8s.io/client-go/informers/autoscaling"
	"k8s.io/client-go/informers/batch"
	"k8s.io/client-go/informers/certificates"
	"k8s.io/client-go/informers/coordination"
	"k8s.io/client-go/informers/core"
	"k8s.io/client-go/informers/discovery"
	"k8s.io/client-go/informers/events"
	"k8s.io/client-go/informers/extensions"
	"k8s.io/client-go/informers/flowcontrol"
	"k8s.io/client-go/informers/internalinterfaces"
	"k8s.io/client-go/informers/networking"
	"k8s.io/client-go/informers/node"
	"k8s.io/client-go/informers/policy"
	"k8s.io/client-go/informers/rbac"
	"k8s.io/client-go/informers/scheduling"
	"k8s.io/client-go/informers/storage"
	"k8s.io/client-go/informers/storagemigration"
	"k8s.io/client-go/tools/cache"
)

// Interface allows us to hold onto a strongly-typed cluster-aware informer factory here, while
// passing in a cluster-unaware (but non-functional) factory to k8s libraries. We export this type so that we
// can get the cluster-aware factory back using casting in admission plugin initialization.
type Interface interface {
	informers.SharedInformerFactory
	ClusterAware() kcpkubernetesinformers.SharedInformerFactory
}

var _ Interface = (*hack)(nil)

// Wrap adapts a cluster-aware informer factory to a cluster-unaware wrapper that can divulge it after casting.
func Wrap(clusterAware kcpkubernetesinformers.SharedInformerFactory) Interface {
	return &hack{clusterAware: clusterAware}
}

// Unwrap extracts a cluster-aware informer factory from the cluster-unaware wrapper, or panics if we get the wrong input.
func Unwrap(clusterUnaware informers.SharedInformerFactory) kcpkubernetesinformers.SharedInformerFactory {
	return clusterUnaware.(Interface).ClusterAware()
}

type hack struct {
	clusterAware kcpkubernetesinformers.SharedInformerFactory
}

func (s *hack) Shutdown() {
	panic("not implemented yet")
}

func (s *hack) Resource() resource.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Start(stopCh <-chan struct{}) {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) InformerFor(obj runtime.Object, newFunc internalinterfaces.NewInformerFunc) cache.SharedIndexInformer {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) ExtraClusterScopedIndexers() cache.Indexers {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) ExtraNamespaceScopedIndexers() cache.Indexers {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) KeyFunction() cache.KeyFunc {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) ForResource(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) WaitForCacheSync(stopCh <-chan struct{}) map[reflect.Type]bool {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Admissionregistration() admissionregistration.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Internal() apiserverinternal.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Apps() apps.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Autoscaling() autoscaling.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Batch() batch.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Certificates() certificates.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Coordination() coordination.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Core() core.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Discovery() discovery.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Events() events.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Extensions() extensions.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Flowcontrol() flowcontrol.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Networking() networking.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Node() node.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Policy() policy.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Rbac() rbac.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Scheduling() scheduling.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Storage() storage.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) Storagemigration() storagemigration.Interface {
	panic("programmer error: using a cluster-unaware informer factory, need to cast this to use the cluster-aware one!")
}

func (s *hack) ClusterAware() kcpkubernetesinformers.SharedInformerFactory {
	return s.clusterAware
}
