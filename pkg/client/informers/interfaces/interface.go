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

package interfaces

import (
	"time"

	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/runtime"
)

type NewInternalInformerFunc func(internalclientset.Interface, time.Duration) cache.SharedIndexInformer
type NewVersionedInformerFunc func(release_1_5.Interface, time.Duration) cache.SharedIndexInformer

type SharedInformerFactory interface {
	Start(stopCh <-chan struct{})
	InternalInformerFor(obj runtime.Object, newFunc NewInternalInformerFunc) cache.SharedIndexInformer
	VersionedInformerFor(obj runtime.Object, newFunc NewVersionedInformerFunc) cache.SharedIndexInformer
}
