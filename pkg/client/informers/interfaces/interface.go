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
