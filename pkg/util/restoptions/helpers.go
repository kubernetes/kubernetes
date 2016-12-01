package restoptions

import (
	"strings"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// defaultKeyFunctions sets the default behavior for storage key generation onto a Store.
func defaultKeyFunctions(store *registry.Store, prefix string) {
	if store.CreateStrategy.NamespaceScoped() {
		if store.KeyRootFunc == nil {
			store.KeyRootFunc = func(ctx api.Context) string {
				return registry.NamespaceKeyRootFunc(ctx, prefix)
			}
		}
		if store.KeyFunc == nil {
			store.KeyFunc = func(ctx api.Context, name string) (string, error) {
				return registry.NamespaceKeyFunc(ctx, prefix, name)
			}
		}
	} else {
		if store.KeyRootFunc == nil {
			store.KeyRootFunc = func(ctx api.Context) string {
				return prefix
			}
		}
		if store.KeyFunc == nil {
			store.KeyFunc = func(ctx api.Context, name string) (string, error) {
				return registry.NoNamespaceKeyFunc(ctx, prefix, name)
			}
		}
	}
}

// ApplyOptions updates the given generic storage from the provided rest options
func ApplyOptions(optsGetter genericapiserver.RESTOptionsGetter, store *registry.Store, triggerFn storage.TriggerPublisherFunc) {
	if store.QualifiedResource.Empty() {
		glog.Fatalf("store %#v must have a non-empty qualified resource", store)
	}
	if store.NewFunc == nil {
		glog.Fatalf("store for %s must have NewFunc set", store.QualifiedResource.String())
	}
	if store.NewListFunc == nil {
		glog.Fatalf("store for %s must have NewListFunc set", store.QualifiedResource.String())
	}
	if store.CreateStrategy == nil {
		glog.Fatalf("store for %s must have CreateStrategy set", store.QualifiedResource.String())
	}

	opts := optsGetter(store.QualifiedResource)

	// Resource prefix must come from the underlying factory
	prefix := opts.ResourcePrefix
	if !strings.HasPrefix(prefix, "/") {
		prefix = "/" + prefix
	}

	defaultKeyFunctions(store, prefix)

	keyFunc := func(obj runtime.Object) (string, error) {
		accessor, err := meta.Accessor(obj)
		if err != nil {
			return "", err
		}

		if store.CreateStrategy.NamespaceScoped() {
			return store.KeyFunc(api.WithNamespace(api.NewContext(), accessor.GetNamespace()), accessor.GetName())
		}

		return store.KeyFunc(api.NewContext(), accessor.GetName())
	}

	store.DeleteCollectionWorkers = opts.DeleteCollectionWorkers
	store.EnableGarbageCollection = opts.EnableGarbageCollection
	store.Storage, store.DestroyFunc = opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.Resource(store.QualifiedResource.Resource)),
		store.NewFunc(),
		prefix,
		keyFunc,
		store.NewListFunc,
		triggerFn,
	)
}
