package etcd

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	ingressPoint "k8s.io/kubernetes/pkg/registry/ingresspoint"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

const (
	IngressPointPath string = "/ingressPoints"
)

// rest implements a RESTStorage for replication controllers against etcd
type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against replication controllers.
func NewREST(s storage.Interface) *REST {
	store := &etcdgeneric.Etcd{
		NewFunc: func() runtime.Object { return &api.IngressPoint{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: func() runtime.Object { return &api.IngressPointList{} },
		// Produces a ingressPoint that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, IngressPointPath)
		},
		// Produces a ingressPoint that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, IngressPointPath, name)
		},
		// Retrieve the name field of a replication controller
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.IngressPoint).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return ingressPoint.MatchIngressPoint(label, field)
		},
		EndpointName: "ingressPoints",

		// Used to validate controller creation
		CreateStrategy: ingressPoint.Strategy,

		// Used to validate controller updates
		UpdateStrategy: ingressPoint.Strategy,

		Storage: s,
	}

	return &REST{store}
}
