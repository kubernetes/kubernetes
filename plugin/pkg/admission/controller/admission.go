package controller

import (
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	controller "k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

)

var (
	selectorKeyFunc = controller.NamespaceSelectorKeyFunc
)


func init() {
	admission.RegisterPlugin("ReplicationControllerExists", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewExists(client), nil
	})
}

// exists is an implementation of admission.Interface.
// It rejects all incoming CREATE or UPDATE requests of replication controller if it exists
// It is useful in deployments that want to filter out repeated replication controller
type exists struct {
	*admission.Handler
	client client.Interface
	store  cache.Store
}

func (e *exists) Admit(a admission.Attributes) (err error) {
	if a.GetResource() == "replicationcontrollers" {
		_, ok := a.GetObject().(*api.ReplicationController)
		if !ok {
			return nil
		}
		key, err := selectorKeyFunc(a.GetObject())
		_, exists, err := e.store.GetByKey(key)
		if err != nil {
			return admission.NewForbidden(a, err)
		}
		if exists {
			return admission.NewForbidden(a, fmt.Errorf("A replication with the same selctor has exist"))
		}
		return nil
	}
	return nil
}


// NewExists creates a new replication controller exists admission control handler
func NewExists(client client.Interface) admission.Interface {
	store := cache.NewStore(selectorKeyFunc)
	reflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return client.ReplicationControllers(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return client.ReplicationControllers(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.ReplicationController{},
		store,
		0,
	)
	reflector.Run()
	return &exists{
		client:  client,
		store:   store,
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}
