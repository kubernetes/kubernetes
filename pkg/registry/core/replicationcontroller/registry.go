/*
Copyright 2014 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicaSet.

package replicationcontroller

import (
	"fmt"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
)

// Registry is an interface for things that know how to store ReplicationControllers.
type Registry interface {
	ListControllers(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*api.ReplicationControllerList, error)
	WatchControllers(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
	GetController(ctx genericapirequest.Context, controllerID string, options *metav1.GetOptions) (*api.ReplicationController, error)
	CreateController(ctx genericapirequest.Context, controller *api.ReplicationController) (*api.ReplicationController, error)
	UpdateController(ctx genericapirequest.Context, controller *api.ReplicationController) (*api.ReplicationController, error)
	DeleteController(ctx genericapirequest.Context, controllerID string) error
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListControllers(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*api.ReplicationControllerList, error) {
	if options != nil && options.FieldSelector != nil && !options.FieldSelector.Empty() {
		return nil, fmt.Errorf("field selector not supported yet")
	}
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ReplicationControllerList), err
}

func (s *storage) WatchControllers(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetController(ctx genericapirequest.Context, controllerID string, options *metav1.GetOptions) (*api.ReplicationController, error) {
	obj, err := s.Get(ctx, controllerID, options)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ReplicationController), nil
}

func (s *storage) CreateController(ctx genericapirequest.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, err := s.Create(ctx, controller)
	if err != nil {
		return nil, err
	}
	return obj.(*api.ReplicationController), nil
}

func (s *storage) UpdateController(ctx genericapirequest.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	obj, _, err := s.Update(ctx, controller.Name, rest.DefaultUpdatedObjectInfo(controller, api.Scheme))
	if err != nil {
		return nil, err
	}
	return obj.(*api.ReplicationController), nil
}

func (s *storage) DeleteController(ctx genericapirequest.Context, controllerID string) error {
	_, err := s.Delete(ctx, controllerID, nil)
	return err
}
