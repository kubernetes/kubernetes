/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"

	"k8s.io/kubernetes/pkg/registry/controller"
	"k8s.io/kubernetes/pkg/registry/controller/etcd"

	"k8s.io/kubernetes/pkg/apis/experimental"
)

// Container includes dummy storage for RC pods and experimental storage for Scale.
type ContainerStorage struct {
	ReplicationController *RcREST
	Scale                 *ScaleREST
}

func NewStorage(s storage.Interface) ContainerStorage {
	rcRegistry := controller.NewRegistry(etcd.NewREST(s))

	return ContainerStorage{
		ReplicationController: &RcREST{},
		Scale: &ScaleREST{registry: &rcRegistry},
	}
}

type ScaleREST struct {
	registry *controller.Registry
}

// LogREST implements GetterWithOptions
var _ = rest.Patcher(&ScaleREST{})

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &experimental.Scale{}
}

func (r *ScaleREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	rc, err := (*r.registry).GetController(ctx, name)
	if err != nil {
		return nil, errors.NewNotFound("scale", name)
	}
	return &experimental.Scale{
		ObjectMeta: api.ObjectMeta{
			Name:              name,
			Namespace:         rc.Namespace,
			CreationTimestamp: rc.CreationTimestamp,
		},
		Spec: experimental.ScaleSpec{
			Replicas: rc.Spec.Replicas,
		},
		Status: experimental.ScaleStatus{
			Replicas: rc.Status.Replicas,
			Selector: rc.Spec.Selector,
		},
	}, nil
}

func (r *ScaleREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	if obj == nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("nil update passed to Scale"))
	}
	scale, ok := obj.(*experimental.Scale)
	if !ok {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("wrong object passed to Scale update: %v", obj))
	}
	rc, err := (*r.registry).GetController(ctx, scale.Name)
	if err != nil {
		return nil, false, errors.NewNotFound("scale", scale.Name)
	}
	rc.Spec.Replicas = scale.Spec.Replicas
	rc, err = (*r.registry).UpdateController(ctx, rc)
	if err != nil {
		return nil, false, errors.NewConflict("scale", scale.Name, err)
	}
	return &experimental.Scale{
		ObjectMeta: api.ObjectMeta{
			Name:              rc.Name,
			Namespace:         rc.Namespace,
			CreationTimestamp: rc.CreationTimestamp,
		},
		Spec: experimental.ScaleSpec{
			Replicas: rc.Spec.Replicas,
		},
		Status: experimental.ScaleStatus{
			Replicas: rc.Status.Replicas,
			Selector: rc.Spec.Selector,
		},
	}, false, nil
}

// Dummy implementation
type RcREST struct{}

func (r *RcREST) New() runtime.Object {
	return &experimental.ReplicationControllerDummy{}
}
