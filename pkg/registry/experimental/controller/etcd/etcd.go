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

package etcd

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/registry/controller"
	"k8s.io/kubernetes/pkg/registry/controller/etcd"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"

	"k8s.io/kubernetes/pkg/apis/extensions"

	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
)

// Container includes dummy storage for RC pods and experimental storage for Scale.
type ContainerStorage struct {
	ReplicationController *RcREST
	Scale                 *ScaleREST
}

func NewStorage(opts generic.RESTOptions) ContainerStorage {
	// scale does not set status, only updates spec so we ignore the status
	controllerREST, _ := etcd.NewREST(opts)
	rcRegistry := controller.NewRegistry(controllerREST)

	return ContainerStorage{
		ReplicationController: &RcREST{},
		Scale: &ScaleREST{registry: &rcRegistry},
	}
}

type ScaleREST struct {
	registry *controller.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &extensions.Scale{}
}

func (r *ScaleREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	rc, err := (*r.registry).GetController(ctx, name)
	if err != nil {
		return nil, errors.NewNotFound(extensions.Resource("replicationcontrollers/scale"), name)
	}
	return scaleFromRC(rc), nil
}

func (r *ScaleREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	rc, err := (*r.registry).GetController(ctx, name)
	if err != nil {
		return nil, false, errors.NewNotFound(extensions.Resource("replicationcontrollers/scale"), name)
	}
	oldScale := scaleFromRC(rc)

	obj, err := objInfo.UpdatedObject(ctx, oldScale)

	if obj == nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("nil update passed to Scale"))
	}
	scale, ok := obj.(*extensions.Scale)
	if !ok {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("wrong object passed to Scale update: %v", obj))
	}

	if errs := extvalidation.ValidateScale(scale); len(errs) > 0 {
		return nil, false, errors.NewInvalid(extensions.Kind("Scale"), scale.Name, errs)
	}

	rc.Spec.Replicas = scale.Spec.Replicas
	rc.ResourceVersion = scale.ResourceVersion
	rc, err = (*r.registry).UpdateController(ctx, rc)
	if err != nil {
		return nil, false, errors.NewConflict(extensions.Resource("replicationcontrollers/scale"), scale.Name, err)
	}
	return scaleFromRC(rc), false, nil
}

// scaleFromRC returns a scale subresource for a replication controller.
func scaleFromRC(rc *api.ReplicationController) *extensions.Scale {
	return &extensions.Scale{
		ObjectMeta: api.ObjectMeta{
			Name:              rc.Name,
			Namespace:         rc.Namespace,
			UID:               rc.UID,
			ResourceVersion:   rc.ResourceVersion,
			CreationTimestamp: rc.CreationTimestamp,
		},
		Spec: extensions.ScaleSpec{
			Replicas: rc.Spec.Replicas,
		},
		Status: extensions.ScaleStatus{
			Replicas: rc.Status.Replicas,
			Selector: &unversioned.LabelSelector{
				MatchLabels: rc.Spec.Selector,
			},
		},
	}
}

// Dummy implementation
type RcREST struct{}

func (r *RcREST) New() runtime.Object {
	return &extensions.ReplicationControllerDummy{}
}
