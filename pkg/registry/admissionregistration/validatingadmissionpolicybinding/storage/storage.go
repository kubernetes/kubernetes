/*
Copyright 2022 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/validatingadmissionpolicybinding"
	"k8s.io/kubernetes/pkg/registry/rbac"
)

// REST implements a RESTStorage for policyBinding against etcd
type REST struct {
	*genericregistry.Store
	authorize AuthorizationFunc
}

// AuthorizationFunc checks the user from the context
// to determine if the user can perform the requested action.
// It returns no error if the user is authorized, an error
// indicating the reason of rejection otherwise.
type AuthorizationFunc func(ctx context.Context) error

var groupResource = admissionregistration.Resource("validatingadmissionpolicybindings")

// NewREST returns a RESTStorage object that will work against policyBinding.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, error) {
	r := &REST{authorize: superuserOnly}
	store := &genericregistry.Store{
		NewFunc:     func() runtime.Object { return &admissionregistration.ValidatingAdmissionPolicyBinding{} },
		NewListFunc: func() runtime.Object { return &admissionregistration.ValidatingAdmissionPolicyBindingList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*admissionregistration.ValidatingAdmissionPolicyBinding).Name, nil
		},
		DefaultQualifiedResource: groupResource,

		CreateStrategy: validatingadmissionpolicybinding.Strategy,
		UpdateStrategy: validatingadmissionpolicybinding.Strategy,
		DeleteStrategy: validatingadmissionpolicybinding.Strategy,

		BeginCreate: r.beginCreateFunc(),
		BeginUpdate: r.beginUpdateFunc(),

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}
	r.Store = store
	return r, nil
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"api-extensions"}
}

func (r *REST) beginCreateFunc() genericregistry.BeginCreateFunc {
	return func(ctx context.Context, obj runtime.Object, options *metav1.CreateOptions) (genericregistry.FinishFunc, error) {
		err := r.authorize(ctx)
		if err == nil {
			return noop, nil
		}
		name := ""
		if b, ok := obj.(*admissionregistration.ValidatingAdmissionPolicyBinding); ok && b != nil {
			name = b.Name
		}
		return nil, errors.NewForbidden(groupResource, name, err)
	}
}

func (r *REST) beginUpdateFunc() genericregistry.BeginUpdateFunc {
	return func(ctx context.Context, obj, old runtime.Object, options *metav1.UpdateOptions) (genericregistry.FinishFunc, error) {
		err := r.authorize(ctx)
		if err == nil {
			return noop, nil
		}
		name := ""
		if b, ok := obj.(*admissionregistration.ValidatingAdmissionPolicyBinding); ok && b != nil {
			name = b.Name
		} else if b, ok := obj.(*admissionregistration.ValidatingAdmissionPolicyBinding); ok && b != nil {
			name = b.Name
		}
		return nil, errors.NewForbidden(groupResource, name, err)
	}
}

func noop(context.Context, bool) {}

func superuserOnly(ctx context.Context) error {
	if rbac.EscalationAllowed(ctx) {
		return nil
	}
	return ErrNotSuperuser
}

var _ genericregistry.FinishFunc = noop
var _ AuthorizationFunc = superuserOnly

// ErrNotSuperuser is returned if the user sending the request is not considered a superuser.
var ErrNotSuperuser = fmt.Errorf("not a superuser")
