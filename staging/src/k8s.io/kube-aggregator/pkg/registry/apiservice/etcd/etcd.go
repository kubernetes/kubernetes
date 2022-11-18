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

package etcd

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	metatable "k8s.io/apimachinery/pkg/api/meta/table"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	"k8s.io/kube-aggregator/pkg/registry/apiservice"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// REST implements a RESTStorage for API services against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against API services.
func NewREST(scheme *runtime.Scheme, optsGetter generic.RESTOptionsGetter) *REST {
	strategy := apiservice.NewStrategy(scheme)
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &apiregistration.APIService{} },
		NewListFunc:               func() runtime.Object { return &apiregistration.APIServiceList{} },
		PredicateFunc:             apiservice.MatchAPIService,
		DefaultQualifiedResource:  apiregistration.Resource("apiservices"),
		SingularQualifiedResource: apiregistration.Resource("apiservice"),

		CreateStrategy:      strategy,
		UpdateStrategy:      strategy,
		DeleteStrategy:      strategy,
		ResetFieldsStrategy: strategy,

		// TODO: define table converter that exposes more than name/creation timestamp
		TableConvertor: rest.NewDefaultTableConvertor(apiregistration.Resource("apiservices")),
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: apiservice.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}
	return &REST{store}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (c *REST) Categories() []string {
	return []string{"api-extensions"}
}

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// ConvertToTable implements the TableConvertor interface for REST.
func (c *REST) ConvertToTable(ctx context.Context, obj runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	table := &metav1.Table{
		ColumnDefinitions: []metav1.TableColumnDefinition{
			{Name: "Name", Type: "string", Format: "name", Description: swaggerMetadataDescriptions["name"]},
			{Name: "Service", Type: "string", Description: "The reference to the service that hosts this API endpoint."},
			{Name: "Available", Type: "string", Description: "Whether this service is available."},
			{Name: "Age", Type: "string", Description: swaggerMetadataDescriptions["creationTimestamp"]},
		},
	}
	if m, err := meta.ListAccessor(obj); err == nil {
		table.ResourceVersion = m.GetResourceVersion()
		table.Continue = m.GetContinue()
		table.RemainingItemCount = m.GetRemainingItemCount()
	} else {
		if m, err := meta.CommonAccessor(obj); err == nil {
			table.ResourceVersion = m.GetResourceVersion()
		}
	}

	var err error
	table.Rows, err = metatable.MetaToTableRow(obj, func(obj runtime.Object, m metav1.Object, name, age string) ([]interface{}, error) {
		svc := obj.(*apiregistration.APIService)
		service := "Local"
		if svc.Spec.Service != nil {
			service = fmt.Sprintf("%s/%s", svc.Spec.Service.Namespace, svc.Spec.Service.Name)
		}
		status := string(apiregistration.ConditionUnknown)
		if condition := getCondition(svc.Status.Conditions, "Available"); condition != nil {
			switch {
			case condition.Status == apiregistration.ConditionTrue:
				status = string(condition.Status)
			case len(condition.Reason) > 0:
				status = fmt.Sprintf("%s (%s)", condition.Status, condition.Reason)
			default:
				status = string(condition.Status)
			}
		}
		return []interface{}{name, service, status, age}, nil
	})
	return table, err
}

func getCondition(conditions []apiregistration.APIServiceCondition, conditionType apiregistration.APIServiceConditionType) *apiregistration.APIServiceCondition {
	for i, condition := range conditions {
		if condition.Type == conditionType {
			return &conditions[i]
		}
	}
	return nil
}

// NewStatusREST makes a RESTStorage for status that has more limited options.
// It is based on the original REST so that we can share the same underlying store
func NewStatusREST(scheme *runtime.Scheme, rest *REST) *StatusREST {
	strategy := apiservice.NewStatusStrategy(scheme)
	statusStore := *rest.Store
	statusStore.CreateStrategy = nil
	statusStore.DeleteStrategy = nil
	statusStore.UpdateStrategy = strategy
	statusStore.ResetFieldsStrategy = strategy
	return &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of an APIService.
type StatusREST struct {
	store *genericregistry.Store
}

var _ = rest.Patcher(&StatusREST{})

// New creates a new APIService object.
func (r *StatusREST) New() runtime.Object {
	return &apiregistration.APIService{}
}

// Destroy cleans up resources on shutdown.
func (r *StatusREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// We are explicitly setting forceAllowCreate to false in the call to the underlying storage because
	// subresources should never allow create on update.
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, false, options)
}

// GetResetFields implements rest.ResetFieldsStrategy
func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}
