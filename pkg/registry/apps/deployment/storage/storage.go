/*
Copyright 2015 The Kubernetes Authors.

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
	"net/http"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	storeerr "k8s.io/apiserver/pkg/storage/errors"
	"k8s.io/apiserver/pkg/util/dryrun"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/apps"
	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	appsv1beta2 "k8s.io/kubernetes/pkg/apis/apps/v1beta2"
	appsvalidation "k8s.io/kubernetes/pkg/apis/apps/validation"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	autoscalingvalidation "k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/apps/deployment"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// DeploymentStorage includes dummy storage for Deployments and for Scale subresource.
type DeploymentStorage struct {
	Deployment *REST
	Status     *StatusREST
	Scale      *ScaleREST
	Rollback   *RollbackREST
}

// ReplicasPathMappings returns the mappings between each group version and a replicas path
func ReplicasPathMappings() managedfields.ResourcePathMappings {
	return replicasPathInDeployment
}

// maps a group version to the replicas path in a deployment object
var replicasPathInDeployment = managedfields.ResourcePathMappings{
	schema.GroupVersion{Group: "apps", Version: "v1beta1"}.String(): fieldpath.MakePathOrDie("spec", "replicas"),
	schema.GroupVersion{Group: "apps", Version: "v1beta2"}.String(): fieldpath.MakePathOrDie("spec", "replicas"),
	schema.GroupVersion{Group: "apps", Version: "v1"}.String():      fieldpath.MakePathOrDie("spec", "replicas"),
}

// NewStorage returns new instance of DeploymentStorage.
func NewStorage(optsGetter generic.RESTOptionsGetter) (DeploymentStorage, error) {
	deploymentRest, deploymentStatusRest, deploymentRollbackRest, err := NewREST(optsGetter)
	if err != nil {
		return DeploymentStorage{}, err
	}

	return DeploymentStorage{
		Deployment: deploymentRest,
		Status:     deploymentStatusRest,
		Scale:      &ScaleREST{store: deploymentRest.Store},
		Rollback:   deploymentRollbackRest,
	}, nil
}

// REST implements a RESTStorage for Deployments.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against deployments.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, *RollbackREST, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &apps.Deployment{} },
		NewListFunc:               func() runtime.Object { return &apps.DeploymentList{} },
		DefaultQualifiedResource:  apps.Resource("deployments"),
		SingularQualifiedResource: apps.Resource("deployment"),

		CreateStrategy:      deployment.Strategy,
		UpdateStrategy:      deployment.Strategy,
		DeleteStrategy:      deployment.Strategy,
		ResetFieldsStrategy: deployment.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = deployment.StatusStrategy
	statusStore.ResetFieldsStrategy = deployment.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}, &RollbackREST{store: store}, nil
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"deploy"}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the REST endpoint for changing the status of a deployment
type StatusREST struct {
	store *genericregistry.Store
}

// New returns empty Deployment object.
func (r *StatusREST) New() runtime.Object {
	return &apps.Deployment{}
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

func (r *StatusREST) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return r.store.ConvertToTable(ctx, object, tableOptions)
}

// RollbackREST implements the REST endpoint for initiating the rollback of a deployment
type RollbackREST struct {
	store *genericregistry.Store
}

// ProducesMIMETypes returns a list of the MIME types the specified HTTP verb (GET, POST, DELETE,
// PATCH) can respond with.
func (r *RollbackREST) ProducesMIMETypes(verb string) []string {
	return nil
}

// ProducesObject returns an object the specified HTTP verb respond with. It will overwrite storage object if
// it is not nil. Only the type of the return object matters, the value will be ignored.
func (r *RollbackREST) ProducesObject(verb string) interface{} {
	return metav1.Status{}
}

var _ = rest.StorageMetadata(&RollbackREST{})

// New creates a rollback
func (r *RollbackREST) New() runtime.Object {
	return &apps.DeploymentRollback{}
}

// Destroy cleans up resources on shutdown.
func (r *RollbackREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

var _ = rest.NamedCreater(&RollbackREST{})

// Create runs rollback for deployment
func (r *RollbackREST) Create(ctx context.Context, name string, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	rollback, ok := obj.(*apps.DeploymentRollback)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("not a DeploymentRollback: %#v", obj))
	}

	if errs := appsvalidation.ValidateDeploymentRollback(rollback); len(errs) != 0 {
		return nil, errors.NewInvalid(apps.Kind("DeploymentRollback"), rollback.Name, errs)
	}
	if name != rollback.Name {
		return nil, errors.NewBadRequest("name in URL does not match name in DeploymentRollback object")
	}

	if createValidation != nil {
		if err := createValidation(ctx, obj.DeepCopyObject()); err != nil {
			return nil, err
		}
	}

	// Update the Deployment with information in DeploymentRollback to trigger rollback
	err := r.rollbackDeployment(ctx, rollback.Name, &rollback.RollbackTo, rollback.UpdatedAnnotations, dryrun.IsDryRun(options.DryRun))
	if err != nil {
		return nil, err
	}
	return &metav1.Status{
		Status:  metav1.StatusSuccess,
		Message: fmt.Sprintf("rollback request for deployment %q succeeded", rollback.Name),
		Code:    http.StatusOK,
	}, nil
}

func (r *RollbackREST) rollbackDeployment(ctx context.Context, deploymentID string, config *apps.RollbackConfig, annotations map[string]string, dryRun bool) error {
	if _, err := r.setDeploymentRollback(ctx, deploymentID, config, annotations, dryRun); err != nil {
		err = storeerr.InterpretGetError(err, apps.Resource("deployments"), deploymentID)
		err = storeerr.InterpretUpdateError(err, apps.Resource("deployments"), deploymentID)
		if _, ok := err.(*errors.StatusError); !ok {
			err = errors.NewInternalError(err)
		}
		return err
	}
	return nil
}

func (r *RollbackREST) setDeploymentRollback(ctx context.Context, deploymentID string, config *apps.RollbackConfig, annotations map[string]string, dryRun bool) (*apps.Deployment, error) {
	dKey, err := r.store.KeyFunc(ctx, deploymentID)
	if err != nil {
		return nil, err
	}
	var finalDeployment *apps.Deployment
	err = r.store.Storage.GuaranteedUpdate(ctx, dKey, &apps.Deployment{}, false, nil, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
		d, ok := obj.(*apps.Deployment)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		if d.Annotations == nil {
			d.Annotations = make(map[string]string)
		}
		for k, v := range annotations {
			d.Annotations[k] = v
		}
		d.Spec.RollbackTo = config
		finalDeployment = d
		return d, nil
	}), dryRun, nil)
	return finalDeployment, err
}

// ScaleREST implements a Scale for Deployment.
type ScaleREST struct {
	store *genericregistry.Store
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

// GroupVersionKind returns GroupVersionKind for Deployment Scale object
func (r *ScaleREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	switch containingGV {
	case extensionsv1beta1.SchemeGroupVersion:
		return extensionsv1beta1.SchemeGroupVersion.WithKind("Scale")
	case appsv1beta1.SchemeGroupVersion:
		return appsv1beta1.SchemeGroupVersion.WithKind("Scale")
	case appsv1beta2.SchemeGroupVersion:
		return appsv1beta2.SchemeGroupVersion.WithKind("Scale")
	default:
		return autoscalingv1.SchemeGroupVersion.WithKind("Scale")
	}
}

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &autoscaling.Scale{}
}

// Destroy cleans up resources on shutdown.
func (r *ScaleREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Get retrieves object from Scale storage.
func (r *ScaleREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	obj, err := r.store.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	deployment := obj.(*apps.Deployment)
	scale, err := scaleFromDeployment(deployment)
	if err != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return scale, nil
}

// Update alters scale subset of Deployment object.
func (r *ScaleREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, _, err := r.store.Update(
		ctx,
		name,
		&scaleUpdatedObjectInfo{name, objInfo},
		toScaleCreateValidation(createValidation),
		toScaleUpdateValidation(updateValidation),
		false,
		options,
	)
	if err != nil {
		return nil, false, err
	}
	deployment := obj.(*apps.Deployment)
	newScale, err := scaleFromDeployment(deployment)
	if err != nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return newScale, false, nil
}

func (r *ScaleREST) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return r.store.ConvertToTable(ctx, object, tableOptions)
}

func toScaleCreateValidation(f rest.ValidateObjectFunc) rest.ValidateObjectFunc {
	return func(ctx context.Context, obj runtime.Object) error {
		scale, err := scaleFromDeployment(obj.(*apps.Deployment))
		if err != nil {
			return err
		}
		return f(ctx, scale)
	}
}

func toScaleUpdateValidation(f rest.ValidateObjectUpdateFunc) rest.ValidateObjectUpdateFunc {
	return func(ctx context.Context, obj, old runtime.Object) error {
		newScale, err := scaleFromDeployment(obj.(*apps.Deployment))
		if err != nil {
			return err
		}
		oldScale, err := scaleFromDeployment(old.(*apps.Deployment))
		if err != nil {
			return err
		}
		return f(ctx, newScale, oldScale)
	}
}

// scaleFromDeployment returns a scale subresource for a deployment.
func scaleFromDeployment(deployment *apps.Deployment) (*autoscaling.Scale, error) {
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, err
	}

	return &autoscaling.Scale{
		// TODO: Create a variant of ObjectMeta type that only contains the fields below.
		ObjectMeta: metav1.ObjectMeta{
			Name:              deployment.Name,
			Namespace:         deployment.Namespace,
			UID:               deployment.UID,
			ResourceVersion:   deployment.ResourceVersion,
			CreationTimestamp: deployment.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: deployment.Spec.Replicas,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: deployment.Status.Replicas,
			Selector: selector.String(),
		},
	}, nil
}

// scaleUpdatedObjectInfo transforms existing deployment -> existing scale -> new scale -> new deployment
type scaleUpdatedObjectInfo struct {
	name       string
	reqObjInfo rest.UpdatedObjectInfo
}

func (i *scaleUpdatedObjectInfo) Preconditions() *metav1.Preconditions {
	return i.reqObjInfo.Preconditions()
}

func (i *scaleUpdatedObjectInfo) UpdatedObject(ctx context.Context, oldObj runtime.Object) (runtime.Object, error) {
	deployment, ok := oldObj.DeepCopyObject().(*apps.Deployment)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("expected existing object type to be Deployment, got %T", deployment))
	}
	// if zero-value, the existing object does not exist
	if len(deployment.ResourceVersion) == 0 {
		return nil, errors.NewNotFound(apps.Resource("deployments/scale"), i.name)
	}

	groupVersion := schema.GroupVersion{Group: "apps", Version: "v1"}
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		requestGroupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		if _, ok := replicasPathInDeployment[requestGroupVersion.String()]; ok {
			groupVersion = requestGroupVersion
		} else {
			klog.Fatalf("Unrecognized group/version in request info %q", requestGroupVersion.String())
		}
	}

	managedFieldsHandler := managedfields.NewScaleHandler(
		deployment.ManagedFields,
		groupVersion,
		replicasPathInDeployment,
	)

	// deployment -> old scale
	oldScale, err := scaleFromDeployment(deployment)
	if err != nil {
		return nil, err
	}
	scaleManagedFields, err := managedFieldsHandler.ToSubresource()
	if err != nil {
		return nil, err
	}
	oldScale.ManagedFields = scaleManagedFields

	// old scale -> new scale
	newScaleObj, err := i.reqObjInfo.UpdatedObject(ctx, oldScale)
	if err != nil {
		return nil, err
	}
	if newScaleObj == nil {
		return nil, errors.NewBadRequest("nil update passed to Scale")
	}
	scale, ok := newScaleObj.(*autoscaling.Scale)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("expected input object type to be Scale, but %T", newScaleObj))
	}

	// validate
	if errs := autoscalingvalidation.ValidateScale(scale); len(errs) > 0 {
		return nil, errors.NewInvalid(autoscaling.Kind("Scale"), deployment.Name, errs)
	}

	// validate precondition if specified (resourceVersion matching is handled by storage)
	if len(scale.UID) > 0 && scale.UID != deployment.UID {
		return nil, errors.NewConflict(
			apps.Resource("deployments/scale"),
			deployment.Name,
			fmt.Errorf("Precondition failed: UID in precondition: %v, UID in object meta: %v", scale.UID, deployment.UID),
		)
	}

	// move replicas/resourceVersion fields to object and return
	deployment.Spec.Replicas = scale.Spec.Replicas
	deployment.ResourceVersion = scale.ResourceVersion

	updatedEntries, err := managedFieldsHandler.ToParent(scale.ManagedFields)
	if err != nil {
		return nil, err
	}
	deployment.ManagedFields = updatedEntries

	return deployment, nil
}
