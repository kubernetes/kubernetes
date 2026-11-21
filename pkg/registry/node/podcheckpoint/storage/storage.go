/*
Copyright The Kubernetes Authors.

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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/node/podcheckpoint"
	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"
)

// REST implements a RESTStorage for PodCheckpoints against etcd.
type REST struct {
	*genericregistry.Store
}

// StatusREST implements the REST endpoint for changing the status of a PodCheckpoint.
type StatusREST struct {
	store *genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against PodCheckpoints.
func NewREST(optsGetter generic.RESTOptionsGetter, pods corev1client.PodsGetter) (*REST, *StatusREST, error) {
	if pods == nil {
		return nil, nil, fmt.Errorf("PodCheckpoint storage requires a Pod client")
	}
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &nodeapi.PodCheckpoint{} },
		NewListFunc:               func() runtime.Object { return &nodeapi.PodCheckpointList{} },
		DefaultQualifiedResource:  nodeapi.Resource("podcheckpoints"),
		SingularQualifiedResource: nodeapi.Resource("podcheckpoint"),

		PredicateFunc: podcheckpoint.MatchPodCheckpoint,

		CreateStrategy:      podcheckpoint.Strategy,
		UpdateStrategy:      podcheckpoint.Strategy,
		DeleteStrategy:      podcheckpoint.Strategy,
		ResetFieldsStrategy: podcheckpoint.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	// Server-side apply creates through Update, so assign in the hook that
	// both create paths call.
	store.BeginCreate = func(ctx context.Context, obj runtime.Object, _ *metav1.CreateOptions) (genericregistry.FinishFunc, error) {
		return func(context.Context, bool) {}, assignSourceNode(ctx, pods, obj.(*nodeapi.PodCheckpoint))
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: podcheckpoint.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = podcheckpoint.StatusStrategy
	statusStore.ResetFieldsStrategy = podcheckpoint.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, nil
}

// New creates a new PodCheckpoint object.
func (r *StatusREST) New() runtime.Object {
	return &nodeapi.PodCheckpoint{}
}

// Destroy cleans up resources on shutdown.
func (r *StatusREST) Destroy() {
	// Given that underlying store is shared with REST, we don't need to destroy it here.
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// Status updates must not create objects, including server-side apply.
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, false, options)
}

// GetResetFields implements rest.ResetFieldsStrategy
func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}

// assignSourceNode replaces any client-supplied status with the node and UID
// of the source Pod, which must exist and be Running. The kubelet's
// node-scoped watch and the node authorizer route the checkpoint by this
// assignment, so it is set when the PodCheckpoint is created.
func assignSourceNode(ctx context.Context, pods corev1client.PodsGetter, pc *nodeapi.PodCheckpoint) error {
	pc.Status = nil
	namespace, _ := genericapirequest.NamespaceFrom(ctx)
	ref := &pc.Spec.SourcePod
	if ref.Name == "" {
		// Validation reports the missing name.
		return nil
	}
	pod, err := pods.Pods(namespace).Get(ctx, ref.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return apierrors.NewForbidden(nodeapi.Resource("podcheckpoints"), pc.Name, fmt.Errorf("source Pod %q does not exist", ref.Name))
		}
		return apierrors.NewInternalError(fmt.Errorf("cannot read source Pod %q: %w", ref.Name, err))
	}
	if err := checkpointutil.ValidateSourcePod(pod, ref.Name, ref.UID); err != nil {
		return apierrors.NewForbidden(nodeapi.Resource("podcheckpoints"), pc.Name, err)
	}
	pc.Status = &nodeapi.PodCheckpointStatus{NodeName: &pod.Spec.NodeName, SourcePodUID: &pod.UID}
	return nil
}
