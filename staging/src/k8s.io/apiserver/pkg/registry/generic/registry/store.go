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

package registry

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	kubeerr "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/validation/path"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage"
	storeerr "k8s.io/apiserver/pkg/storage/errors"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"

	"github.com/golang/glog"
)

// defaultWatchCacheSize is the default size of a watch catch per resource in number of entries.
const DefaultWatchCacheSize = 100

// ObjectFunc is a function to act on a given object. An error may be returned
// if the hook cannot be completed. An ObjectFunc may transform the provided
// object.
type ObjectFunc func(obj runtime.Object) error

// Store implements pkg/api/rest.StandardStorage. It's intended to be
// embeddable and allows the consumer to implement any non-generic functions
// that are required. This object is intended to be copyable so that it can be
// used in different ways but share the same underlying behavior.
//
// All fields are required unless specified.
//
// The intended use of this type is embedding within a Kind specific
// RESTStorage implementation. This type provides CRUD semantics on a Kubelike
// resource, handling details like conflict detection with ResourceVersion and
// semantics. The RESTCreateStrategy, RESTUpdateStrategy, and
// RESTDeleteStrategy are generic across all backends, and encapsulate logic
// specific to the API.
//
// TODO: make the default exposed methods exactly match a generic RESTStorage
type Store struct {
	// Copier is used to make some storage caching decorators work
	Copier runtime.ObjectCopier

	// NewFunc returns a new instance of the type this registry returns for a
	// GET of a single object, e.g.:
	//
	// curl GET /apis/group/version/namespaces/my-ns/myresource/name-of-object
	NewFunc func() runtime.Object

	// NewListFunc returns a new list of the type this registry; it is the
	// type returned when the resource is listed, e.g.:
	//
	// curl GET /apis/group/version/namespaces/my-ns/myresource
	NewListFunc func() runtime.Object

	// QualifiedResource is the pluralized name of the resource.
	QualifiedResource schema.GroupResource

	// KeyRootFunc returns the root etcd key for this resource; should not
	// include trailing "/".  This is used for operations that work on the
	// entire collection (listing and watching).
	//
	// KeyRootFunc and KeyFunc must be supplied together or not at all.
	KeyRootFunc func(ctx genericapirequest.Context) string

	// KeyFunc returns the key for a specific object in the collection.
	// KeyFund is dalled for Create/Update/Get/Delete. Note that 'namespace'
	// can be gotten from ctx.
	//
	// KeyFunc and KeyRootFunc must be supplied together or not at all.
	KeyFunc func(ctx genericapirequest.Context, name string) (string, error)

	// ObjectNameFunc returns the name of an object or an error.
	ObjectNameFunc func(obj runtime.Object) (string, error)

	// TTLFunc returns the TTL (time to live) that objects should be persisted
	// with. The existing parameter is the current TTL or the default for this
	// operation. The update parameter indicates whether this is an operation
	// against an existing object.
	//
	// Objects that are persisted with a TTL are evicted once the TTL expires.
	TTLFunc func(obj runtime.Object, existing uint64, update bool) (uint64, error)

	// PredicateFunc returns a matcher corresponding to the provided labels
	// and fields. The SelectionPredicate returned should return true if the
	// object matches the given field and label selectors.
	PredicateFunc func(label labels.Selector, field fields.Selector) storage.SelectionPredicate

	// EnableGarbageCollection affects the handling of Update and Delete
	// requests. Enabling garbage collection allows finalizers to do work to
	// finalize this object before the store deletes it.
	//
	// If any store has garbage collection enabled, it must also be enabled in
	// the kube-controller-manager.
	EnableGarbageCollection bool

	// DeleteCollectionWorkers is the maximum number of workers in a single
	// DeleteCollection call. Delete requests for the items in a collection
	// are issued in parallel.
	DeleteCollectionWorkers int

	// Decorator is an optional exit hook on an object returned from the
	// underlying storage. The returned object could be an individual object
	// (e.g. Pod) or a list type (e.g. PodList). Decorator is intended for
	// integrations that are above storage and should only be used for
	// specific cases where storage of the value is not appropriate, since
	// they cannot be watched.
	Decorator ObjectFunc
	// CreateStrategy implements resource-specific behavior during creation.
	CreateStrategy rest.RESTCreateStrategy
	// AfterCreate implements a further operation to run after a resource is
	// created and before it is decorated, optional.
	AfterCreate ObjectFunc
	// UpdateStrategy implements resource-specific behavior during updates.
	UpdateStrategy rest.RESTUpdateStrategy
	// AfterUpdate implements a further operation to run after a resource is
	// updated and before it is decorated, optional.
	AfterUpdate ObjectFunc
	// DeleteStrategy implements resource-specific behavior during deletion,
	// optional.
	DeleteStrategy rest.RESTDeleteStrategy
	// AfterDelete implements a further operation to run after a resource is
	// deleted and before it is decorated, optional.
	AfterDelete ObjectFunc
	// ReturnDeletedObject determines whether the Store returns the object
	// that was deleted. Otherwise, return a generic success status response.
	ReturnDeletedObject bool
	// ExportStrategy implements resource-specific behavior during export,
	// optional. Exported objects are not decorated.
	ExportStrategy rest.RESTExportStrategy

	// Storage is the interface for the underlying storage for the resource.
	Storage storage.Interface
	// Called to cleanup clients used by the underlying Storage; optional.
	DestroyFunc func()
	// Maximum size of the watch history cached in memory, in number of entries.
	// A zero value here means that a default is used. This value is ignored if
	// Storage is non-nil.
	WatchCacheSize int
}

// Note: the rest.StandardStorage interface aggregates the common REST verbs
var _ rest.StandardStorage = &Store{}
var _ rest.Exporter = &Store{}

const OptimisticLockErrorMsg = "the object has been modified; please apply your changes to the latest version and try again"

// NamespaceKeyRootFunc is the default function for constructing storage paths
// to resource directories enforcing namespace rules.
func NamespaceKeyRootFunc(ctx genericapirequest.Context, prefix string) string {
	key := prefix
	ns, ok := genericapirequest.NamespaceFrom(ctx)
	if ok && len(ns) > 0 {
		key = key + "/" + ns
	}
	return key
}

// NamespaceKeyFunc is the default function for constructing storage paths to
// a resource relative to the given prefix enforcing namespace rules. If the
// context does not contain a namespace, it errors.
func NamespaceKeyFunc(ctx genericapirequest.Context, prefix string, name string) (string, error) {
	key := NamespaceKeyRootFunc(ctx, prefix)
	ns, ok := genericapirequest.NamespaceFrom(ctx)
	if !ok || len(ns) == 0 {
		return "", kubeerr.NewBadRequest("Namespace parameter required.")
	}
	if len(name) == 0 {
		return "", kubeerr.NewBadRequest("Name parameter required.")
	}
	if msgs := path.IsValidPathSegmentName(name); len(msgs) != 0 {
		return "", kubeerr.NewBadRequest(fmt.Sprintf("Name parameter invalid: %q: %s", name, strings.Join(msgs, ";")))
	}
	key = key + "/" + name
	return key, nil
}

// NoNamespaceKeyFunc is the default function for constructing storage paths
// to a resource relative to the given prefix without a namespace.
func NoNamespaceKeyFunc(ctx genericapirequest.Context, prefix string, name string) (string, error) {
	if len(name) == 0 {
		return "", kubeerr.NewBadRequest("Name parameter required.")
	}
	if msgs := path.IsValidPathSegmentName(name); len(msgs) != 0 {
		return "", kubeerr.NewBadRequest(fmt.Sprintf("Name parameter invalid: %q: %s", name, strings.Join(msgs, ";")))
	}
	key := prefix + "/" + name
	return key, nil
}

// New implements RESTStorage.New.
func (e *Store) New() runtime.Object {
	return e.NewFunc()
}

// NewList implements rest.Lister.
func (e *Store) NewList() runtime.Object {
	return e.NewListFunc()
}

// List returns a list of items matching labels and field according to the
// store's PredicateFunc.
func (e *Store) List(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	label := labels.Everything()
	if options != nil && options.LabelSelector != nil {
		label = options.LabelSelector
	}
	field := fields.Everything()
	if options != nil && options.FieldSelector != nil {
		field = options.FieldSelector
	}
	out, err := e.ListPredicate(ctx, e.PredicateFunc(label, field), options)
	if err != nil {
		return nil, err
	}
	if e.Decorator != nil {
		if err := e.Decorator(out); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ListPredicate returns a list of all the items matching the given
// SelectionPredicate.
func (e *Store) ListPredicate(ctx genericapirequest.Context, p storage.SelectionPredicate, options *metainternalversion.ListOptions) (runtime.Object, error) {
	if options == nil {
		// By default we should serve the request from etcd.
		options = &metainternalversion.ListOptions{ResourceVersion: ""}
	}
	list := e.NewListFunc()
	if name, ok := p.MatchesSingle(); ok {
		if key, err := e.KeyFunc(ctx, name); err == nil {
			err := e.Storage.GetToList(ctx, key, options.ResourceVersion, p, list)
			return list, storeerr.InterpretListError(err, e.QualifiedResource)
		}
		// if we cannot extract a key based on the current context, the optimization is skipped
	}

	err := e.Storage.List(ctx, e.KeyRootFunc(ctx), options.ResourceVersion, p, list)
	return list, storeerr.InterpretListError(err, e.QualifiedResource)
}

// Create inserts a new item according to the unique key from the object.
func (e *Store) Create(ctx genericapirequest.Context, obj runtime.Object) (runtime.Object, error) {
	if err := rest.BeforeCreate(e.CreateStrategy, ctx, obj); err != nil {
		return nil, err
	}
	name, err := e.ObjectNameFunc(obj)
	if err != nil {
		return nil, err
	}
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return nil, err
	}
	ttl, err := e.calculateTTL(obj, 0, false)
	if err != nil {
		return nil, err
	}
	out := e.NewFunc()
	if err := e.Storage.Create(ctx, key, obj, out, ttl); err != nil {
		err = storeerr.InterpretCreateError(err, e.QualifiedResource, name)
		err = rest.CheckGeneratedNameError(e.CreateStrategy, err, obj)
		if !kubeerr.IsAlreadyExists(err) {
			return nil, err
		}
		if errGet := e.Storage.Get(ctx, key, "", out, false); errGet != nil {
			return nil, err
		}
		accessor, errGetAcc := meta.Accessor(out)
		if errGetAcc != nil {
			return nil, err
		}
		if accessor.GetDeletionTimestamp() != nil {
			msg := &err.(*kubeerr.StatusError).ErrStatus.Message
			*msg = fmt.Sprintf("object is being deleted: %s", *msg)
		}
		return nil, err
	}
	if e.AfterCreate != nil {
		if err := e.AfterCreate(out); err != nil {
			return nil, err
		}
	}
	if e.Decorator != nil {
		if err := e.Decorator(obj); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// shouldDeleteDuringUpdate checks if a Update is removing all the object's
// finalizers. If so, it further checks if the object's
// DeletionGracePeriodSeconds is 0. If so, it returns true.
//
// If the store does not have garbage collection enabled,
// shouldDeleteDuringUpdate will always return false.
func (e *Store) shouldDeleteDuringUpdate(ctx genericapirequest.Context, key string, obj, existing runtime.Object) bool {
	if !e.EnableGarbageCollection {
		return false
	}
	newMeta, err := metav1.ObjectMetaFor(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return false
	}
	oldMeta, err := metav1.ObjectMetaFor(existing)
	if err != nil {
		utilruntime.HandleError(err)
		return false
	}
	return len(newMeta.Finalizers) == 0 && oldMeta.DeletionGracePeriodSeconds != nil && *oldMeta.DeletionGracePeriodSeconds == 0
}

// deleteForEmptyFinalizers handles deleting an object once its finalizer list
// becomes empty due to an update.
func (e *Store) deleteForEmptyFinalizers(ctx genericapirequest.Context, name, key string, obj runtime.Object, preconditions *storage.Preconditions) (runtime.Object, bool, error) {
	out := e.NewFunc()
	glog.V(6).Infof("going to delete %s from registry, triggered by update", name)
	if err := e.Storage.Delete(ctx, key, out, preconditions); err != nil {
		// Deletion is racy, i.e., there could be multiple update
		// requests to remove all finalizers from the object, so we
		// ignore the NotFound error.
		if storage.IsNotFound(err) {
			_, err := e.finalizeDelete(obj, true)
			// clients are expecting an updated object if a PUT succeeded,
			// but finalizeDelete returns a metav1.Status, so return
			// the object in the request instead.
			return obj, false, err
		}
		return nil, false, storeerr.InterpretDeleteError(err, e.QualifiedResource, name)
	}
	_, err := e.finalizeDelete(out, true)
	// clients are expecting an updated object if a PUT succeeded, but
	// finalizeDelete returns a metav1.Status, so return the object in
	// the request instead.
	return obj, false, err
}

// Update performs an atomic update and set of the object. Returns the result of the update
// or an error. If the registry allows create-on-update, the create flow will be executed.
// A bool is returned along with the object and any errors, to indicate object creation.
func (e *Store) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return nil, false, err
	}

	var (
		creatingObj runtime.Object
		creating    = false
	)

	storagePreconditions := &storage.Preconditions{}
	if preconditions := objInfo.Preconditions(); preconditions != nil {
		storagePreconditions.UID = preconditions.UID
	}

	out := e.NewFunc()
	// deleteObj is only used in case a deletion is carried out
	var deleteObj runtime.Object
	err = e.Storage.GuaranteedUpdate(ctx, key, out, true, storagePreconditions, func(existing runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		// Given the existing object, get the new object
		obj, err := objInfo.UpdatedObject(ctx, existing)
		if err != nil {
			return nil, nil, err
		}

		// If AllowUnconditionalUpdate() is true and the object specified by
		// the user does not have a resource version, then we populate it with
		// the latest version. Else, we check that the version specified by
		// the user matches the version of latest storage object.
		resourceVersion, err := e.Storage.Versioner().ObjectResourceVersion(obj)
		if err != nil {
			return nil, nil, err
		}
		doUnconditionalUpdate := resourceVersion == 0 && e.UpdateStrategy.AllowUnconditionalUpdate()

		version, err := e.Storage.Versioner().ObjectResourceVersion(existing)
		if err != nil {
			return nil, nil, err
		}
		if version == 0 {
			if !e.UpdateStrategy.AllowCreateOnUpdate() {
				return nil, nil, kubeerr.NewNotFound(e.QualifiedResource, name)
			}
			creating = true
			creatingObj = obj
			if err := rest.BeforeCreate(e.CreateStrategy, ctx, obj); err != nil {
				return nil, nil, err
			}
			ttl, err := e.calculateTTL(obj, 0, false)
			if err != nil {
				return nil, nil, err
			}
			return obj, &ttl, nil
		}

		creating = false
		creatingObj = nil
		if doUnconditionalUpdate {
			// Update the object's resource version to match the latest
			// storage object's resource version.
			err = e.Storage.Versioner().UpdateObject(obj, res.ResourceVersion)
			if err != nil {
				return nil, nil, err
			}
		} else {
			// Check if the object's resource version matches the latest
			// resource version.
			newVersion, err := e.Storage.Versioner().ObjectResourceVersion(obj)
			if err != nil {
				return nil, nil, err
			}
			if newVersion == 0 {
				// TODO: The Invalid error should have a field for Resource.
				// After that field is added, we should fill the Resource and
				// leave the Kind field empty. See the discussion in #18526.
				qualifiedKind := schema.GroupKind{Group: e.QualifiedResource.Group, Kind: e.QualifiedResource.Resource}
				fieldErrList := field.ErrorList{field.Invalid(field.NewPath("metadata").Child("resourceVersion"), newVersion, "must be specified for an update")}
				return nil, nil, kubeerr.NewInvalid(qualifiedKind, name, fieldErrList)
			}
			if newVersion != version {
				return nil, nil, kubeerr.NewConflict(e.QualifiedResource, name, fmt.Errorf(OptimisticLockErrorMsg))
			}
		}
		if err := rest.BeforeUpdate(e.UpdateStrategy, ctx, obj, existing); err != nil {
			return nil, nil, err
		}
		if e.shouldDeleteDuringUpdate(ctx, key, obj, existing) {
			deleteObj = obj
			return nil, nil, errEmptiedFinalizers
		}
		ttl, err := e.calculateTTL(obj, res.TTL, true)
		if err != nil {
			return nil, nil, err
		}
		if int64(ttl) != res.TTL {
			return obj, &ttl, nil
		}
		return obj, nil, nil
	})

	if err != nil {
		// delete the object
		if err == errEmptiedFinalizers {
			return e.deleteForEmptyFinalizers(ctx, name, key, deleteObj, storagePreconditions)
		}
		if creating {
			err = storeerr.InterpretCreateError(err, e.QualifiedResource, name)
			err = rest.CheckGeneratedNameError(e.CreateStrategy, err, creatingObj)
		} else {
			err = storeerr.InterpretUpdateError(err, e.QualifiedResource, name)
		}
		return nil, false, err
	}
	if creating {
		if e.AfterCreate != nil {
			if err := e.AfterCreate(out); err != nil {
				return nil, false, err
			}
		}
	} else {
		if e.AfterUpdate != nil {
			if err := e.AfterUpdate(out); err != nil {
				return nil, false, err
			}
		}
	}
	if e.Decorator != nil {
		if err := e.Decorator(out); err != nil {
			return nil, false, err
		}
	}
	return out, creating, nil
}

// Get retrieves the item from storage.
func (e *Store) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	obj := e.NewFunc()
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return nil, err
	}
	if err := e.Storage.Get(ctx, key, options.ResourceVersion, obj, false); err != nil {
		return nil, storeerr.InterpretGetError(err, e.QualifiedResource, name)
	}
	if e.Decorator != nil {
		if err := e.Decorator(obj); err != nil {
			return nil, err
		}
	}
	return obj, nil
}

var (
	errAlreadyDeleting   = fmt.Errorf("abort delete")
	errDeleteNow         = fmt.Errorf("delete now")
	errEmptiedFinalizers = fmt.Errorf("emptied finalizers")
)

// shouldUpdateFinalizers returns if we need to update the finalizers of the
// object, and the desired list of finalizers. When deciding whether to add
// the OrphanDependent finalizer, factors in the order of highest to lowest
// priority are:
//
// - options.OrphanDependents,
// - existing finalizers of the object
// - e.DeleteStrategy.DefaultGarbageCollectionPolicy
func shouldUpdateFinalizers(e *Store, accessor metav1.Object, options *metav1.DeleteOptions) (shouldUpdate bool, newFinalizers []string) {
	shouldOrphan := false
	// Get default orphan policy from this REST object type
	if gcStrategy, ok := e.DeleteStrategy.(rest.GarbageCollectionDeleteStrategy); ok {
		if gcStrategy.DefaultGarbageCollectionPolicy() == rest.OrphanDependents {
			shouldOrphan = true
		}
	}
	// If a finalizer is set in the object, it overrides the default
	hasOrphanFinalizer := false
	finalizers := accessor.GetFinalizers()
	for _, f := range finalizers {
		if f == metav1.FinalizerOrphan {
			shouldOrphan = true
			hasOrphanFinalizer = true
			break
		}
		// TODO: update this when we add a finalizer indicating a preference for the other behavior
	}
	// If an explicit policy was set at deletion time, that overrides both
	if options != nil && options.OrphanDependents != nil {
		shouldOrphan = *options.OrphanDependents
	}
	if shouldOrphan && !hasOrphanFinalizer {
		finalizers = append(finalizers, metav1.FinalizerOrphan)
		return true, finalizers
	}
	if !shouldOrphan && hasOrphanFinalizer {
		var newFinalizers []string
		for _, f := range finalizers {
			if f == metav1.FinalizerOrphan {
				continue
			}
			newFinalizers = append(newFinalizers, f)
		}
		return true, newFinalizers
	}
	return false, finalizers
}

// markAsDeleting sets the obj's DeletionGracePeriodSeconds to 0, and sets the
// DeletionTimestamp to "now". Finalizers are watching for such updates and will
// finalize the object if their IDs are present in the object's Finalizers list.
func markAsDeleting(obj runtime.Object) (err error) {
	objectMeta, kerr := metav1.ObjectMetaFor(obj)
	if kerr != nil {
		return kerr
	}
	now := metav1.NewTime(time.Now())
	// This handles Generation bump for resources that don't support graceful
	// deletion. For resources that support graceful deletion is handle in
	// pkg/api/rest/delete.go
	if objectMeta.DeletionTimestamp == nil && objectMeta.Generation > 0 {
		objectMeta.Generation++
	}
	objectMeta.DeletionTimestamp = &now
	var zero int64 = 0
	objectMeta.DeletionGracePeriodSeconds = &zero
	return nil
}

// updateForGracefulDeletion and updateForGracefulDeletionAndFinalizers both
// implement deletion flows for graceful deletion.  Graceful deletion is
// implemented as setting the deletion timestamp in an update.  If the
// implementation of graceful deletion is changed, both of these methods
// should be changed together.

// updateForGracefulDeletion updates the given object for graceful deletion by
// setting the deletion timestamp and grace period seconds and returns:
//
// 1. an error
// 2. a boolean indicating that the object was not found, but it should be
//    ignored
// 3. a boolean indicating that the object's grace period is exhausted and it
//    should be deleted immediately
// 4. a new output object with the state that was updated
// 5. a copy of the last existing state of the object
func (e *Store) updateForGracefulDeletion(ctx genericapirequest.Context, name, key string, options *metav1.DeleteOptions, preconditions storage.Preconditions, in runtime.Object) (err error, ignoreNotFound, deleteImmediately bool, out, lastExisting runtime.Object) {
	lastGraceful := int64(0)
	out = e.NewFunc()
	err = e.Storage.GuaranteedUpdate(
		ctx,
		key,
		out,
		false, /* ignoreNotFound */
		&preconditions,
		storage.SimpleUpdate(func(existing runtime.Object) (runtime.Object, error) {
			graceful, pendingGraceful, err := rest.BeforeDelete(e.DeleteStrategy, ctx, existing, options)
			if err != nil {
				return nil, err
			}
			if pendingGraceful {
				return nil, errAlreadyDeleting
			}
			if !graceful {
				return nil, errDeleteNow
			}
			lastGraceful = *options.GracePeriodSeconds
			lastExisting = existing
			return existing, nil
		}),
	)
	switch err {
	case nil:
		if lastGraceful > 0 {
			return nil, false, false, out, lastExisting
		}
		// If we are here, the registry supports grace period mechanism and
		// we are intentionally delete gracelessly. In this case, we may
		// enter a race with other k8s components. If other component wins
		// the race, the object will not be found, and we should tolerate
		// the NotFound error. See
		// https://github.com/kubernetes/kubernetes/issues/19403 for
		// details.
		return nil, true, true, out, lastExisting
	case errDeleteNow:
		// we've updated the object to have a zero grace period, or it's already at 0, so
		// we should fall through and truly delete the object.
		return nil, false, true, out, lastExisting
	case errAlreadyDeleting:
		out, err = e.finalizeDelete(in, true)
		return err, false, false, out, lastExisting
	default:
		return storeerr.InterpretUpdateError(err, e.QualifiedResource, name), false, false, out, lastExisting
	}
}

// updateForGracefulDeletionAndFinalizers updates the given object for
// graceful deletion and finalization by setting the deletion timestamp and
// grace period seconds (graceful deletion) and updating the list of
// finalizers (finalization); it returns:
//
// 1. an error
// 2. a boolean indicating that the object was not found, but it should be
//    ignored
// 3. a boolean indicating that the object's grace period is exhausted and it
//    should be deleted immediately
// 4. a new output object with the state that was updated
// 5. a copy of the last existing state of the object
func (e *Store) updateForGracefulDeletionAndFinalizers(ctx genericapirequest.Context, name, key string, options *metav1.DeleteOptions, preconditions storage.Preconditions, in runtime.Object) (err error, ignoreNotFound, deleteImmediately bool, out, lastExisting runtime.Object) {
	lastGraceful := int64(0)
	var pendingFinalizers bool
	out = e.NewFunc()
	err = e.Storage.GuaranteedUpdate(
		ctx,
		key,
		out,
		false, /* ignoreNotFound */
		&preconditions,
		storage.SimpleUpdate(func(existing runtime.Object) (runtime.Object, error) {
			graceful, pendingGraceful, err := rest.BeforeDelete(e.DeleteStrategy, ctx, existing, options)
			if err != nil {
				return nil, err
			}
			if pendingGraceful {
				return nil, errAlreadyDeleting
			}

			// Add/remove the orphan finalizer as the options dictates.
			// Note that this occurs after checking pendingGraceufl, so
			// finalizers cannot be updated via DeleteOptions if deletion has
			// started.
			existingAccessor, err := meta.Accessor(existing)
			if err != nil {
				return nil, err
			}
			shouldUpdate, newFinalizers := shouldUpdateFinalizers(e, existingAccessor, options)
			if shouldUpdate {
				existingAccessor.SetFinalizers(newFinalizers)
			}

			pendingFinalizers = len(existingAccessor.GetFinalizers()) != 0
			if !graceful {
				// set the DeleteGracePeriods to 0 if the object has pendingFinalizers but not supporting graceful deletion
				if pendingFinalizers {
					glog.V(6).Infof("update the DeletionTimestamp to \"now\" and GracePeriodSeconds to 0 for object %s, because it has pending finalizers", name)
					err = markAsDeleting(existing)
					if err != nil {
						return nil, err
					}
					return existing, nil
				}
				return nil, errDeleteNow
			}
			lastGraceful = *options.GracePeriodSeconds
			lastExisting = existing
			return existing, nil
		}),
	)
	switch err {
	case nil:
		// If there are pending finalizers, we never delete the object immediately.
		if pendingFinalizers {
			return nil, false, false, out, lastExisting
		}
		if lastGraceful > 0 {
			return nil, false, false, out, lastExisting
		}
		// If we are here, the registry supports grace period mechanism and
		// we are intentionally delete gracelessly. In this case, we may
		// enter a race with other k8s components. If other component wins
		// the race, the object will not be found, and we should tolerate
		// the NotFound error. See
		// https://github.com/kubernetes/kubernetes/issues/19403 for
		// details.
		return nil, true, true, out, lastExisting
	case errDeleteNow:
		// we've updated the object to have a zero grace period, or it's already at 0, so
		// we should fall through and truly delete the object.
		return nil, false, true, out, lastExisting
	case errAlreadyDeleting:
		out, err = e.finalizeDelete(in, true)
		return err, false, false, out, lastExisting
	default:
		return storeerr.InterpretUpdateError(err, e.QualifiedResource, name), false, false, out, lastExisting
	}
}

// Delete removes the item from storage.
func (e *Store) Delete(ctx genericapirequest.Context, name string, options *metav1.DeleteOptions) (runtime.Object, error) {
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return nil, err
	}

	obj := e.NewFunc()
	if err := e.Storage.Get(ctx, key, "", obj, false); err != nil {
		return nil, storeerr.InterpretDeleteError(err, e.QualifiedResource, name)
	}
	// support older consumers of delete by treating "nil" as delete immediately
	if options == nil {
		options = metav1.NewDeleteOptions(0)
	}
	var preconditions storage.Preconditions
	if options.Preconditions != nil {
		preconditions.UID = options.Preconditions.UID
	}
	graceful, pendingGraceful, err := rest.BeforeDelete(e.DeleteStrategy, ctx, obj, options)
	if err != nil {
		return nil, err
	}
	// this means finalizers cannot be updated via DeleteOptions if a deletion is already pending
	if pendingGraceful {
		return e.finalizeDelete(obj, false)
	}
	// check if obj has pending finalizers
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, kubeerr.NewInternalError(err)
	}
	pendingFinalizers := len(accessor.GetFinalizers()) != 0
	var ignoreNotFound bool
	var deleteImmediately bool = true
	var lastExisting, out runtime.Object

	// Handle combinations of graceful deletion and finalization by issuing
	// the correct updates.
	if e.EnableGarbageCollection {
		shouldUpdateFinalizers, _ := shouldUpdateFinalizers(e, accessor, options)
		// TODO: remove the check, because we support no-op updates now.
		if graceful || pendingFinalizers || shouldUpdateFinalizers {
			err, ignoreNotFound, deleteImmediately, out, lastExisting = e.updateForGracefulDeletionAndFinalizers(ctx, name, key, options, preconditions, obj)
		}
	} else {
		// TODO: remove the check on graceful, because we support no-op updates now.
		if graceful {
			err, ignoreNotFound, deleteImmediately, out, lastExisting = e.updateForGracefulDeletion(ctx, name, key, options, preconditions, obj)
		}
	}
	// !deleteImmediately covers all cases where err != nil. We keep both to be future-proof.
	if !deleteImmediately || err != nil {
		return out, err
	}

	// delete immediately, or no graceful deletion supported
	glog.V(6).Infof("going to delete %s from registry: ", name)
	out = e.NewFunc()
	if err := e.Storage.Delete(ctx, key, out, &preconditions); err != nil {
		// Please refer to the place where we set ignoreNotFound for the reason
		// why we ignore the NotFound error .
		if storage.IsNotFound(err) && ignoreNotFound && lastExisting != nil {
			// The lastExisting object may not be the last state of the object
			// before its deletion, but it's the best approximation.
			return e.finalizeDelete(lastExisting, true)
		}
		return nil, storeerr.InterpretDeleteError(err, e.QualifiedResource, name)
	}
	return e.finalizeDelete(out, true)
}

// DeleteCollection removes all items returned by List with a given ListOptions from storage.
//
// DeleteCollection is currently NOT atomic. It can happen that only subset of objects
// will be deleted from storage, and then an error will be returned.
// In case of success, the list of deleted objects will be returned.
//
// TODO: Currently, there is no easy way to remove 'directory' entry from storage (if we
// are removing all objects of a given type) with the current API (it's technically
// possibly with storage API, but watch is not delivered correctly then).
// It will be possible to fix it with v3 etcd API.
func (e *Store) DeleteCollection(ctx genericapirequest.Context, options *metav1.DeleteOptions, listOptions *metainternalversion.ListOptions) (runtime.Object, error) {
	listObj, err := e.List(ctx, listOptions)
	if err != nil {
		return nil, err
	}
	items, err := meta.ExtractList(listObj)
	if err != nil {
		return nil, err
	}
	// Spawn a number of goroutines, so that we can issue requests to storage
	// in parallel to speed up deletion.
	// TODO: Make this proportional to the number of items to delete, up to
	// DeleteCollectionWorkers (it doesn't make much sense to spawn 16
	// workers to delete 10 items).
	workersNumber := e.DeleteCollectionWorkers
	if workersNumber < 1 {
		workersNumber = 1
	}
	wg := sync.WaitGroup{}
	toProcess := make(chan int, 2*workersNumber)
	errs := make(chan error, workersNumber+1)

	go func() {
		defer utilruntime.HandleCrash(func(panicReason interface{}) {
			errs <- fmt.Errorf("DeleteCollection distributor panicked: %v", panicReason)
		})
		for i := 0; i < len(items); i++ {
			toProcess <- i
		}
		close(toProcess)
	}()

	wg.Add(workersNumber)
	for i := 0; i < workersNumber; i++ {
		go func() {
			// panics don't cross goroutine boundaries
			defer utilruntime.HandleCrash(func(panicReason interface{}) {
				errs <- fmt.Errorf("DeleteCollection goroutine panicked: %v", panicReason)
			})
			defer wg.Done()

			for {
				index, ok := <-toProcess
				if !ok {
					return
				}
				accessor, err := meta.Accessor(items[index])
				if err != nil {
					errs <- err
					return
				}
				if _, err := e.Delete(ctx, accessor.GetName(), options); err != nil && !kubeerr.IsNotFound(err) {
					glog.V(4).Infof("Delete %s in DeleteCollection failed: %v", accessor.GetName(), err)
					errs <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	select {
	case err := <-errs:
		return nil, err
	default:
		return listObj, nil
	}
}

// finalizeDelete runs the Store's AfterDelete hook if runHooks is set and
// returns the decorated deleted object if appropriate.
func (e *Store) finalizeDelete(obj runtime.Object, runHooks bool) (runtime.Object, error) {
	if runHooks && e.AfterDelete != nil {
		if err := e.AfterDelete(obj); err != nil {
			return nil, err
		}
	}
	if e.ReturnDeletedObject {
		if e.Decorator != nil {
			if err := e.Decorator(obj); err != nil {
				return nil, err
			}
		}
		return obj, nil
	}
	return &metav1.Status{Status: metav1.StatusSuccess}, nil
}

// Watch makes a matcher for the given label and field, and calls
// WatchPredicate. If possible, you should customize PredicateFunc to produce
// a matcher that matches by key. SelectionPredicate does this for you
// automatically.
func (e *Store) Watch(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	label := labels.Everything()
	if options != nil && options.LabelSelector != nil {
		label = options.LabelSelector
	}
	field := fields.Everything()
	if options != nil && options.FieldSelector != nil {
		field = options.FieldSelector
	}
	resourceVersion := ""
	if options != nil {
		resourceVersion = options.ResourceVersion
	}
	return e.WatchPredicate(ctx, e.PredicateFunc(label, field), resourceVersion)
}

// WatchPredicate starts a watch for the items that m matches.
func (e *Store) WatchPredicate(ctx genericapirequest.Context, p storage.SelectionPredicate, resourceVersion string) (watch.Interface, error) {
	if name, ok := p.MatchesSingle(); ok {
		if key, err := e.KeyFunc(ctx, name); err == nil {
			w, err := e.Storage.Watch(ctx, key, resourceVersion, p)
			if err != nil {
				return nil, err
			}
			if e.Decorator != nil {
				return newDecoratedWatcher(w, e.Decorator), nil
			}
			return w, nil
		}
		// if we cannot extract a key based on the current context, the
		// optimization is skipped
	}

	w, err := e.Storage.WatchList(ctx, e.KeyRootFunc(ctx), resourceVersion, p)
	if err != nil {
		return nil, err
	}
	if e.Decorator != nil {
		return newDecoratedWatcher(w, e.Decorator), nil
	}
	return w, nil
}

// calculateTTL is a helper for retrieving the updated TTL for an object or
// returning an error if the TTL cannot be calculated. The defaultTTL is
// changed to 1 if less than zero. Zero means no TTL, not expire immediately.
func (e *Store) calculateTTL(obj runtime.Object, defaultTTL int64, update bool) (ttl uint64, err error) {
	// TODO: validate this is assertion is still valid.

	// etcd may return a negative TTL for a node if the expiration has not
	// occurred due to server lag - we will ensure that the value is at least
	// set.
	if defaultTTL < 0 {
		defaultTTL = 1
	}
	ttl = uint64(defaultTTL)
	if e.TTLFunc != nil {
		ttl, err = e.TTLFunc(obj, ttl, update)
	}
	return ttl, err
}

// exportObjectMeta unsets the fields on the given object that should not be
// present when the object is exported.
func exportObjectMeta(accessor metav1.Object, exact bool) {
	accessor.SetUID("")
	if !exact {
		accessor.SetNamespace("")
	}
	accessor.SetCreationTimestamp(metav1.Time{})
	accessor.SetDeletionTimestamp(nil)
	accessor.SetResourceVersion("")
	accessor.SetSelfLink("")
	if len(accessor.GetGenerateName()) > 0 && !exact {
		accessor.SetName("")
	}
}

// Export implements the rest.Exporter interface
func (e *Store) Export(ctx genericapirequest.Context, name string, opts metav1.ExportOptions) (runtime.Object, error) {
	obj, err := e.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	if accessor, err := meta.Accessor(obj); err == nil {
		exportObjectMeta(accessor, opts.Exact)
	} else {
		glog.V(4).Infof("Object of type %v does not have ObjectMeta: %v", reflect.TypeOf(obj), err)
	}

	if e.ExportStrategy != nil {
		if err = e.ExportStrategy.Export(ctx, obj, opts.Exact); err != nil {
			return nil, err
		}
	} else {
		e.CreateStrategy.PrepareForCreate(ctx, obj)
	}
	return obj, nil
}

// CompleteWithOptions updates the store with the provided options and
// defaults common fields.
func (e *Store) CompleteWithOptions(options *generic.StoreOptions) error {
	if e.QualifiedResource.Empty() {
		return fmt.Errorf("store %#v must have a non-empty qualified resource", e)
	}
	if e.NewFunc == nil {
		return fmt.Errorf("store for %s must have NewFunc set", e.QualifiedResource.String())
	}
	if e.NewListFunc == nil {
		return fmt.Errorf("store for %s must have NewListFunc set", e.QualifiedResource.String())
	}
	if (e.KeyRootFunc == nil) != (e.KeyFunc == nil) {
		return fmt.Errorf("store for %s must set both KeyRootFunc and KeyFunc or neither", e.QualifiedResource.String())
	}

	var isNamespaced bool
	switch {
	case e.CreateStrategy != nil:
		isNamespaced = e.CreateStrategy.NamespaceScoped()
	case e.UpdateStrategy != nil:
		isNamespaced = e.UpdateStrategy.NamespaceScoped()
	default:
		return fmt.Errorf("store for %s must have CreateStrategy or UpdateStrategy set", e.QualifiedResource.String())
	}

	if options.RESTOptions == nil {
		return fmt.Errorf("options for %s must have RESTOptions set", e.QualifiedResource.String())
	}
	if options.AttrFunc == nil {
		return fmt.Errorf("options for %s must have AttrFunc set", e.QualifiedResource.String())
	}

	opts, err := options.RESTOptions.GetRESTOptions(e.QualifiedResource)
	if err != nil {
		return err
	}

	// Resource prefix must come from the underlying factory
	prefix := opts.ResourcePrefix
	if !strings.HasPrefix(prefix, "/") {
		prefix = "/" + prefix
	}
	if prefix == "/" {
		return fmt.Errorf("store for %s has an invalid prefix %q", e.QualifiedResource.String(), opts.ResourcePrefix)
	}

	// Set the default behavior for storage key generation
	if e.KeyRootFunc == nil && e.KeyFunc == nil {
		if isNamespaced {
			e.KeyRootFunc = func(ctx genericapirequest.Context) string {
				return NamespaceKeyRootFunc(ctx, prefix)
			}
			e.KeyFunc = func(ctx genericapirequest.Context, name string) (string, error) {
				return NamespaceKeyFunc(ctx, prefix, name)
			}
		} else {
			e.KeyRootFunc = func(ctx genericapirequest.Context) string {
				return prefix
			}
			e.KeyFunc = func(ctx genericapirequest.Context, name string) (string, error) {
				return NoNamespaceKeyFunc(ctx, prefix, name)
			}
		}
	}

	// We adapt the store's keyFunc so that we can use it with the StorageDecorator
	// without making any assumptions about where objects are stored in etcd
	keyFunc := func(obj runtime.Object) (string, error) {
		accessor, err := meta.Accessor(obj)
		if err != nil {
			return "", err
		}

		if isNamespaced {
			return e.KeyFunc(genericapirequest.WithNamespace(genericapirequest.NewContext(), accessor.GetNamespace()), accessor.GetName())
		}

		return e.KeyFunc(genericapirequest.NewContext(), accessor.GetName())
	}

	triggerFunc := options.TriggerFunc
	if triggerFunc == nil {
		triggerFunc = storage.NoTriggerPublisher
	}

	if e.DeleteCollectionWorkers == 0 {
		e.DeleteCollectionWorkers = opts.DeleteCollectionWorkers
	}

	e.EnableGarbageCollection = opts.EnableGarbageCollection

	if e.Storage == nil {
		capacity := DefaultWatchCacheSize
		if e.WatchCacheSize != 0 {
			capacity = DefaultWatchCacheSize
		}
		e.Storage, e.DestroyFunc = opts.Decorator(
			e.Copier,
			opts.StorageConfig,
			capacity,
			e.NewFunc(),
			prefix,
			keyFunc,
			e.NewListFunc,
			options.AttrFunc,
			triggerFunc,
		)
	}

	return nil
}
