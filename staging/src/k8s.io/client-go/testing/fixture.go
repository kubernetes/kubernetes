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

package testing

import (
	"fmt"
	"reflect"
	"sigs.k8s.io/yaml"
	"sort"
	"strings"
	"sync"

	jsonpatch "gopkg.in/evanphx/json-patch.v4"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apimachinery/pkg/util/managedfields/managedfieldstest"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/watch"
	restclient "k8s.io/client-go/rest"
)

// ObjectTracker keeps track of objects. It is intended to be used to
// fake calls to a server by returning objects based on their kind,
// namespace and name.
type ObjectTracker interface {
	// Add adds an object to the tracker. If object being added
	// is a list, its items are added separately.
	Add(obj runtime.Object) error

	// Get retrieves the object by its kind, namespace and name.
	Get(gvr schema.GroupVersionResource, ns, name string) (runtime.Object, error)

	// Create adds an object to the tracker in the specified namespace.
	Create(gvr schema.GroupVersionResource, obj runtime.Object, ns string) error

	// Update updates an existing object in the tracker in the specified namespace.
	Update(gvr schema.GroupVersionResource, obj runtime.Object, ns string) error

	// List retrieves all objects of a given kind in the given
	// namespace. Only non-List kinds are accepted.
	List(gvr schema.GroupVersionResource, gvk schema.GroupVersionKind, ns string) (runtime.Object, error)

	// Delete deletes an existing object from the tracker. If object
	// didn't exist in the tracker prior to deletion, Delete returns
	// no error.
	Delete(gvr schema.GroupVersionResource, ns, name string) error

	// Watch watches objects from the tracker. Watch returns a channel
	// which will push added / modified / deleted object.
	Watch(gvr schema.GroupVersionResource, ns string) (watch.Interface, error)
}

// ManagedFieldObjectTracker keeps track of objects and managed fields. It is intended to be used to
// fake calls to a server by returning objects based on their kind, namespace and name.
type ManagedFieldObjectTracker interface {
	ObjectTracker

	// CreateWithFieldManager adds an object to the tracker in the specified namespace using the provided fieldManager.
	CreateWithFieldManager(gvr schema.GroupVersionResource, obj runtime.Object, ns string, fieldManager string) error

	// UpdateWithFieldManager updates an existing object in the tracker in the specified namespace using the provided fieldManager.
	UpdateWithFieldManager(gvr schema.GroupVersionResource, obj runtime.Object, ns string, fieldManager string) error

	// Apply applies an object in the tracker in the specified namespace.
	Apply(gvr schema.GroupVersionResource, applyConfiguration runtime.Object, ns string, fieldManager string, force bool) error
}

// ObjectScheme abstracts the implementation of common operations on objects.
type ObjectScheme interface {
	runtime.ObjectCreater
	runtime.ObjectTyper
}

// ObjectReaction returns a ReactionFunc that applies core.Action to
// the given tracker.
//
// If tracker also implements ManagedFieldObjectTracker, then managed fields
// will be handled by the tracker and apply patch actions will be evaluated
// using the field manager and will take field ownership into consideration.
// Without a ManagedFieldObjectTracker, apply patch actions do not consider
// field ownership.
//
// WARNING: There is no server side defaulting, validation, or conversion handled
// by the fake client and subresources are not handled accurately (fields in the
// root resource are not automatically updated when a scale resource is updated, for example).
func ObjectReaction(tracker ObjectTracker) ReactionFunc {
	return func(action Action) (bool, runtime.Object, error) {
		fieldManager := "default-fake-field-manager"
		ns := action.GetNamespace()
		gvr := action.GetResource()

		// Here and below we need to switch on implementation types,
		// not on interfaces, as some interfaces are identical
		// (e.g. UpdateAction and CreateAction), so if we use them,
		// updates and creates end up matching the same case branch.
		switch action := action.(type) {

		case ListActionImpl:
			obj, err := tracker.List(gvr, action.GetKind(), ns)
			return true, obj, err

		case GetActionImpl:
			obj, err := tracker.Get(gvr, ns, action.GetName())
			return true, obj, err

		case CreateActionImpl:
			if len(action.CreateOptions.FieldManager) > 0 {
				fieldManager = action.CreateOptions.FieldManager
			}
			objMeta, err := meta.Accessor(action.GetObject())
			if err != nil {
				return true, nil, err
			}
			if action.GetSubresource() == "" {
				if fieldManagedTracker, ok := tracker.(ManagedFieldObjectTracker); ok {
					err = fieldManagedTracker.CreateWithFieldManager(gvr, action.GetObject(), ns, fieldManager)
				} else {
					err = tracker.Create(gvr, action.GetObject(), ns)
				}
				if err != nil {
					return true, nil, err
				}
			} else {
				oldObj, getOldObjErr := tracker.Get(gvr, ns, objMeta.GetName())
				if getOldObjErr != nil {
					return true, nil, getOldObjErr
				}
				// Check whether the existing historical object type is the same as the current operation object type that needs to be updated, and if it is the same, perform the update operation.
				if reflect.TypeOf(oldObj) == reflect.TypeOf(action.GetObject()) {
					// TODO: Currently we're handling subresource creation as an update
					// on the enclosing resource. This works for some subresources but
					// might not be generic enough.
					err = tracker.Update(gvr, action.GetObject(), ns)
				} else {
					// If the historical object type is different from the current object type, need to make sure we return the object submitted,don't persist the submitted object in the tracker.
					return true, action.GetObject(), nil
				}
			}
			if err != nil {
				return true, nil, err
			}
			obj, err := tracker.Get(gvr, ns, objMeta.GetName())
			return true, obj, err

		case UpdateActionImpl:
			if len(action.UpdateOptions.FieldManager) > 0 {
				fieldManager = action.UpdateOptions.FieldManager
			}
			objMeta, err := meta.Accessor(action.GetObject())
			if err != nil {
				return true, nil, err
			}

			if fieldManagedTracker, ok := tracker.(ManagedFieldObjectTracker); ok {
				err = fieldManagedTracker.UpdateWithFieldManager(gvr, action.GetObject(), ns, fieldManager)
				if err != nil {
					return true, nil, err
				}
			} else {
				err = tracker.Update(gvr, action.GetObject(), ns)
				if err != nil {
					return true, nil, err
				}
			}

			obj, err := tracker.Get(gvr, ns, objMeta.GetName())
			return true, obj, err

		case DeleteActionImpl:
			err := tracker.Delete(gvr, ns, action.GetName())
			if err != nil {
				return true, nil, err
			}
			return true, nil, nil

		case PatchActionImpl:
			if len(action.PatchOptions.FieldManager) > 0 {
				fieldManager = action.PatchOptions.FieldManager
			}
			force := false
			if action.PatchOptions.Force != nil {
				force = *action.PatchOptions.Force
			}

			fieldManagedTracker, hasManagedFieldTracker := tracker.(ManagedFieldObjectTracker)
			if hasManagedFieldTracker && action.GetPatchType() == types.ApplyPatchType {
				// Handle field managed apply:
				patchObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
				if err := yaml.Unmarshal(action.GetPatch(), &patchObj.Object); err != nil {
					return true, nil, err
				}
				err := fieldManagedTracker.Apply(gvr, patchObj, ns, fieldManager, force)
				if err != nil {
					return true, nil, err
				}
				applyConfigurationMeta, err := meta.Accessor(patchObj)
				if err != nil {
					return true, nil, err
				}
				obj, err := tracker.Get(gvr, ns, applyConfigurationMeta.GetName())
				return true, obj, err
			}

			obj, err := tracker.Get(gvr, ns, action.GetName())
			if err != nil {
				return true, nil, err
			}

			old, err := json.Marshal(obj)
			if err != nil {
				return true, nil, err
			}

			// reset the object in preparation to unmarshal, since unmarshal does not guarantee that fields
			// in obj that are removed by patch are cleared
			value := reflect.ValueOf(obj)
			value.Elem().Set(reflect.New(value.Type().Elem()).Elem())

			switch action.GetPatchType() {
			case types.JSONPatchType:
				patch, err := jsonpatch.DecodePatch(action.GetPatch())
				if err != nil {
					return true, nil, err
				}
				modified, err := patch.Apply(old)
				if err != nil {
					return true, nil, err
				}

				if err = json.Unmarshal(modified, obj); err != nil {
					return true, nil, err
				}
			case types.MergePatchType:
				modified, err := jsonpatch.MergePatch(old, action.GetPatch())
				if err != nil {
					return true, nil, err
				}

				if err := json.Unmarshal(modified, obj); err != nil {
					return true, nil, err
				}
			case types.StrategicMergePatchType, types.ApplyPatchType:
				// For backward compatibility with behavior 1.30 and earlier, continue to handle apply
				// via strategic merge patch if a ManagedFieldObjectTracker is not used.
				mergedByte, err := strategicpatch.StrategicMergePatch(old, action.GetPatch(), obj)
				if err != nil {
					return true, nil, err
				}
				if err = json.Unmarshal(mergedByte, obj); err != nil {
					return true, nil, err
				}
			default:
				return true, nil, fmt.Errorf("PatchType is not supported")
			}

			if fieldManagedTracker, ok := tracker.(ManagedFieldObjectTracker); ok {
				if err = fieldManagedTracker.UpdateWithFieldManager(gvr, obj, ns, fieldManager); err != nil {
					return true, nil, err
				}
			} else {
				if err = tracker.Update(gvr, obj, ns); err != nil {
					return true, nil, err
				}
			}

			return true, obj, nil

		default:
			return false, nil, fmt.Errorf("no reaction implemented for %s", action)
		}
	}
}

type tracker struct {
	scheme  ObjectScheme
	decoder runtime.Decoder
	lock    sync.RWMutex
	objects map[schema.GroupVersionResource]map[types.NamespacedName]runtime.Object
	// The value type of watchers is a map of which the key is either a namespace or
	// all/non namespace aka "" and its value is list of fake watchers.
	// Manipulations on resources will broadcast the notification events into the
	// watchers' channel. Note that too many unhandled events (currently 100,
	// see apimachinery/pkg/watch.DefaultChanSize) will cause a panic.
	watchers map[schema.GroupVersionResource]map[string][]*watch.RaceFreeFakeWatcher
}

var _ ObjectTracker = &tracker{}

// NewObjectTracker returns an ObjectTracker that can be used to keep track
// of objects for the fake clientset. Mostly useful for unit tests.
func NewObjectTracker(scheme ObjectScheme, decoder runtime.Decoder) ObjectTracker {
	return &tracker{
		scheme:   scheme,
		decoder:  decoder,
		objects:  make(map[schema.GroupVersionResource]map[types.NamespacedName]runtime.Object),
		watchers: make(map[schema.GroupVersionResource]map[string][]*watch.RaceFreeFakeWatcher),
	}
}

func (t *tracker) List(gvr schema.GroupVersionResource, gvk schema.GroupVersionKind, ns string) (runtime.Object, error) {
	// Heuristic for list kind: original kind + List suffix. Might
	// not always be true but this tracker has a pretty limited
	// understanding of the actual API model.
	listGVK := gvk
	listGVK.Kind = listGVK.Kind + "List"
	// GVK does have the concept of "internal version". The scheme recognizes
	// the runtime.APIVersionInternal, but not the empty string.
	if listGVK.Version == "" {
		listGVK.Version = runtime.APIVersionInternal
	}

	list, err := t.scheme.New(listGVK)
	if err != nil {
		return nil, err
	}

	if !meta.IsListType(list) {
		return nil, fmt.Errorf("%q is not a list type", listGVK.Kind)
	}

	t.lock.RLock()
	defer t.lock.RUnlock()

	objs, ok := t.objects[gvr]
	if !ok {
		return list, nil
	}

	matchingObjs, err := filterByNamespace(objs, ns)
	if err != nil {
		return nil, err
	}
	if err := meta.SetList(list, matchingObjs); err != nil {
		return nil, err
	}
	return list.DeepCopyObject(), nil
}

func (t *tracker) Watch(gvr schema.GroupVersionResource, ns string) (watch.Interface, error) {
	t.lock.Lock()
	defer t.lock.Unlock()

	fakewatcher := watch.NewRaceFreeFake()

	if _, exists := t.watchers[gvr]; !exists {
		t.watchers[gvr] = make(map[string][]*watch.RaceFreeFakeWatcher)
	}
	t.watchers[gvr][ns] = append(t.watchers[gvr][ns], fakewatcher)
	return fakewatcher, nil
}

func (t *tracker) Get(gvr schema.GroupVersionResource, ns, name string) (runtime.Object, error) {
	errNotFound := apierrors.NewNotFound(gvr.GroupResource(), name)

	t.lock.RLock()
	defer t.lock.RUnlock()

	objs, ok := t.objects[gvr]
	if !ok {
		return nil, errNotFound
	}

	matchingObj, ok := objs[types.NamespacedName{Namespace: ns, Name: name}]
	if !ok {
		return nil, errNotFound
	}

	// Only one object should match in the tracker if it works
	// correctly, as Add/Update methods enforce kind/namespace/name
	// uniqueness.
	obj := matchingObj.DeepCopyObject()
	if status, ok := obj.(*metav1.Status); ok {
		if status.Status != metav1.StatusSuccess {
			return nil, &apierrors.StatusError{ErrStatus: *status}
		}
	}

	return obj, nil
}

func (t *tracker) Add(obj runtime.Object) error {
	if meta.IsListType(obj) {
		return t.addList(obj, false)
	}
	objMeta, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	gvks, _, err := t.scheme.ObjectKinds(obj)
	if err != nil {
		return err
	}

	if partial, ok := obj.(*metav1.PartialObjectMetadata); ok && len(partial.TypeMeta.APIVersion) > 0 {
		gvks = []schema.GroupVersionKind{partial.TypeMeta.GroupVersionKind()}
	}

	if len(gvks) == 0 {
		return fmt.Errorf("no registered kinds for %v", obj)
	}
	for _, gvk := range gvks {
		// NOTE: UnsafeGuessKindToResource is a heuristic and default match. The
		// actual registration in apiserver can specify arbitrary route for a
		// gvk. If a test uses such objects, it cannot preset the tracker with
		// objects via Add(). Instead, it should trigger the Create() function
		// of the tracker, where an arbitrary gvr can be specified.
		gvr, _ := meta.UnsafeGuessKindToResource(gvk)
		// Resource doesn't have the concept of "__internal" version, just set it to "".
		if gvr.Version == runtime.APIVersionInternal {
			gvr.Version = ""
		}

		err := t.add(gvr, obj, objMeta.GetNamespace(), false)
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *tracker) Create(gvr schema.GroupVersionResource, obj runtime.Object, ns string) error {
	return t.add(gvr, obj, ns, false)
}

func (t *tracker) Update(gvr schema.GroupVersionResource, obj runtime.Object, ns string) error {
	return t.add(gvr, obj, ns, true)
}

func (t *tracker) getWatches(gvr schema.GroupVersionResource, ns string) []*watch.RaceFreeFakeWatcher {
	watches := []*watch.RaceFreeFakeWatcher{}
	if t.watchers[gvr] != nil {
		if w := t.watchers[gvr][ns]; w != nil {
			watches = append(watches, w...)
		}
		if ns != metav1.NamespaceAll {
			if w := t.watchers[gvr][metav1.NamespaceAll]; w != nil {
				watches = append(watches, w...)
			}
		}
	}
	return watches
}

func (t *tracker) add(gvr schema.GroupVersionResource, obj runtime.Object, ns string, replaceExisting bool) error {
	t.lock.Lock()
	defer t.lock.Unlock()

	gr := gvr.GroupResource()

	// To avoid the object from being accidentally modified by caller
	// after it's been added to the tracker, we always store the deep
	// copy.
	obj = obj.DeepCopyObject()

	newMeta, err := meta.Accessor(obj)
	if err != nil {
		return err
	}

	// Propagate namespace to the new object if hasn't already been set.
	if len(newMeta.GetNamespace()) == 0 {
		newMeta.SetNamespace(ns)
	}

	if ns != newMeta.GetNamespace() {
		msg := fmt.Sprintf("request namespace does not match object namespace, request: %q object: %q", ns, newMeta.GetNamespace())
		return apierrors.NewBadRequest(msg)
	}

	_, ok := t.objects[gvr]
	if !ok {
		t.objects[gvr] = make(map[types.NamespacedName]runtime.Object)
	}

	namespacedName := types.NamespacedName{Namespace: newMeta.GetNamespace(), Name: newMeta.GetName()}
	if _, ok = t.objects[gvr][namespacedName]; ok {
		if replaceExisting {
			for _, w := range t.getWatches(gvr, ns) {
				// To avoid the object from being accidentally modified by watcher
				w.Modify(obj.DeepCopyObject())
			}
			t.objects[gvr][namespacedName] = obj
			return nil
		}
		return apierrors.NewAlreadyExists(gr, newMeta.GetName())
	}

	if replaceExisting {
		// Tried to update but no matching object was found.
		return apierrors.NewNotFound(gr, newMeta.GetName())
	}

	t.objects[gvr][namespacedName] = obj

	for _, w := range t.getWatches(gvr, ns) {
		// To avoid the object from being accidentally modified by watcher
		w.Add(obj.DeepCopyObject())
	}

	return nil
}

func (t *tracker) addList(obj runtime.Object, replaceExisting bool) error {
	list, err := meta.ExtractList(obj)
	if err != nil {
		return err
	}
	errs := runtime.DecodeList(list, t.decoder)
	if len(errs) > 0 {
		return errs[0]
	}
	for _, obj := range list {
		if err := t.Add(obj); err != nil {
			return err
		}
	}
	return nil
}

func (t *tracker) Delete(gvr schema.GroupVersionResource, ns, name string) error {
	t.lock.Lock()
	defer t.lock.Unlock()

	objs, ok := t.objects[gvr]
	if !ok {
		return apierrors.NewNotFound(gvr.GroupResource(), name)
	}

	namespacedName := types.NamespacedName{Namespace: ns, Name: name}
	obj, ok := objs[namespacedName]
	if !ok {
		return apierrors.NewNotFound(gvr.GroupResource(), name)
	}

	delete(objs, namespacedName)
	for _, w := range t.getWatches(gvr, ns) {
		w.Delete(obj.DeepCopyObject())
	}
	return nil
}

type managedFieldObjectTracker struct {
	ObjectTracker
	scheme        ObjectScheme
	mapper        meta.RESTMapper
	typeConverter managedfields.TypeConverter
}

var _ ManagedFieldObjectTracker = &managedFieldObjectTracker{}

// NewFieldManagedObjectTracker returns an ObjectTracker that can be used to keep track
// of objects and managed fields for the fake clientset. Mostly useful for unit tests.
func NewFieldManagedObjectTracker(tracker ObjectTracker, scheme ObjectScheme, mapper meta.RESTMapper, typeConverter managedfields.TypeConverter) ManagedFieldObjectTracker {
	return &managedFieldObjectTracker{ObjectTracker: tracker, scheme: scheme, mapper: mapper, typeConverter: typeConverter}
}

func (t *managedFieldObjectTracker) CreateWithFieldManager(gvr schema.GroupVersionResource, obj runtime.Object, ns string, fieldManager string) error {
	gvk, err := t.mapper.KindFor(gvr)
	if err != nil {
		return err
	}
	mgr := managedfieldstest.NewFakeFieldManager(t.typeConverter, gvk)

	objMeta, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	oldObj, err := t.ObjectTracker.Get(gvr, ns, objMeta.GetName())
	if apierrors.IsNotFound(err) {
		oldObj, err = t.scheme.New(gvk)
		if err != nil {
			return err
		}
		oldObj.GetObjectKind().SetGroupVersionKind(gvk)
	} else if err != nil {
		return err
	}
	objWithManagedFields, err := mgr.Update(oldObj, obj, fieldManager)
	if err != nil {
		return err
	}
	return t.ObjectTracker.Create(gvr, objWithManagedFields, ns)
}

func (t *managedFieldObjectTracker) UpdateWithFieldManager(gvr schema.GroupVersionResource, obj runtime.Object, ns string, fieldManager string) error {
	gvk, err := t.mapper.KindFor(gvr)
	if err != nil {
		return err
	}
	mgr := managedfieldstest.NewFakeFieldManager(t.typeConverter, gvk)

	objMeta, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	oldObj, err := t.ObjectTracker.Get(gvr, ns, objMeta.GetName())
	if err != nil {
		return err
	}
	objWithManagedFields, err := mgr.Update(oldObj, obj, fieldManager)
	if err != nil {
		return err
	}

	return t.ObjectTracker.Update(gvr, objWithManagedFields, ns)
}

func (t *managedFieldObjectTracker) Apply(gvr schema.GroupVersionResource, applyConfiguration runtime.Object, ns string, fieldManager string, force bool) error {
	gvk, err := t.mapper.KindFor(gvr)
	if err != nil {
		return err
	}
	applyConfigurationMeta, err := meta.Accessor(applyConfiguration)
	if err != nil {
		return err
	}

	exists := true
	liveObject, err := t.ObjectTracker.Get(gvr, ns, applyConfigurationMeta.GetName())
	if apierrors.IsNotFound(err) {
		exists = false
		liveObject, err = t.scheme.New(gvk)
		if err != nil {
			return err
		}
		liveObject.GetObjectKind().SetGroupVersionKind(gvk)
	} else if err != nil {
		return err
	}
	mgr := managedfieldstest.NewFakeFieldManager(t.typeConverter, gvk)
	objWithManagedFields, err := mgr.Apply(liveObject, applyConfiguration, fieldManager, force)
	if err != nil {
		return err
	}

	if !exists {
		return t.ObjectTracker.Create(gvr, objWithManagedFields, ns)
	} else {
		return t.ObjectTracker.Update(gvr, objWithManagedFields, ns)
	}
}

// filterByNamespace returns all objects in the collection that
// match provided namespace. Empty namespace matches
// non-namespaced objects.
func filterByNamespace(objs map[types.NamespacedName]runtime.Object, ns string) ([]runtime.Object, error) {
	var res []runtime.Object

	for _, obj := range objs {
		acc, err := meta.Accessor(obj)
		if err != nil {
			return nil, err
		}
		if ns != "" && acc.GetNamespace() != ns {
			continue
		}
		res = append(res, obj)
	}

	// Sort res to get deterministic order.
	sort.Slice(res, func(i, j int) bool {
		acc1, _ := meta.Accessor(res[i])
		acc2, _ := meta.Accessor(res[j])
		if acc1.GetNamespace() != acc2.GetNamespace() {
			return acc1.GetNamespace() < acc2.GetNamespace()
		}
		return acc1.GetName() < acc2.GetName()
	})
	return res, nil
}

func DefaultWatchReactor(watchInterface watch.Interface, err error) WatchReactionFunc {
	return func(action Action) (bool, watch.Interface, error) {
		return true, watchInterface, err
	}
}

// SimpleReactor is a Reactor.  Each reaction function is attached to a given verb,resource tuple.  "*" in either field matches everything for that value.
// For instance, *,pods matches all verbs on pods.  This allows for easier composition of reaction functions
type SimpleReactor struct {
	Verb     string
	Resource string

	Reaction ReactionFunc
}

func (r *SimpleReactor) Handles(action Action) bool {
	verbCovers := r.Verb == "*" || r.Verb == action.GetVerb()
	if !verbCovers {
		return false
	}

	return resourceCovers(r.Resource, action)
}

func (r *SimpleReactor) React(action Action) (bool, runtime.Object, error) {
	return r.Reaction(action)
}

// SimpleWatchReactor is a WatchReactor.  Each reaction function is attached to a given resource.  "*" matches everything for that value.
// For instance, *,pods matches all verbs on pods.  This allows for easier composition of reaction functions
type SimpleWatchReactor struct {
	Resource string

	Reaction WatchReactionFunc
}

func (r *SimpleWatchReactor) Handles(action Action) bool {
	return resourceCovers(r.Resource, action)
}

func (r *SimpleWatchReactor) React(action Action) (bool, watch.Interface, error) {
	return r.Reaction(action)
}

// SimpleProxyReactor is a ProxyReactor.  Each reaction function is attached to a given resource.  "*" matches everything for that value.
// For instance, *,pods matches all verbs on pods.  This allows for easier composition of reaction functions.
type SimpleProxyReactor struct {
	Resource string

	Reaction ProxyReactionFunc
}

func (r *SimpleProxyReactor) Handles(action Action) bool {
	return resourceCovers(r.Resource, action)
}

func (r *SimpleProxyReactor) React(action Action) (bool, restclient.ResponseWrapper, error) {
	return r.Reaction(action)
}

func resourceCovers(resource string, action Action) bool {
	if resource == "*" {
		return true
	}

	if resource == action.GetResource().Resource {
		return true
	}

	if index := strings.Index(resource, "/"); index != -1 &&
		resource[:index] == action.GetResource().Resource &&
		resource[index+1:] == action.GetSubresource() {
		return true
	}

	return false
}
