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
	"sort"
	"sync"

	jsonpatch "github.com/evanphx/json-patch"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
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

// ObjectScheme abstracts the implementation of common operations on objects.
type ObjectScheme interface {
	runtime.ObjectCreater
	runtime.ObjectTyper
}

// ObjectReaction returns a ReactionFunc that applies core.Action to
// the given tracker.
func ObjectReaction(tracker ObjectTracker) ReactionFunc {
	return func(action Action) (bool, runtime.Object, error) {
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
			objMeta, err := meta.Accessor(action.GetObject())
			if err != nil {
				return true, nil, err
			}
			if action.GetSubresource() == "" {
				err = tracker.Create(gvr, action.GetObject(), ns)
			} else {
				// TODO: Currently we're handling subresource creation as an update
				// on the enclosing resource. This works for some subresources but
				// might not be generic enough.
				err = tracker.Update(gvr, action.GetObject(), ns)
			}
			if err != nil {
				return true, nil, err
			}
			obj, err := tracker.Get(gvr, ns, objMeta.GetName())
			return true, obj, err

		case UpdateActionImpl:
			objMeta, err := meta.Accessor(action.GetObject())
			if err != nil {
				return true, nil, err
			}
			err = tracker.Update(gvr, action.GetObject(), ns)
			if err != nil {
				return true, nil, err
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
			case types.StrategicMergePatchType:
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

			if err = tracker.Update(gvr, obj, ns); err != nil {
				return true, nil, err
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
	errNotFound := errors.NewNotFound(gvr.GroupResource(), name)

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
			return nil, &errors.StatusError{ErrStatus: *status}
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
		return errors.NewBadRequest(msg)
	}

	_, ok := t.objects[gvr]
	if !ok {
		t.objects[gvr] = make(map[types.NamespacedName]runtime.Object)
	}

	namespacedName := types.NamespacedName{Namespace: newMeta.GetNamespace(), Name: newMeta.GetName()}
	if _, ok = t.objects[gvr][namespacedName]; ok {
		if replaceExisting {
			for _, w := range t.getWatches(gvr, ns) {
				w.Modify(obj)
			}
			t.objects[gvr][namespacedName] = obj
			return nil
		}
		return errors.NewAlreadyExists(gr, newMeta.GetName())
	}

	if replaceExisting {
		// Tried to update but no matching object was found.
		return errors.NewNotFound(gr, newMeta.GetName())
	}

	t.objects[gvr][namespacedName] = obj

	for _, w := range t.getWatches(gvr, ns) {
		w.Add(obj)
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
		return errors.NewNotFound(gvr.GroupResource(), name)
	}

	namespacedName := types.NamespacedName{Namespace: ns, Name: name}
	obj, ok := objs[namespacedName]
	if !ok {
		return errors.NewNotFound(gvr.GroupResource(), name)
	}

	delete(objs, namespacedName)
	for _, w := range t.getWatches(gvr, ns) {
		w.Delete(obj)
	}
	return nil
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
	resourceCovers := r.Resource == "*" || r.Resource == action.GetResource().Resource
	if !resourceCovers {
		return false
	}

	return true
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
	resourceCovers := r.Resource == "*" || r.Resource == action.GetResource().Resource
	if !resourceCovers {
		return false
	}

	return true
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
	resourceCovers := r.Resource == "*" || r.Resource == action.GetResource().Resource
	if !resourceCovers {
		return false
	}

	return true
}

func (r *SimpleProxyReactor) React(action Action) (bool, restclient.ResponseWrapper, error) {
	return r.Reaction(action)
}
