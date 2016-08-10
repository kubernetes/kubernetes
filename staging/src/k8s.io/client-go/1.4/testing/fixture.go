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
	"sync"

	"k8s.io/client-go/1.4/pkg/api/errors"
	"k8s.io/client-go/1.4/pkg/api/meta"
	"k8s.io/client-go/1.4/pkg/api/unversioned"
	"k8s.io/client-go/1.4/pkg/apimachinery/registered"
	"k8s.io/client-go/1.4/pkg/runtime"
	"k8s.io/client-go/1.4/pkg/watch"
	"k8s.io/client-go/1.4/rest"
)

// ObjectTracker keeps track of objects. It is intended to be used to
// fake calls to a server by returning objects based on their kind,
// namespace and name.
type ObjectTracker interface {
	// Add adds an object to the tracker. If object being added
	// is a list, its items are added separately.
	Add(obj runtime.Object) error

	// Get retrieves the object by its kind, namespace and name.
	Get(gvk unversioned.GroupVersionKind, ns, name string) (runtime.Object, error)

	// Update updates an existing object in the tracker.
	Update(obj runtime.Object) error

	// List retrieves all objects of a given kind in the given
	// namespace. Only non-List kinds are accepted.
	List(gvk unversioned.GroupVersionKind, ns string) (runtime.Object, error)

	// Delete deletes an existing object from the tracker. If object
	// didn't exist in the tracker prior to deletion, Delete returns
	// no error.
	Delete(gvk unversioned.GroupVersionKind, ns, name string) error
}

// ObjectScheme abstracts the implementation of common operations on objects.
type ObjectScheme interface {
	runtime.ObjectCreater
	runtime.ObjectCopier
	runtime.ObjectTyper
}

// ObjectReaction returns a ReactionFunc that applies testing.Action to
// the given tracker.
func ObjectReaction(tracker ObjectTracker, mapper meta.RESTMapper) ReactionFunc {
	return func(action Action) (bool, runtime.Object, error) {
		ns := action.GetNamespace()
		gvr := action.GetResource()

		gvk, err := mapper.KindFor(gvr)
		if err != nil {
			return false, nil, fmt.Errorf("error getting kind for resource %q: %s", gvr, err)
		}

		// This is a temporary fix. Because there is no internal resource, so
		// the caller has no way to express that it expects to get an internal
		// kind back. A more proper fix will be directly specify the Kind when
		// build the action.
		gvk.Version = gvr.Version
		if len(gvk.Version) == 0 {
			gvk.Version = runtime.APIVersionInternal
		}

		// Here and below we need to switch on implementation types,
		// not on interfaces, as some interfaces are identical
		// (e.g. UpdateAction and CreateAction), so if we use them,
		// updates and creates end up matching the same case branch.
		switch action := action.(type) {

		case ListActionImpl:
			obj, err := tracker.List(gvk, ns)
			return true, obj, err

		case GetActionImpl:
			obj, err := tracker.Get(gvk, ns, action.GetName())
			return true, obj, err

		case CreateActionImpl:
			objMeta, err := meta.Accessor(action.GetObject())
			if err != nil {
				return true, nil, err
			}
			if action.GetSubresource() == "" {
				err = tracker.Add(action.GetObject())
			} else {
				// TODO: Currently we're handling subresource creation as an update
				// on the enclosing resource. This works for some subresources but
				// might not be generic enough.
				err = tracker.Update(action.GetObject())
			}
			if err != nil {
				return true, nil, err
			}
			obj, err := tracker.Get(gvk, ns, objMeta.GetName())
			return true, obj, err

		case UpdateActionImpl:
			objMeta, err := meta.Accessor(action.GetObject())
			if err != nil {
				return true, nil, err
			}
			err = tracker.Update(action.GetObject())
			if err != nil {
				return true, nil, err
			}
			obj, err := tracker.Get(gvk, ns, objMeta.GetName())
			return true, obj, err

		case DeleteActionImpl:
			err := tracker.Delete(gvk, ns, action.GetName())
			if err != nil {
				return true, nil, err
			}
			return true, nil, nil

		default:
			return false, nil, fmt.Errorf("no reaction implemented for %s", action)
		}
	}
}

type tracker struct {
	scheme  ObjectScheme
	decoder runtime.Decoder
	lock    sync.RWMutex
	objects map[unversioned.GroupVersionKind][]runtime.Object
}

var _ ObjectTracker = &tracker{}

// NewObjectTracker returns an ObjectTracker that can be used to keep track
// of objects for the fake clientset. Mostly useful for unit tests.
func NewObjectTracker(scheme ObjectScheme, decoder runtime.Decoder) ObjectTracker {
	return &tracker{
		scheme:  scheme,
		decoder: decoder,
		objects: make(map[unversioned.GroupVersionKind][]runtime.Object),
	}
}

func (t *tracker) List(gvk unversioned.GroupVersionKind, ns string) (runtime.Object, error) {
	// Heuristic for list kind: original kind + List suffix. Might
	// not always be true but this tracker has a pretty limited
	// understanding of the actual API model.
	listGVK := gvk
	listGVK.Kind = listGVK.Kind + "List"

	list, err := t.scheme.New(listGVK)
	if err != nil {
		return nil, err
	}

	if !meta.IsListType(list) {
		return nil, fmt.Errorf("%q is not a list type", listGVK.Kind)
	}

	t.lock.RLock()
	defer t.lock.RUnlock()

	objs, ok := t.objects[gvk]
	if !ok {
		return list, nil
	}

	matchingObjs, err := filterByNamespaceAndName(objs, ns, "")
	if err != nil {
		return nil, err
	}
	if err := meta.SetList(list, matchingObjs); err != nil {
		return nil, err
	}
	if list, err = t.scheme.Copy(list); err != nil {
		return nil, err
	}
	return list, nil
}

func (t *tracker) Get(gvk unversioned.GroupVersionKind, ns, name string) (runtime.Object, error) {
	if err := checkNamespace(gvk, ns); err != nil {
		return nil, err
	}

	errNotFound := errors.NewNotFound(unversioned.GroupResource{Group: gvk.Group, Resource: gvk.Kind}, name)

	t.lock.RLock()
	defer t.lock.RUnlock()

	objs, ok := t.objects[gvk]
	if !ok {
		return nil, errNotFound
	}

	matchingObjs, err := filterByNamespaceAndName(objs, ns, name)
	if err != nil {
		return nil, err
	}
	if len(matchingObjs) == 0 {
		return nil, errNotFound
	}
	if len(matchingObjs) > 1 {
		return nil, fmt.Errorf("more than one object matched gvk %s, ns: %q name: %q", gvk, ns, name)
	}

	// Only one object should match in the tracker if it works
	// correctly, as Add/Update methods enforce kind/namespace/name
	// uniqueness.
	obj, err := t.scheme.Copy(matchingObjs[0])
	if err != nil {
		return nil, err
	}

	if status, ok := obj.(*unversioned.Status); ok {
		if status.Details != nil {
			status.Details.Kind = gvk.Kind
		}
		if status.Status != unversioned.StatusSuccess {
			return nil, &errors.StatusError{ErrStatus: *status}
		}
	}

	return obj, nil
}

func (t *tracker) Add(obj runtime.Object) error {
	return t.add(obj, false)
}

func (t *tracker) Update(obj runtime.Object) error {
	return t.add(obj, true)
}

func (t *tracker) add(obj runtime.Object, replaceExisting bool) error {
	if meta.IsListType(obj) {
		return t.addList(obj, replaceExisting)
	}

	gvks, _, err := t.scheme.ObjectKinds(obj)
	if err != nil {
		return err
	}
	if len(gvks) == 0 {
		return fmt.Errorf("no registered kinds for %v", obj)
	}

	t.lock.Lock()
	defer t.lock.Unlock()

	for _, gvk := range gvks {
		gr := unversioned.GroupResource{Group: gvk.Group, Resource: gvk.Kind}

		// To avoid the object from being accidentally modified by caller
		// after it's been added to the tracker, we always store the deep
		// copy.
		obj, err = t.scheme.Copy(obj)
		if err != nil {
			return err
		}

		if status, ok := obj.(*unversioned.Status); ok && status.Details != nil {
			gvk.Kind = status.Details.Kind
		}

		newMeta, err := meta.Accessor(obj)
		if err != nil {
			return err
		}

		if err := checkNamespace(gvk, newMeta.GetNamespace()); err != nil {
			return err
		}

		for i, existingObj := range t.objects[gvk] {
			oldMeta, err := meta.Accessor(existingObj)
			if err != nil {
				return err
			}
			if oldMeta.GetNamespace() == newMeta.GetNamespace() && oldMeta.GetName() == newMeta.GetName() {
				if replaceExisting {
					t.objects[gvk][i] = obj
					return nil
				}
				return errors.NewAlreadyExists(gr, newMeta.GetName())
			}
		}

		if replaceExisting {
			// Tried to update but no matching object was found.
			return errors.NewNotFound(gr, newMeta.GetName())
		}

		t.objects[gvk] = append(t.objects[gvk], obj)
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
		err := t.add(obj, replaceExisting)
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *tracker) Delete(gvk unversioned.GroupVersionKind, ns, name string) error {
	if err := checkNamespace(gvk, ns); err != nil {
		return err
	}

	t.lock.Lock()
	defer t.lock.Unlock()

	found := false

	for i, existingObj := range t.objects[gvk] {
		objMeta, err := meta.Accessor(existingObj)
		if err != nil {
			return err
		}
		if objMeta.GetNamespace() == ns && objMeta.GetName() == name {
			t.objects[gvk] = append(t.objects[gvk][:i], t.objects[gvk][i+1:]...)
			found = true
			break
		}
	}

	if found {
		return nil
	}

	return errors.NewNotFound(unversioned.GroupResource{Group: gvk.Group, Resource: gvk.Kind}, name)
}

// filterByNamespaceAndName returns all objects in the collection that
// match provided namespace and name. Empty namespace matches
// non-namespaced objects.
func filterByNamespaceAndName(objs []runtime.Object, ns, name string) ([]runtime.Object, error) {
	var res []runtime.Object

	for _, obj := range objs {
		acc, err := meta.Accessor(obj)
		if err != nil {
			return nil, err
		}
		if ns != "" && acc.GetNamespace() != ns {
			continue
		}
		if name != "" && acc.GetName() != name {
			continue
		}
		res = append(res, obj)
	}

	return res, nil
}

// checkNamespace makes sure that the scope of gvk matches ns. It
// returns an error if namespace is empty but gvk is a namespaced
// kind, or if ns is non-empty and gvk is a namespaced kind.
func checkNamespace(gvk unversioned.GroupVersionKind, ns string) error {
	group, err := registered.Group(gvk.Group)
	if err != nil {
		return err
	}
	mapping, err := group.RESTMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return err
	}
	switch mapping.Scope.Name() {
	case meta.RESTScopeNameRoot:
		if ns != "" {
			return fmt.Errorf("namespace specified for a non-namespaced kind %s", gvk)
		}
	case meta.RESTScopeNameNamespace:
		if ns == "" {
			// Skipping this check for Events, since
			// controllers emit events that have no namespace,
			// even though Event is a namespaced resource.
			if gvk.Kind != "Event" {
				return fmt.Errorf("no namespace specified for a namespaced kind %s", gvk)
			}
		}
	}

	return nil
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

func (r *SimpleProxyReactor) React(action Action) (bool, rest.ResponseWrapper, error) {
	return r.Reaction(action)
}
