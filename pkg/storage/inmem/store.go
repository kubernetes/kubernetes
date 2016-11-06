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

package inmem

import (
	"k8s.io/kubernetes/pkg/storage"

	"errors"
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"

	"bytes"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/watch"
	"time"
)

var errorLSNMismatch = errors.New("LSN mismatch")
var errorItemNotFound = errors.New("Item not found")
var errorAlreadyExists = errors.New("Item already exists")

type LSN uint64

type store struct {
	versioner storage.Versioner
	codec     runtime.Codec

	backend *Backend
}

func NewStore(prefix string, codec runtime.Codec, backend *Backend) *store {
	glog.Infof("building inmem store prefix=%q", prefix)
	versioner := etcd.APIObjectVersioner{}
	s := &store{
		backend:   backend,
		versioner: versioner,
		codec:     codec,
	}
	return s
}

var _ storage.Interface = &store{}

// Versioner implements storage.Interface.Versioner.
func (s *store) Versioner() storage.Versioner {
	return s.versioner
}

// Create adds a new object at a key unless it already exists. 'ttl' is time-to-live
// in seconds (0 means forever). If no error is returned and out is not nil, out will be
// set to the read value from database.
func (s *store) Create(ctx context.Context, path string, obj, out runtime.Object, ttl uint64) error {
	glog.Infof("Create %s", path)
	if version, err := s.versioner.ObjectResourceVersion(obj); err == nil && version != 0 {
		return errors.New("resourceVersion should not be set on objects to be created")
	}

	glog.Infof("1")

	data, err := runtime.Encode(s.codec, obj)
	if err != nil {
		return err
	}

	glog.Infof("2")

	objMeta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return storage.NewInternalErrorf("can't get meta on un-introspectable object %v, got error: %v", obj, err)
	}

	glog.Infof("3")

	bucket, k := splitPath(path)
	item := itemData{objMeta.UID, data, 0, 0}
	if ttl != 0 {
		item.expiry = uint64(time.Now().Unix()) + ttl
	}

	lsn, err := s.backend.Create(bucket, k, item)
	if err != nil {
		if err == errorAlreadyExists {
			return storage.NewKeyExistsError(path, 0)
		}
		return err
	}

	if out != nil {
		return decode(s.codec, s.versioner, data, out, lsn)
	}
	return nil
}

// Delete removes the specified key and returns the value that existed at that spot.
// If key didn't exist, it will return NotFound storage error.
func (s *store) Delete(ctx context.Context, path string, out runtime.Object, preconditions *storage.Preconditions) error {
	// TODO: Needed?
	_, err := conversion.EnforcePtr(out)
	if err != nil {
		panic("unable to convert output object to pointer")
	}

	bucket, k := splitPath(path)

	oldItem, err := s.backend.Delete(bucket, k, preconditions)
	if err != nil {
		if err == errorItemNotFound {
			return storage.NewKeyNotFoundError(path, 0)
		}
		return err
	}

	return decode(s.codec, s.versioner, oldItem.data, out, oldItem.lsn)
}

// Watch begins watching the specified key. Events are decoded into API objects,
// and any items selected by 'p' are sent down to returned watch.Interface.
// resourceVersion may be used to specify what version to begin watching,
// which should be the current resourceVersion, and no longer rv+1
// (e.g. reconnecting without missing any updates).
func (s *store) Watch(ctx context.Context, path string, resourceVersion string, p storage.SelectionPredicate) (watch.Interface, error) {
	glog.V(4).Infof("Watch %s from %s", path, resourceVersion)
	rev, err := storage.ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}

	return s.backend.log.newWatcher(s, LSN(rev), p, path, false)
}

// WatchList begins watching the specified key's items. Items are decoded into API
// objects and any item selected by 'p' are sent down to returned watch.Interface.
// resourceVersion may be used to specify what version to begin watching,
// which should be the current resourceVersion, and no longer rv+1
// (e.g. reconnecting without missing any updates).
func (s *store) WatchList(ctx context.Context, path string, resourceVersion string, p storage.SelectionPredicate) (watch.Interface, error) {
	glog.V(4).Infof("WatchList %s from %s", path, resourceVersion)
	rev, err := storage.ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}

	return s.backend.log.newWatcher(s, LSN(rev), p, path, true)
}

// Get implements storage.Interface.Get.
func (s *store) Get(ctx context.Context, path string, out runtime.Object, ignoreNotFound bool) error {
	glog.V(4).Infof("Get %s", path)
	bucket, k := splitPath(path)

	item, _, err := s.backend.Get(bucket, k)
	if err != nil {
		if err == errorItemNotFound {
			if ignoreNotFound {
				return runtime.SetZeroValue(out)
			} else {
				return storage.NewKeyNotFoundError(path, 0)
			}
		}
		return err
	}

	return decode(s.codec, s.versioner, item.data, out, item.lsn)
}

// GetToList implements storage.Interface.GetToList.
func (s *store) GetToList(ctx context.Context, path string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	glog.V(4).Infof("GetToList %s", path)
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}

	bucket, k := splitPath(path)

	item, systemLSN, err := s.backend.Get(bucket, k)
	if err != nil {
		if err == errorItemNotFound {
			return nil
		}
		return err
	}

	if err != nil {
		return err
	}

	elems := []itemData{item}
	if err := decodeList(elems, storage.SimpleFilter(pred), listPtr, s.codec, s.versioner); err != nil {
		return err
	}
	// update version with cluster level revision
	return s.versioner.UpdateList(listObj, uint64(systemLSN))
}

// List implements storage.Interface.List.
func (s *store) List(ctx context.Context, path, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	glog.V(4).Infof("List %s", path)

	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}

	items, lsn, err := s.backend.List(path)
	if err != nil {
		return err
	}

	if err := decodeList(items, storage.SimpleFilter(pred), listPtr, s.codec, s.versioner); err != nil {
		return err
	}

	// update version with cluster level revision
	return s.versioner.UpdateList(listObj, uint64(lsn))
}

// decodeList decodes a list of values into a list of objects, with resource version set to corresponding rev.
// On success, ListPtr would be set to the list of objects.
func decodeList(elems []itemData, filter storage.FilterFunc, ListPtr interface{}, codec runtime.Codec, versioner storage.Versioner) error {
	v, err := conversion.EnforcePtr(ListPtr)
	if err != nil || v.Kind() != reflect.Slice {
		panic("need ptr to slice")
	}
	for _, elem := range elems {
		obj, _, err := codec.Decode(elem.data, nil, reflect.New(v.Type().Elem()).Interface().(runtime.Object))
		if err != nil {
			return err
		}
		// being unable to set the version does not prevent the object from being extracted
		versioner.UpdateObject(obj, uint64(elem.lsn))
		if filter(obj) {
			v.Set(reflect.Append(v, reflect.ValueOf(obj).Elem()))
		}
	}
	return nil
}

// GuaranteedUpdate keeps calling 'tryUpdate()' to update key 'key' (of type 'ptrToType')
// retrying the update until success if there is index conflict.
// Note that object passed to tryUpdate may change across invocations of tryUpdate() if
// other writers are simultaneously updating it, so tryUpdate() needs to take into account
// the current contents of the object when deciding how the update object should look.
// If the key doesn't exist, it will return NotFound storage error if ignoreNotFound=false
// or zero value in 'ptrToType' parameter otherwise.
// If the object to update has the same value as previous, it won't do any update
// but will return the object in 'ptrToType' parameter.
// If 'suggestion' can contain zero or one element - in such case this can be used as
// a suggestion about the current version of the object to avoid read operation from
// storage to get it.
//
// Example:
//
// s := /* implementation of Interface */
// err := s.GuaranteedUpdate(
//     "myKey", &MyType{}, true,
//     func(input runtime.Object, res ResponseMeta) (runtime.Object, *uint64, error) {
//       // Before each incovation of the user defined function, "input" is reset to
//       // current contents for "myKey" in database.
//       curr := input.(*MyType)  // Guaranteed to succeed.
//
//       // Make the modification
//       curr.Counter++
//
//       // Return the modified object - return an error to stop iterating. Return
//       // a uint64 to alter the TTL on the object, or nil to keep it the same value.
//       return cur, nil, nil
//    }
// })
// GuaranteedUpdate implements storage.Interface.GuaranteedUpdate.
func (s *store) GuaranteedUpdate(
	ctx context.Context, path string, out runtime.Object, ignoreNotFound bool,
	precondtions *storage.Preconditions, tryUpdate storage.UpdateFunc, suggestion ...runtime.Object) error {
	glog.Infof("GuaranteedUpdate %s", path)

	v, err := conversion.EnforcePtr(out)
	if err != nil {
		panic("unable to convert output object to pointer")
	}

	bucket, k := splitPath(path)

	var uid types.UID
	var origState *objState
	if len(suggestion) == 1 && suggestion[0] != nil {
		origState, err = s.getStateFromObject(suggestion[0])
		if err != nil {
			return err
		}

		objMeta, err := api.ObjectMetaFor(origState.obj)
		if err != nil {
			return storage.NewInternalErrorf("can't get meta on un-introspectable object %v, got error: %v", origState.obj, err)
		}

		uid = objMeta.UID
	} else {
		item, _, err := s.backend.Get(bucket, k)
		if err != nil {
			if err == errorItemNotFound {
				if ignoreNotFound {
					// keep goig
				} else {
					return storage.NewKeyNotFoundError(path, 0)
				}
			} else {
				return err
			}
		}
		origState, err = s.getState(item, v)
		if err != nil {
			return err
		}

		uid = item.uid
	}

	for {
		if err := checkPreconditions(path, precondtions, uid); err != nil {
			return err
		}

		ret, ttl, err := s.updateState(origState, tryUpdate)
		if err != nil {
			return err
		}

		data, err := runtime.Encode(s.codec, ret)
		if err != nil {
			return err
		}
		if bytes.Equal(data, origState.data) {
			return decode(s.codec, s.versioner, origState.data, out, origState.rev)
		}

		newItem := itemData{uid, data, 0, 0}
		if ttl != 0 {
			newItem.expiry = uint64(time.Now().Unix()) + ttl
		}

		// response will be the new item if we swapped,
		// or the existing item if err==errorLSNMismatch
		response, err := s.backend.Update(bucket, k, origState.rev, newItem)
		if err != nil {
			if err == errorLSNMismatch {
				glog.V(4).Infof("GuaranteedUpdate of %s failed because of a conflict, going to retry", path)

				origState, err = s.getState(response, v)
				if err != nil {
					return err
				}

				continue
			} else if err == errorItemNotFound {
				if ignoreNotFound {
					origState = &objState{
						obj:  reflect.New(v.Type()).Interface().(runtime.Object),
						meta: &storage.ResponseMeta{},
					}
					if err := runtime.SetZeroValue(origState.obj); err != nil {
						return err
					}
					continue
				}

				return storage.NewKeyNotFoundError(path, 0)
			} else {
				return err
			}
		}

		return decode(s.codec, s.versioner, data, out, response.lsn)
	}
}

type objState struct {
	obj  runtime.Object
	meta *storage.ResponseMeta
	rev  LSN
	data []byte
}

func (s *store) getState(item itemData, v reflect.Value) (*objState, error) {
	state := &objState{
		obj:  reflect.New(v.Type()).Interface().(runtime.Object),
		meta: &storage.ResponseMeta{},
	}
	state.rev = item.lsn
	state.meta.ResourceVersion = uint64(item.lsn)
	state.data = item.data
	if err := decode(s.codec, s.versioner, state.data, state.obj, state.rev); err != nil {
		return nil, err
	}
	return state, nil
}

func (s *store) getStateFromObject(obj runtime.Object) (*objState, error) {
	state := &objState{
		obj:  obj,
		meta: &storage.ResponseMeta{},
	}

	rv, err := s.versioner.ObjectResourceVersion(obj)
	if err != nil {
		return nil, fmt.Errorf("couldn't get resource version: %v", err)
	}
	state.rev = LSN(rv)
	state.meta.ResourceVersion = rv

	// Compute the serialized form - for that we need to temporarily clean
	// its resource version field (those are not stored in etcd).
	if err := s.versioner.UpdateObject(obj, 0); err != nil {
		return nil, errors.New("resourceVersion cannot be set on objects store in etcd")
	}
	state.data, err = runtime.Encode(s.codec, obj)
	if err != nil {
		return nil, err
	}
	s.versioner.UpdateObject(state.obj, uint64(rv))
	return state, nil
}

func (s *store) updateState(st *objState, userUpdate storage.UpdateFunc) (runtime.Object, uint64, error) {
	ret, ttlPtr, err := userUpdate(st.obj, *st.meta)
	if err != nil {
		return nil, 0, err
	}

	version, err := s.versioner.ObjectResourceVersion(ret)
	if err != nil {
		return nil, 0, err
	}
	if version != 0 {
		// We cannot store object with resourceVersion in etcd. We need to reset it.
		if err := s.versioner.UpdateObject(ret, 0); err != nil {
			return nil, 0, fmt.Errorf("UpdateObject failed: %v", err)
		}
	}
	var ttl uint64
	if ttlPtr != nil {
		ttl = *ttlPtr
	}
	return ret, ttl, nil
}

// decode decodes value of bytes into object. It will also set the object resource version to rev.
// On success, objPtr would be set to the object.
func decode(codec runtime.Codec, versioner storage.Versioner, value []byte, objPtr runtime.Object, rev LSN) error {
	if _, err := conversion.EnforcePtr(objPtr); err != nil {
		panic("unable to convert output object to pointer")
	}
	_, _, err := codec.Decode(value, nil, objPtr)
	if err != nil {
		return err
	}
	// being unable to set the version does not prevent the object from being extracted
	versioner.UpdateObject(objPtr, uint64(rev))
	return nil
}

func checkPreconditions(path string, preconditions *storage.Preconditions, objectUID types.UID) error {
	if preconditions == nil {
		return nil
	}
	if preconditions.UID != nil && *preconditions.UID != objectUID {
		errMsg := fmt.Sprintf("Precondition failed: UID in precondition: %v, UID in object meta: %v", *preconditions.UID, objectUID)
		return storage.NewInvalidObjError(path, errMsg)
	}
	return nil
}
