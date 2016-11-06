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

package native

import (
	"bytes"
	"errors"
	"fmt"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/watch"
	"path"
	"reflect"
	"time"
)

type LSN uint64

type store struct {
	versioner storage.Versioner
	codec     runtime.Codec

	backend StorageServiceClient

	prefix string
}

func NewStore(prefix string, codec runtime.Codec, backend StorageServiceClient) *store {
	glog.V(4).Infof("building native store prefix=%q", prefix)

	if prefix == "" {
		prefix = "/"
	}
	versioner := etcd.APIObjectVersioner{}
	s := &store{
		backend:   backend,
		versioner: versioner,
		codec:     codec,
		prefix:    prefix,
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
func (s *store) Create(ctx context.Context, p string, obj, out runtime.Object, ttl uint64) error {
	p = path.Join(s.prefix, p)

	if version, err := s.versioner.ObjectResourceVersion(obj); err == nil && version != 0 {
		return errors.New("resourceVersion should not be set on objects to be created")
	}

	data, err := runtime.Encode(s.codec, obj)
	if err != nil {
		return err
	}

	objMeta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return storage.NewInternalErrorf("can't get meta on un-introspectable object %v, got error: %v", obj, err)
	}

	itemData := &ItemData{
		Uid:  string(objMeta.UID),
		Data: data,
	}
	if ttl != 0 {
		itemData.Ttl = ttl
	}

	op := &StorageOperation{
		OpType:   StorageOperationType_CREATE,
		Path:     p,
		ItemData: itemData,
	}
	result, err := s.doOperation(ctx, op)

	if err != nil {
		return err
	}

	if result.ErrorCode != 0 {
		switch result.ErrorCode {
		case ErrorCode_ALREADY_EXISTS:
			return storage.NewKeyExistsError(p, 0)

		default:
			return fmt.Errorf("unexpected error code: %v", result.ErrorCode)
		}
	}

	if out != nil {
		return decode(s.codec, s.versioner, data, out, LSN(result.ItemData.Lsn))
	}
	return nil
}

// Delete removes the specified key and returns the value that existed at that spot.
// If key didn't exist, it will return NotFound storage error.
func (s *store) Delete(ctx context.Context, p string, out runtime.Object, preconditions *storage.Preconditions) error {
	p = path.Join(s.prefix, p)

	// TODO: Needed?
	_, err := conversion.EnforcePtr(out)
	if err != nil {
		panic("unable to convert output object to pointer")
	}

	op := &StorageOperation{
		OpType: StorageOperationType_DELETE,
		Path:   p,
	}
	if preconditions != nil {
		if preconditions.UID != nil {
			op.PreconditionUid = string(*preconditions.UID)
		}
	}
	result, err := s.doOperation(ctx, op)
	if err != nil {
		return err
	}

	if result.ErrorCode != 0 {
		switch result.ErrorCode {
		case ErrorCode_NOT_FOUND:
			return storage.NewKeyNotFoundError(p, 0)

		case ErrorCode_PRECONDITION_NOT_MET_UID:
			errMsg := fmt.Sprintf("Precondition failed: UID in precondition: %v, UID in object meta: %v", *preconditions.UID, result.ItemData.Uid)
			return storage.NewInvalidObjError(p, errMsg)

		default:
			return fmt.Errorf("unexpected error code: %v", result.ErrorCode)
		}
	}

	oldItem := result.ItemData
	return decode(s.codec, s.versioner, oldItem.Data, out, LSN(oldItem.Lsn))
}

// Watch begins watching the specified key. Events are decoded into API objects,
// and any items selected by 'p' are sent down to returned watch.Interface.
// resourceVersion may be used to specify what version to begin watching,
// which should be the current resourceVersion, and no longer rv+1
// (e.g. reconnecting without missing any updates).
func (s *store) Watch(ctx context.Context, p string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	p = path.Join(s.prefix, p)

	glog.V(4).Infof("Watch %s from %s", p, resourceVersion)
	rev, err := storage.ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}

	return newGrpcWatcher(ctx, s.backend, LSN(rev), p, false, pred, s.versioner, s.codec)
}

// WatchList begins watching the specified key's items. Items are decoded into API
// objects and any item selected by 'p' are sent down to returned watch.Interface.
// resourceVersion may be used to specify what version to begin watching,
// which should be the current resourceVersion, and no longer rv+1
// (e.g. reconnecting without missing any updates).
func (s *store) WatchList(ctx context.Context, p string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	p = path.Join(s.prefix, p)

	glog.V(4).Infof("WatchList %s from %s", p, resourceVersion)
	rev, err := storage.ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}

	return newGrpcWatcher(ctx, s.backend, LSN(rev), p, true, pred, s.versioner, s.codec)
}

// Get implements storage.Interface.Get.
func (s *store) Get(ctx context.Context, p string, out runtime.Object, ignoreNotFound bool) error {
	p = path.Join(s.prefix, p)

	glog.V(4).Infof("Get %s", p)

	op := &StorageOperation{
		OpType: StorageOperationType_GET,
		Path:   p,
	}
	result, err := s.doOperation(ctx, op)
	if err != nil {
		glog.V(4).Infof("Get %s => error %v", p, err)
		return err
	}

	if result.ErrorCode != 0 {
		switch result.ErrorCode {
		case ErrorCode_NOT_FOUND:
			glog.V(4).Infof("Get %s => NOT_FOUND", p)
			if ignoreNotFound {
				return runtime.SetZeroValue(out)
			} else {
				return storage.NewKeyNotFoundError(p, 0)
			}

		default:
			glog.V(4).Infof("Get %s => error code %s", result.ErrorCode)
			return fmt.Errorf("unexpected error code: %v", result.ErrorCode)
		}
	}

	return decode(s.codec, s.versioner, result.ItemData.Data, out, LSN(result.ItemData.Lsn))
}

// GetToList implements storage.Interface.GetToList.
func (s *store) GetToList(ctx context.Context, p string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	p = path.Join(s.prefix, p)

	glog.V(4).Infof("GetToList %s", p)
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}

	op := &StorageOperation{
		OpType: StorageOperationType_GET,
		Path:   p,
	}
	result, err := s.doOperation(ctx, op)
	if err != nil {
		return err
	}

	if result.ErrorCode != 0 {
		switch result.ErrorCode {
		case ErrorCode_NOT_FOUND:
			return nil

		default:
			return fmt.Errorf("unexpected error code: %v", result.ErrorCode)
		}
	}

	elems := []*ItemData{result.ItemData}
	if err := decodeList(elems, storage.SimpleFilter(pred), listPtr, s.codec, s.versioner); err != nil {
		return err
	}
	// update version with cluster level revision
	return s.versioner.UpdateList(listObj, uint64(result.CurrentLsn))
}

// List implements storage.Interface.List.
func (s *store) List(ctx context.Context, p, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	p = path.Join(s.prefix, p)

	glog.V(4).Infof("List %s", p)

	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}

	op := &StorageOperation{
		OpType: StorageOperationType_LIST,
		Path:   p,
	}

	result, err := s.doOperation(ctx, op)
	if err != nil {
		return err
	}

	if result.ErrorCode != 0 {
		switch result.ErrorCode {
		case ErrorCode_NOT_FOUND:
			// Return empty list

		default:
			return fmt.Errorf("unexpected error code: %v", result.ErrorCode)
		}
	} else {
		items := result.ItemList
		if err := decodeList(items, storage.SimpleFilter(pred), listPtr, s.codec, s.versioner); err != nil {
			return err
		}
	}

	// update version with cluster level revision
	return s.versioner.UpdateList(listObj, uint64(result.CurrentLsn))
}

// decodeList decodes a list of values into a list of objects, with resource version set to corresponding rev.
// On success, ListPtr would be set to the list of objects.
func decodeList(elems []*ItemData, filter storage.FilterFunc, ListPtr interface{}, codec runtime.Codec, versioner storage.Versioner) error {
	v, err := conversion.EnforcePtr(ListPtr)
	if err != nil || v.Kind() != reflect.Slice {
		panic("need ptr to slice")
	}
	for _, elem := range elems {
		obj, _, err := codec.Decode(elem.Data, nil, reflect.New(v.Type().Elem()).Interface().(runtime.Object))
		if err != nil {
			return err
		}
		// being unable to set the version does not prevent the object from being extracted
		versioner.UpdateObject(obj, uint64(elem.Lsn))
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
	ctx context.Context, p string, out runtime.Object, ignoreNotFound bool,
	precondtions *storage.Preconditions, tryUpdate storage.UpdateFunc, suggestion ...runtime.Object) error {
	p = path.Join(s.prefix, p)

	glog.V(4).Infof("GuaranteedUpdate %s", p)

	v, err := conversion.EnforcePtr(out)
	if err != nil {
		panic("unable to convert output object to pointer")
	}

	var doCreate bool
	var origState *objState
	if len(suggestion) == 1 && suggestion[0] != nil {
		origState, err = s.getStateFromObject(suggestion[0])
		if err != nil {
			return err
		}

		if origState.rev == 0 {
			glog.Infof("Object was %v", suggestion[0])
			glog.Fatalf("origState.rev unexpectedly 0")
		}
	} else {
		op := &StorageOperation{
			OpType: StorageOperationType_GET,
			Path:   p,
		}
		result, err := s.doOperation(ctx, op)
		if err != nil {
			return err
		}

		if result.ErrorCode != 0 {
			switch result.ErrorCode {
			case ErrorCode_NOT_FOUND:
				if ignoreNotFound {
					// keep going
					doCreate = true

					origState = &objState{
						obj:  reflect.New(v.Type()).Interface().(runtime.Object),
						meta: &storage.ResponseMeta{},
					}
				} else {
					return storage.NewKeyNotFoundError(p, 0)
				}

			default:
				return fmt.Errorf("unexpected error code: %v", result.ErrorCode)
			}
		} else {
			origState, err = s.getState(result, v)
			if err != nil {
				return err
			}

			if origState.rev == 0 {
				glog.Fatalf("origState.rev unexpectedly 0")
			}
		}
	}

	for {
		if err := checkPreconditions(p, precondtions, origState.uid); err != nil {
			return err
		}

		newObj, ttl, err := s.updateState(origState, tryUpdate)
		if err != nil {
			return err
		}

		data, err := runtime.Encode(s.codec, newObj)
		if err != nil {
			return err
		}
		if bytes.Equal(data, origState.data) {
			return decode(s.codec, s.versioner, origState.data, out, origState.rev)
		}

		objMeta, err := api.ObjectMetaFor(newObj)
		if err != nil {
			return storage.NewInternalErrorf("can't get meta on un-introspectable object %v, got error: %v", newObj, err)
		}

		newUid := objMeta.UID
		newItem := itemData{newUid, data, 0, 0}
		if ttl != 0 {
			newItem.expiry = uint64(time.Now().Unix()) + ttl
		}

		var result *StorageOperationResult
		if doCreate {
			glog.V(6).Infof("Trying create of %s", p)

			op := &StorageOperation{
				OpType:   StorageOperationType_CREATE,
				ItemData: toProto(newItem),
				Path:     p,
			}
			result, err = s.doOperation(ctx, op)
		} else {
			// response will be the new item if we swapped,
			// or the existing item if err==errorLSNMismatch
			glog.V(6).Infof("Trying update of %s with lsn %d", p, origState.rev)

			if origState.rev == 0 {
				glog.Fatalf("origState.rev unexpectedly 0")
			}

			op := &StorageOperation{
				OpType:          StorageOperationType_UPDATE,
				ItemData:        toProto(newItem),
				Path:            p,
				PreconditionLsn: uint64(origState.rev),
			}
			result, err = s.doOperation(ctx, op)
		}

		if err != nil {
			return err
		}

		if result.ErrorCode != 0 {
			switch result.ErrorCode {
			case ErrorCode_PRECONDITION_NOT_MET_LSN:
				glog.V(4).Infof("GuaranteedUpdate of %s failed because of a conflict, going to retry", p)

				origState, err = s.getState(result, v)
				if err != nil {
					return err
				}

				doCreate = false
				continue

			case ErrorCode_NOT_FOUND:
				if ignoreNotFound {
					origState = &objState{
						obj:  reflect.New(v.Type()).Interface().(runtime.Object),
						meta: &storage.ResponseMeta{},
					}
					if err := runtime.SetZeroValue(origState.obj); err != nil {
						return err
					}
					doCreate = true
					continue
				}

				return storage.NewKeyNotFoundError(p, 0)

			case ErrorCode_ALREADY_EXISTS:
				// this is the create path
				glog.V(4).Infof("GuaranteedUpdate of %s failed because of a conflict (on create), going to retry", p)

				origState, err = s.getState(result, v)
				if err != nil {
					return err
				}

				doCreate = false
				continue

			default:
				return fmt.Errorf("unexpected error code: %v", result.ErrorCode)
			}
		}

		return decode(s.codec, s.versioner, data, out, LSN(result.ItemData.Lsn))
	}
}

type objState struct {
	obj  runtime.Object
	meta *storage.ResponseMeta
	rev  LSN
	uid  types.UID
	data []byte
}

func (s *store) getState(result *StorageOperationResult, v reflect.Value) (*objState, error) {
	state := &objState{
		obj:  reflect.New(v.Type()).Interface().(runtime.Object),
		meta: &storage.ResponseMeta{},
	}
	state.uid = types.UID(result.ItemData.Uid)
	state.rev = LSN(result.ItemData.Lsn)
	state.meta.ResourceVersion = result.ItemData.Lsn
	state.data = result.ItemData.Data
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

	objMeta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return nil, storage.NewInternalErrorf("can't get meta on un-introspectable object %v, got error: %v", obj, err)
	}
	state.uid = objMeta.UID

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
	err = versioner.UpdateObject(objPtr, uint64(rev))
	if err != nil {
		glog.V(2).Infof("ignoring error setting version on object %T: %v", objPtr, err)
	}
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

func (s *store) doOperation(ctx context.Context, op *StorageOperation) (*StorageOperationResult, error) {
	return s.backend.DoOperation(ctx, op)
}
