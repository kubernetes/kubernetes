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

package etcd3

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/golang/glog"
	"golang.org/x/net/context"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd"
	"k8s.io/apiserver/pkg/storage/value"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
)

// authenticatedDataString satisfies the value.Context interface. It uses the key to
// authenticate the stored data. This does not defend against reuse of previously
// encrypted values under the same key, but will prevent an attacker from using an
// encrypted value from a different key. A stronger authenticated data segment would
// include the etcd3 Version field (which is incremented on each write to a key and
// reset when the key is deleted), but an attacker with write access to etcd can
// force deletion and recreation of keys to weaken that angle.
type authenticatedDataString string

// AuthenticatedData implements the value.Context interface.
func (d authenticatedDataString) AuthenticatedData() []byte {
	return []byte(string(d))
}

var _ value.Context = authenticatedDataString("")

type store struct {
	client *clientv3.Client
	// getOpts contains additional options that should be passed
	// to all Get() calls.
	getOps        []clientv3.OpOption
	codec         runtime.Codec
	versioner     storage.Versioner
	transformer   value.Transformer
	pathPrefix    string
	watcher       *watcher
	pagingEnabled bool
}

type elemForDecode struct {
	data []byte
	rev  uint64
}

type objState struct {
	obj   runtime.Object
	meta  *storage.ResponseMeta
	rev   int64
	data  []byte
	stale bool
}

// New returns an etcd3 implementation of storage.Interface.
func New(c *clientv3.Client, codec runtime.Codec, prefix string, transformer value.Transformer, pagingEnabled bool) storage.Interface {
	return newStore(c, true, pagingEnabled, codec, prefix, transformer)
}

// NewWithNoQuorumRead returns etcd3 implementation of storage.Interface
// where Get operations don't require quorum read.
func NewWithNoQuorumRead(c *clientv3.Client, codec runtime.Codec, prefix string, transformer value.Transformer, pagingEnabled bool) storage.Interface {
	return newStore(c, false, pagingEnabled, codec, prefix, transformer)
}

func newStore(c *clientv3.Client, quorumRead, pagingEnabled bool, codec runtime.Codec, prefix string, transformer value.Transformer) *store {
	versioner := etcd.APIObjectVersioner{}
	result := &store{
		client:        c,
		codec:         codec,
		versioner:     versioner,
		transformer:   transformer,
		pagingEnabled: pagingEnabled,
		// for compatibility with etcd2 impl.
		// no-op for default prefix of '/registry'.
		// keeps compatibility with etcd2 impl for custom prefixes that don't start with '/'
		pathPrefix: path.Join("/", prefix),
		watcher:    newWatcher(c, codec, versioner, transformer),
	}
	if !quorumRead {
		// In case of non-quorum reads, we can set WithSerializable()
		// options for all Get operations.
		result.getOps = append(result.getOps, clientv3.WithSerializable())
	}
	return result
}

// Versioner implements storage.Interface.Versioner.
func (s *store) Versioner() storage.Versioner {
	return s.versioner
}

// Get implements storage.Interface.Get.
func (s *store) Get(ctx context.Context, key string, resourceVersion string, out runtime.Object, ignoreNotFound bool) error {
	key = path.Join(s.pathPrefix, key)
	getResp, err := s.client.KV.Get(ctx, key, s.getOps...)
	if err != nil {
		return err
	}

	if len(getResp.Kvs) == 0 {
		if ignoreNotFound {
			return runtime.SetZeroValue(out)
		}
		return storage.NewKeyNotFoundError(key, 0)
	}
	kv := getResp.Kvs[0]

	data, _, err := s.transformer.TransformFromStorage(kv.Value, authenticatedDataString(key))
	if err != nil {
		return storage.NewInternalError(err.Error())
	}

	return decode(s.codec, s.versioner, data, out, kv.ModRevision)
}

// Create implements storage.Interface.Create.
func (s *store) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	if version, err := s.versioner.ObjectResourceVersion(obj); err == nil && version != 0 {
		return errors.New("resourceVersion should not be set on objects to be created")
	}
	if err := s.versioner.PrepareObjectForStorage(obj); err != nil {
		return fmt.Errorf("PrepareObjectForStorage failed: %v", err)
	}
	data, err := runtime.Encode(s.codec, obj)
	if err != nil {
		return err
	}
	key = path.Join(s.pathPrefix, key)

	opts, err := s.ttlOpts(ctx, int64(ttl))
	if err != nil {
		return err
	}

	newData, err := s.transformer.TransformToStorage(data, authenticatedDataString(key))
	if err != nil {
		return storage.NewInternalError(err.Error())
	}

	txnResp, err := s.client.KV.Txn(ctx).If(
		notFound(key),
	).Then(
		clientv3.OpPut(key, string(newData), opts...),
	).Commit()
	if err != nil {
		return err
	}
	if !txnResp.Succeeded {
		return storage.NewKeyExistsError(key, 0)
	}

	if out != nil {
		putResp := txnResp.Responses[0].GetResponsePut()
		return decode(s.codec, s.versioner, data, out, putResp.Header.Revision)
	}
	return nil
}

// Delete implements storage.Interface.Delete.
func (s *store) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions) error {
	v, err := conversion.EnforcePtr(out)
	if err != nil {
		panic("unable to convert output object to pointer")
	}
	key = path.Join(s.pathPrefix, key)
	if preconditions == nil {
		return s.unconditionalDelete(ctx, key, out)
	}
	return s.conditionalDelete(ctx, key, out, v, preconditions)
}

func (s *store) unconditionalDelete(ctx context.Context, key string, out runtime.Object) error {
	// We need to do get and delete in single transaction in order to
	// know the value and revision before deleting it.
	txnResp, err := s.client.KV.Txn(ctx).If().Then(
		clientv3.OpGet(key),
		clientv3.OpDelete(key),
	).Commit()
	if err != nil {
		return err
	}
	getResp := txnResp.Responses[0].GetResponseRange()
	if len(getResp.Kvs) == 0 {
		return storage.NewKeyNotFoundError(key, 0)
	}

	kv := getResp.Kvs[0]
	data, _, err := s.transformer.TransformFromStorage(kv.Value, authenticatedDataString(key))
	if err != nil {
		return storage.NewInternalError(err.Error())
	}
	return decode(s.codec, s.versioner, data, out, kv.ModRevision)
}

func (s *store) conditionalDelete(ctx context.Context, key string, out runtime.Object, v reflect.Value, preconditions *storage.Preconditions) error {
	getResp, err := s.client.KV.Get(ctx, key)
	if err != nil {
		return err
	}
	for {
		origState, err := s.getState(getResp, key, v, false)
		if err != nil {
			return err
		}
		if err := checkPreconditions(key, preconditions, origState.obj); err != nil {
			return err
		}
		txnResp, err := s.client.KV.Txn(ctx).If(
			clientv3.Compare(clientv3.ModRevision(key), "=", origState.rev),
		).Then(
			clientv3.OpDelete(key),
		).Else(
			clientv3.OpGet(key),
		).Commit()
		if err != nil {
			return err
		}
		if !txnResp.Succeeded {
			getResp = (*clientv3.GetResponse)(txnResp.Responses[0].GetResponseRange())
			glog.V(4).Infof("deletion of %s failed because of a conflict, going to retry", key)
			continue
		}
		return decode(s.codec, s.versioner, origState.data, out, origState.rev)
	}
}

// GuaranteedUpdate implements storage.Interface.GuaranteedUpdate.
func (s *store) GuaranteedUpdate(
	ctx context.Context, key string, out runtime.Object, ignoreNotFound bool,
	preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, suggestion ...runtime.Object) error {
	trace := utiltrace.New(fmt.Sprintf("GuaranteedUpdate etcd3: %s", reflect.TypeOf(out).String()))
	defer trace.LogIfLong(500 * time.Millisecond)

	v, err := conversion.EnforcePtr(out)
	if err != nil {
		panic("unable to convert output object to pointer")
	}
	key = path.Join(s.pathPrefix, key)

	getCurrentState := func() (*objState, error) {
		getResp, err := s.client.KV.Get(ctx, key, s.getOps...)
		if err != nil {
			return nil, err
		}
		return s.getState(getResp, key, v, ignoreNotFound)
	}

	var origState *objState
	var mustCheckData bool
	if len(suggestion) == 1 && suggestion[0] != nil {
		origState, err = s.getStateFromObject(suggestion[0])
		if err != nil {
			return err
		}
		mustCheckData = true
	} else {
		origState, err = getCurrentState()
		if err != nil {
			return err
		}
	}
	trace.Step("initial value restored")

	transformContext := authenticatedDataString(key)
	for {
		if err := checkPreconditions(key, preconditions, origState.obj); err != nil {
			return err
		}

		ret, ttl, err := s.updateState(origState, tryUpdate)
		if err != nil {
			// It's possible we were working with stale data
			if mustCheckData && apierrors.IsConflict(err) {
				// Actually fetch
				origState, err = getCurrentState()
				if err != nil {
					return err
				}
				mustCheckData = false
				// Retry
				continue
			}

			return err
		}

		data, err := runtime.Encode(s.codec, ret)
		if err != nil {
			return err
		}
		if !origState.stale && bytes.Equal(data, origState.data) {
			// if we skipped the original Get in this loop, we must refresh from
			// etcd in order to be sure the data in the store is equivalent to
			// our desired serialization
			if mustCheckData {
				origState, err = getCurrentState()
				if err != nil {
					return err
				}
				mustCheckData = false
				continue
			}
			return decode(s.codec, s.versioner, origState.data, out, origState.rev)
		}

		newData, err := s.transformer.TransformToStorage(data, transformContext)
		if err != nil {
			return storage.NewInternalError(err.Error())
		}

		opts, err := s.ttlOpts(ctx, int64(ttl))
		if err != nil {
			return err
		}
		trace.Step("Transaction prepared")

		txnResp, err := s.client.KV.Txn(ctx).If(
			clientv3.Compare(clientv3.ModRevision(key), "=", origState.rev),
		).Then(
			clientv3.OpPut(key, string(newData), opts...),
		).Else(
			clientv3.OpGet(key),
		).Commit()
		if err != nil {
			return err
		}
		trace.Step("Transaction committed")
		if !txnResp.Succeeded {
			getResp := (*clientv3.GetResponse)(txnResp.Responses[0].GetResponseRange())
			glog.V(4).Infof("GuaranteedUpdate of %s failed because of a conflict, going to retry", key)
			origState, err = s.getState(getResp, key, v, ignoreNotFound)
			if err != nil {
				return err
			}
			trace.Step("Retry value restored")
			mustCheckData = false
			continue
		}
		putResp := txnResp.Responses[0].GetResponsePut()

		return decode(s.codec, s.versioner, data, out, putResp.Header.Revision)
	}
}

// GetToList implements storage.Interface.GetToList.
func (s *store) GetToList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	key = path.Join(s.pathPrefix, key)

	getResp, err := s.client.KV.Get(ctx, key, s.getOps...)
	if err != nil {
		return err
	}
	if len(getResp.Kvs) == 0 {
		return nil
	}
	data, _, err := s.transformer.TransformFromStorage(getResp.Kvs[0].Value, authenticatedDataString(key))
	if err != nil {
		return storage.NewInternalError(err.Error())
	}
	elems := []*elemForDecode{{
		data: data,
		rev:  uint64(getResp.Kvs[0].ModRevision),
	}}
	if err := decodeList(elems, storage.SimpleFilter(pred), listPtr, s.codec, s.versioner); err != nil {
		return err
	}
	// update version with cluster level revision
	return s.versioner.UpdateList(listObj, uint64(getResp.Header.Revision), "")
}

// continueToken is a simple structured object for encoding the state of a continue token.
// TODO: if we change the version of the encoded from, we can't start encoding the new version
// until all other servers are upgraded (i.e. we need to support rolling schema)
// This is a public API struct and cannot change.
type continueToken struct {
	APIVersion      string `json:"v"`
	ResourceVersion int64  `json:"rv"`
	StartKey        string `json:"start"`
}

// parseFrom transforms an encoded predicate from into a versioned struct.
// TODO: return a typed error that instructs clients that they must relist
func decodeContinue(continueValue, keyPrefix string) (fromKey string, rv int64, err error) {
	data, err := base64.RawURLEncoding.DecodeString(continueValue)
	if err != nil {
		return "", 0, fmt.Errorf("continue key is not valid: %v", err)
	}
	var c continueToken
	if err := json.Unmarshal(data, &c); err != nil {
		return "", 0, fmt.Errorf("continue key is not valid: %v", err)
	}
	switch c.APIVersion {
	case "v1alpha1":
		if c.ResourceVersion == 0 {
			return "", 0, fmt.Errorf("continue key is not valid: incorrect encoded start resourceVersion (version v1alpha1)")
		}
		if len(c.StartKey) == 0 {
			return "", 0, fmt.Errorf("continue key is not valid: encoded start key empty (version v1alpha1)")
		}
		// defend against path traversal attacks by clients - path.Clean will ensure that startKey cannot
		// be at a higher level of the hierarchy, and so when we append the key prefix we will end up with
		// continue start key that is fully qualified and cannot range over anything less specific than
		// keyPrefix.
		cleaned := path.Clean(c.StartKey)
		if cleaned != c.StartKey || cleaned == "." || cleaned == "/" {
			return "", 0, fmt.Errorf("continue key is not valid: %s", cleaned)
		}
		if len(cleaned) == 0 {
			return "", 0, fmt.Errorf("continue key is not valid: encoded start key empty (version 0)")
		}
		return keyPrefix + cleaned, c.ResourceVersion, nil
	default:
		return "", 0, fmt.Errorf("continue key is not valid: server does not recognize this encoded version %q", c.APIVersion)
	}
}

// encodeContinue returns a string representing the encoded continuation of the current query.
func encodeContinue(key, keyPrefix string, resourceVersion int64) (string, error) {
	nextKey := strings.TrimPrefix(key, keyPrefix)
	if nextKey == key {
		return "", fmt.Errorf("unable to encode next field: the key and key prefix do not match")
	}
	out, err := json.Marshal(&continueToken{APIVersion: "v1alpha1", ResourceVersion: resourceVersion, StartKey: nextKey})
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(out), nil
}

// List implements storage.Interface.List.
func (s *store) List(ctx context.Context, key, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	key = path.Join(s.pathPrefix, key)
	// We need to make sure the key ended with "/" so that we only get children "directories".
	// e.g. if we have key "/a", "/a/b", "/ab", getting keys with prefix "/a" will return all three,
	// while with prefix "/a/" will return only "/a/b" which is the correct answer.
	if !strings.HasSuffix(key, "/") {
		key += "/"
	}
	keyPrefix := key

	// set the appropriate clientv3 options to filter the returned data set
	options := make([]clientv3.OpOption, 0, 4)
	if s.pagingEnabled && pred.Limit > 0 {
		options = append(options, clientv3.WithLimit(pred.Limit))
	}
	var returnedRV int64
	switch {
	case s.pagingEnabled && len(pred.Continue) > 0:
		continueKey, continueRV, err := decodeContinue(pred.Continue, keyPrefix)
		if err != nil {
			return err
		}

		options = append(options, clientv3.WithRange(clientv3.GetPrefixRangeEnd(key)))
		key = continueKey

		options = append(options, clientv3.WithRev(continueRV))
		returnedRV = continueRV

	case len(resourceVersion) > 0:
		fromRV, err := strconv.ParseInt(resourceVersion, 10, 64)
		if err != nil {
			return fmt.Errorf("invalid resource version: %v", err)
		}

		options = append(options, clientv3.WithPrefix(), clientv3.WithRev(fromRV))
		returnedRV = fromRV

	default:
		options = append(options, clientv3.WithPrefix())
	}

	getResp, err := s.client.KV.Get(ctx, key, options...)
	if err != nil {
		return interpretListError(err, len(pred.Continue) > 0)
	}

	elems := make([]*elemForDecode, 0, len(getResp.Kvs))
	for _, kv := range getResp.Kvs {
		data, _, err := s.transformer.TransformFromStorage(kv.Value, authenticatedDataString(kv.Key))
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("unable to transform key %q: %v", kv.Key, err))
			continue
		}

		elems = append(elems, &elemForDecode{
			data: data,
			rev:  uint64(kv.ModRevision),
		})
	}

	if err := decodeList(elems, storage.SimpleFilter(pred), listPtr, s.codec, s.versioner); err != nil {
		return err
	}

	// indicate to the client which resource version was returned
	if returnedRV == 0 {
		returnedRV = getResp.Header.Revision
	}
	switch {
	case !getResp.More:
		// no continuation
		return s.versioner.UpdateList(listObj, uint64(returnedRV), "")
	case len(getResp.Kvs) == 0:
		return fmt.Errorf("no results were found, but etcd indicated there were more values")
	default:
		// we want to start immediately after the last key
		// TODO: this reveals info about certain keys
		key := string(getResp.Kvs[len(getResp.Kvs)-1].Key)
		next, err := encodeContinue(key+"\x00", keyPrefix, returnedRV)
		if err != nil {
			return err
		}
		return s.versioner.UpdateList(listObj, uint64(returnedRV), next)
	}
}

// Watch implements storage.Interface.Watch.
func (s *store) Watch(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	return s.watch(ctx, key, resourceVersion, pred, false)
}

// WatchList implements storage.Interface.WatchList.
func (s *store) WatchList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	return s.watch(ctx, key, resourceVersion, pred, true)
}

func (s *store) watch(ctx context.Context, key string, rv string, pred storage.SelectionPredicate, recursive bool) (watch.Interface, error) {
	rev, err := storage.ParseWatchResourceVersion(rv)
	if err != nil {
		return nil, err
	}
	key = path.Join(s.pathPrefix, key)
	return s.watcher.Watch(ctx, key, int64(rev), recursive, pred)
}

func (s *store) getState(getResp *clientv3.GetResponse, key string, v reflect.Value, ignoreNotFound bool) (*objState, error) {
	state := &objState{
		obj:  reflect.New(v.Type()).Interface().(runtime.Object),
		meta: &storage.ResponseMeta{},
	}
	if len(getResp.Kvs) == 0 {
		if !ignoreNotFound {
			return nil, storage.NewKeyNotFoundError(key, 0)
		}
		if err := runtime.SetZeroValue(state.obj); err != nil {
			return nil, err
		}
	} else {
		data, stale, err := s.transformer.TransformFromStorage(getResp.Kvs[0].Value, authenticatedDataString(key))
		if err != nil {
			return nil, storage.NewInternalError(err.Error())
		}
		state.rev = getResp.Kvs[0].ModRevision
		state.meta.ResourceVersion = uint64(state.rev)
		state.data = data
		state.stale = stale
		if err := decode(s.codec, s.versioner, state.data, state.obj, state.rev); err != nil {
			return nil, err
		}
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
	state.rev = int64(rv)
	state.meta.ResourceVersion = uint64(state.rev)

	// Compute the serialized form - for that we need to temporarily clean
	// its resource version field (those are not stored in etcd).
	if err := s.versioner.PrepareObjectForStorage(obj); err != nil {
		return nil, fmt.Errorf("PrepareObjectForStorage failed: %v", err)
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

	if err := s.versioner.PrepareObjectForStorage(ret); err != nil {
		return nil, 0, fmt.Errorf("PrepareObjectForStorage failed: %v", err)
	}
	var ttl uint64
	if ttlPtr != nil {
		ttl = *ttlPtr
	}
	return ret, ttl, nil
}

// ttlOpts returns client options based on given ttl.
// ttl: if ttl is non-zero, it will attach the key to a lease with ttl of roughly the same length
func (s *store) ttlOpts(ctx context.Context, ttl int64) ([]clientv3.OpOption, error) {
	if ttl == 0 {
		return nil, nil
	}
	// TODO: one lease per ttl key is expensive. Based on current use case, we can have a long window to
	// put keys within into same lease. We shall benchmark this and optimize the performance.
	lcr, err := s.client.Lease.Grant(ctx, ttl)
	if err != nil {
		return nil, err
	}
	return []clientv3.OpOption{clientv3.WithLease(clientv3.LeaseID(lcr.ID))}, nil
}

// decode decodes value of bytes into object. It will also set the object resource version to rev.
// On success, objPtr would be set to the object.
func decode(codec runtime.Codec, versioner storage.Versioner, value []byte, objPtr runtime.Object, rev int64) error {
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

// decodeList decodes a list of values into a list of objects, with resource version set to corresponding rev.
// On success, ListPtr would be set to the list of objects.
func decodeList(elems []*elemForDecode, filter storage.FilterFunc, listPtr interface{}, codec runtime.Codec, versioner storage.Versioner) error {
	v, err := conversion.EnforcePtr(listPtr)
	if err != nil || v.Kind() != reflect.Slice {
		panic("need ptr to slice")
	}
	for _, elem := range elems {
		obj, _, err := codec.Decode(elem.data, nil, reflect.New(v.Type().Elem()).Interface().(runtime.Object))
		if err != nil {
			return err
		}
		// being unable to set the version does not prevent the object from being extracted
		versioner.UpdateObject(obj, elem.rev)
		if filter(obj) {
			v.Set(reflect.Append(v, reflect.ValueOf(obj).Elem()))
		}
	}
	return nil
}

func checkPreconditions(key string, preconditions *storage.Preconditions, out runtime.Object) error {
	if preconditions == nil {
		return nil
	}
	objMeta, err := meta.Accessor(out)
	if err != nil {
		return storage.NewInternalErrorf("can't enforce preconditions %v on un-introspectable object %v, got error: %v", *preconditions, out, err)
	}
	if preconditions.UID != nil && *preconditions.UID != objMeta.GetUID() {
		errMsg := fmt.Sprintf("Precondition failed: UID in precondition: %v, UID in object meta: %v", *preconditions.UID, objMeta.GetUID())
		return storage.NewInvalidObjError(key, errMsg)
	}
	return nil
}

func notFound(key string) clientv3.Cmp {
	return clientv3.Compare(clientv3.ModRevision(key), "=", 0)
}
