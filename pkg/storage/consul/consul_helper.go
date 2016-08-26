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

package consul

import (
	"bytes"
	"encoding/base64"
	"encoding/gob"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"path"
	"reflect"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/consul/metrics"
	"k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/util"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
	consulapi "github.com/hashicorp/consul/api"
	"golang.org/x/net/context"
)

//implements storage.Interface
type consulHelper struct {
	Client             consulapi.Client
	Config             consulapi.Config
	codec              runtime.Codec
	consulKv           *consulapi.KV
	versioner          storage.Versioner
	quorum             bool
	pathPrefix         string
	eventHistoryPrefix string
}

type ConsulEvent struct {
	Object []byte
	Type   string
}

func init() {
	metrics.Register()
	gob.Register(watch.Event{})
	gob.Register(api.Pod{})
}

func NewConsulStorage(client consulapi.Client, codec runtime.Codec, prefix string, quorum bool, config consulapi.Config) storage.Interface {
	//ensure we have some prefix otherwise watcher on / will fail because this results in an empty key to watch
	if prefix == "" {
		prefix = fmt.Sprintf("pref%d", rand.Intn(800000))
	}
	return &consulHelper{
		Client:             client,
		Config:             config,
		codec:              codec,
		consulKv:           client.KV(),
		versioner:          etcd.APIObjectVersioner{},
		pathPrefix:         path.Join("/", prefix),
		quorum:             quorum,
		eventHistoryPrefix: "past_events",
	}
}

func (h *consulHelper) extractObj(kvPair *consulapi.KVPair, inErr error, objPtr runtime.Object, ignoreNotFound bool, key string) error {
	if inErr != nil || kvPair == nil || len(kvPair.Value) == 0 {
		if ignoreNotFound {
			v, err := conversion.EnforcePtr(objPtr)
			if err != nil {
				return err
			}
			v.Set(reflect.Zero(v.Type()))
			return nil
		} else if inErr != nil {
			return inErr
		}

		return toStorageErr(NewNotFoundError(), key, 0)
	}

	out, gvk, err := h.codec.Decode(kvPair.Value, nil, objPtr)
	if err != nil {
		return toStorageErr(err, key, 0)
	}
	if out != objPtr {
		return toStorageErr(NewInvalidObjectError(fmt.Sprintf("unable to decode object %s into %v", gvk.String(), reflect.TypeOf(objPtr))), key, 0)
	}

	// being unable to set the version does not prevent the object from being extracted
	_ = h.versioner.UpdateObject(objPtr, kvPair.ModifyIndex)
	return err
}

func (h *consulHelper) prefixConsulKey(key string) string {
	if strings.HasPrefix(key, h.pathPrefix) {
		return key
	}
	return path.Join(h.pathPrefix, key)
}

func (h *consulHelper) Backends(ctx context.Context) []string {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}

	peers, err := h.Client.Status().Peers()
	if err != nil {
		return []string{}
	}

	return peers
}

func (h *consulHelper) Versioner() storage.Versioner {
	return h.versioner
}

func (h *consulHelper) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	trace := util.NewTrace("consulHelper::Create " + getTypeName(obj))
	defer trace.LogIfLong(250 * time.Millisecond)
	if ctx == nil {
		glog.Errorf("Context is nil")
	}

	originalKey := key

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	data, err := runtime.Encode(h.codec, obj)
	trace.Step("Object encoded")
	if err != nil {
		return toStorageErr(NewInvalidObjectError("Encoding object failed: "+err.Error()), key, 0)
	}

	if version, err := h.versioner.ObjectResourceVersion(obj); err == nil && version != 0 {
		return toStorageErr(NewInvalidObjectError("resourceVersion may not be set on objects to be created"), key, 0)
	}
	trace.Step("Version checked")

	queryOpt := &consulapi.QueryOptions{}
	if h.quorum == true {
		queryOpt.RequireConsistent = true
	}

	startTime := time.Now()
	pair := &consulapi.KVPair{
		Key:         key,
		Value:       data,
		ModifyIndex: 0, // setting to 0 only creats the key if NOT already existent
	}

	ok, _, err := h.consulKv.CAS(pair, nil)
	metrics.RecordConsulRequestLatency("cas", getTypeName(obj), startTime)
	if !ok {
		return toStorageErr(NewExistsError(fmt.Sprintf("Key: %s already exists", key)), key, 0)
	}
	if err != nil {
		return toStorageErr(err, key, 0)
	}
	trace.Step("Object created")

	storedPair, _, err := h.consulKv.Get(key, queryOpt)
	if err != nil {
		return toStorageErr(err, key, 0)
	}
	if storedPair == nil {
		return toStorageErr(fmt.Errorf("Failed to retrive previsouly saved KVPAir"), key, 0)
	}

	h.archiveEvent(originalKey, reflect.ValueOf(obj).Type().Elem(), watch.Added)

	if out != nil {
		if _, err := conversion.EnforcePtr(out); err != nil {
			panic("unable to convert output object to pointer")
		}

		//check against len == 0, if yes what to do?
		err = h.extractObj(storedPair, err, out, false, key)
	}
	return toStorageErr(err, storedPair.Key, 0)
}

func (h *consulHelper) Get(ctx context.Context, key string, out runtime.Object, ignoreNotFound bool) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	startTime := time.Now()

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	queryOpt := &consulapi.QueryOptions{}
	if h.quorum == true {
		queryOpt.RequireConsistent = true
	}

	kv, _, err := h.consulKv.Get(key, queryOpt)
	metrics.RecordConsulRequestLatency("get", getTypeName(out), startTime)
	if err != nil {
		return toStorageErr(err, key, 0)
	}
	err = h.extractObj(kv, err, out, ignoreNotFound, key)
	return toStorageErr(err, key, 0)
}

func (h *consulHelper) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	v, err := conversion.EnforcePtr(out)
	if err != nil {
		panic("unable to convert output object to pointer: " + err.Error())
	}

	queryOpt := &consulapi.QueryOptions{}
	if h.quorum == true {
		queryOpt.RequireConsistent = true
	}

	if preconditions == nil {
		kv, _, err := h.consulKv.Get(key, queryOpt)
		if err != nil {
			return toStorageErr(err, key, 0)
		}
		if kv == nil {
			return toStorageErr(NewNotFoundError(), key, 0)
		}

		err = h.extractObj(kv, err, out, false, key)
		if err != nil {
			return toStorageErr(NewInvalidObjectError(err.Error()), key, 0)
		}

		startTime := time.Now()
		_, err = h.consulKv.Delete(key, nil)
		metrics.RecordConsulRequestLatency("delete", getTypeName(out), startTime)

		return toStorageErr(err, key, 0)
	}

	// kv declared outside of the spin-loop so that we can decode subsequent successful Gets
	// in the event that another client deletes our key before we do.. this value is possibly
	// lacking certified freshness
	var kvPrev *consulapi.KVPair
	var succeeded bool
	// TODO: perhaps a timeout or spincount would be wise here
	for {
		// empty QueryOptions is explicitly setting AllowStale to false
		startTime := time.Now()
		kv, _, err := h.consulKv.Get(key, queryOpt)
		metrics.RecordConsulRequestLatency("get", getTypeName(out), startTime)
		if err != nil {
			return toStorageErr(err, key, 0)
		}
		if kv == nil {
			break
		}
		kvPrev = kv

		obj := reflect.New(v.Type()).Interface().(runtime.Object)
		h.extractObj(kv, err, obj, false, key)
		if err = checkPreconditions(preconditions, obj); err != nil {
			return toStorageErr(err, kv.Key, 0)
		}

		startTime = time.Now()
		succeeded, _, err = h.consulKv.DeleteCAS(kv, nil)
		metrics.RecordConsulRequestLatency("delete", getTypeName(out), startTime)
		if err != nil {
			return toStorageErr(err, key, 0)
		}
		if succeeded {
			h.archiveEvent(key, reflect.ValueOf(obj).Type().Elem(), watch.Deleted)
			break
		}
		glog.Infof("delection of %s failed because of a conflict, going to retry", key)
	}

	if kvPrev != nil && out != nil {
		if _, err := conversion.EnforcePtr(out); err != nil {
			panic("unable to convert output object to pointer")
		}

		_ = h.extractObj(kvPrev, nil, out, false, key)
	}
	if kvPrev == nil || succeeded == false {
		return toStorageErr(NewNotFoundError(), key, 0)
	}
	return nil
}

func (h *consulHelper) Watch(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	watchRV, err := storage.ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, toStorageErr(err, key, 0)
	}

	return h.newConsulWatch(key, watchRV, false, h.versioner, h.Config.Address, storage.SimpleFilter(pred))
}

func (h *consulHelper) WatchList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	watchRV, err := storage.ParseWatchResourceVersion(resourceVersion)
	if err != nil {
		return nil, err
	}

	return h.newConsulWatch(key, watchRV, true, h.versioner, h.Config.Address, storage.SimpleFilter(pred))
}

func (h *consulHelper) GetToList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	startTime := time.Now()
	trace := util.NewTrace("GetToList " + getTypeName(listObj))
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return toStorageErr(err, key, 0)
	}

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	queryOpt := &consulapi.QueryOptions{}
	if h.quorum == true {
		queryOpt.RequireConsistent = true
	}

	trace.Step("About ot read from Consul KV")
	kv, _, err := h.consulKv.Get(key, queryOpt)
	trace.Step("Consul KV read")
	metrics.RecordConsulRequestLatency("get", getTypeName(listPtr), startTime)
	if err != nil {
		return toStorageErr(err, key, 0)
	}
	if kv == nil || len(kv.Value) == 0 {
		return nil
	}

	kvs := make([]*consulapi.KVPair, 0)
	kvs = append(kvs, kv)

	if err := h.decodeKVPairList(kvs, storage.SimpleFilter(pred), listPtr); err != nil {
		return toStorageErr(err, key, 0)
	}
	trace.Step("Object decoded")

	if err := h.versioner.UpdateList(listObj, kv.ModifyIndex); err != nil {
		return toStorageErr(err, key, 0)
	}

	return nil
}

func (h *consulHelper) GuaranteedUpdate(ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, _ ...runtime.Object) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	v, err := conversion.EnforcePtr(ptrToType)
	if err != nil {
		// Panic is appropriate, because this is a programming error.
		panic("need ptr to type")
	}

	originalKey := key

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	queryOpt := &consulapi.QueryOptions{}
	if h.quorum == true {
		queryOpt.RequireConsistent = true
	}

	for {
		startTime := time.Now()
		kv, _, err := h.consulKv.Get(key, queryOpt)
		metrics.RecordConsulRequestLatency("get", getTypeName(v), startTime)
		if err != nil {
			return toStorageErr(err, key, 0)
		}

		obj := reflect.New(v.Type()).Interface().(runtime.Object)
		err = h.extractObj(kv, err, obj, ignoreNotFound, key)
		if err != nil {
			if kv == nil {
				return toStorageErr(err, key, 0)
			}
			return toStorageErr(err, key, int64(kv.ModifyIndex))
		}

		if err := checkPreconditions(preconditions, obj); err != nil {
			return toStorageErr(NewConflictError(err.Error()), key, int64(kv.ModifyIndex))
		}

		responseMeta := storage.ResponseMeta{}
		if kv != nil {
			responseMeta.TTL = 0
			responseMeta.ResourceVersion = kv.ModifyIndex
		}
		// Get the object to be written by calling tryUpdate.
		// ignore newTTL because Consul does not support ttl
		ret, _, err := tryUpdate(obj, responseMeta)
		if err != nil {
			return toStorageErr(err, key, 0)
		}

		index := uint64(0)
		if kv != nil {
			index = kv.ModifyIndex
		}

		// Since update object may have a resourceVersion set, we need to clear it here.
		if err := h.versioner.UpdateObject(ret, 0); err != nil {
			return toStorageErr(NewInvalidObjectError("resourceVersion cannot be set on objects store in consul"), key, 0)
		}

		data, err := runtime.Encode(h.codec, ret)
		if err != nil {
			return err
		}

		// First time this key has been used, try creating new value.
		if index == 0 {
			startTime := time.Now()
			kvPair := &consulapi.KVPair{
				Key:   key,
				Value: data,
			}
			_, err = h.consulKv.Put(kvPair, nil)
			metrics.RecordConsulRequestLatency("put", getTypeName(v), startTime)
			if err != nil {
				glog.Infof("Error putting new key in GuaranteeUpdaed: %v", err)
				continue
			}

			//retrieve value from consul again because the Put call does not return the added KVPair
			startTime = time.Now()
			addedKV, _, err := h.consulKv.Get(key, queryOpt)
			metrics.RecordConsulRequestLatency("get", getTypeName(v), startTime)
			if err != nil {
				glog.Infof("Error getting new key in GuaranteeUpdaed: %v", err)
				continue
			}

			err = h.extractObj(addedKV, err, ptrToType, ignoreNotFound, key)
			return toStorageErr(err, key, 0)
		}

		if bytes.Equal(data, kv.Value) {
			err = h.extractObj(kv, nil, ptrToType, ignoreNotFound, key)
			return toStorageErr(err, key, 0)
		}

		kvPair := &consulapi.KVPair{
			ModifyIndex: index,
			Key:         key,
			Value:       data,
		}

		ok, _, err := h.consulKv.CAS(kvPair, nil)
		if !ok {
			continue
		}

		startTime = time.Now()
		kvPair, _, err = h.consulKv.Get(key, queryOpt)
		metrics.RecordConsulRequestLatency("get", getTypeName(v), startTime)
		if err != nil {
			return toStorageErr(err, key, 0)
		}
		h.archiveEvent(originalKey, reflect.ValueOf(ptrToType).Type().Elem(), watch.Modified)

		err = h.extractObj(kvPair, err, ptrToType, false, key)
		return toStorageErr(err, key, 0)
	}
}

func (h *consulHelper) Codec() runtime.Codec {
	return h.codec
}

func (h *consulHelper) List(ctx context.Context, key, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	_, err := h.listInternal("List ", key, storage.SimpleFilter(pred), listObj)

	return toStorageErr(err, key, 0)
}

func (h *consulHelper) listInternal(fnName string, key string, filter storage.FilterFunc, listObj runtime.Object) (uint64, error) {
	trace := util.NewTrace(fnName + key)
	defer trace.LogIfLong(time.Second)

	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return 0, err
	}

	queryOpt := &consulapi.QueryOptions{}
	if h.quorum == true {
		queryOpt.RequireConsistent = true
	}

	kvPairs, queryMeta, err := h.consulKv.List(key, queryOpt)

	// TODO: record metrics
	if err != nil {
		return 0, toStorageErr(err, key, 0)
	}

	err = h.decodeKVPairList(kvPairs, filter, listPtr)
	if err != nil {
		return 0, err
	}

	if err := h.versioner.UpdateList(listObj, queryMeta.LastIndex); err != nil {
		return 0, err
	}

	return queryMeta.LastIndex, nil
}

func (h *consulHelper) decodeKVPairList(kvPairs []*consulapi.KVPair, filter storage.FilterFunc, slicePtr interface{}) error {
	trace := util.NewTrace("decodeNodeList " + getTypeName(slicePtr))
	defer trace.LogIfLong(400 * time.Millisecond)
	v, err := conversion.EnforcePtr(slicePtr)
	if err != nil || v.Kind() != reflect.Slice {
		// This should not happen at runtime.
		panic("need ptr to slice")
	}

	trace.Step("Decoding KVPairs")
	for _, kv := range kvPairs {
		trace.Step("Decoding key " + kv.Key + " START")
		obj, _, err := h.codec.Decode(kv.Value, nil, reflect.New(v.Type().Elem()).Interface().(runtime.Object))
		trace.Step("Decoding key " + kv.Key + " END")
		if err != nil {
			return err
		}

		// being unable to set the version does not prevent the object from being extracted
		_ = h.versioner.UpdateObject(obj, kv.ModifyIndex)
		if filter(obj) {
			v.Set(reflect.Append(v, reflect.ValueOf(obj).Elem()))
		}
	}
	trace.Step("Decoded KVPairs")

	return nil
}

func (h *consulHelper) transformKeyName(keyIn string) string {
	return strings.Trim(keyIn, "/")
}

func (h *consulHelper) archiveEvent(key string, objType reflect.Type, eventType watch.EventType) {
	objOut := reflect.New(objType).Interface().(runtime.Object)
	err := h.Get(context.Background(), key, objOut, false)
	if err != nil {
		//TODO: no panic
		log.Panicf("Failed to get %s", err)
	}

	key = h.prefixConsulKey(key)
	key = h.transformKeyName(key)

	data, err := runtime.Encode(h.codec, objOut)
	if err != nil {
		panic("Failed ot encode object")
	}

	rv := objOut.(meta.Object).GetResourceVersion()

	log.Printf("Archived %s with %s at %s", key, string(eventType), rv)
	//wrap that into a dummy object and encode globally
	dummy := ConsulEvent{
		Object: data,
		Type:   string(eventType),
	}
	dummyEnc, err := ToGOB64(dummy)
	if err != nil {
		utilruntime.HandleError(err)
		//continue or stop then?
	}

	eventKey := fmt.Sprintf("%s/%s/%s", h.eventHistoryPrefix, key, rv)

	//create KVPair to store it
	pair := &consulapi.KVPair{
		Key:   eventKey,
		Value: []byte(dummyEnc),
	}

	//store the event with it's version for possible later needs
	_, err = h.consulKv.Put(pair, nil)
	if err != nil {
		//TODO: don't panic
		panic(fmt.Sprintf("Failed to store event: %v", err))
	}
}

func toStorageErr(err error, key string, rv int64) error {
	switch {
	case isInvalidObjectError(err):
		return storage.NewInvalidObjError(key, err.Error())
	case isNotFoundError(err):
		return storage.NewKeyNotFoundError(key, rv)
	case isURLError(err):
		storeErr := storage.NewUnreachableError(key, rv)
		storeErr.AdditionalErrorMsg = "Consul agent not responding"
		return storeErr
	case isExists(err):
		return storage.NewKeyExistsError(key, rv)
	case isConflict(err):
		return storage.NewResourceVersionConflictsError(key, rv)
	default:
		return err
	}
}

func toStorageTxnErr(what string, key string, rv int64) error {
	if strings.HasSuffix(what, ", index is stale") {
		return storage.NewResourceVersionConflictsError(key, rv)
	}
	return storage.NewInternalError(what)
}

func getTypeName(obj interface{}) string {
	return reflect.TypeOf(obj).String()
}

func checkPreconditions(preconditions *storage.Preconditions, out runtime.Object) error {
	if preconditions == nil {
		return nil
	}
	objMeta, err := api.ObjectMetaFor(out)
	if err != nil {
		return storage.NewInternalErrorf("can't enforce preconditions %v on un-introspectable object %v, got error: %v", *preconditions, out, err)
	}
	if preconditions.UID != nil && *preconditions.UID != objMeta.UID {
		return errors.New(fmt.Sprintf("the UID in the precondition (%s) does not match the UID in record (%s). The object might have been deleted and then recreated", *preconditions.UID, objMeta.UID))
	}
	return nil
}

// go binary encoder
func ToGOB64(m ConsulEvent) (string, error) {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	err := e.Encode(m)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(b.Bytes()), nil
}

// go binary decoder
func FromGOB64(str string) (ConsulEvent, error) {
	m := ConsulEvent{}
	by, err := base64.StdEncoding.DecodeString(str)
	if err != nil {
		return ConsulEvent{}, err
	}
	b := bytes.Buffer{}
	b.Write(by)
	d := gob.NewDecoder(&b)
	err = d.Decode(&m)
	if err != nil {
		return ConsulEvent{}, err
	}
	return m, nil
}
