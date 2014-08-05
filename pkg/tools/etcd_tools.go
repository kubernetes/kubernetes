/*
Copyright 2014 Google Inc. All rights reserved.

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

package tools

import (
	"errors"
	"fmt"
	"reflect"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

const (
	EtcdErrorCodeNotFound      = 100
	EtcdErrorCodeTestFailed    = 101
	EtcdErrorCodeNodeExist     = 105
	EtcdErrorCodeValueRequired = 200
)

var (
	EtcdErrorNotFound      = &etcd.EtcdError{ErrorCode: EtcdErrorCodeNotFound}
	EtcdErrorTestFailed    = &etcd.EtcdError{ErrorCode: EtcdErrorCodeTestFailed}
	EtcdErrorNodeExist     = &etcd.EtcdError{ErrorCode: EtcdErrorCodeNodeExist}
	EtcdErrorValueRequired = &etcd.EtcdError{ErrorCode: EtcdErrorCodeValueRequired}
)

type Encoding interface {
	Encode(obj interface{}) (data []byte, err error)
	Decode(data []byte) (interface{}, error)
	DecodeInto(data []byte, obj interface{}) error
}

type Versioning interface {
	SetResourceVersion(obj interface{}, version uint64) error
	ResourceVersion(obj interface{}) (uint64, error)
}

// EtcdClient is an injectable interface for testing.
type EtcdClient interface {
	AddChild(key, data string, ttl uint64) (*etcd.Response, error)
	Get(key string, sort, recursive bool) (*etcd.Response, error)
	Set(key, value string, ttl uint64) (*etcd.Response, error)
	Create(key, value string, ttl uint64) (*etcd.Response, error)
	CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error)
	Delete(key string, recursive bool) (*etcd.Response, error)
	// I'd like to use directional channels here (e.g. <-chan) but this interface mimics
	// the etcd client interface which doesn't, and it doesn't seem worth it to wrap the api.
	Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error)
}

// Interface exposing only the etcd operations needed by EtcdHelper.
type EtcdGetSet interface {
	Get(key string, sort, recursive bool) (*etcd.Response, error)
	Set(key, value string, ttl uint64) (*etcd.Response, error)
	Create(key, value string, ttl uint64) (*etcd.Response, error)
	CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error)
	Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error)
}

// EtcdHelper offers common object marshalling/unmarshalling operations on an etcd client.
type EtcdHelper struct {
	Client   EtcdGetSet
	Encoding Encoding
	// optional
	Versioning Versioning
}

// IsEtcdNotFound returns true iff err is an etcd not found error.
func IsEtcdNotFound(err error) bool {
	return isEtcdErrorNum(err, EtcdErrorCodeNotFound)
}

// Returns true iff err is an etcd key node exists error.
func IsEtcdNodeExist(err error) bool {
	return isEtcdErrorNum(err, EtcdErrorCodeNodeExist)
}

// IsEtcdTestFailed returns true iff err is an etcd write conflict.
func IsEtcdTestFailed(err error) bool {
	return isEtcdErrorNum(err, EtcdErrorCodeTestFailed)
}

// IsEtcdWatchStoppedByUser returns true iff err is a client triggered stop.
func IsEtcdWatchStoppedByUser(err error) bool {
	return etcd.ErrWatchStoppedByUser == err
}

// Returns true iff err is an etcd error, whose errorCode matches errorCode
func isEtcdErrorNum(err error, errorCode int) bool {
	etcdError, ok := err.(*etcd.EtcdError)
	return ok && etcdError != nil && etcdError.ErrorCode == errorCode
}

func (h *EtcdHelper) listEtcdNode(key string) ([]*etcd.Node, error) {
	result, err := h.Client.Get(key, false, true)
	if err != nil {
		nodes := make([]*etcd.Node, 0)
		if IsEtcdNotFound(err) {
			return nodes, nil
		} else {
			return nodes, err
		}
	}
	return result.Node.Nodes, nil
}

// Extract a go object per etcd node into a slice.
func (h *EtcdHelper) ExtractList(key string, slicePtr interface{}) error {
	nodes, err := h.listEtcdNode(key)
	if err != nil {
		return err
	}
	pv := reflect.ValueOf(slicePtr)
	if pv.Type().Kind() != reflect.Ptr || pv.Type().Elem().Kind() != reflect.Slice {
		// This should not happen at runtime.
		panic("need ptr to slice")
	}
	v := pv.Elem()
	for _, node := range nodes {
		obj := reflect.New(v.Type().Elem())
		err = h.Encoding.DecodeInto([]byte(node.Value), obj.Interface())
		if err != nil {
			return err
		}
		v.Set(reflect.Append(v, obj.Elem()))
	}
	return nil
}

// Unmarshals json found at key into objPtr. On a not found error, will either return
// a zero object of the requested type, or an error, depending on ignoreNotFound. Treats
// empty responses and nil response nodes exactly like a not found error.
func (h *EtcdHelper) ExtractObj(key string, objPtr interface{}, ignoreNotFound bool) error {
	_, _, err := h.bodyAndExtractObj(key, objPtr, ignoreNotFound)
	return err
}

func (h *EtcdHelper) bodyAndExtractObj(key string, objPtr interface{}, ignoreNotFound bool) (body string, modifiedIndex uint64, err error) {
	response, err := h.Client.Get(key, false, false)

	if err != nil && !IsEtcdNotFound(err) {
		return "", 0, err
	}
	if err != nil || response.Node == nil || len(response.Node.Value) == 0 {
		if ignoreNotFound {
			pv := reflect.ValueOf(objPtr)
			pv.Elem().Set(reflect.Zero(pv.Type().Elem()))
			return "", 0, nil
		} else if err != nil {
			return "", 0, err
		}
		return "", 0, fmt.Errorf("key '%v' found no nodes field: %#v", key, response)
	}
	body = response.Node.Value
	err = h.Encoding.DecodeInto([]byte(body), objPtr)
	if h.Versioning != nil {
		_ = h.Versioning.SetResourceVersion(objPtr, response.Node.ModifiedIndex)
		// being unable to set the version does not prevent the object from being extracted
	}
	return body, response.Node.ModifiedIndex, err
}

func (h *EtcdHelper) CreateObj(key string, obj interface{}) error {
	data, err := h.Encoding.Encode(obj)
	if err != nil {
		return err
	}
	if h.Versioning != nil {
		if version, err := h.Versioning.ResourceVersion(obj); err == nil && version != 0 {
			return errors.New("resourceVersion may not be set on objects to be created")
		}
	}

	_, err = h.Client.Create(key, string(data), 0)
	return err
}

// SetObj marshals obj via json, and stores under key. Will do an
// atomic update if obj's ResourceVersion field is set.
func (h *EtcdHelper) SetObj(key string, obj interface{}) error {
	data, err := h.Encoding.Encode(obj)
	if err != nil {
		return err
	}
	if h.Versioning != nil {
		if version, err := h.Versioning.ResourceVersion(obj); err == nil && version != 0 {
			_, err = h.Client.CompareAndSwap(key, string(data), 0, "", version)
			return err // err is shadowed!
		}
	}

	// Create will fail if a key already exists.
	_, err = h.Client.Create(key, string(data), 0)
	return err
}

// Pass an EtcdUpdateFunc to EtcdHelper.AtomicUpdate to make an atomic etcd update.
// See the comment for AtomicUpdate for more detail.
type EtcdUpdateFunc func(input interface{}) (output interface{}, err error)

// AtomicUpdate generalizes the pattern that allows for making atomic updates to etcd objects.
// Note, tryUpdate may be called more than once.
//
// Example:
//
// h := &util.EtcdHelper{client, encoding, versioning}
// err := h.AtomicUpdate("myKey", &MyType{}, func(input interface{}) (interface{}, error) {
//	// Before this function is called, currentObj has been reset to etcd's current
//	// contents for "myKey".
//
//	cur := input.(*MyType) // Gauranteed to work.
//
//	// Make a *modification*.
//	cur.Counter++
//
//	// Return the modified object. Return an error to stop iterating.
//	return cur, nil
// })
//
func (h *EtcdHelper) AtomicUpdate(key string, ptrToType interface{}, tryUpdate EtcdUpdateFunc) error {
	pt := reflect.TypeOf(ptrToType)
	if pt.Kind() != reflect.Ptr {
		// Panic is appropriate, because this is a programming error.
		panic("need ptr to type")
	}
	for {
		obj := reflect.New(pt.Elem()).Interface()
		origBody, index, err := h.bodyAndExtractObj(key, obj, true)
		if err != nil {
			return err
		}

		ret, err := tryUpdate(obj)
		if err != nil {
			return err
		}

		// First time this key has been used, just set.
		if index == 0 {
			return h.SetObj(key, ret)
		}

		data, err := h.Encoding.Encode(ret)
		if err != nil {
			return err
		}
		_, err = h.Client.CompareAndSwap(key, string(data), 0, origBody, index)
		if IsEtcdTestFailed(err) {
			continue
		}
		return err
	}
}

// FilterFunc is a predicate which takes an API object and returns true
// iff the object should remain in the set.
type FilterFunc func(obj interface{}) bool

// Everything is a FilterFunc which accepts all objects.
func Everything(interface{}) bool {
	return true
}

// WatchList begins watching the specified key's items. Items are decoded into
// API objects, and any items passing 'filter' are sent down the returned
// watch.Interface.
func (h *EtcdHelper) WatchList(key string, filter FilterFunc) (watch.Interface, error) {
	w := newEtcdWatcher(true, filter, h.Encoding)
	go w.etcdWatch(h.Client, key)
	return w, nil
}

// Watch begins watching the specified key. Events are decoded into
// API objects and sent down the returned watch.Interface.
func (h *EtcdHelper) Watch(key string) (watch.Interface, error) {
	w := newEtcdWatcher(false, nil, h.Encoding)
	go w.etcdWatch(h.Client, key)
	return w, nil
}

// etcdWatcher converts a native etcd watch to a watch.Interface.
type etcdWatcher struct {
	encoding Encoding

	list   bool // If we're doing a recursive watch, should be true.
	filter FilterFunc

	etcdIncoming  chan *etcd.Response
	etcdStop      chan bool
	etcdCallEnded chan struct{}

	outgoing chan watch.Event
	userStop chan struct{}
	stopped  bool
	stopLock sync.Mutex

	// Injectable for testing. Send the event down the outgoing channel.
	emit func(watch.Event)
}

// Returns a new etcdWatcher; if list is true, watch sub-nodes.
func newEtcdWatcher(list bool, filter FilterFunc, encoding Encoding) *etcdWatcher {
	w := &etcdWatcher{
		encoding:      encoding,
		list:          list,
		filter:        filter,
		etcdIncoming:  make(chan *etcd.Response),
		etcdStop:      make(chan bool),
		etcdCallEnded: make(chan struct{}),
		outgoing:      make(chan watch.Event),
		userStop:      make(chan struct{}),
	}
	w.emit = func(e watch.Event) { w.outgoing <- e }
	go w.translate()
	return w
}

// etcdWatch calls etcd's Watch function, and handles any errors. Meant to be called
// as a goroutine.
func (w *etcdWatcher) etcdWatch(client EtcdGetSet, key string) {
	defer util.HandleCrash()
	defer close(w.etcdCallEnded)
	_, err := client.Watch(key, 0, w.list, w.etcdIncoming, w.etcdStop)
	if err != etcd.ErrWatchStoppedByUser {
		glog.Errorf("etcd.Watch stopped unexpectedly: %v (%#v)", err, err)
	}
}

// Pull stuff from etcd, convert, and push out the outgoing channel. Meant to be
// called as a goroutine.
func (w *etcdWatcher) translate() {
	defer close(w.outgoing)
	defer util.HandleCrash()

	for {
		select {
		case <-w.etcdCallEnded:
			return
		case <-w.userStop:
			w.etcdStop <- true
			return
		case res, ok := <-w.etcdIncoming:
			if !ok {
				return
			}
			w.sendResult(res)
		}
	}
}

func (w *etcdWatcher) sendResult(res *etcd.Response) {
	var action watch.EventType
	var data []byte
	switch res.Action {
	case "create", "set":
		if res.Node == nil {
			glog.Errorf("unexpected nil node: %#v", res)
			return
		}
		data = []byte(res.Node.Value)
		// TODO: Is this conditional correct?
		if res.EtcdIndex > 0 {
			action = watch.Modified
		} else {
			action = watch.Added
		}
	case "delete":
		if res.PrevNode == nil {
			glog.Errorf("unexpected nil prev node: %#v", res)
			return
		}
		data = []byte(res.PrevNode.Value)
		action = watch.Deleted
	default:
		glog.Errorf("unknown action: %v", res.Action)
		return
	}

	obj, err := w.encoding.Decode(data)
	if err != nil {
		glog.Errorf("failure to decode api object: '%v' from %#v %#v", string(data), res, res.Node)
		// TODO: expose an error through watch.Interface?
		w.Stop()
		return
	}
	w.emit(watch.Event{
		Type:   action,
		Object: obj,
	})
}

// ResultChannel implements watch.Interface.
func (w *etcdWatcher) ResultChan() <-chan watch.Event {
	return w.outgoing
}

// Stop implements watch.Interface.
func (w *etcdWatcher) Stop() {
	w.stopLock.Lock()
	defer w.stopLock.Unlock()
	// Prevent double channel closes.
	if !w.stopped {
		w.stopped = true
		close(w.userStop)
	}
}
