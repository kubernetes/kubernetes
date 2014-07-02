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
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/coreos/go-etcd/etcd"
)

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
	CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error)
}

// EtcdHelper offers common object marshalling/unmarshalling operations on an etcd client.
type EtcdHelper struct {
	Client EtcdGetSet
}

// Returns true iff err is an etcd not found error.
func IsEtcdNotFound(err error) bool {
	return isEtcdErrorNum(err, 100)
}

// Returns true iff err is an etcd write conflict.
func IsEtcdConflict(err error) bool {
	return isEtcdErrorNum(err, 101)
}

// Returns true iff err is an etcd error, whose errorCode matches errorCode
func isEtcdErrorNum(err error, errorCode int) bool {
	if err == nil {
		return false
	}
	switch err.(type) {
	case *etcd.EtcdError:
		etcdError := err.(*etcd.EtcdError)
		if etcdError == nil {
			return false
		}
		if etcdError.ErrorCode == errorCode {
			return true
		}
	}
	return false
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
		err = json.Unmarshal([]byte(node.Value), obj.Interface())
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
	err = json.Unmarshal([]byte(body), objPtr)
	if jsonBase, err := api.FindJSONBase(objPtr); err == nil {
		jsonBase.ResourceVersion = response.Node.ModifiedIndex
		// Note that err shadows the err returned below, so we won't
		// return an error just because we failed to find a JSONBase.
		// This is intentional.
	}
	return body, response.Node.ModifiedIndex, err
}

// SetObj marshals obj via json, and stores under key. Will do an
// atomic update if obj's ResourceVersion field is set.
func (h *EtcdHelper) SetObj(key string, obj interface{}) error {
	data, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	if jsonBase, err := api.FindJSONBaseRO(obj); err == nil && jsonBase.ResourceVersion != 0 {
		_, err = h.Client.CompareAndSwap(key, string(data), 0, "", jsonBase.ResourceVersion)
		return err // err is shadowed!
	}

	// TODO: when client supports atomic creation, integrate this with the above.
	_, err = h.Client.Set(key, string(data), 0)
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
// h := &util.EtcdHelper{client}
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
		// TODO: This is racy. Fix when our client supports prevExist. See:
		// https://github.com/coreos/etcd/blob/master/Documentation/api.md#atomic-compare-and-swap
		if index == 0 {
			return h.SetObj(key, ret)
		}

		data, err := json.Marshal(ret)
		if err != nil {
			return err
		}
		_, err = h.Client.CompareAndSwap(key, string(data), 0, origBody, index)
		if IsEtcdConflict(err) {
			continue
		}
		return err
	}
}
