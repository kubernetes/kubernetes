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

package util

import (
	"encoding/json"
	"fmt"
	"reflect"

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
func (h *EtcdHelper) ExtractObj(key string, objPtr interface{}, ignoreNotFound bool) (error, uint64) {
	response, err := h.Client.Get(key, false, false)

	if err != nil && !IsEtcdNotFound(err) {
		return err, 0
	}
	if err != nil || response.Node == nil || len(response.Node.Value) == 0 {
		if ignoreNotFound {
			pv := reflect.ValueOf(objPtr)
			pv.Elem().Set(reflect.Zero(pv.Type().Elem()))
			return nil, 0
		} else if err != nil {
			return err, 0
		}
		return fmt.Errorf("key '%v' found no nodes field: %#v", key, response), 0
	}
	return json.Unmarshal([]byte(response.Node.Value), objPtr), response.Node.ModifiedIndex
}

// CompareAndSwapObj marshals obj via json, and stores under key so long as index matches the previous modified index
func (h *EtcdHelper) CompareAndSwapObj(key string, obj interface{}, index uint64) error {
	data, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	_, err = h.Client.CompareAndSwap(key, string(data), 0, "", index)
	return err
}

// SetObj marshals obj via json, and stores under key.
func (h *EtcdHelper) SetObj(key string, obj interface{}) error {
	data, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	_, err = h.Client.Set(key, string(data), 0)
	return err
}
