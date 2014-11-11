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
	"io/ioutil"
	"net/http"
	"os/exec"
	"reflect"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
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

// EtcdClient is an injectable interface for testing.
type EtcdClient interface {
	GetCluster() []string
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

// EtcdGetSet interface exposes only the etcd operations needed by EtcdHelper.
type EtcdGetSet interface {
	GetCluster() []string
	Get(key string, sort, recursive bool) (*etcd.Response, error)
	Set(key, value string, ttl uint64) (*etcd.Response, error)
	Create(key, value string, ttl uint64) (*etcd.Response, error)
	Delete(key string, recursive bool) (*etcd.Response, error)
	CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error)
	Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error)
}

type EtcdResourceVersioner interface {
	SetResourceVersion(obj runtime.Object, version uint64) error
	ResourceVersion(obj runtime.Object) (uint64, error)
}

// RuntimeVersionAdapter converts a string based versioner to EtcdResourceVersioner
type RuntimeVersionAdapter struct {
	Versioner runtime.ResourceVersioner
}

// SetResourceVersion implements EtcdResourceVersioner
func (a RuntimeVersionAdapter) SetResourceVersion(obj runtime.Object, version uint64) error {
	if version == 0 {
		return a.Versioner.SetResourceVersion(obj, "")
	}
	s := strconv.FormatUint(version, 10)
	return a.Versioner.SetResourceVersion(obj, s)
}

// SetResourceVersion implements EtcdResourceVersioner
func (a RuntimeVersionAdapter) ResourceVersion(obj runtime.Object) (uint64, error) {
	version, err := a.Versioner.ResourceVersion(obj)
	if err != nil {
		return 0, err
	}
	if version == "" {
		return 0, nil
	}
	return strconv.ParseUint(version, 10, 64)
}

// EtcdHelper offers common object marshalling/unmarshalling operations on an etcd client.
type EtcdHelper struct {
	Client EtcdGetSet
	Codec  runtime.Codec
	// optional, no atomic operations can be performed without this interface
	ResourceVersioner EtcdResourceVersioner
}

// IsEtcdNotFound returns true iff err is an etcd not found error.
func IsEtcdNotFound(err error) bool {
	return isEtcdErrorNum(err, EtcdErrorCodeNotFound)
}

// IsEtcdNodeExist returns true iff err is an etcd node aleady exist error.
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

// isEtcdErrorNum returns true iff err is an etcd error, whose errorCode matches errorCode
func isEtcdErrorNum(err error, errorCode int) bool {
	etcdError, ok := err.(*etcd.EtcdError)
	return ok && etcdError != nil && etcdError.ErrorCode == errorCode
}

// etcdErrorIndex returns the index associated with the error message and whether the
// index was available.
func etcdErrorIndex(err error) (uint64, bool) {
	if etcdError, ok := err.(*etcd.EtcdError); ok {
		return etcdError.Index, true
	}
	return 0, false
}

func (h *EtcdHelper) listEtcdNode(key string) ([]*etcd.Node, uint64, error) {
	result, err := h.Client.Get(key, false, true)
	if err != nil {
		index, ok := etcdErrorIndex(err)
		if !ok {
			index = 0
		}
		nodes := make([]*etcd.Node, 0)
		if IsEtcdNotFound(err) {
			return nodes, index, nil
		} else {
			return nodes, index, err
		}
	}
	return result.Node.Nodes, result.EtcdIndex, nil
}

// ExtractList extracts a go object per etcd node into a slice with the resource version.
// DEPRECATED: Use ExtractToList instead, it's more convenient.
func (h *EtcdHelper) ExtractList(key string, slicePtr interface{}, resourceVersion *uint64) error {
	nodes, index, err := h.listEtcdNode(key)
	if resourceVersion != nil {
		*resourceVersion = index
	}
	if err != nil {
		return err
	}
	return h.decodeNodeList(nodes, slicePtr)
}

// decodeNodeList walks the tree of each node in the list and decodes into the specified object
func (h *EtcdHelper) decodeNodeList(nodes []*etcd.Node, slicePtr interface{}) error {
	v, err := conversion.EnforcePtr(slicePtr)
	if err != nil || v.Kind() != reflect.Slice {
		// This should not happen at runtime.
		panic("need ptr to slice")
	}
	for _, node := range nodes {
		if node.Dir {
			if err := h.decodeNodeList(node.Nodes, slicePtr); err != nil {
				return err
			}
			continue
		}
		obj := reflect.New(v.Type().Elem())
		if err := h.Codec.DecodeInto([]byte(node.Value), obj.Interface().(runtime.Object)); err != nil {
			return err
		}
		if h.ResourceVersioner != nil {
			_ = h.ResourceVersioner.SetResourceVersion(obj.Interface().(runtime.Object), node.ModifiedIndex)
			// being unable to set the version does not prevent the object from being extracted
		}
		v.Set(reflect.Append(v, obj.Elem()))
	}
	return nil
}

// ExtractToList is just like ExtractList, but it works on a ThingyList api object.
// extracts a go object per etcd node into a slice with the resource version.
func (h *EtcdHelper) ExtractToList(key string, listObj runtime.Object) error {
	var resourceVersion uint64
	listPtr, err := runtime.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	if err := h.ExtractList(key, listPtr, &resourceVersion); err != nil {
		return err
	}
	if h.ResourceVersioner != nil {
		if err := h.ResourceVersioner.SetResourceVersion(listObj, resourceVersion); err != nil {
			return err
		}
	}
	return nil
}

// ExtractObj unmarshals json found at key into objPtr. On a not found error, will either return
// a zero object of the requested type, or an error, depending on ignoreNotFound. Treats
// empty responses and nil response nodes exactly like a not found error.
func (h *EtcdHelper) ExtractObj(key string, objPtr runtime.Object, ignoreNotFound bool) error {
	_, _, err := h.bodyAndExtractObj(key, objPtr, ignoreNotFound)
	return err
}

func (h *EtcdHelper) bodyAndExtractObj(key string, objPtr runtime.Object, ignoreNotFound bool) (body string, modifiedIndex uint64, err error) {
	response, err := h.Client.Get(key, false, false)

	if err != nil && !IsEtcdNotFound(err) {
		return "", 0, err
	}
	if err != nil || response.Node == nil || len(response.Node.Value) == 0 {
		if ignoreNotFound {
			v, err := conversion.EnforcePtr(objPtr)
			if err != nil {
				return "", 0, err
			}
			v.Set(reflect.Zero(v.Type()))
			return "", 0, nil
		} else if err != nil {
			return "", 0, err
		}
		return "", 0, fmt.Errorf("key '%v' found no nodes field: %#v", key, response)
	}
	body = response.Node.Value
	err = h.Codec.DecodeInto([]byte(body), objPtr)
	if h.ResourceVersioner != nil {
		_ = h.ResourceVersioner.SetResourceVersion(objPtr, response.Node.ModifiedIndex)
		// being unable to set the version does not prevent the object from being extracted
	}
	return body, response.Node.ModifiedIndex, err
}

// CreateObj adds a new object at a key unless it already exists. 'ttl' is time-to-live in seconds,
// and 0 means forever.
func (h *EtcdHelper) CreateObj(key string, obj runtime.Object, ttl uint64) error {
	data, err := h.Codec.Encode(obj)
	if err != nil {
		return err
	}
	if h.ResourceVersioner != nil {
		if version, err := h.ResourceVersioner.ResourceVersion(obj); err == nil && version != 0 {
			return errors.New("resourceVersion may not be set on objects to be created")
		}
	}

	_, err = h.Client.Create(key, string(data), ttl)
	return err
}

// Delete removes the specified key.
func (h *EtcdHelper) Delete(key string, recursive bool) error {
	_, err := h.Client.Delete(key, recursive)
	return err
}

// SetObj marshals obj via json, and stores under key. Will do an
// atomic update if obj's ResourceVersion field is set.
func (h *EtcdHelper) SetObj(key string, obj runtime.Object) error {
	data, err := h.Codec.Encode(obj)
	if err != nil {
		return err
	}
	if h.ResourceVersioner != nil {
		if version, err := h.ResourceVersioner.ResourceVersion(obj); err == nil && version != 0 {
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
type EtcdUpdateFunc func(input runtime.Object) (output runtime.Object, err error)

// AtomicUpdate generalizes the pattern that allows for making atomic updates to etcd objects.
// Note, tryUpdate may be called more than once.
//
// Example:
//
// h := &util.EtcdHelper{client, encoding, versioning}
// err := h.AtomicUpdate("myKey", &MyType{}, func(input runtime.Object) (runtime.Object, error) {
//	// Before this function is called, currentObj has been reset to etcd's current
//	// contents for "myKey".
//
//	cur := input.(*MyType) // Guaranteed to work.
//
//	// Make a *modification*.
//	cur.Counter++
//
//	// Return the modified object. Return an error to stop iterating.
//	return cur, nil
// })
//
func (h *EtcdHelper) AtomicUpdate(key string, ptrToType runtime.Object, tryUpdate EtcdUpdateFunc) error {
	v, err := conversion.EnforcePtr(ptrToType)
	if err != nil {
		// Panic is appropriate, because this is a programming error.
		panic("need ptr to type")
	}
	for {
		obj := reflect.New(v.Type()).Interface().(runtime.Object)
		origBody, index, err := h.bodyAndExtractObj(key, obj, true)
		if err != nil {
			return err
		}

		ret, err := tryUpdate(obj)
		if err != nil {
			return err
		}

		data, err := h.Codec.Encode(ret)
		if err != nil {
			return err
		}

		// First time this key has been used, try creating new value.
		if index == 0 {
			_, err = h.Client.Create(key, string(data), 0)
			if IsEtcdNodeExist(err) {
				continue
			}
			return err
		}

		if string(data) == origBody {
			return nil
		}

		_, err = h.Client.CompareAndSwap(key, string(data), 0, origBody, index)
		if IsEtcdTestFailed(err) {
			continue
		}
		return err
	}
}

func checkEtcd(host string) error {
	response, err := http.Get(host + "/version")
	if err != nil {
		return err
	}
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return err
	}
	if !strings.HasPrefix("etcd", string(body)) {
		return fmt.Errorf("Unknown server: %s", string(body))
	}
	return nil
}

func startEtcd() (*exec.Cmd, error) {
	cmd := exec.Command("etcd")
	err := cmd.Start()
	if err != nil {
		return nil, err
	}
	return cmd, nil
}

func NewEtcdClientStartServerIfNecessary(server string) (EtcdClient, error) {
	err := checkEtcd(server)
	if err != nil {
		glog.Infof("Failed to find etcd, attempting to start.")
		_, err := startEtcd()
		if err != nil {
			return nil, err
		}
	}

	servers := []string{server}
	return etcd.NewClient(servers), nil
}
