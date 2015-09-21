/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"sort"
	"sync"
	"time"

	"github.com/coreos/go-etcd/etcd"
)

type EtcdResponseWithError struct {
	R *etcd.Response
	E error
	// if N is non-null, it will be assigned into the map after this response is used for an operation
	N *EtcdResponseWithError
}

// TestLogger is a type passed to Test functions to support formatted test logs.
type TestLogger interface {
	Fatalf(format string, args ...interface{})
	Errorf(format string, args ...interface{})
	Logf(format string, args ...interface{})
}

type FakeEtcdClient struct {
	watchCompletedChan chan bool

	Data                 map[string]EtcdResponseWithError
	DeletedKeys          []string
	expectNotFoundGetSet map[string]struct{}
	sync.Mutex
	Err         error
	CasErr      error
	t           TestLogger
	Ix          int
	TestIndex   bool
	ChangeIndex uint64
	LastSetTTL  uint64
	// Will avoid setting the expires header on objects to make comparison easier
	HideExpires bool
	Machines    []string

	// Will become valid after Watch is called; tester may write to it. Tester may
	// also read from it to verify that it's closed after injecting an error.
	WatchResponse chan *etcd.Response
	WatchIndex    uint64
	// Write to this to prematurely stop a Watch that is running in a goroutine.
	WatchInjectError chan<- error
	WatchStop        chan<- bool
	// If non-nil, will be returned immediately when Watch is called.
	WatchImmediateError error
}

func NewFakeEtcdClient(t TestLogger) *FakeEtcdClient {
	ret := &FakeEtcdClient{
		t:                    t,
		expectNotFoundGetSet: map[string]struct{}{},
		Data:                 map[string]EtcdResponseWithError{},
	}
	// There are three publicly accessible channels in FakeEtcdClient:
	//  - WatchResponse
	//  - WatchInjectError
	//  - WatchStop
	// They are only available when Watch() is called.  If users of
	// FakeEtcdClient want to use any of these channels, they have to call
	// WaitForWatchCompletion before any operation on these channels.
	// Internally, FakeEtcdClient use watchCompletedChan to indicate if the
	// Watch() method has been called. WaitForWatchCompletion() will wait
	// on this channel. WaitForWatchCompletion() will return only when
	// WatchResponse, WatchInjectError and WatchStop are ready to read/write.
	ret.watchCompletedChan = make(chan bool)
	return ret
}

func (f *FakeEtcdClient) SetError(err error) {
	f.Err = err
}

func (f *FakeEtcdClient) GetCluster() []string {
	return f.Machines
}

func (f *FakeEtcdClient) ExpectNotFoundGet(key string) {
	f.expectNotFoundGetSet[key] = struct{}{}
}

func (f *FakeEtcdClient) NewError(code int) *etcd.EtcdError {
	return &etcd.EtcdError{
		ErrorCode: code,
		Index:     f.ChangeIndex,
	}
}

func (f *FakeEtcdClient) generateIndex() uint64 {
	if !f.TestIndex {
		return 0
	}

	f.ChangeIndex++
	f.t.Logf("generating index %v", f.ChangeIndex)
	return f.ChangeIndex
}

// Requires that f.Mutex be held.
func (f *FakeEtcdClient) updateResponse(key string) {
	resp, found := f.Data[key]
	if !found || resp.N == nil {
		return
	}
	f.Data[key] = *resp.N
}

func (f *FakeEtcdClient) Get(key string, sort, recursive bool) (*etcd.Response, error) {
	if f.Err != nil {
		return nil, f.Err
	}

	f.Mutex.Lock()
	defer f.Mutex.Unlock()
	defer f.updateResponse(key)

	result := f.Data[key]
	if result.R == nil {
		if _, ok := f.expectNotFoundGetSet[key]; !ok {
			f.t.Logf("data for %s was not defined prior to invoking Get", key)
		}
		return &etcd.Response{}, f.NewError(EtcdErrorCodeNotFound)
	}
	f.t.Logf("returning %v: %#v %#v", key, result.R, result.E)

	// Sort response, note this will alter result.R.
	if result.R.Node != nil && result.R.Node.Nodes != nil && sort {
		f.sortResponse(result.R.Node.Nodes)
	}
	return result.R, result.E
}

func (f *FakeEtcdClient) sortResponse(nodes etcd.Nodes) {
	for i := range nodes {
		if nodes[i].Dir {
			f.sortResponse(nodes[i].Nodes)
		}
	}
	sort.Sort(nodes)
}

func (f *FakeEtcdClient) nodeExists(key string) bool {
	result, ok := f.Data[key]
	return ok && result.R != nil && result.R.Node != nil && result.E == nil
}

func (f *FakeEtcdClient) setLocked(key, value string, ttl uint64) (*etcd.Response, error) {
	f.LastSetTTL = ttl
	if f.Err != nil {
		return nil, f.Err
	}

	i := f.generateIndex()

	if f.nodeExists(key) {
		prevResult := f.Data[key]
		createdIndex := prevResult.R.Node.CreatedIndex
		f.t.Logf("updating %v, index %v -> %v (ttl: %d)", key, createdIndex, i, ttl)
		var expires *time.Time
		if !f.HideExpires && ttl > 0 {
			now := time.Now()
			expires = &now
		}
		result := EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         value,
					CreatedIndex:  createdIndex,
					ModifiedIndex: i,
					TTL:           int64(ttl),
					Expiration:    expires,
				},
			},
		}
		f.Data[key] = result
		return result.R, nil
	}

	f.t.Logf("creating %v, index %v (ttl: %d)", key, i, ttl)
	result := EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         value,
				CreatedIndex:  i,
				ModifiedIndex: i,
				TTL:           int64(ttl),
			},
		},
	}
	f.Data[key] = result
	return result.R, nil
}

func (f *FakeEtcdClient) Set(key, value string, ttl uint64) (*etcd.Response, error) {
	f.Mutex.Lock()
	defer f.Mutex.Unlock()
	defer f.updateResponse(key)

	return f.setLocked(key, value, ttl)
}

func (f *FakeEtcdClient) CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error) {
	if f.Err != nil {
		f.t.Logf("c&s: returning err %v", f.Err)
		return nil, f.Err
	}
	if f.CasErr != nil {
		f.t.Logf("c&s: returning err %v", f.CasErr)
		return nil, f.CasErr
	}

	if !f.TestIndex {
		f.t.Errorf("Enable TestIndex for test involving CompareAndSwap")
		return nil, errors.New("Enable TestIndex for test involving CompareAndSwap")
	}

	if prevValue == "" && prevIndex == 0 {
		return nil, errors.New("Either prevValue or prevIndex must be specified.")
	}

	f.Mutex.Lock()
	defer f.Mutex.Unlock()
	defer f.updateResponse(key)

	if !f.nodeExists(key) {
		f.t.Logf("c&s: node doesn't exist")
		return nil, EtcdErrorNotFound
	}

	prevNode := f.Data[key].R.Node

	if prevValue != "" && prevValue != prevNode.Value {
		f.t.Logf("body didn't match")
		return nil, EtcdErrorTestFailed
	}

	if prevIndex != 0 && prevIndex != prevNode.ModifiedIndex {
		f.t.Logf("got index %v but needed %v", prevIndex, prevNode.ModifiedIndex)
		return nil, EtcdErrorTestFailed
	}

	return f.setLocked(key, value, ttl)
}

func (f *FakeEtcdClient) Create(key, value string, ttl uint64) (*etcd.Response, error) {
	f.Mutex.Lock()
	defer f.Mutex.Unlock()
	defer f.updateResponse(key)

	if f.nodeExists(key) {
		return nil, EtcdErrorNodeExist
	}

	return f.setLocked(key, value, ttl)
}

func (f *FakeEtcdClient) Delete(key string, recursive bool) (*etcd.Response, error) {
	if f.Err != nil {
		return nil, f.Err
	}

	f.Mutex.Lock()
	defer f.Mutex.Unlock()
	existing, ok := f.Data[key]
	if !ok {
		return &etcd.Response{}, &etcd.EtcdError{
			ErrorCode: EtcdErrorCodeNotFound,
			Index:     f.ChangeIndex,
		}
	}
	etcdError, ok := existing.E.(*etcd.EtcdError)
	if ok && etcdError != nil && etcdError.ErrorCode == EtcdErrorCodeNotFound {
		f.DeletedKeys = append(f.DeletedKeys, key)
		return existing.R, existing.E
	}
	index := f.generateIndex()
	f.Data[key] = EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{
			ErrorCode: EtcdErrorCodeNotFound,
			Index:     index,
		},
	}
	res := &etcd.Response{
		Action:    "delete",
		Node:      nil,
		PrevNode:  nil,
		EtcdIndex: index,
	}
	if existing.R != nil && existing.R.Node != nil {
		res.PrevNode = existing.R.Node
	}

	f.DeletedKeys = append(f.DeletedKeys, key)
	return res, nil
}

func (f *FakeEtcdClient) WaitForWatchCompletion() {
	<-f.watchCompletedChan
}

func (f *FakeEtcdClient) Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error) {
	f.Mutex.Lock()
	if f.WatchImmediateError != nil {
		return nil, f.WatchImmediateError
	}
	f.WatchResponse = receiver
	f.WatchStop = stop
	f.WatchIndex = waitIndex
	injectedError := make(chan error)

	defer close(injectedError)
	f.WatchInjectError = injectedError

	f.Mutex.Unlock()
	if receiver == nil {
		return f.Get(prefix, false, recursive)
	} else {
		// Emulate etcd's behavior. (I think.)
		defer close(receiver)
	}

	f.watchCompletedChan <- true
	select {
	case <-stop:
		return nil, etcd.ErrWatchStoppedByUser
	case err := <-injectedError:
		return nil, err
	}
}
