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
	"fmt"
	"testing"

	"github.com/coreos/go-etcd/etcd"
)

type EtcdResponseWithError struct {
	R *etcd.Response
	E error
}

type FakeEtcdClient struct {
	watchCompletedChan chan bool

	Data        map[string]EtcdResponseWithError
	DeletedKeys []string
	Err         error
	t           *testing.T
	Ix          int

	// Will become valid after Watch is called; tester may write to it. Tester may
	// also read from it to verify that it's closed after injecting an error.
	WatchResponse chan *etcd.Response
	// Write to this to prematurely stop a Watch that is running in a goroutine.
	WatchInjectError chan<- error
	WatchStop        chan<- bool
}

func MakeFakeEtcdClient(t *testing.T) *FakeEtcdClient {
	ret := &FakeEtcdClient{
		t:    t,
		Data: map[string]EtcdResponseWithError{},
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

func (f *FakeEtcdClient) AddChild(key, data string, ttl uint64) (*etcd.Response, error) {
	f.Ix = f.Ix + 1
	return f.Set(fmt.Sprintf("%s/%d", key, f.Ix), data, ttl)
}

func (f *FakeEtcdClient) Get(key string, sort, recursive bool) (*etcd.Response, error) {
	result := f.Data[key]
	if result.R == nil {
		f.t.Errorf("Unexpected get for %s", key)
		return &etcd.Response{}, &etcd.EtcdError{ErrorCode: 100} // Key not found
	}
	f.t.Logf("returning %v: %v %#v", key, result.R, result.E)
	return result.R, result.E
}

func (f *FakeEtcdClient) Set(key, value string, ttl uint64) (*etcd.Response, error) {
	result := EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: value,
			},
		},
	}
	f.Data[key] = result
	return result.R, f.Err
}

func (f *FakeEtcdClient) CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error) {
	// TODO: Maybe actually implement compare and swap here?
	return f.Set(key, value, ttl)
}

func (f *FakeEtcdClient) Create(key, value string, ttl uint64) (*etcd.Response, error) {
	return f.Set(key, value, ttl)
}
func (f *FakeEtcdClient) Delete(key string, recursive bool) (*etcd.Response, error) {
	f.Data[key] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{ErrorCode: 100},
	}

	f.DeletedKeys = append(f.DeletedKeys, key)
	return &etcd.Response{}, f.Err
}

func (f *FakeEtcdClient) WaitForWatchCompletion() {
	<-f.watchCompletedChan
}

func (f *FakeEtcdClient) Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error) {
	f.WatchResponse = receiver
	f.WatchStop = stop
	injectedError := make(chan error)

	defer close(injectedError)
	f.WatchInjectError = injectedError

	f.watchCompletedChan <- true
	select {
	case <-stop:
		return nil, etcd.ErrWatchStoppedByUser
	case err := <-injectedError:
		// Emulate etcd's behavior.
		close(receiver)
		return nil, err
	}
	// Never get here.
	return nil, nil
}
