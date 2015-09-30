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
	"sort"
	"strings"
	"sync"
	"time"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
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
	WatchInjectError chan error
	// If non-nil, will be returned immediately when Watch is called.
	WatchImmediateError error
}

type FakeWatcher struct {
	Opts   etcd.WatcherOptions
	Key    string
	Client FakeEtcdClient
}

func (f *FakeWatcher) Next(ctx context.Context) (*etcd.Response, error) {
	if f.Client.WatchImmediateError != nil {
		return nil, f.Client.WatchImmediateError
	}

	select {
	case <-ctx.Done():
		return nil, EtcdErrWatchStoppedByUser
	case err := <-f.Client.WatchInjectError:
		return nil, err
	case resp := <-f.Client.WatchResponse:
		return resp, nil
	}
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
	// They are only available when Watch() is called.  If users of
	// FakeEtcdClient want to use any of these channels, they have to call
	// WaitForWatchCompletion before any operation on these channels.
	// Internally, FakeEtcdClient use watchCompletedChan to indicate if the
	// Watch() method has been called. WaitForWatchCompletion() will wait
	// on this channel. WaitForWatchCompletion() will return only when
	// WatchResponse, WatchInjectError are ready to read/write.
	ret.watchCompletedChan = make(chan bool)
	ret.WatchResponse = make(chan *etcd.Response)
	ret.WatchInjectError = make(chan error)
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

func (f *FakeEtcdClient) NewError(code int) *etcd.Error {
	return &etcd.Error{
		Code:  code,
		Index: f.ChangeIndex,
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

func (f *FakeEtcdClient) Get(ctx context.Context, key string, opts *etcd.GetOptions) (*etcd.Response, error) {
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
	if result.R.Node != nil && result.R.Node.Nodes != nil && opts.Sort {
		f.sortResponse(result.R.Node)
	}
	return result.R, result.E
}

func (f *FakeEtcdClient) sortResponse(node *etcd.Node) {
	nodes := node.Nodes
	for i := range nodes {
		if nodes[i].Dir {
			f.sortResponse(nodes[i])
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

	f.t.Logf("creating %v, index %v, value %v (ttl: %d)", key, i, value, ttl)
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

// NOTE: Compare and Swap behavior is now behind the options, but follow on PR's should remove this interface
//       entirely for the purposes of data hiding behind the storage layer.
func (f *FakeEtcdClient) Set(ctx context.Context, key, value string, opts *etcd.SetOptions) (*etcd.Response, error) {
	var ttl uint64

	f.Mutex.Lock()
	defer f.Mutex.Unlock()
	defer f.updateResponse(key)

	ttl = 0
	if opts != nil {
		ttl = uint64(opts.TTL)
		if opts.PrevExist == etcd.PrevNoExist && f.nodeExists(key) {
			return nil, EtcdErrorNodeExist
		}
	}

	return f.setLocked(key, value, ttl)
}

//NOTE: This is unused and will be ripped out in follow on PRs
func (f *FakeEtcdClient) Update(ctx context.Context, key, value string) (*etcd.Response, error) {
	return f.Set(ctx, key, value, nil)
}

func (f *FakeEtcdClient) Create(ctx context.Context, key, value string) (*etcd.Response, error) {
	f.Mutex.Lock()
	defer f.Mutex.Unlock()
	defer f.updateResponse(key)

	if f.nodeExists(key) {
		return nil, EtcdErrorNodeExist
	}

	return f.setLocked(key, value, 0)

}

func (f *FakeEtcdClient) CreateInOrder(ctx context.Context, key, value string, opts *etcd.CreateInOrderOptions) (*etcd.Response, error) {
	var ttl uint64

	f.Mutex.Lock()
	defer f.Mutex.Unlock()
	defer f.updateResponse(key)
	ttl = 0
	if opts != nil {
		ttl = uint64(opts.TTL)
	}

	if f.nodeExists(key) {
		return nil, EtcdErrorNodeExist
	}

	return f.setLocked(key, value, ttl)
}

func (f *FakeEtcdClient) Delete(ctx context.Context, key string, opts *etcd.DeleteOptions) (*etcd.Response, error) {
	if f.Err != nil {
		return nil, f.Err
	}

	f.Mutex.Lock()
	defer f.Mutex.Unlock()

	if opts != nil {
		// NOTE: this only exists for kube2sky tests, whose behavior I completely question
		// The entire kube2sky client imho is suspect
		for p := range f.Data {
			if (opts.Recursive && strings.HasPrefix(p, key)) || (!opts.Recursive && p == key) {
				f.DeletedKeys = append(f.DeletedKeys, key)
			}
		}
		return nil, nil
	}

	existing, ok := f.Data[key]
	if !ok {
		return &etcd.Response{}, &etcd.Error{
			Code:  EtcdErrorCodeNotFound,
			Index: f.ChangeIndex,
		}
	}
	etcdError, ok := existing.E.(*etcd.Error)
	if ok && etcdError != nil && etcdError.Code == EtcdErrorCodeNotFound {
		f.DeletedKeys = append(f.DeletedKeys, key)
		return existing.R, existing.E
	}
	index := f.generateIndex()
	f.Data[key] = EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.Error{
			Code:  EtcdErrorCodeNotFound,
			Index: index,
		},
	}
	res := &etcd.Response{
		Action:   "delete",
		Node:     nil,
		PrevNode: nil,
		Index:    index,
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

func (f *FakeEtcdClient) Watcher(key string, opts *etcd.WatcherOptions) etcd.Watcher {

	if opts != nil {
		f.WatchIndex = opts.AfterIndex
	}

	watcher := FakeWatcher{
		Key:    key,
		Opts:   *opts,
		Client: *f,
	}

	select {
	case f.watchCompletedChan <- true:
		// Buffer on free list; nothing more to do.
	default:
		// b/c of the difference in clients carry on if full
	}
	return &watcher
}
