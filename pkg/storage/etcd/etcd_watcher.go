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

package etcd

import (
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdutil "k8s.io/kubernetes/pkg/storage/etcd/util"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/watch"

	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

// Etcd watch event actions
const (
	EtcdCreate = "create"
	EtcdGet    = "get"
	EtcdSet    = "set"
	EtcdCAS    = "compareAndSwap"
	EtcdDelete = "delete"
	EtcdCAD    = "compareAndDelete"
	EtcdExpire = "expire"
)

// HighWaterMark is a thread-safe object for tracking the maximum value seen
// for some quantity.
type HighWaterMark int64

// Update returns true if and only if 'current' is the highest value ever seen.
func (hwm *HighWaterMark) Update(current int64) bool {
	for {
		old := atomic.LoadInt64((*int64)(hwm))
		if current <= old {
			return false
		}
		if atomic.CompareAndSwapInt64((*int64)(hwm), old, current) {
			return true
		}
	}
}

// TransformFunc attempts to convert an object to another object for use with a watcher.
type TransformFunc func(runtime.Object) (runtime.Object, error)

// includeFunc returns true if the given key should be considered part of a watch
type includeFunc func(key string) bool

// exceptKey is an includeFunc that returns false when the provided key matches the watched key
func exceptKey(except string) includeFunc {
	return func(key string) bool {
		return key != except
	}
}

// etcdWatcher converts a native etcd watch to a watch.Interface.
type etcdWatcher struct {
	encoding runtime.Codec
	// Note that versioner is required for etcdWatcher to work correctly.
	// There is no public constructor of it, so be careful when manipulating
	// with it manually.
	versioner storage.Versioner
	transform TransformFunc

	list    bool // If we're doing a recursive watch, should be true.
	quorum  bool // If we enable quorum, shoule be true
	include includeFunc
	filter  storage.FilterFunc

	etcdIncoming  chan *etcd.Response
	etcdError     chan error
	ctx           context.Context
	cancel        context.CancelFunc
	etcdCallEnded chan struct{}

	outgoing chan watch.Event
	userStop chan struct{}
	stopped  bool
	stopLock sync.Mutex
	// wg is used to avoid calls to etcd after Stop(), and to make sure
	// that the translate goroutine is not leaked.
	wg sync.WaitGroup

	// Injectable for testing. Send the event down the outgoing channel.
	emit func(watch.Event)

	cache etcdCache
}

// watchWaitDuration is the amount of time to wait for an error from watch.
const watchWaitDuration = 100 * time.Millisecond

// newEtcdWatcher returns a new etcdWatcher; if list is true, watch sub-nodes.
// The versioner must be able to handle the objects that transform creates.
func newEtcdWatcher(
	list bool, quorum bool, include includeFunc, filter storage.FilterFunc,
	encoding runtime.Codec, versioner storage.Versioner, transform TransformFunc,
	cache etcdCache) *etcdWatcher {
	w := &etcdWatcher{
		encoding:  encoding,
		versioner: versioner,
		transform: transform,
		list:      list,
		quorum:    quorum,
		include:   include,
		filter:    filter,
		// Buffer this channel, so that the etcd client is not forced
		// to context switch with every object it gets, and so that a
		// long time spent decoding an object won't block the *next*
		// object. Basically, we see a lot of "401 window exceeded"
		// errors from etcd, and that's due to the client not streaming
		// results but rather getting them one at a time. So we really
		// want to never block the etcd client, if possible. The 100 is
		// mostly arbitrary--we know it goes as high as 50, though.
		// There's a V(2) log message that prints the length so we can
		// monitor how much of this buffer is actually used.
		etcdIncoming: make(chan *etcd.Response, 100),
		etcdError:    make(chan error, 1),
		outgoing:     make(chan watch.Event),
		userStop:     make(chan struct{}),
		stopped:      false,
		wg:           sync.WaitGroup{},
		cache:        cache,
		ctx:          nil,
		cancel:       nil,
	}
	w.emit = func(e watch.Event) {
		// Give up on user stop, without this we leak a lot of goroutines in tests.
		select {
		case w.outgoing <- e:
		case <-w.userStop:
		}
	}
	// translate will call done. We need to Add() here because otherwise,
	// if Stop() gets called before translate gets started, there'd be a
	// problem.
	w.wg.Add(1)
	go w.translate()
	return w
}

// etcdWatch calls etcd's Watch function, and handles any errors. Meant to be called
// as a goroutine.
func (w *etcdWatcher) etcdWatch(ctx context.Context, client etcd.KeysAPI, key string, resourceVersion uint64) {
	defer utilruntime.HandleCrash()
	defer close(w.etcdError)
	defer close(w.etcdIncoming)

	// All calls to etcd are coming from this function - once it is finished
	// no other call to etcd should be generated by this watcher.
	done := func() {}

	// We need to be prepared, that Stop() can be called at any time.
	// It can potentially also be called, even before this function is called.
	// If that is the case, we simply skip all the code here.
	// See #18928 for more details.
	var watcher etcd.Watcher
	returned := func() bool {
		w.stopLock.Lock()
		defer w.stopLock.Unlock()
		if w.stopped {
			// Watcher has already been stopped - don't event initiate it here.
			return true
		}
		w.wg.Add(1)
		done = w.wg.Done
		// Perform initialization of watcher under lock - we want to avoid situation when
		// Stop() is called in the meantime (which in tests can cause etcd termination and
		// strange behavior here).
		if resourceVersion == 0 {
			latest, err := etcdGetInitialWatchState(ctx, client, key, w.list, w.quorum, w.etcdIncoming)
			if err != nil {
				w.etcdError <- err
				return true
			}
			resourceVersion = latest
		}

		opts := etcd.WatcherOptions{
			Recursive:  w.list,
			AfterIndex: resourceVersion,
		}
		watcher = client.Watcher(key, &opts)
		w.ctx, w.cancel = context.WithCancel(ctx)
		return false
	}()
	defer done()
	if returned {
		return
	}

	for {
		resp, err := watcher.Next(w.ctx)
		if err != nil {
			w.etcdError <- err
			return
		}
		w.etcdIncoming <- resp
	}
}

// etcdGetInitialWatchState turns an etcd Get request into a watch equivalent
func etcdGetInitialWatchState(ctx context.Context, client etcd.KeysAPI, key string, recursive bool, quorum bool, incoming chan<- *etcd.Response) (resourceVersion uint64, err error) {
	opts := etcd.GetOptions{
		Recursive: recursive,
		Sort:      false,
		Quorum:    quorum,
	}
	resp, err := client.Get(ctx, key, &opts)
	if err != nil {
		if !etcdutil.IsEtcdNotFound(err) {
			utilruntime.HandleError(fmt.Errorf("watch was unable to retrieve the current index for the provided key (%q): %v", key, err))
			return resourceVersion, toStorageErr(err, key, 0)
		}
		if etcdError, ok := err.(etcd.Error); ok {
			resourceVersion = etcdError.Index
		}
		return resourceVersion, nil
	}
	resourceVersion = resp.Index
	convertRecursiveResponse(resp.Node, resp, incoming)
	return
}

// convertRecursiveResponse turns a recursive get response from etcd into individual response objects
// by copying the original response.  This emulates the behavior of a recursive watch.
func convertRecursiveResponse(node *etcd.Node, response *etcd.Response, incoming chan<- *etcd.Response) {
	if node.Dir {
		for i := range node.Nodes {
			convertRecursiveResponse(node.Nodes[i], response, incoming)
		}
		return
	}
	copied := *response
	copied.Action = "get"
	copied.Node = node
	incoming <- &copied
}

var (
	watchChannelHWM HighWaterMark
)

// translate pulls stuff from etcd, converts, and pushes out the outgoing channel. Meant to be
// called as a goroutine.
func (w *etcdWatcher) translate() {
	defer w.wg.Done()
	defer close(w.outgoing)
	defer utilruntime.HandleCrash()

	for {
		select {
		case err := <-w.etcdError:
			if err != nil {
				var status *unversioned.Status
				switch {
				case etcdutil.IsEtcdWatchExpired(err):
					status = &unversioned.Status{
						Status:  unversioned.StatusFailure,
						Message: err.Error(),
						Code:    http.StatusGone, // Gone
						Reason:  unversioned.StatusReasonExpired,
					}
				// TODO: need to generate errors using api/errors which has a circular dependency on this package
				//   no other way to inject errors
				// case etcdutil.IsEtcdUnreachable(err):
				//   status = errors.NewServerTimeout(...)
				default:
					status = &unversioned.Status{
						Status:  unversioned.StatusFailure,
						Message: err.Error(),
						Code:    http.StatusInternalServerError,
						Reason:  unversioned.StatusReasonInternalError,
					}
				}
				w.emit(watch.Event{
					Type:   watch.Error,
					Object: status,
				})
			}
			return
		case <-w.userStop:
			return
		case res, ok := <-w.etcdIncoming:
			if ok {
				if curLen := int64(len(w.etcdIncoming)); watchChannelHWM.Update(curLen) {
					// Monitor if this gets backed up, and how much.
					glog.V(2).Infof("watch: %v objects queued in channel.", curLen)
				}
				w.sendResult(res)
			}
			// If !ok, don't return here-- must wait for etcdError channel
			// to give an error or be closed.
		}
	}
}

func (w *etcdWatcher) decodeObject(node *etcd.Node) (runtime.Object, error) {
	if obj, found := w.cache.getFromCache(node.ModifiedIndex, storage.Everything); found {
		return obj, nil
	}

	obj, err := runtime.Decode(w.encoding, []byte(node.Value))
	if err != nil {
		return nil, err
	}

	// ensure resource version is set on the object we load from etcd
	if err := w.versioner.UpdateObject(obj, node.ModifiedIndex); err != nil {
		utilruntime.HandleError(fmt.Errorf("failure to version api object (%d) %#v: %v", node.ModifiedIndex, obj, err))
	}

	// perform any necessary transformation
	if w.transform != nil {
		obj, err = w.transform(obj)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("failure to transform api object %#v: %v", obj, err))
			return nil, err
		}
	}

	if node.ModifiedIndex != 0 {
		w.cache.addToCache(node.ModifiedIndex, obj)
	}
	return obj, nil
}

func (w *etcdWatcher) sendAdd(res *etcd.Response) {
	if res.Node == nil {
		utilruntime.HandleError(fmt.Errorf("unexpected nil node: %#v", res))
		return
	}
	if w.include != nil && !w.include(res.Node.Key) {
		return
	}
	obj, err := w.decodeObject(res.Node)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failure to decode api object: %v\n'%v' from %#v %#v", err, string(res.Node.Value), res, res.Node))
		// TODO: expose an error through watch.Interface?
		// Ignore this value. If we stop the watch on a bad value, a client that uses
		// the resourceVersion to resume will never be able to get past a bad value.
		return
	}
	if !w.filter(obj) {
		return
	}
	action := watch.Added
	if res.Node.ModifiedIndex != res.Node.CreatedIndex {
		action = watch.Modified
	}
	w.emit(watch.Event{
		Type:   action,
		Object: obj,
	})
}

func (w *etcdWatcher) sendModify(res *etcd.Response) {
	if res.Node == nil {
		glog.Errorf("unexpected nil node: %#v", res)
		return
	}
	if w.include != nil && !w.include(res.Node.Key) {
		return
	}
	curObj, err := w.decodeObject(res.Node)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failure to decode api object: %v\n'%v' from %#v %#v", err, string(res.Node.Value), res, res.Node))
		// TODO: expose an error through watch.Interface?
		// Ignore this value. If we stop the watch on a bad value, a client that uses
		// the resourceVersion to resume will never be able to get past a bad value.
		return
	}
	curObjPasses := w.filter(curObj)
	oldObjPasses := false
	var oldObj runtime.Object
	if res.PrevNode != nil && res.PrevNode.Value != "" {
		// Ignore problems reading the old object.
		if oldObj, err = w.decodeObject(res.PrevNode); err == nil {
			oldObjPasses = w.filter(oldObj)
		}
	}
	// Some changes to an object may cause it to start or stop matching a filter.
	// We need to report those as adds/deletes. So we have to check both the previous
	// and current value of the object.
	switch {
	case curObjPasses && oldObjPasses:
		w.emit(watch.Event{
			Type:   watch.Modified,
			Object: curObj,
		})
	case curObjPasses && !oldObjPasses:
		w.emit(watch.Event{
			Type:   watch.Added,
			Object: curObj,
		})
	case !curObjPasses && oldObjPasses:
		w.emit(watch.Event{
			Type:   watch.Deleted,
			Object: oldObj,
		})
	}
	// Do nothing if neither new nor old object passed the filter.
}

func (w *etcdWatcher) sendDelete(res *etcd.Response) {
	if res.PrevNode == nil {
		utilruntime.HandleError(fmt.Errorf("unexpected nil prev node: %#v", res))
		return
	}
	if w.include != nil && !w.include(res.PrevNode.Key) {
		return
	}
	node := *res.PrevNode
	if res.Node != nil {
		// Note that this sends the *old* object with the etcd index for the time at
		// which it gets deleted. This will allow users to restart the watch at the right
		// index.
		node.ModifiedIndex = res.Node.ModifiedIndex
	}
	obj, err := w.decodeObject(&node)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failure to decode api object: %v\nfrom %#v %#v", err, res, res.Node))
		// TODO: expose an error through watch.Interface?
		// Ignore this value. If we stop the watch on a bad value, a client that uses
		// the resourceVersion to resume will never be able to get past a bad value.
		return
	}
	if !w.filter(obj) {
		return
	}
	w.emit(watch.Event{
		Type:   watch.Deleted,
		Object: obj,
	})
}

func (w *etcdWatcher) sendResult(res *etcd.Response) {
	switch res.Action {
	case EtcdCreate, EtcdGet:
		w.sendAdd(res)
	case EtcdSet, EtcdCAS:
		w.sendModify(res)
	case EtcdDelete, EtcdExpire, EtcdCAD:
		w.sendDelete(res)
	default:
		utilruntime.HandleError(fmt.Errorf("unknown action: %v", res.Action))
	}
}

// ResultChan implements watch.Interface.
func (w *etcdWatcher) ResultChan() <-chan watch.Event {
	return w.outgoing
}

// Stop implements watch.Interface.
func (w *etcdWatcher) Stop() {
	w.stopLock.Lock()
	if w.cancel != nil {
		w.cancel()
		w.cancel = nil
	}
	if !w.stopped {
		w.stopped = true
		close(w.userStop)
	}
	w.stopLock.Unlock()

	// Wait until all calls to etcd are finished and no other
	// will be issued.
	w.wg.Wait()
}
