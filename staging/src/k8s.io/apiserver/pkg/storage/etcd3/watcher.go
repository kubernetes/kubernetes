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
	"fmt"
	"net/http"
	"strings"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"

	"github.com/coreos/etcd/clientv3"
	etcdrpc "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

const (
	// We have set a buffer in order to reduce times of context switches.
	incomingBufSize = 100
	outgoingBufSize = 100
)

type watcher struct {
	client      *clientv3.Client
	codec       runtime.Codec
	versioner   storage.Versioner
	transformer ValueTransformer
}

// watchChan implements watch.Interface.
type watchChan struct {
	watcher           *watcher
	key               string
	initialRev        int64
	recursive         bool
	internalFilter    storage.FilterFunc
	ctx               context.Context
	cancel            context.CancelFunc
	incomingEventChan chan *event
	resultChan        chan watch.Event
	errChan           chan error
}

func newWatcher(client *clientv3.Client, codec runtime.Codec, versioner storage.Versioner, transformer ValueTransformer) *watcher {
	return &watcher{
		client:      client,
		codec:       codec,
		versioner:   versioner,
		transformer: transformer,
	}
}

// Watch watches on a key and returns a watch.Interface that transfers relevant notifications.
// If rev is zero, it will return the existing object(s) and then start watching from
// the maximum revision+1 from returned objects.
// If rev is non-zero, it will watch events happened after given revision.
// If recursive is false, it watches on given key.
// If recursive is true, it watches any children and directories under the key, excluding the root key itself.
// pred must be non-nil. Only if pred matches the change, it will be returned.
func (w *watcher) Watch(ctx context.Context, key string, rev int64, recursive bool, pred storage.SelectionPredicate) (watch.Interface, error) {
	if recursive && !strings.HasSuffix(key, "/") {
		key += "/"
	}
	wc := w.createWatchChan(ctx, key, rev, recursive, pred)
	go wc.run()
	return wc, nil
}

func (w *watcher) createWatchChan(ctx context.Context, key string, rev int64, recursive bool, pred storage.SelectionPredicate) *watchChan {
	wc := &watchChan{
		watcher:           w,
		key:               key,
		initialRev:        rev,
		recursive:         recursive,
		internalFilter:    storage.SimpleFilter(pred),
		incomingEventChan: make(chan *event, incomingBufSize),
		resultChan:        make(chan watch.Event, outgoingBufSize),
		errChan:           make(chan error, 1),
	}
	if pred.Label.Empty() && pred.Field.Empty() {
		// The filter doesn't filter out any object.
		wc.internalFilter = nil
	}
	wc.ctx, wc.cancel = context.WithCancel(ctx)
	return wc
}

func (wc *watchChan) run() {
	watchClosedCh := make(chan struct{})
	go wc.startWatching(watchClosedCh)

	var resultChanWG sync.WaitGroup
	resultChanWG.Add(1)
	go wc.processEvent(&resultChanWG)

	select {
	case err := <-wc.errChan:
		if err == context.Canceled {
			break
		}
		errResult := parseError(err)
		if errResult != nil {
			// error result is guaranteed to be received by user before closing ResultChan.
			select {
			case wc.resultChan <- *errResult:
			case <-wc.ctx.Done(): // user has given up all results
			}
		}
	case <-watchClosedCh:
	case <-wc.ctx.Done(): // user cancel
	}

	// We use wc.ctx to reap all goroutines. Under whatever condition, we should stop them all.
	// It's fine to double cancel.
	wc.cancel()

	// we need to wait until resultChan wouldn't be used anymore
	resultChanWG.Wait()
	close(wc.resultChan)
}

func (wc *watchChan) Stop() {
	wc.cancel()
}

func (wc *watchChan) ResultChan() <-chan watch.Event {
	return wc.resultChan
}

// sync tries to retrieve existing data and send them to process.
// The revision to watch will be set to the revision in response.
// All events sent will have isCreated=true
func (wc *watchChan) sync() error {
	opts := []clientv3.OpOption{}
	if wc.recursive {
		opts = append(opts, clientv3.WithPrefix())
	}
	getResp, err := wc.watcher.client.Get(wc.ctx, wc.key, opts...)
	if err != nil {
		return err
	}
	wc.initialRev = getResp.Header.Revision
	for _, kv := range getResp.Kvs {
		wc.sendEvent(parseKV(kv))
	}
	return nil
}

// startWatching does:
// - get current objects if initialRev=0; set initialRev to current rev
// - watch on given key and send events to process.
func (wc *watchChan) startWatching(watchClosedCh chan struct{}) {
	if wc.initialRev == 0 {
		if err := wc.sync(); err != nil {
			glog.Errorf("failed to sync with latest state: %v", err)
			wc.sendError(err)
			return
		}
	}
	opts := []clientv3.OpOption{clientv3.WithRev(wc.initialRev + 1), clientv3.WithPrevKV()}
	if wc.recursive {
		opts = append(opts, clientv3.WithPrefix())
	}
	wch := wc.watcher.client.Watch(wc.ctx, wc.key, opts...)
	for wres := range wch {
		if wres.Err() != nil {
			err := wres.Err()
			// If there is an error on server (e.g. compaction), the channel will return it before closed.
			glog.Errorf("watch chan error: %v", err)
			wc.sendError(err)
			return
		}
		for _, e := range wres.Events {
			wc.sendEvent(parseEvent(e))
		}
	}
	// When we come to this point, it's only possible that client side ends the watch.
	// e.g. cancel the context, close the client.
	// If this watch chan is broken and context isn't cancelled, other goroutines will still hang.
	// We should notify the main thread that this goroutine has exited.
	close(watchClosedCh)
}

// processEvent processes events from etcd watcher and sends results to resultChan.
func (wc *watchChan) processEvent(wg *sync.WaitGroup) {
	defer wg.Done()

	for {
		select {
		case e := <-wc.incomingEventChan:
			res := wc.transform(e)
			if res == nil {
				continue
			}
			if len(wc.resultChan) == outgoingBufSize {
				glog.Warningf("Fast watcher, slow processing. Number of buffered events: %d."+
					"Probably caused by slow dispatching events to watchers", outgoingBufSize)
			}
			// If user couldn't receive results fast enough, we also block incoming events from watcher.
			// Because storing events in local will cause more memory usage.
			// The worst case would be closing the fast watcher.
			select {
			case wc.resultChan <- *res:
			case <-wc.ctx.Done():
				return
			}
		case <-wc.ctx.Done():
			return
		}
	}
}

func (wc *watchChan) filter(obj runtime.Object) bool {
	if wc.internalFilter == nil {
		return true
	}
	return wc.internalFilter(obj)
}

func (wc *watchChan) acceptAll() bool {
	return wc.internalFilter == nil
}

// transform transforms an event into a result for user if not filtered.
func (wc *watchChan) transform(e *event) (res *watch.Event) {
	curObj, oldObj, err := wc.prepareObjs(e)
	if err != nil {
		glog.Errorf("failed to prepare current and previous objects: %v", err)
		wc.sendError(err)
		return nil
	}

	switch {
	case e.isDeleted:
		if !wc.filter(oldObj) {
			return nil
		}
		res = &watch.Event{
			Type:   watch.Deleted,
			Object: oldObj,
		}
	case e.isCreated:
		if !wc.filter(curObj) {
			return nil
		}
		res = &watch.Event{
			Type:   watch.Added,
			Object: curObj,
		}
	default:
		if wc.acceptAll() {
			res = &watch.Event{
				Type:   watch.Modified,
				Object: curObj,
			}
			return res
		}
		curObjPasses := wc.filter(curObj)
		oldObjPasses := wc.filter(oldObj)
		switch {
		case curObjPasses && oldObjPasses:
			res = &watch.Event{
				Type:   watch.Modified,
				Object: curObj,
			}
		case curObjPasses && !oldObjPasses:
			res = &watch.Event{
				Type:   watch.Added,
				Object: curObj,
			}
		case !curObjPasses && oldObjPasses:
			res = &watch.Event{
				Type:   watch.Deleted,
				Object: oldObj,
			}
		}
	}
	return res
}

func parseError(err error) *watch.Event {
	var status *metav1.Status
	switch {
	case err == etcdrpc.ErrCompacted:
		status = &metav1.Status{
			Status:  metav1.StatusFailure,
			Message: err.Error(),
			Code:    http.StatusGone,
			Reason:  metav1.StatusReasonExpired,
		}
	default:
		status = &metav1.Status{
			Status:  metav1.StatusFailure,
			Message: err.Error(),
			Code:    http.StatusInternalServerError,
			Reason:  metav1.StatusReasonInternalError,
		}
	}

	return &watch.Event{
		Type:   watch.Error,
		Object: status,
	}
}

func (wc *watchChan) sendError(err error) {
	select {
	case wc.errChan <- err:
	case <-wc.ctx.Done():
	}
}

func (wc *watchChan) sendEvent(e *event) {
	if len(wc.incomingEventChan) == incomingBufSize {
		glog.Warningf("Fast watcher, slow processing. Number of buffered events: %d."+
			"Probably caused by slow decoding, user not receiving fast, or other processing logic",
			incomingBufSize)
	}
	select {
	case wc.incomingEventChan <- e:
	case <-wc.ctx.Done():
	}
}

func (wc *watchChan) prepareObjs(e *event) (curObj runtime.Object, oldObj runtime.Object, err error) {
	if !e.isDeleted {
		data, _, err := wc.watcher.transformer.TransformFromStorage(e.value)
		if err != nil {
			return nil, nil, err
		}
		curObj, err = decodeObj(wc.watcher.codec, wc.watcher.versioner, data, e.rev)
		if err != nil {
			return nil, nil, err
		}
	}
	// We need to decode prevValue, only if this is deletion event or
	// the underlying filter doesn't accept all objects (otherwise we
	// know that the filter for previous object will return true and
	// we need the object only to compute whether it was filtered out
	// before).
	if len(e.prevValue) > 0 && (e.isDeleted || !wc.acceptAll()) {
		data, _, err := wc.watcher.transformer.TransformFromStorage(e.prevValue)
		if err != nil {
			return nil, nil, err
		}
		// Note that this sends the *old* object with the etcd revision for the time at
		// which it gets deleted.
		oldObj, err = decodeObj(wc.watcher.codec, wc.watcher.versioner, data, e.rev)
		if err != nil {
			return nil, nil, err
		}
	}
	return curObj, oldObj, nil
}

func decodeObj(codec runtime.Codec, versioner storage.Versioner, data []byte, rev int64) (runtime.Object, error) {
	obj, err := runtime.Decode(codec, []byte(data))
	if err != nil {
		return nil, err
	}
	// ensure resource version is set on the object we load from etcd
	if err := versioner.UpdateObject(obj, uint64(rev)); err != nil {
		return nil, fmt.Errorf("failure to version api object (%d) %#v: %v", rev, obj, err)
	}
	return obj, nil
}
