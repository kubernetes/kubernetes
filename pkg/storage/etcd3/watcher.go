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

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/coreos/etcd/clientv3"
	etcdrpc "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

const (
	// We have set a buffer in order to reduce times of context switches.
	incomingBufSize = 100
	outgoingBufSize = 100
)

type watcher struct {
	client    *clientv3.Client
	codec     runtime.Codec
	versioner storage.Versioner
}

// watchChan implements watch.Interface.
type watchChan struct {
	watcher           *watcher
	key               string
	initialRev        int64
	recursive         bool
	filter            storage.Filter
	ctx               context.Context
	cancel            context.CancelFunc
	incomingEventChan chan *event
	resultChan        chan watch.Event
	errChan           chan error
}

func newWatcher(client *clientv3.Client, codec runtime.Codec, versioner storage.Versioner) *watcher {
	return &watcher{
		client:    client,
		codec:     codec,
		versioner: versioner,
	}
}

// Watch watches on a key and returns a watch.Interface that transfers relevant notifications.
// If rev is zero, it will return the existing object(s) and then start watching from
// the maximum revision+1 from returned objects.
// If rev is non-zero, it will watch events happened after given revision.
// If recursive is false, it watches on given key.
// If recursive is true, it watches any children and directories under the key, excluding the root key itself.
// filter must be non-nil. Only if filter returns true will the changes be returned.
func (w *watcher) Watch(ctx context.Context, key string, rev int64, recursive bool, filter storage.Filter) (watch.Interface, error) {
	if recursive && !strings.HasSuffix(key, "/") {
		key += "/"
	}
	wc := w.createWatchChan(ctx, key, rev, recursive, filter)
	go wc.run()
	return wc, nil
}

func (w *watcher) createWatchChan(ctx context.Context, key string, rev int64, recursive bool, filter storage.Filter) *watchChan {
	wc := &watchChan{
		watcher:           w,
		key:               key,
		initialRev:        rev,
		recursive:         recursive,
		filter:            filter,
		incomingEventChan: make(chan *event, incomingBufSize),
		resultChan:        make(chan watch.Event, outgoingBufSize),
		errChan:           make(chan error, 1),
	}
	wc.ctx, wc.cancel = context.WithCancel(ctx)
	return wc
}

func (wc *watchChan) run() {
	go wc.startWatching()

	var resultChanWG sync.WaitGroup
	resultChanWG.Add(1)
	go wc.processEvent(&resultChanWG)

	select {
	case err := <-wc.errChan:
		errResult := parseError(err)
		if errResult != nil {
			// error result is guaranteed to be received by user before closing ResultChan.
			select {
			case wc.resultChan <- *errResult:
			case <-wc.ctx.Done(): // user has given up all results
			}
		}
		wc.cancel()
	case <-wc.ctx.Done():
	}
	// we need to wait until resultChan wouldn't be sent to anymore
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
func (wc *watchChan) startWatching() {
	if wc.initialRev == 0 {
		if err := wc.sync(); err != nil {
			wc.sendError(err)
			return
		}
	}
	opts := []clientv3.OpOption{clientv3.WithRev(wc.initialRev + 1)}
	if wc.recursive {
		opts = append(opts, clientv3.WithPrefix())
	}
	wch := wc.watcher.client.Watch(wc.ctx, wc.key, opts...)
	for wres := range wch {
		if wres.Err() != nil {
			// If there is an error on server (e.g. compaction), the channel will return it before closed.
			wc.sendError(wres.Err())
			return
		}
		for _, e := range wres.Events {
			wc.sendEvent(parseEvent(e))
		}
	}
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

// transform transforms an event into a result for user if not filtered.
// TODO (Optimization):
// - Save remote round-trip.
//   Currently, DELETE and PUT event don't contain the previous value.
//   We need to do another Get() in order to get previous object and have logic upon it.
//   We could potentially do some optimizations:
//   - For PUT, we can save current and previous objects into the value.
//   - For DELETE, See https://github.com/coreos/etcd/issues/4620
func (wc *watchChan) transform(e *event) (res *watch.Event) {
	curObj, oldObj, err := prepareObjs(wc.ctx, e, wc.watcher.client, wc.watcher.codec, wc.watcher.versioner)
	if err != nil {
		wc.sendError(err)
		return nil
	}

	switch {
	case e.isDeleted:
		if !wc.filter.Filter(oldObj) {
			return nil
		}
		res = &watch.Event{
			Type:   watch.Deleted,
			Object: oldObj,
		}
	case e.isCreated:
		if !wc.filter.Filter(curObj) {
			return nil
		}
		res = &watch.Event{
			Type:   watch.Added,
			Object: curObj,
		}
	default:
		curObjPasses := wc.filter.Filter(curObj)
		oldObjPasses := wc.filter.Filter(oldObj)
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
	var status *unversioned.Status
	switch {
	case err == etcdrpc.ErrCompacted:
		status = &unversioned.Status{
			Status:  unversioned.StatusFailure,
			Message: err.Error(),
			Code:    http.StatusGone,
			Reason:  unversioned.StatusReasonExpired,
		}
	default:
		status = &unversioned.Status{
			Status:  unversioned.StatusFailure,
			Message: err.Error(),
			Code:    http.StatusInternalServerError,
			Reason:  unversioned.StatusReasonInternalError,
		}
	}

	return &watch.Event{
		Type:   watch.Error,
		Object: status,
	}
}

func (wc *watchChan) sendError(err error) {
	// Context.canceled is an expected behavior.
	// We should just stop all goroutines in watchChan without returning error.
	// TODO: etcd client should return context.Canceled instead of grpc specific error.
	if grpc.Code(err) == codes.Canceled || err == context.Canceled {
		return
	}
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

func prepareObjs(ctx context.Context, e *event, client *clientv3.Client, codec runtime.Codec, versioner storage.Versioner) (curObj runtime.Object, oldObj runtime.Object, err error) {
	if !e.isDeleted {
		curObj, err = decodeObj(codec, versioner, e.value, e.rev)
		if err != nil {
			return nil, nil, err
		}
	}
	if e.isDeleted || !e.isCreated {
		getResp, err := client.Get(ctx, e.key, clientv3.WithRev(e.rev-1))
		if err != nil {
			return nil, nil, err
		}
		// Note that this sends the *old* object with the etcd revision for the time at
		// which it gets deleted.
		// We assume old object is returned only in Deleted event. Users (e.g. cacher) need
		// to have larger than previous rev to tell the ordering.
		oldObj, err = decodeObj(codec, versioner, getResp.Kvs[0].Value, e.rev)
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
