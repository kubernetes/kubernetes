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
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	grpccodes "google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3/metrics"
	"k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	"k8s.io/klog/v2"
)

const (
	// We have set a buffer in order to reduce times of context switches.
	incomingBufSize         = 100
	outgoingBufSize         = 100
	processEventConcurrency = 10
)

// defaultWatcherMaxLimit is used to facilitate construction tests
var defaultWatcherMaxLimit int64 = maxLimit

// fatalOnDecodeError is used during testing to panic the server if watcher encounters a decoding error
var fatalOnDecodeError = false

func init() {
	// check to see if we are running in a test environment
	TestOnlySetFatalOnDecodeError(true)
	fatalOnDecodeError, _ = strconv.ParseBool(os.Getenv("KUBE_PANIC_WATCH_DECODE_ERROR"))
}

// TestOnlySetFatalOnDecodeError should only be used for cases where decode errors are expected and need to be tested. e.g. conversion webhooks.
func TestOnlySetFatalOnDecodeError(b bool) {
	fatalOnDecodeError = b
}

type watcher struct {
	client              *clientv3.Client
	codec               runtime.Codec
	newFunc             func() runtime.Object
	objectType          string
	groupResource       schema.GroupResource
	versioner           storage.Versioner
	transformer         value.Transformer
	getCurrentStorageRV func(context.Context) (uint64, error)
	stats               *statsCache
}

// watchChan implements watch.Interface.
type watchChan struct {
	watcher           *watcher
	key               string
	initialRev        int64
	recursive         bool
	progressNotify    bool
	internalPred      storage.SelectionPredicate
	ctx               context.Context
	cancel            context.CancelFunc
	incomingEventChan chan *event
	resultChan        chan watch.Event
	stats             *statsCache
}

// Watch watches on a key and returns a watch.Interface that transfers relevant notifications.
// If rev is zero, it will return the existing object(s) and then start watching from
// the maximum revision+1 from returned objects.
// If rev is non-zero, it will watch events happened after given revision.
// If opts.Recursive is false, it watches on given key.
// If opts.Recursive is true, it watches any children and directories under the key, excluding the root key itself.
// pred must be non-nil. Only if opts.Predicate matches the change, it will be returned.
func (w *watcher) Watch(ctx context.Context, key string, rev int64, opts storage.ListOptions) (watch.Interface, error) {
	if opts.Recursive && !strings.HasSuffix(key, "/") {
		key += "/"
	}
	if opts.ProgressNotify && w.newFunc == nil {
		return nil, apierrors.NewInternalError(errors.New("progressNotify for watch is unsupported by the etcd storage because no newFunc was provided"))
	}
	startWatchRV, err := w.getStartWatchResourceVersion(ctx, rev, opts)
	if err != nil {
		return nil, err
	}
	wc := w.createWatchChan(ctx, key, startWatchRV, opts.Recursive, opts.ProgressNotify, opts.Predicate)
	go wc.run(isInitialEventsEndBookmarkRequired(opts), areInitialEventsRequired(rev, opts))

	// For etcd watch we don't have an easy way to answer whether the watch
	// has already caught up. So in the initial version (given that watchcache
	// is by default enabled for all resources but Events), we just deliver
	// the initialization signal immediately. Improving this will be explored
	// in the future.
	utilflowcontrol.WatchInitialized(ctx)

	return wc, nil
}

func (w *watcher) createWatchChan(ctx context.Context, key string, rev int64, recursive, progressNotify bool, pred storage.SelectionPredicate) *watchChan {
	wc := &watchChan{
		watcher:           w,
		key:               key,
		initialRev:        rev,
		recursive:         recursive,
		progressNotify:    progressNotify,
		internalPred:      pred,
		incomingEventChan: make(chan *event, incomingBufSize),
		resultChan:        make(chan watch.Event, outgoingBufSize),
		stats:             w.stats,
	}
	if pred.Empty() {
		// The filter doesn't filter out any object.
		wc.internalPred = storage.Everything
	}
	wc.ctx, wc.cancel = context.WithCancel(ctx)
	return wc
}

// getStartWatchResourceVersion returns a ResourceVersion
// the watch will be started from.
// Depending on the input parameters the semantics of the returned ResourceVersion are:
//   - start at Exact (return resourceVersion)
//   - start at Most Recent (return an RV from etcd)
func (w *watcher) getStartWatchResourceVersion(ctx context.Context, resourceVersion int64, opts storage.ListOptions) (int64, error) {
	if resourceVersion > 0 {
		return resourceVersion, nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.WatchList) {
		return 0, nil
	}
	if opts.SendInitialEvents == nil || *opts.SendInitialEvents {
		// note that when opts.SendInitialEvents=true
		// we will be issuing a consistent LIST request
		// against etcd followed by the special bookmark event
		return 0, nil
	}
	// at this point the clients is interested
	// only in getting a stream of events
	// starting at the MostRecent point in time (RV)
	currentStorageRV, err := w.getCurrentStorageRV(ctx)
	if err != nil {
		return 0, err
	}
	// currentStorageRV is taken from resp.Header.Revision (int64)
	// and cast to uint64, so it is safe to do reverse
	// at some point we should unify the interface but that
	// would require changing  Versioner.UpdateList
	return int64(currentStorageRV), nil
}

// isInitialEventsEndBookmarkRequired since there is no way to directly set
// opts.ProgressNotify from the API and the etcd3 impl doesn't support
// notification for external clients we simply return initialEventsEndBookmarkRequired
// to only send the bookmark event after the initial list call.
//
// see: https://github.com/kubernetes/kubernetes/issues/120348
func isInitialEventsEndBookmarkRequired(opts storage.ListOptions) bool {
	if !utilfeature.DefaultFeatureGate.Enabled(features.WatchList) {
		return false
	}
	return opts.SendInitialEvents != nil && *opts.SendInitialEvents && opts.Predicate.AllowWatchBookmarks
}

// areInitialEventsRequired returns true if all events from the etcd should be returned.
func areInitialEventsRequired(resourceVersion int64, opts storage.ListOptions) bool {
	if opts.SendInitialEvents == nil && resourceVersion == 0 {
		return true // legacy case
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.WatchList) {
		return false
	}
	return opts.SendInitialEvents != nil && *opts.SendInitialEvents
}

type etcdError interface {
	Code() grpccodes.Code
	Error() string
}

type grpcError interface {
	GRPCStatus() *grpcstatus.Status
}

func isCancelError(err error) bool {
	if err == nil {
		return false
	}
	if err == context.Canceled {
		return true
	}
	if etcdErr, ok := err.(etcdError); ok && etcdErr.Code() == grpccodes.Canceled {
		return true
	}
	if grpcErr, ok := err.(grpcError); ok && grpcErr.GRPCStatus().Code() == grpccodes.Canceled {
		return true
	}
	return false
}

func (wc *watchChan) run(initialEventsEndBookmarkRequired, forceInitialEvents bool) {
	watchClosedCh := make(chan struct{})
	var resultChanWG sync.WaitGroup

	resultChanWG.Add(1)
	go func() {
		defer resultChanWG.Done()
		wc.startWatching(watchClosedCh, initialEventsEndBookmarkRequired, forceInitialEvents)
	}()
	wc.processEvents(&resultChanWG)

	select {
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

func (wc *watchChan) RequestWatchProgress() error {
	return wc.watcher.client.RequestProgress(wc.ctx)
}

// sync tries to retrieve existing data and send them to process.
// The revision to watch will be set to the revision in response.
// All events sent will have isCreated=true
func (wc *watchChan) sync() error {
	opts := []clientv3.OpOption{}
	if wc.recursive {
		opts = append(opts, clientv3.WithLimit(defaultWatcherMaxLimit))
		rangeEnd := clientv3.GetPrefixRangeEnd(wc.key)
		opts = append(opts, clientv3.WithRange(rangeEnd))
	}

	var err error
	var lastKey []byte
	var withRev int64
	var getResp *clientv3.GetResponse

	metricsOp := "get"
	if wc.recursive {
		metricsOp = "list"
	}

	preparedKey := wc.key

	for {
		startTime := time.Now()
		getResp, err = wc.watcher.client.KV.Get(wc.ctx, preparedKey, opts...)
		metrics.RecordEtcdRequest(metricsOp, wc.watcher.groupResource, err, startTime)
		if err != nil {
			return interpretListError(err, true, preparedKey, wc.key)
		}

		if len(getResp.Kvs) == 0 && getResp.More {
			return fmt.Errorf("no results were found, but etcd indicated there were more values remaining")
		}

		// send items from the response until no more results
		for i, kv := range getResp.Kvs {
			lastKey = kv.Key
			wc.queueEvent(parseKV(kv))
			// free kv early. Long lists can take O(seconds) to decode.
			getResp.Kvs[i] = nil
		}

		if withRev == 0 {
			wc.initialRev = getResp.Header.Revision
		}

		// no more results remain
		if !getResp.More {
			return nil
		}

		preparedKey = string(lastKey) + "\x00"
		if withRev == 0 {
			withRev = getResp.Header.Revision
			opts = append(opts, clientv3.WithRev(withRev))
		}
	}
}

func logWatchChannelErr(err error) {
	switch {
	case strings.Contains(err.Error(), "mvcc: required revision has been compacted"):
		// mvcc revision compaction which is regarded as warning, not error
		klog.Warningf("watch chan error: %v", err)
	case isCancelError(err):
		// expected when watches close, no need to log
	default:
		klog.Errorf("watch chan error: %v", err)
	}
}

// startWatching does:
// - get current objects if initialRev=0; set initialRev to current rev
// - watch on given key and send events to process.
//
// initialEventsEndBookmarkSent helps us keep track
// of whether we have sent an annotated bookmark event.
//
// it's important to note that we don't
// need to track the actual RV because
// we only send the bookmark event
// after the initial list call.
//
// when this variable is set to false,
// it means we don't have any specific
// preferences for delivering bookmark events.
func (wc *watchChan) startWatching(watchClosedCh chan struct{}, initialEventsEndBookmarkRequired, forceInitialEvents bool) {
	if wc.initialRev > 0 && forceInitialEvents {
		currentStorageRV, err := wc.watcher.getCurrentStorageRV(wc.ctx)
		if err != nil {
			wc.sendError(err)
			return
		}
		if uint64(wc.initialRev) > currentStorageRV {
			wc.sendError(storage.NewTooLargeResourceVersionError(uint64(wc.initialRev), currentStorageRV, int(wait.Jitter(1*time.Second, 3).Seconds())))
			return
		}
	}
	if forceInitialEvents {
		if err := wc.sync(); err != nil {
			klog.Errorf("failed to sync with latest state: %v", err)
			wc.sendError(err)
			return
		}
	}
	if initialEventsEndBookmarkRequired {
		wc.queueEvent(func() *event {
			e := progressNotifyEvent(wc.initialRev)
			e.isInitialEventsEndBookmark = true
			return e
		}())
	}
	opts := []clientv3.OpOption{clientv3.WithRev(wc.initialRev + 1), clientv3.WithPrevKV()}
	if wc.recursive {
		opts = append(opts, clientv3.WithPrefix())
	}
	if wc.progressNotify {
		opts = append(opts, clientv3.WithProgressNotify())
	}
	wch := wc.watcher.client.Watch(wc.ctx, wc.key, opts...)
	for wres := range wch {
		if wres.Err() != nil {
			err := wres.Err()
			// If there is an error on server (e.g. compaction), the channel will return it before closed.
			logWatchChannelErr(err)
			// sendError doesn't guarantee that no more items will be put into resultChan.
			// However, by returning from startWatching here, we guarantee, that events
			// with higher resourceVersion than the error will not be queue and thus also
			// processed and send to the user.
			// TODO(wojtek-t): Figure out if we can synchronously prevent more events.
			wc.sendError(err)
			return
		}
		if wres.IsProgressNotify() {
			wc.queueEvent(progressNotifyEvent(wres.Header.GetRevision()))
			metrics.RecordEtcdBookmark(wc.watcher.groupResource)
			continue
		}

		for _, e := range wres.Events {
			if wc.stats != nil {
				switch e.Type {
				case clientv3.EventTypePut:
					wc.stats.UpdateKey(e.Kv)
				case clientv3.EventTypeDelete:
					wc.stats.DeleteKey(e.Kv)
				}
			}
			metrics.RecordEtcdEvent(wc.watcher.groupResource)
			parsedEvent, err := parseEvent(e)
			if err != nil {
				logWatchChannelErr(err)
				// sendError doesn't guarantee that no more items will be put into resultChan.
				// However, by returning from startWatching here, we guarantee, that events
				// with higher resourceVersion than the error will not be queue and thus also
				// processed and send to the user.
				// TODO(wojtek-t): Figure out if we can synchronously prevent more events.
				wc.sendError(err)
				return
			}
			wc.queueEvent(parsedEvent)
		}
	}
	// When we come to this point, it's only possible that client side ends the watch.
	// e.g. cancel the context, close the client.
	// If this watch chan is broken and context isn't cancelled, other goroutines will still hang.
	// We should notify the main thread that this goroutine has exited.
	close(watchClosedCh)
}

// processEvents processes events from etcd watcher and sends results to resultChan.
func (wc *watchChan) processEvents(wg *sync.WaitGroup) {
	if utilfeature.DefaultFeatureGate.Enabled(features.ConcurrentWatchObjectDecode) {
		wc.concurrentProcessEvents(wg)
	} else {
		wg.Add(1)
		go wc.serialProcessEvents(wg)
	}
}
func (wc *watchChan) serialProcessEvents(wg *sync.WaitGroup) {
	defer wg.Done()
	for {
		select {
		case e := <-wc.incomingEventChan:
			res, err := wc.transform(e)
			if err != nil {
				wc.sendError(err)
				return
			}

			if res == nil {
				continue
			}
			if !wc.sendEvent(res) {
				return
			}
		case <-wc.ctx.Done():
			return
		}
	}
}

func (wc *watchChan) concurrentProcessEvents(wg *sync.WaitGroup) {
	p := concurrentOrderedEventProcessing{
		wc:              wc,
		processingQueue: make(chan chan *processingResult, processEventConcurrency-1),

		objectType:    wc.watcher.objectType,
		groupResource: wc.watcher.groupResource,
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		p.scheduleEventProcessing(wc.ctx, wg)
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		p.collectEventProcessing(wc.ctx)
	}()
}

type processingResult struct {
	event *watch.Event
	err   error
}

type concurrentOrderedEventProcessing struct {
	wc *watchChan

	processingQueue chan chan *processingResult
	// Metadata for logging
	objectType    string
	groupResource schema.GroupResource
}

func (p *concurrentOrderedEventProcessing) scheduleEventProcessing(ctx context.Context, wg *sync.WaitGroup) {
	var e *event
	for {
		select {
		case <-ctx.Done():
			return
		case e = <-p.wc.incomingEventChan:
		}
		processingResponse := make(chan *processingResult, 1)
		select {
		case <-ctx.Done():
			return
		case p.processingQueue <- processingResponse:
		}
		wg.Add(1)
		go func(e *event, response chan<- *processingResult) {
			defer wg.Done()
			responseEvent, err := p.wc.transform(e)
			select {
			case <-ctx.Done():
			case response <- &processingResult{event: responseEvent, err: err}:
			}
		}(e, processingResponse)
	}
}

func (p *concurrentOrderedEventProcessing) collectEventProcessing(ctx context.Context) {
	var processingResponse chan *processingResult
	var r *processingResult
	for {
		select {
		case <-ctx.Done():
			return
		case processingResponse = <-p.processingQueue:
		}
		select {
		case <-ctx.Done():
			return
		case r = <-processingResponse:
		}
		if r.err != nil {
			p.wc.sendError(r.err)
			return
		}
		if r.event == nil {
			continue
		}
		if !p.wc.sendEvent(r.event) {
			return
		}
	}
}

func (wc *watchChan) filter(obj runtime.Object) bool {
	if wc.internalPred.Empty() {
		return true
	}
	matched, err := wc.internalPred.Matches(obj)
	return err == nil && matched
}

func (wc *watchChan) acceptAll() bool {
	return wc.internalPred.Empty()
}

// transform transforms an event into a result for user if not filtered.
func (wc *watchChan) transform(e *event) (res *watch.Event, err error) {
	curObj, oldObj, err := wc.prepareObjs(e)
	if err != nil {
		klog.Errorf("failed to prepare current and previous objects: %v", err)
		return nil, err
	}

	switch {
	case e.isProgressNotify:
		object := wc.watcher.newFunc()
		if err := wc.watcher.versioner.UpdateObject(object, uint64(e.rev)); err != nil {
			klog.Errorf("failed to propagate object version: %v", err)
			return nil, fmt.Errorf("failed to propagate object resource version: %w", err)
		}
		if e.isInitialEventsEndBookmark {
			if err := storage.AnnotateInitialEventsEndBookmark(object); err != nil {
				return nil, fmt.Errorf("error while accessing object's metadata gr: %v, type: %v, obj: %#v, err: %w", wc.watcher.groupResource, wc.watcher.objectType, object, err)
			}
		}
		res = &watch.Event{
			Type:   watch.Bookmark,
			Object: object,
		}
	case e.isDeleted:
		if !wc.filter(oldObj) {
			return nil, nil
		}
		res = &watch.Event{
			Type:   watch.Deleted,
			Object: oldObj,
		}
	case e.isCreated:
		if !wc.filter(curObj) {
			return nil, nil
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
			return res, nil
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
	return res, nil
}

func transformErrorToEvent(err error) *watch.Event {
	err = interpretWatchError(err)
	if _, ok := err.(apierrors.APIStatus); !ok {
		err = apierrors.NewInternalError(err)
	}
	status := err.(apierrors.APIStatus).Status()
	return &watch.Event{
		Type:   watch.Error,
		Object: &status,
	}
}

// sendError synchronously puts an error event into resultChan and
// trigger cancelling all goroutines.
func (wc *watchChan) sendError(err error) {
	// We use wc.ctx to reap all goroutines. Under whatever condition, we should stop them all.
	// It's fine to double cancel.
	defer wc.cancel()

	if isCancelError(err) {
		return
	}
	errResult := transformErrorToEvent(err)
	if errResult != nil {
		// error result is guaranteed to be received by user before closing ResultChan.
		select {
		case wc.resultChan <- *errResult:
		case <-wc.ctx.Done(): // user has given up all results
		}
	}
}

// sendEvent synchronously puts an event into resultChan.
// Returns true if it was successful.
func (wc *watchChan) sendEvent(event *watch.Event) bool {
	if len(wc.resultChan) == cap(wc.resultChan) {
		klog.V(3).InfoS("Fast watcher, slow processing. Probably caused by slow dispatching events to watchers", "outgoingEvents", outgoingBufSize, "objectType", wc.watcher.objectType, "groupResource", wc.watcher.groupResource)
	}
	// If user couldn't receive results fast enough, we also block incoming events from watcher.
	// Because storing events in local will cause more memory usage.
	// The worst case would be closing the fast watcher.
	select {
	case wc.resultChan <- *event:
		return true
	case <-wc.ctx.Done():
		return false
	}
}

func (wc *watchChan) queueEvent(e *event) {
	if len(wc.incomingEventChan) == incomingBufSize {
		klog.V(3).InfoS("Fast watcher, slow processing. Probably caused by slow decoding, user not receiving fast, or other processing logic", "incomingEvents", incomingBufSize, "objectType", wc.watcher.objectType, "groupResource", wc.watcher.groupResource)
	}
	select {
	case wc.incomingEventChan <- e:
	case <-wc.ctx.Done():
	}
}

func (wc *watchChan) prepareObjs(e *event) (curObj runtime.Object, oldObj runtime.Object, err error) {
	if e.isProgressNotify {
		// progressNotify events doesn't contain neither current nor previous object version,
		return nil, nil, nil
	}

	if !e.isDeleted {
		data, _, err := wc.watcher.transformer.TransformFromStorage(wc.ctx, e.value, authenticatedDataString(e.key))
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
		data, _, err := wc.watcher.transformer.TransformFromStorage(wc.ctx, e.prevValue, authenticatedDataString(e.key))
		if err != nil {
			return nil, nil, wc.watcher.transformIfCorruptObjectError(e, err)
		}
		// Note that this sends the *old* object with the etcd revision for the time at
		// which it gets deleted.
		oldObj, err = decodeObj(wc.watcher.codec, wc.watcher.versioner, data, e.rev)
		if err != nil {
			return nil, nil, wc.watcher.transformIfCorruptObjectError(e, err)
		}
	}
	return curObj, oldObj, nil
}

type corruptObjectDeletedError struct {
	err error
}

func (e *corruptObjectDeletedError) Error() string {
	return fmt.Sprintf("saw a DELETED event, but object data is corrupt - %v", e.err)
}
func (e *corruptObjectDeletedError) Unwrap() error { return e.err }

func (w *watcher) transformIfCorruptObjectError(e *event, err error) error {
	var corruptObjErr *corruptObjectError
	if !e.isDeleted || !errors.As(err, &corruptObjErr) {
		return err
	}

	// if we are here it means we received a DELETED event but the object
	// associated with it is corrupt because we failed to transform or
	// decode the data associated with the object.
	// wrap the original error so we can send a proper watch Error event.
	return &corruptObjectDeletedError{err: corruptObjErr}
}

func decodeObj(codec runtime.Codec, versioner storage.Versioner, data []byte, rev int64) (_ runtime.Object, err error) {
	obj, err := runtime.Decode(codec, []byte(data))
	if err != nil {
		if fatalOnDecodeError {
			// we are running in a test environment and thus an
			// error here is due to a coder mistake if the defer
			// does not catch it
			panic(err)
		}
		return nil, err
	}
	// ensure resource version is set on the object we load from etcd
	if err := versioner.UpdateObject(obj, uint64(rev)); err != nil {
		return nil, fmt.Errorf("failure to version api object (%d) %#v: %v", rev, obj, err)
	}
	return obj, nil
}
