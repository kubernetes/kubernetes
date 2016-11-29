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

package native

import (
	"k8s.io/kubernetes/pkg/storage"

	"fmt"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
	"strings"
	"sync"
	"sync/atomic"
)

type grpcWatcher struct {
	versioner storage.Versioner
	codec     runtime.Codec

	position   LSN
	pathPrefix string
	recursive  bool
	selection  storage.SelectionPredicate

	cancelFunction context.CancelFunc

	outgoing chan watch.Event
	//userStop chan struct{}
	//stopped  bool
	//stopLock sync.Mutex
	stop int32
	// wg is used to avoid calls to etcd after Stop(), and to make sure
	// that the translate goroutine is not leaked.
	wg sync.WaitGroup
}

var _ watch.Interface = &grpcWatcher{}

func newGrpcWatcher(ctx context.Context,
	client StorageServiceClient,
	startPosition LSN,
	path string,
	recursive bool,
	selection storage.SelectionPredicate,
	versioner storage.Versioner,
	codec runtime.Codec) (watch.Interface, error) {
	// TODO: etc3 code has this
	//if pred.Label.Empty() && pred.Field.Empty() {
	//	// The filter doesn't filter out any object.
	//	wc.internalFilter = nil
	//}

	pathPrefix := path
	if recursive {
		if !strings.HasSuffix(pathPrefix, "/") {
			pathPrefix += "/"
		}
	}

	watchRequest := &WatchRequest{
		Path:          pathPrefix,
		Recursive:     recursive,
		StartPosition: uint64(startPosition),
	}

	// Prevent goroutine thrashing
	bufferSize := 16

	watchContext, cancelFunction := context.WithCancel(ctx)
	watchClient, err := client.Watch(watchContext, watchRequest)
	if err != nil {
		return nil, fmt.Errorf("error starting watch: %v", err)
	}

	w := &grpcWatcher{
		outgoing:   make(chan watch.Event, bufferSize),
		position:   startPosition,
		pathPrefix: pathPrefix,
		recursive:  recursive,
		selection:  selection,
		versioner:  versioner,
		codec:      codec,

		cancelFunction: cancelFunction,
	}
	w.wg.Add(1)
	go w.run(watchClient)

	return w, nil
}

// Stops watching. Will close the channel returned by ResultChan(). Releases
// any resources used by the watch.
func (w *grpcWatcher) Stop() {
	//w.stopLock.Lock()
	//if w.cancel != nil {
	//	w.cancel()
	//	w.cancel = nil
	//}
	//if !w.stopped {
	//	w.stopped = true
	//	close(w.userStop)
	//}
	//w.stopLock.Unlock()

	atomic.StoreInt32(&w.stop, 1)

	w.cancelFunction()

	// Wait until all calls to etcd are finished and no other
	// will be issued.
	w.wg.Wait()
}

// Returns a chan which will receive all the events. If an error occurs
// or Stop() is called, this channel will be closed, in which case the
// watch should be completely cleaned up.
func (w *grpcWatcher) ResultChan() <-chan watch.Event {
	return w.outgoing
}

func (w *grpcWatcher) emit(e watch.Event) {
	//if curLen := int64(len(w.outgoing)); w.outgoingHWM.Update(curLen) {
	//	// Monitor if this gets backed up, and how much.
	//	glog.V(1).Infof("watch (%v): %v objects queued in outgoing channel.", reflect.TypeOf(e.Object).String(), curLen)
	//}
	//// Give up on user stop, without this we leak a lot of goroutines in tests.
	//select {
	//case w.outgoing <- e:
	//case <-w.userStop:
	//}

	w.outgoing <- e
}

func (w *grpcWatcher) run(watchClient StorageService_WatchClient) {
	defer w.wg.Done()

	for {
		event, err := watchClient.Recv()

		if err != nil {
			if atomic.LoadInt32(&w.stop) != 0 {
				return
			}

			glog.Warningf("error from watch receieve: %v", err)
			w.emit(watch.Event{
				Type: watch.Error,
				Object: &unversioned.Status{
					Status:  unversioned.StatusFailure,
					Message: err.Error(),
					Reason:  unversioned.StatusReasonInternalError,
				},
			})
			return
		}

		if event.Error != nil {
			glog.Warningf("error from watch: %v", event.Error)
			w.emit(watch.Event{
				Type: watch.Error,
				Object: &unversioned.Status{
					Status:  unversioned.StatusFailure,
					Message: event.Error.Message,
					Reason:  unversioned.StatusReasonInternalError,
				},
			})
			return
		}

		op := event.Op

		// Double check
		if w.recursive {
			if !strings.HasPrefix(op.Path, w.pathPrefix) {
				continue
			}
		} else {
			if op.Path != w.pathPrefix {
				continue
			}
		}

		if op.ItemData == nil || op.ItemData.Data == nil {
			glog.Fatalf("itemdata nil in %v", op)
		}

		obj, err := w.decode(op.ItemData.Data, LSN(op.ItemData.Lsn))
		if err != nil {
			glog.Warningf("error decoding change event: %v", err)
			w.emit(watch.Event{
				Type: watch.Error,
				Object: &unversioned.Status{
					Status:  unversioned.StatusFailure,
					Message: err.Error(),
					Reason:  unversioned.StatusReasonInternalError,
				},
			})
			return
		}

		// TODO: Expose these fields without a full parse?
		match, err := w.selection.Matches(obj)
		if err != nil {
			glog.Warningf("predicate returned error: %v", err)
		}
		if !match {
			continue
		}

		var eventType watch.EventType

		// TODO: Map event type
		// TODO: record if success (or maybe just record failed applies)
		switch op.OpType {
		case StorageOperationType_DELETE:
			eventType = watch.Deleted
			break

		case StorageOperationType_UPDATE:
			eventType = watch.Modified
			break

		case StorageOperationType_CREATE:
			eventType = watch.Added
			break

		default:
			glog.Warningf("error decoding change event - unknown event: %v", op.OpType)
			w.emit(watch.Event{
				Type: watch.Error,
				Object: &unversioned.Status{
					Status:  unversioned.StatusFailure,
					Message: fmt.Sprintf("Unknown event %v", op.OpType),
					Reason:  unversioned.StatusReasonInternalError,
				},
			})
			return
		}
		glog.V(4).Infof("Sending %s for %v", eventType, obj)

		w.emit(watch.Event{
			Type:   eventType,
			Object: obj,
		})
	}
}

func (w *grpcWatcher) decode(data []byte, lsn LSN) (runtime.Object, error) {
	obj, err := runtime.Decode(w.codec, []byte(data))
	if err != nil {
		return nil, err
	}
	// ensure resource version is set on the object we load from etcd
	if err := w.versioner.UpdateObject(obj, uint64(lsn)); err != nil {
		return nil, fmt.Errorf("failure to version api object (%d) %#v: %v", lsn, obj, err)
	}
	return obj, nil
}
