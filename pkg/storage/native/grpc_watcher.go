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
	"sync/atomic"
)

type grpcWatcher struct {
	client StorageServiceClient

	versioner storage.Versioner
	codec     runtime.Codec

	log        ReadableLog
	resultChan chan watch.Event

	stop int32

	position   LSN
	pathPrefix string
	recursive  bool
	selection  storage.SelectionPredicate
}

var _ watch.Interface = &grpcWatcher{}

func newGrpcWatcher(ctx context.Context, client StorageServiceClient, startPosition LSN, path string, recursive bool, selection storage.SelectionPredicate) (watch.Interface, error) {
	// TODO: etc3 code has this
	//if pred.Label.Empty() && pred.Field.Empty() {
	//	// The filter doesn't filter out any object.
	//	wc.internalFilter = nil
	//}

	pathPrefix := normalizePath(path)
	if recursive {
		pathPrefix += "/"
	}

	watchRequest := &WatchRequest{
		Path:          pathPrefix,
		Recursive:     recursive,
		StartPosition: uint64(startPosition),
	}

	// Prevent goroutine thrashing
	bufferSize := 16

	watchClient, err := client.Watch(ctx, watchRequest)
	if err != nil {
		return nil, fmt.Errorf("error starting watch: %v", err)
	}

	w := &grpcWatcher{
		resultChan: make(chan watch.Event, bufferSize),
		position:   startPosition,
		pathPrefix: pathPrefix,
		recursive:  recursive,

		selection: selection,
	}
	go w.run(ctx, watchClient)

	return w, nil
}

// Stops watching. Will close the channel returned by ResultChan(). Releases
// any resources used by the watch.
func (w *grpcWatcher) Stop() {
	close(w.resultChan)
	atomic.StoreInt32(&w.stop, 1)
}

// Returns a chan which will receive all the events. If an error occurs
// or Stop() is called, this channel will be closed, in which case the
// watch should be completely cleaned up.
func (w *grpcWatcher) ResultChan() <-chan watch.Event {
	return w.resultChan
}

func (w *grpcWatcher) run(ctx context.Context, watchClient StorageService_WatchClient) {
	for {
		event, err := watchClient.Recv()

		if atomic.LoadInt32(&w.stop) != 0 {
			return
		}

		if err != nil {
			glog.Warningf("error from watch receieve: %v", err)
			w.resultChan <- watch.Event{
				Type: watch.Error,
				Object: &unversioned.Status{
					Status:  unversioned.StatusFailure,
					Message: err.Error(),
					Reason:  unversioned.StatusReasonInternalError,
				},
			}
			return
		}

		if event.Error != nil {
			glog.Warningf("error from watch: %v", event.Error)
			w.resultChan <- watch.Event{
				Type: watch.Error,
				Object: &unversioned.Status{
					Status:  unversioned.StatusFailure,
					Message: event.Error.Message,
					Reason:  unversioned.StatusReasonInternalError,
				},
			}
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

		obj, err := w.decode(op.ItemData.Data, LSN(op.ItemData.Lsn))
		if err != nil {
			glog.Warningf("error decoding change event: %v", err)
			w.resultChan <- watch.Event{
				Type: watch.Error,
				Object: &unversioned.Status{
					Status:  unversioned.StatusFailure,
					Message: err.Error(),
					Reason:  unversioned.StatusReasonInternalError,
				},
			}
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
			w.resultChan <- watch.Event{
				Type: watch.Error,
				Object: &unversioned.Status{
					Status:  unversioned.StatusFailure,
					Message: fmt.Sprintf("Unknown event %v", op.OpType),
					Reason:  unversioned.StatusReasonInternalError,
				},
			}
			return
		}
		glog.V(4).Infof("Sending %s for %v", eventType, obj)

		w.resultChan <- watch.Event{
			Type:   eventType,
			Object: obj,
		}
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
