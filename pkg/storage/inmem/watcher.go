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

package inmem

import (
	"k8s.io/kubernetes/pkg/storage"

	"fmt"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
	"strings"
	"sync/atomic"
)

type watcher struct {
	versioner storage.Versioner
	codec     runtime.Codec

	log        *changeLog
	resultChan chan watch.Event

	stop int32

	position   LSN
	pathPrefix string
	pred       storage.SelectionPredicate
}

var _ watch.Interface = &watcher{}

// Stops watching. Will close the channel returned by ResultChan(). Releases
// any resources used by the watch.
func (w *watcher) Stop() {
	close(w.resultChan)
	atomic.StoreInt32(&w.stop, 1)
}

// Returns a chan which will receive all the events. If an error occurs
// or Stop() is called, this channel will be closed, in which case the
// watch should be completely cleaned up.
func (w *watcher) ResultChan() <-chan watch.Event {
	return w.resultChan
}

func (w *watcher) run() {
	for {
		entry, err := w.log.read(w.position)

		if atomic.LoadInt32(&w.stop) != 0 {
			return
		}

		if err != nil {
			glog.Warningf("out of range read; will return error to watch")
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
		for _, i := range entry.items {
			if !strings.HasPrefix(i.path, w.pathPrefix) {
				continue
			}

			obj, err := w.decode(i.data, entry.lsn)
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
			match, err := w.pred.Matches(obj)
			if err != nil {
				glog.Warningf("predicate returned error: %v", err)
			}
			if !match {
				continue
			}

			glog.V(4).Infof("Sending %s for %v", i.eventType, obj)

			w.resultChan <- watch.Event{
				Type:   i.eventType,
				Object: obj,
			}
		}
		w.position++
	}
}

func (w *watcher) decode(data []byte, lsn LSN) (runtime.Object, error) {
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
