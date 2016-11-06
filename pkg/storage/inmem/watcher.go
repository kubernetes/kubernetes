package inmem

import (
	"k8s.io/kubernetes/pkg/storage"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/watch"
	"strings"
	"sync/atomic"
)

type watcher struct {
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

			match, err := w.pred.Matches(i.object)
			if err != nil {
				glog.Warningf("predicate returned error: %v", err)
			}
			if !match {
				continue
			}

			w.resultChan <- watch.Event{
				Type:   i.eventType,
				Object: i.object,
			}
		}
		w.position++
	}
}
