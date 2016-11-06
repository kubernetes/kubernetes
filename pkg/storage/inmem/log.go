package inmem

import (
	"fmt"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/watch"
	"sync"
)

type changeLog struct {
	mutex  sync.Mutex
	cond   *sync.Cond
	minLsn LSN
	maxLsn LSN
	log    map[LSN]*logEntry
}

type logEntry struct {
	lsn   LSN
	items []logItem
}

type logItem struct {
	path      string
	eventType watch.EventType
	data      []byte
}

func (i logItem) String() string {
	return fmt.Sprintf("logItem [path=%s, eventType=%s]", i.path, i.eventType)
}

func newChangeLog() *changeLog {
	l := &changeLog{
		log: make(map[LSN]*logEntry),
	}

	l.cond = sync.NewCond(&l.mutex)

	return l
}

func (l *changeLog) append(e *logEntry) {
	glog.Infof("appending logentry at %d: %v", e.lsn, e.items)
	l.mutex.Lock()
	defer l.mutex.Unlock()

	lsn := e.lsn
	if lsn <= l.maxLsn {
		panic("out of order LSN appends")
	}
	l.maxLsn = lsn
	if l.minLsn == 0 {
		l.minLsn = lsn
	}
	l.log[lsn] = e

	l.cond.Broadcast()
}

func (l *changeLog) read(pos LSN) (*logEntry, error) {
	l.mutex.Lock()

	for {
		e := l.log[pos]
		if e != nil {
			l.mutex.Unlock()
			return e, nil
		}

		if l.minLsn > pos {
			l.mutex.Unlock()
			return nil, fmt.Errorf("out of range read at position=%d min=%d", pos, l.minLsn)
		}

		l.cond.Wait()
	}
}

func (l *changeLog) newWatcher(s *store, startPosition LSN, predicate storage.SelectionPredicate, path string, recursive bool) (watch.Interface, error) {
	// TODO: etc3 code has this
	//if pred.Label.Empty() && pred.Field.Empty() {
	//	// The filter doesn't filter out any object.
	//	wc.internalFilter = nil
	//}

	pathPrefix := normalizePath(path)
	if recursive {
		pathPrefix += "/"
	}

	// Prevent goroutine thrashing
	bufferSize := 16

	w := &watcher{
		versioner: s.versioner,
		codec:     s.codec,

		log:        l,
		resultChan: make(chan watch.Event, bufferSize),
		position:   startPosition,
		pred:       predicate,
		pathPrefix: pathPrefix,
	}
	w.run()

	return w, nil
}
