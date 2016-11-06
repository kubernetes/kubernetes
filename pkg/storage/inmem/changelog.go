package inmem

import (

	"fmt"

	"k8s.io/kubernetes/pkg/runtime"

	"sync"
	"k8s.io/kubernetes/pkg/watch"
)

type changeLog struct {
	mutex sync.RWMutex
	cond  sync.Cond
	minLsn LSN
	maxLsn LSN
	log    map[LSN]*logEntry
}


type logEntry struct {
	lsn   LSN
	items []logItem
}

type logItem struct {
	path string
	eventType watch.EventType
	data []byte

	object runtime.Object
}


func newChangeLog() (*changeLog) {
	l := &changeLog{
		log: make(map[LSN]*logEntry),
	}

	l.cond = sync.NewCond(&l.mutex)

	return l
}

func (l*changeLog) append(e *logEntry) {
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

func (l*changeLog) read(pos LSN) (*logEntry, error) {
	l.mutex.RLock()

	for {
		e := l.log[pos]
		if e != nil {
			l.mutex.RUnlock()
			return e, nil
		}

		if l.minLsn > pos {
			l.mutex.RUnlock()
			return nil, fmt.Errorf("out of range read at position %d", pos)
		}

		l.cond.Wait()
	}
}

func (e*logEntry) addItem(s* store, path string, eventType watch.EventType,  data []byte) {
	// TODO: Is it safe to reuse the event object across events?
	// TODO: Could we also return this object (is it the same type?)
	// TODO: Could we use obj?
	obj, err := s.decodeForWatch(data, e.lsn)
	if err != nil {
		panic(fmt.Errorf("error decoding object: %v", err))
	}

	e.items = append(e.items, logItem{
		path: path,
		data: data,
		object: obj,
		eventType: eventType,
	})
}

