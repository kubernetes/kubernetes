// +build linux solaris

package libcontainerd

import "sync"

type queue struct {
	sync.Mutex
	fns map[string]chan struct{}
}

func (q *queue) append(id string, f func()) {
	q.Lock()
	defer q.Unlock()

	if q.fns == nil {
		q.fns = make(map[string]chan struct{})
	}

	done := make(chan struct{})

	fn, ok := q.fns[id]
	q.fns[id] = done
	go func() {
		if ok {
			<-fn
		}
		f()
		close(done)

		q.Lock()
		if q.fns[id] == done {
			delete(q.fns, id)
		}
		q.Unlock()
	}()
}
