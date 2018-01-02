// +build windows

package windows

import (
	"errors"
	"sync"
)

type pidPool struct {
	sync.Mutex
	pool map[uint32]struct{}
	cur  uint32
}

func newPidPool() *pidPool {
	return &pidPool{
		pool: make(map[uint32]struct{}),
	}
}

func (p *pidPool) Get() (uint32, error) {
	p.Lock()
	defer p.Unlock()

	pid := p.cur + 1
	for pid != p.cur {
		// 0 is reserved and invalid
		if pid == 0 {
			pid = 1
		}
		if _, ok := p.pool[pid]; !ok {
			p.cur = pid
			p.pool[pid] = struct{}{}
			return pid, nil
		}
		pid++
	}

	return 0, errors.New("pid pool exhausted")
}

func (p *pidPool) Put(pid uint32) {
	p.Lock()
	delete(p.pool, pid)
	p.Unlock()
}
