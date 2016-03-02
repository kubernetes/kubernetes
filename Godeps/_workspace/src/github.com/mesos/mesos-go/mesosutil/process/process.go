package process

import (
	"fmt"
	"sync"
)

var (
	pidLock sync.Mutex
	pid     uint64
)

func nextPid() uint64 {
	pidLock.Lock()
	defer pidLock.Unlock()
	pid++
	return pid
}

//TODO(jdef) add lifecycle funcs
//TODO(jdef) add messaging funcs
type Process struct {
	label string
}

func New(kind string) *Process {
	return &Process{
		label: fmt.Sprintf("%s(%d)", kind, nextPid()),
	}
}

func (p *Process) Label() string {
	return p.label
}
