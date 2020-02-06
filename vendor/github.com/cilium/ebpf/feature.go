package ebpf

import (
	"sync"
)

type featureTest struct {
	Fn func() bool

	once   sync.Once
	result bool
}

func (ft *featureTest) Result() bool {
	ft.once.Do(func() {
		ft.result = ft.Fn()
	})
	return ft.result
}
