package load

import (
	"sync"

	"golang.org/x/tools/go/packages"
)

type Guard struct {
	loadMutexes map[*packages.Package]*sync.Mutex
	mutex       sync.Mutex
}

func NewGuard() *Guard {
	return &Guard{
		loadMutexes: map[*packages.Package]*sync.Mutex{},
	}
}

func (g *Guard) AddMutexForPkg(pkg *packages.Package) {
	g.loadMutexes[pkg] = &sync.Mutex{}
}

func (g *Guard) MutexForPkg(pkg *packages.Package) *sync.Mutex {
	return g.loadMutexes[pkg]
}

func (g *Guard) Mutex() *sync.Mutex {
	return &g.mutex
}
