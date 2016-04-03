package mech

import (
	"fmt"
	"sync"

	log "github.com/golang/glog"
)

var (
	mechLock       sync.Mutex
	supportedMechs = make(map[string]Factory)
)

func Register(name string, f Factory) error {
	mechLock.Lock()
	defer mechLock.Unlock()

	if _, found := supportedMechs[name]; found {
		return fmt.Errorf("Mechanism registered twice: %s", name)
	}
	supportedMechs[name] = f
	log.V(1).Infof("Registered mechanism %s", name)
	return nil
}

func ListSupported() (list []string) {
	mechLock.Lock()
	defer mechLock.Unlock()

	for mechname := range supportedMechs {
		list = append(list, mechname)
	}
	return list
}

func SelectSupported(mechanisms []string) (selectedMech string, factory Factory) {
	mechLock.Lock()
	defer mechLock.Unlock()

	for _, m := range mechanisms {
		if f, ok := supportedMechs[m]; ok {
			selectedMech = m
			factory = f
			break
		}
	}
	return
}
