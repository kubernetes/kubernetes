package key

import (
	"errors"
	"sync"
)

var ErrorNoKeys = errors.New("no keys found")

type WritableKeySetRepo interface {
	Set(KeySet) error
}

type ReadableKeySetRepo interface {
	Get() (KeySet, error)
}

type PrivateKeySetRepo interface {
	WritableKeySetRepo
	ReadableKeySetRepo
}

func NewPrivateKeySetRepo() PrivateKeySetRepo {
	return &memPrivateKeySetRepo{}
}

type memPrivateKeySetRepo struct {
	mu  sync.RWMutex
	pks PrivateKeySet
}

func (r *memPrivateKeySetRepo) Set(ks KeySet) error {
	pks, ok := ks.(*PrivateKeySet)
	if !ok {
		return errors.New("unable to cast to PrivateKeySet")
	} else if pks == nil {
		return errors.New("nil KeySet")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.pks = *pks
	return nil
}

func (r *memPrivateKeySetRepo) Get() (KeySet, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.pks.keys == nil {
		return nil, ErrorNoKeys
	}
	return KeySet(&r.pks), nil
}
