package fscache

import (
	"os"
	"path/filepath"

	"github.com/pkg/errors"
)

// NewNaiveCacheBackend is a basic backend implementation for fscache
func NewNaiveCacheBackend(root string) Backend {
	return &naiveCacheBackend{root: root}
}

type naiveCacheBackend struct {
	root string
}

func (tcb *naiveCacheBackend) Get(id string) (string, error) {
	d := filepath.Join(tcb.root, id)
	if err := os.MkdirAll(d, 0700); err != nil {
		return "", errors.Wrapf(err, "failed to create tmp dir for %s", d)
	}
	return d, nil
}
func (tcb *naiveCacheBackend) Remove(id string) error {
	return errors.WithStack(os.RemoveAll(filepath.Join(tcb.root, id)))
}
