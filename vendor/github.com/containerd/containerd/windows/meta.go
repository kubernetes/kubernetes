// +build windows

package windows

// TODO: remove this file (i.e. meta.go) once we have a snapshotter

import (
	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/errdefs"
	"github.com/pkg/errors"
)

func newLayerFolderStore(tx *bolt.Tx) *layerFolderStore {
	return &layerFolderStore{tx}
}

type layerFolderStore struct {
	tx *bolt.Tx
}

func (s *layerFolderStore) Create(id, layer string) error {
	bkt, err := s.tx.CreateBucketIfNotExists([]byte(pluginID))
	if err != nil {
		return errors.Wrapf(err, "failed to create bucket %s", pluginID)
	}
	err = bkt.Put([]byte(id), []byte(layer))
	if err != nil {
		return errors.Wrapf(err, "failed to store entry %s:%s", id, layer)
	}

	return nil
}

func (s *layerFolderStore) Get(id string) (string, error) {
	bkt := s.tx.Bucket([]byte(pluginID))
	if bkt == nil {
		return "", errors.Wrapf(errdefs.ErrNotFound, "bucket %s", pluginID)
	}

	return string(bkt.Get([]byte(id))), nil
}

func (s *layerFolderStore) Delete(id string) error {
	bkt := s.tx.Bucket([]byte(pluginID))
	if bkt == nil {
		return errors.Wrapf(errdefs.ErrNotFound, "bucket %s", pluginID)
	}

	if err := bkt.Delete([]byte(id)); err != nil {
		return errors.Wrapf(err, "failed to delete entry %s", id)
	}

	return nil
}
