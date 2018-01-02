package store

import (
	"encoding/json"

	"github.com/boltdb/bolt"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

var volumeBucketName = []byte("volumes")

type volumeMetadata struct {
	Name    string
	Driver  string
	Labels  map[string]string
	Options map[string]string
}

func (s *VolumeStore) setMeta(name string, meta volumeMetadata) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		return setMeta(tx, name, meta)
	})
}

func setMeta(tx *bolt.Tx, name string, meta volumeMetadata) error {
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	b := tx.Bucket(volumeBucketName)
	return errors.Wrap(b.Put([]byte(name), metaJSON), "error setting volume metadata")
}

func (s *VolumeStore) getMeta(name string) (volumeMetadata, error) {
	var meta volumeMetadata
	err := s.db.View(func(tx *bolt.Tx) error {
		return getMeta(tx, name, &meta)
	})
	return meta, err
}

func getMeta(tx *bolt.Tx, name string, meta *volumeMetadata) error {
	b := tx.Bucket(volumeBucketName)
	val := b.Get([]byte(name))
	if string(val) == "" {
		return nil
	}
	if err := json.Unmarshal(val, meta); err != nil {
		return errors.Wrap(err, "error unmarshaling volume metadata")
	}
	return nil
}

func (s *VolumeStore) removeMeta(name string) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		return removeMeta(tx, name)
	})
}

func removeMeta(tx *bolt.Tx, name string) error {
	b := tx.Bucket(volumeBucketName)
	return errors.Wrap(b.Delete([]byte(name)), "error removing volume metadata")
}

// listMeta is used during restore to get the list of volume metadata
// from the on-disk database.
// Any errors that occur are only logged.
func listMeta(tx *bolt.Tx) []volumeMetadata {
	var ls []volumeMetadata
	b := tx.Bucket(volumeBucketName)
	b.ForEach(func(k, v []byte) error {
		if len(v) == 0 {
			// don't try to unmarshal an empty value
			return nil
		}

		var m volumeMetadata
		if err := json.Unmarshal(v, &m); err != nil {
			// Just log the error
			logrus.Errorf("Error while reading volume metadata for volume %q: %v", string(k), err)
			return nil
		}
		ls = append(ls, m)
		return nil
	})
	return ls
}
