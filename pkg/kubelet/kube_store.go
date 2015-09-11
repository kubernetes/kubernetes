/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kubelet

import (
	"fmt"
	"path"
	"time"

	"github.com/boltdb/bolt"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

type kubeStore struct {
	fileName string
	db       *bolt.DB
}

type ResultValue struct {
	k string
	v interface{}
}

func newKubeStore() *kubeStore {
	return &kubeStore{}
}

func (s *kubeStore) Start(dirName string, name string) error {
	var err error
	s.db, err = bolt.Open(path.Join(dirName, name), 0600, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		glog.Errorf("Failed to open %s %v", path.Join(dirName, name), err)
		return err
	}

	return nil
}

func (s *kubeStore) Close() {
	if s.db != nil {
		s.db.Close()
	}
}

func (s *kubeStore) LoadMap(bucketName string, ch chan ResultValue, ivalue interface{}) {
	defer close(ch)

	if s.db == nil {
		glog.Infof("kube store is not ready")
		return
	}

	// Begin Tx
	err := s.db.Update(func(tx *bolt.Tx) error {
		_, bErr := tx.CreateBucketIfNotExists([]byte(bucketName))
		return bErr
	})
	// End Tx

	if err != nil {
		glog.Warningf(err.Error())
		return
	}

	// Begin Tx
	err = s.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		if b == nil {
			return fmt.Errorf("Bucket %q missing", bucketName)
		}

		c := b.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			if v == nil {
				continue
			}
			keyCopy := append([]byte(nil), k...)
			valueCopy := append([]byte(nil), v...)
			yErr := yaml.Unmarshal(valueCopy, ivalue)
			if yErr != nil {
				glog.Warningf("unable to UnMarshall %v=%v %v", keyCopy, valueCopy, yErr)
				continue
			}
			ch <- ResultValue{k: string(keyCopy), v: ivalue}
		}
		return nil
	})
	// End of Tx

	if err != nil {
		glog.Warning(err)
	}
}

func (s *kubeStore) SaveEntry(bucketName string, key string, o interface{}) error {
	if s.db == nil {
		return fmt.Errorf("kubestore not ready")
	}

	y, err := yaml.Marshal(o)
	if err != nil {
		return fmt.Errorf("unable to marshal object %+v", o)
	}

	// Begin Tx
	err = s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		if b == nil {
			_, bErr := tx.CreateBucketIfNotExists([]byte(bucketName))
			return bErr
		}

		pErr := b.Put([]byte(key), y)
		if pErr != nil {
			return pErr
		}
		return nil
	})
	// End Tx

	return err
}

func (s *kubeStore) DeleteEntry(bucketName string, key string) error {
	if s.db == nil {
		return fmt.Errorf("kubestore not ready")
	}

	// Begin Tx
	err := s.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(bucketName))
		if b == nil {
			_, bErr := tx.CreateBucketIfNotExists([]byte(bucketName))
			return bErr
		}

		pErr := b.Delete([]byte(key))
		if pErr != nil {
			return pErr
		}
		return nil
	})
	// End Tx

	return err
}

func createMyBucket(db *bolt.DB, bucketName string) error {
	err := db.Update(func(tx *bolt.Tx) error {
		_, txErr := tx.CreateBucketIfNotExists([]byte(bucketName))
		return txErr
	})
	return err
}
