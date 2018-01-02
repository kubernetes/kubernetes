//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"github.com/boltdb/bolt"
	"github.com/lpabon/godbc"
)

type DbEntry interface {
	BucketName() string
	Marshal() ([]byte, error)
	Unmarshal(buffer []byte) error
}

// Checks if the key already exists in the database.  If it does not exist,
// then it will save the key value pair in the database bucket.
func EntryRegister(tx *bolt.Tx, entry DbEntry, key string, value []byte) ([]byte, error) {
	godbc.Require(tx != nil)
	godbc.Require(len(key) > 0)

	// Access bucket
	b := tx.Bucket([]byte(entry.BucketName()))
	if b == nil {
		err := ErrDbAccess
		logger.Err(err)
		return nil, err
	}

	// Check if key exists already
	val := b.Get([]byte(key))
	if val != nil {
		return val, ErrKeyExists
	}

	// Key does not exist.  We can save it
	err := b.Put([]byte(key), value)
	if err != nil {
		logger.Err(err)
		return nil, err
	}

	return nil, nil
}

func EntryKeys(tx *bolt.Tx, bucket string) []string {
	list := make([]string, 0)

	// Get all the cluster ids from the DB
	b := tx.Bucket([]byte(bucket))
	if b == nil {
		return nil
	}

	err := b.ForEach(func(k, v []byte) error {
		list = append(list, string(k))
		return nil
	})
	if err != nil {
		return nil
	}

	return list
}

func EntrySave(tx *bolt.Tx, entry DbEntry, key string) error {
	godbc.Require(tx != nil)
	godbc.Require(len(key) > 0)

	// Access bucket
	b := tx.Bucket([]byte(entry.BucketName()))
	if b == nil {
		err := ErrDbAccess
		logger.Err(err)
		return err
	}

	// Save device entry to db
	buffer, err := entry.Marshal()
	if err != nil {
		logger.Err(err)
		return err
	}

	// Save data using the id as the key
	err = b.Put([]byte(key), buffer)
	if err != nil {
		logger.Err(err)
		return err
	}

	return nil
}

func EntryDelete(tx *bolt.Tx, entry DbEntry, key string) error {
	godbc.Require(tx != nil)
	godbc.Require(len(key) > 0)

	// Access bucket
	b := tx.Bucket([]byte(entry.BucketName()))
	if b == nil {
		err := ErrDbAccess
		logger.Err(err)
		return err
	}

	// Delete key
	err := b.Delete([]byte(key))
	if err != nil {
		logger.LogError("Unable to delete key [%v] in db: %v", key, err.Error())
		return err
	}

	return nil
}

func EntryLoad(tx *bolt.Tx, entry DbEntry, key string) error {
	godbc.Require(tx != nil)
	godbc.Require(len(key) > 0)

	b := tx.Bucket([]byte(entry.BucketName()))
	if b == nil {
		err := ErrDbAccess
		logger.Err(err)
		return err
	}

	val := b.Get([]byte(key))
	if val == nil {
		return ErrNotFound
	}

	err := entry.Unmarshal(val)
	if err != nil {
		logger.Err(err)
		return err
	}

	return nil
}
