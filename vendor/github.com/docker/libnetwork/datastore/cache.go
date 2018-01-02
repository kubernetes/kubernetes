package datastore

import (
	"errors"
	"fmt"
	"sync"

	"github.com/docker/libkv/store"
)

type kvMap map[string]KVObject

type cache struct {
	sync.Mutex
	kmm map[string]kvMap
	ds  *datastore
}

func newCache(ds *datastore) *cache {
	return &cache{kmm: make(map[string]kvMap), ds: ds}
}

func (c *cache) kmap(kvObject KVObject) (kvMap, error) {
	var err error

	c.Lock()
	keyPrefix := Key(kvObject.KeyPrefix()...)
	kmap, ok := c.kmm[keyPrefix]
	c.Unlock()

	if ok {
		return kmap, nil
	}

	kmap = kvMap{}

	// Bail out right away if the kvObject does not implement KVConstructor
	ctor, ok := kvObject.(KVConstructor)
	if !ok {
		return nil, errors.New("error while populating kmap, object does not implement KVConstructor interface")
	}

	kvList, err := c.ds.store.List(keyPrefix)
	if err != nil {
		if err == store.ErrKeyNotFound {
			// If the store doesn't have anything then there is nothing to
			// populate in the cache. Just bail out.
			goto out
		}

		return nil, fmt.Errorf("error while populating kmap: %v", err)
	}

	for _, kvPair := range kvList {
		// Ignore empty kvPair values
		if len(kvPair.Value) == 0 {
			continue
		}

		dstO := ctor.New()
		err = dstO.SetValue(kvPair.Value)
		if err != nil {
			return nil, err
		}

		// Make sure the object has a correct view of the DB index in
		// case we need to modify it and update the DB.
		dstO.SetIndex(kvPair.LastIndex)

		kmap[Key(dstO.Key()...)] = dstO
	}

out:
	// There may multiple go routines racing to fill the
	// cache. The one which places the kmap in c.kmm first
	// wins. The others should just use what the first populated.
	c.Lock()
	kmapNew, ok := c.kmm[keyPrefix]
	if ok {
		c.Unlock()
		return kmapNew, nil
	}

	c.kmm[keyPrefix] = kmap
	c.Unlock()

	return kmap, nil
}

func (c *cache) add(kvObject KVObject, atomic bool) error {
	kmap, err := c.kmap(kvObject)
	if err != nil {
		return err
	}

	c.Lock()
	// If atomic is true, cache needs to maintain its own index
	// for atomicity and the add needs to be atomic.
	if atomic {
		if prev, ok := kmap[Key(kvObject.Key()...)]; ok {
			if prev.Index() != kvObject.Index() {
				c.Unlock()
				return ErrKeyModified
			}
		}

		// Increment index
		index := kvObject.Index()
		index++
		kvObject.SetIndex(index)
	}

	kmap[Key(kvObject.Key()...)] = kvObject
	c.Unlock()
	return nil
}

func (c *cache) del(kvObject KVObject, atomic bool) error {
	kmap, err := c.kmap(kvObject)
	if err != nil {
		return err
	}

	c.Lock()
	// If atomic is true, cache needs to maintain its own index
	// for atomicity and del needs to be atomic.
	if atomic {
		if prev, ok := kmap[Key(kvObject.Key()...)]; ok {
			if prev.Index() != kvObject.Index() {
				c.Unlock()
				return ErrKeyModified
			}
		}
	}

	delete(kmap, Key(kvObject.Key()...))
	c.Unlock()
	return nil
}

func (c *cache) get(key string, kvObject KVObject) error {
	kmap, err := c.kmap(kvObject)
	if err != nil {
		return err
	}

	c.Lock()
	defer c.Unlock()

	o, ok := kmap[Key(kvObject.Key()...)]
	if !ok {
		return ErrKeyNotFound
	}

	ctor, ok := o.(KVConstructor)
	if !ok {
		return errors.New("kvobject does not implement KVConstructor interface. could not get object")
	}

	return ctor.CopyTo(kvObject)
}

func (c *cache) list(kvObject KVObject) ([]KVObject, error) {
	kmap, err := c.kmap(kvObject)
	if err != nil {
		return nil, err
	}

	c.Lock()
	defer c.Unlock()

	var kvol []KVObject
	for _, v := range kmap {
		kvol = append(kvol, v)
	}

	return kvol, nil
}
