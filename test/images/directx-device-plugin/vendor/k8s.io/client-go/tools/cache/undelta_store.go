/*
Copyright 2015 The Kubernetes Authors.

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

package cache

// UndeltaStore listens to incremental updates and sends complete state on every change.
// It implements the Store interface so that it can receive a stream of mirrored objects
// from Reflector.  Whenever it receives any complete (Store.Replace) or incremental change
// (Store.Add, Store.Update, Store.Delete), it sends the complete state by calling PushFunc.
// It is thread-safe.  It guarantees that every change (Add, Update, Replace, Delete) results
// in one call to PushFunc, but sometimes PushFunc may be called twice with the same values.
// PushFunc should be thread safe.
type UndeltaStore struct {
	Store
	PushFunc func([]interface{})
}

// Assert that it implements the Store interface.
var _ Store = &UndeltaStore{}

// Note about thread safety.  The Store implementation (cache.cache) uses a lock for all methods.
// In the functions below, the lock gets released and reacquired betweend the {Add,Delete,etc}
// and the List.  So, the following can happen, resulting in two identical calls to PushFunc.
// time            thread 1                  thread 2
// 0               UndeltaStore.Add(a)
// 1                                         UndeltaStore.Add(b)
// 2               Store.Add(a)
// 3                                         Store.Add(b)
// 4               Store.List() -> [a,b]
// 5                                         Store.List() -> [a,b]

func (u *UndeltaStore) Add(obj interface{}) error {
	if err := u.Store.Add(obj); err != nil {
		return err
	}
	u.PushFunc(u.Store.List())
	return nil
}

func (u *UndeltaStore) Update(obj interface{}) error {
	if err := u.Store.Update(obj); err != nil {
		return err
	}
	u.PushFunc(u.Store.List())
	return nil
}

func (u *UndeltaStore) Delete(obj interface{}) error {
	if err := u.Store.Delete(obj); err != nil {
		return err
	}
	u.PushFunc(u.Store.List())
	return nil
}

func (u *UndeltaStore) Replace(list []interface{}, resourceVersion string) error {
	if err := u.Store.Replace(list, resourceVersion); err != nil {
		return err
	}
	u.PushFunc(u.Store.List())
	return nil
}

// NewUndeltaStore returns an UndeltaStore implemented with a Store.
func NewUndeltaStore(pushFunc func([]interface{}), keyFunc KeyFunc) *UndeltaStore {
	return &UndeltaStore{
		Store:    NewStore(keyFunc),
		PushFunc: pushFunc,
	}
}
