/*
Copyright 2024 The Kubernetes Authors.

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

type consistencyStore struct {
	destinationStore Store
	backingStore     Store
}

var _ Store = &consistencyStore{}

func newConsistencyStore(destinationStore, backingStore Store) *consistencyStore {
	return &consistencyStore{destinationStore: destinationStore, backingStore: backingStore}
}

func (c *consistencyStore) Add(obj interface{}) error {
	//TODO lock ?
	if err := c.destinationStore.Add(obj); err != nil {
		return err
	}
	return c.backingStore.Add(obj)
}

func (c *consistencyStore) Update(obj interface{}) error {
	if err := c.destinationStore.Update(obj); err != nil {
		return err
	}
	return c.backingStore.Update(obj)
}

func (c *consistencyStore) Delete(obj interface{}) error {
	if err := c.destinationStore.Delete(obj); err != nil {
		return err
	}
	return c.backingStore.Delete(obj)
}

func (c *consistencyStore) List() []interface{} {
	return c.destinationStore.List()
}

func (c *consistencyStore) ListKeys() []string {
	return c.destinationStore.ListKeys()
}

func (c *consistencyStore) Get(obj interface{}) (item interface{}, exists bool, err error) {
	return c.destinationStore.Get(obj)
}

func (c *consistencyStore) GetByKey(key string) (item interface{}, exists bool, err error) {
	return c.destinationStore.GetByKey(key)
}

func (c *consistencyStore) Replace(items []interface{}, rv string) error {
	if err := c.destinationStore.Replace(items, rv); err != nil {
		return err
	}
	return c.backingStore.Replace(items, rv)
}

func (c *consistencyStore) Resync() error {
	return c.destinationStore.Resync()
}
