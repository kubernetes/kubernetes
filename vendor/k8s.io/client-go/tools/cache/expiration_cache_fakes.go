/*
Copyright 2014 The Kubernetes Authors.

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

import (
	"k8s.io/client-go/pkg/util/clock"
	"k8s.io/client-go/pkg/util/sets"
)

type fakeThreadSafeMap struct {
	ThreadSafeStore
	deletedKeys chan<- string
}

func (c *fakeThreadSafeMap) Delete(key string) {
	if c.deletedKeys != nil {
		c.ThreadSafeStore.Delete(key)
		c.deletedKeys <- key
	}
}

type FakeExpirationPolicy struct {
	NeverExpire     sets.String
	RetrieveKeyFunc KeyFunc
}

func (p *FakeExpirationPolicy) IsExpired(obj *timestampedEntry) bool {
	key, _ := p.RetrieveKeyFunc(obj)
	return !p.NeverExpire.Has(key)
}

func NewFakeExpirationStore(keyFunc KeyFunc, deletedKeys chan<- string, expirationPolicy ExpirationPolicy, cacheClock clock.Clock) Store {
	cacheStorage := NewThreadSafeStore(Indexers{}, Indices{})
	return &ExpirationCache{
		cacheStorage:     &fakeThreadSafeMap{cacheStorage, deletedKeys},
		keyFunc:          keyFunc,
		clock:            cacheClock,
		expirationPolicy: expirationPolicy,
	}
}
