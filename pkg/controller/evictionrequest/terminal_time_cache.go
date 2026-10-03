/*
Copyright The Kubernetes Authors.

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

package evictionrequest

import (
	"sync"
	"time"
)

type TerminalTimeCache struct {
	lock  sync.RWMutex
	cache map[string]*time.Time
}

func NewTerminalTimeCache() *TerminalTimeCache {
	return &TerminalTimeCache{
		cache: make(map[string]*time.Time),
	}
}

func (tc *TerminalTimeCache) recordTargetTerminalTime(targetHolderKey string, terminalTime time.Time) {
	tc.lock.Lock()
	defer tc.lock.Unlock()
	tc.cache[targetHolderKey] = &terminalTime
}

func (tc *TerminalTimeCache) clearTargetTerminalTime(targetHolderKey string) {
	tc.lock.Lock()
	defer tc.lock.Unlock()
	delete(tc.cache, targetHolderKey)
}

func (tc *TerminalTimeCache) getTargetTerminalTime(targetHolderKey string) *time.Time {
	tc.lock.RLock()
	defer tc.lock.RUnlock()
	return tc.cache[targetHolderKey]
}
