// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package jsonclient

import (
	"sync"
	"time"
)

type backoff struct {
	mu         sync.RWMutex
	multiplier uint
	notBefore  time.Time
}

const (
	// maximum backoff is 2^(maxMultiplier-1) = 128 seconds
	maxMultiplier = 8
)

func (b *backoff) set(override *time.Duration) time.Duration {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.notBefore.After(time.Now()) {
		if override != nil {
			// If existing backoff is set but override would be longer than
			// it then set it to that.
			notBefore := time.Now().Add(*override)
			if notBefore.After(b.notBefore) {
				b.notBefore = notBefore
			}
		}
		return time.Until(b.notBefore)
	}
	var wait time.Duration
	if override != nil {
		wait = *override
	} else {
		if b.multiplier < maxMultiplier {
			b.multiplier++
		}
		wait = time.Second * time.Duration(1<<(b.multiplier-1))
	}
	b.notBefore = time.Now().Add(wait)
	return wait
}

func (b *backoff) decreaseMultiplier() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.multiplier > 0 {
		b.multiplier--
	}
}

func (b *backoff) until() time.Time {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.notBefore
}
