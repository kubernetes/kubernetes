/*
Copyright 2022 The Kubernetes Authors.

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

package request

import (
	"fmt"
	"sync"
)

type IsDelegatedFunc func(url string) bool

// DelegatedRequestsTracker is a tracker for urls that are being delegated to external apiservers.
type DelegatedRequestsTracker struct {
	mu sync.RWMutex
	f  IsDelegatedFunc
}

// Binds stores a callback used to determine if call is delegated.
func (d *DelegatedRequestsTracker) Bind(f IsDelegatedFunc) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.f != nil {
		return fmt.Errorf("DelegatedRequestsTracker.Bind called twice. Previous value: %v. Current value: %v", d.f, f)
	}
	d.f = f
	return nil
}

// IsDelegated checks if a given url is delegated to an external apiserver.
func (d *DelegatedRequestsTracker) IsDelegated(url string) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.f == nil {
		return false
	}
	return d.f(url)
}
