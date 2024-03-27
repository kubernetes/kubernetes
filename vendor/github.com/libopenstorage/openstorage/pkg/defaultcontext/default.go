/*
Package defaultcontext manage the default context and timeouts
Copyright 2021 Portworx

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
package defaultcontext

import (
	"sync"
	"time"

	grpcgw "github.com/grpc-ecosystem/grpc-gateway/runtime"
)

var (
	inst     *defaultContextManager
	instLock sync.Mutex

	defaultDuration = 5 * time.Minute

	// Inst returns the singleton to the default context manager
	Inst = func() *defaultContextManager {
		return defaultContextManagerGetInst()
	}
)

func defaultContextManagerGetInst() *defaultContextManager {
	instLock.Lock()
	defer instLock.Unlock()

	if inst == nil {
		inst = newDefaultContextManager()
	}

	return inst
}

type defaultContextManager struct {
	lock    sync.RWMutex
	timeout time.Duration
}

func newDefaultContextManager() *defaultContextManager {
	d := &defaultContextManager{
		timeout: defaultDuration,
	}
	d.apply()

	return d
}

// SetDefaultTimeout sets the default timeout duration used by contexts without a timeout
func (d *defaultContextManager) SetDefaultTimeout(t time.Duration) error {
	d.lock.Lock()
	defer d.lock.Unlock()

	d.timeout = t
	d.apply()

	return nil
}

// GetDefaultTimeout returns the default timeout duration used by contexts without a timeout.
//
// It is recommended to use grpcutils.WithDefaultTimeout(ctx) which calls this function,
// instead of calling this function directly.
func (d *defaultContextManager) GetDefaultTimeout() time.Duration {
	d.lock.RLock()
	defer d.lock.RUnlock()

	return d.timeout
}

// Add here any external default timeouts that need to be set
func (d *defaultContextManager) apply() {

	// Set SDK Gateway default context
	grpcgw.DefaultContextTimeout = d.timeout

}
