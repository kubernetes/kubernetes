/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"sync"
	"time"

	"github.com/pborman/uuid"
	"k8s.io/kubernetes/pkg/types"
)

var uuidLock sync.Mutex

/**
 * The UUID package is naive and can generate identical UUIDs if the time interval is quick enough.
 * Block subsequent UUIDs for 200 Nanoseconds, the UUID uses 100 ns increments, we block for 200 to be safe
 * Blocks in a go routine, so that the caller doesn't have to wait.
 * TODO: save old unused UUIDs so that no one has to block.
 */
func NewUUID() types.UID {
	uuidLock.Lock()
	result := uuid.NewUUID()
	go func() {
		time.Sleep(200 * time.Nanosecond)
		uuidLock.Unlock()
	}()
	return types.UID(result.String())
}
