/*
Copyright 2025 The Kubernetes Authors.

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

package oom

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestOOMCounterKeying verifies the exact-string key contract between the OOM
// watcher (recordOOMKill) and the events reader (OOMEventsForContainer). The
// metrics server reads OOMEventsForContainer(info.Name), so the watcher must
// record under the same string that per-container cAdvisor info.Name uses,
// or the metric is never read back.
func TestOOMCounterKeying(t *testing.T) {
	containerID := "abc123def456"

	// Recording under the same key must be read back exactly.
	recordOOMKill(containerID)
	recordOOMKill(containerID)
	assert.Equal(t, uint64(2), OOMEventsForContainer(containerID))

	// A different key must never leak into this container's count.
	recordOOMKill("other-container-id")
	assert.Equal(t, uint64(2), OOMEventsForContainer(containerID))

	// Unknown keys read 0.
	assert.Equal(t, uint64(0), OOMEventsForContainer("not-present"))

	// Clean the package-global map so a later run does not see stale counts.
	oomEventCounts.Lock()
	clear(oomEventCounts.byContainer)
	oomEventCounts.Unlock()
}
