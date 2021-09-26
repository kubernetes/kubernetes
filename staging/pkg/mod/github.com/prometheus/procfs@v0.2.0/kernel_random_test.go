// Copyright 2020 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !windows

package procfs

import (
	"testing"
)

const procfsFixtures = "fixtures/proc"

func TestKernelRandom(t *testing.T) {
	fs, err := NewFS(procfsFixtures)
	if err != nil {
		t.Fatalf("failed to access %s: %v", procfsFixtures, err)
	}

	random, err := fs.KernelRandom()
	if err != nil {
		t.Fatalf("failed to collect %s/sys/kernel/random: %v", procfsFixtures, err)
	}

	if *random.EntropyAvaliable != 3943 {
		t.Errorf("entropy_avail, want %d got %d", 3943, *random.EntropyAvaliable)
	}
	if *random.PoolSize != 4096 {
		t.Errorf("poolsize, want %d got %d", 4096, *random.PoolSize)
	}
	if *random.URandomMinReseedSeconds != 60 {
		t.Errorf("urandom_min_reseed_secs, want %d got %d", 60, *random.URandomMinReseedSeconds)
	}
	if *random.WriteWakeupThreshold != 3072 {
		t.Errorf("write_wakeup_threshold, want %d got %d", 3072, *random.WriteWakeupThreshold)
	}
	if random.ReadWakeupThreshold != nil {
		t.Errorf("read_wakeup_threshold, want %v got %d", nil, *random.ReadWakeupThreshold)
	}
}
