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
	"os"

	"github.com/prometheus/procfs/internal/util"
)

// KernelRandom contains information about to the kernel's random number generator.
type KernelRandom struct {
	// EntropyAvaliable gives the available entropy, in bits.
	EntropyAvaliable *uint64
	// PoolSize gives the size of the entropy pool, in bytes.
	PoolSize *uint64
	// URandomMinReseedSeconds is the number of seconds after which the DRNG will be reseeded.
	URandomMinReseedSeconds *uint64
	// WriteWakeupThreshold the number of bits of entropy below which we wake up processes
	// that do a select(2) or poll(2) for write access to /dev/random.
	WriteWakeupThreshold *uint64
	// ReadWakeupThreshold is the number of bits of entropy required for waking up processes that sleep
	// waiting for entropy from /dev/random.
	ReadWakeupThreshold *uint64
}

// KernelRandom returns values from /proc/sys/kernel/random.
func (fs FS) KernelRandom() (KernelRandom, error) {
	random := KernelRandom{}

	for file, p := range map[string]**uint64{
		"entropy_avail":           &random.EntropyAvaliable,
		"poolsize":                &random.PoolSize,
		"urandom_min_reseed_secs": &random.URandomMinReseedSeconds,
		"write_wakeup_threshold":  &random.WriteWakeupThreshold,
		"read_wakeup_threshold":   &random.ReadWakeupThreshold,
	} {
		val, err := util.ReadUintFromFile(fs.proc.Path("sys", "kernel", "random", file))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return random, err
		}
		*p = &val
	}

	return random, nil
}
