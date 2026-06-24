// Copyright 2026 Google Inc. All Rights Reserved.
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

package model

import (
	"fmt"
	"time"
)

// Container identifier types accepted by RequestOptions.IdType.
const (
	TypeName   = "name"
	TypeDocker = "docker"
	TypePodman = "podman"
)

// RequestOptions is the subtree query model the kubelet uses to pull recent
// container stats from the manager. Folded in from info/v2 as part of the
// v1+v2 -> model collapse (it is distinct from ContainerInfoRequest, which is
// the count/time-window query of the legacy v1 REST surface).
type RequestOptions struct {
	// Type of container identifier - TypeName (default), TypeDocker, TypePodman.
	IdType string `json:"type"`
	// Number of stats to return; -1 means no limit.
	Count int `json:"count"`
	// Whether to include stats for child subcontainers.
	Recursive bool `json:"recursive"`
	// Update stats if older than MaxAge; nil means no update, 0 always updates.
	MaxAge *time.Duration `json:"max_age"`
}

// InstCpuStats derives instantaneous CPU usage (nanocores/second) from two
// consecutive cumulative samples. Folded in from info/v2: the collapse turns
// v2's read-time derivations into model-owned projections (design §6.1).
func InstCpuStats(last, cur *ContainerStats) (*CpuInstStats, error) {
	if last == nil || cur == nil || last.Cpu == nil || cur.Cpu == nil {
		return nil, nil
	}
	if !cur.Timestamp.After(last.Timestamp) {
		return nil, fmt.Errorf("container stats move backwards in time")
	}
	if len(last.Cpu.Usage.PerCpu) != len(cur.Cpu.Usage.PerCpu) {
		return nil, fmt.Errorf("different number of cpus")
	}
	timeDelta := cur.Timestamp.Sub(last.Timestamp)
	// Nanoseconds to gain precision and avoid having zero seconds if the
	// difference between the timestamps is just under a second.
	timeDeltaNs := uint64(timeDelta.Nanoseconds())
	convertToRate := func(lastValue, curValue uint64) (uint64, error) {
		if curValue < lastValue {
			return 0, fmt.Errorf("cumulative stats decrease")
		}
		valueDelta := curValue - lastValue
		// Use float64 to keep precision.
		return uint64(float64(valueDelta) / float64(timeDeltaNs) * 1e9), nil
	}
	total, err := convertToRate(last.Cpu.Usage.Total, cur.Cpu.Usage.Total)
	if err != nil {
		return nil, err
	}
	percpu := make([]uint64, len(last.Cpu.Usage.PerCpu))
	for i := range percpu {
		var err error
		percpu[i], err = convertToRate(last.Cpu.Usage.PerCpu[i], cur.Cpu.Usage.PerCpu[i])
		if err != nil {
			return nil, err
		}
	}
	user, err := convertToRate(last.Cpu.Usage.User, cur.Cpu.Usage.User)
	if err != nil {
		return nil, err
	}
	system, err := convertToRate(last.Cpu.Usage.System, cur.Cpu.Usage.System)
	if err != nil {
		return nil, err
	}
	return &CpuInstStats{
		Usage: CpuInstUsage{
			Total:  total,
			PerCpu: percpu,
			User:   user,
			System: system,
		},
	}, nil
}

// FsInfo is per-filesystem runtime capacity/usage returned by GetFsInfo. Folded in
// from info/v2 (named FsInfo there too); the machine-level filesystem type is named
// FilesystemInfo here to avoid the collision.
type FsInfo struct {
	Timestamp  time.Time `json:"timestamp"`
	Device     string    `json:"device"`
	Mountpoint string    `json:"mountpoint"`
	Capacity   uint64    `json:"capacity"`
	Available  uint64    `json:"available"`
	Usage      uint64    `json:"usage"`
	Labels     []string  `json:"labels"`
	Inodes     *uint64   `json:"inodes,omitempty"`
	InodesFree *uint64   `json:"inodes_free,omitempty"`
}
