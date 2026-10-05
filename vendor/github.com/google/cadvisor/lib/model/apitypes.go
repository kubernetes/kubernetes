// Copyright 2024 Google Inc. All Rights Reserved.
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

import "time"

// These types back the full cAdvisor binary's v2 REST API. They live in the
// library (rather than info/v2, which the library cannot import) so the manager
// can return them; the root binary's info/v2 aliases each one. They carry no
// behavior and no dependencies. The kubelet does not use them.

// ProcessInfo describes a single process in a container (v2 /ps endpoint).
type ProcessInfo struct {
	User          string  `json:"user"`
	Pid           int     `json:"pid"`
	Ppid          int     `json:"parent_pid"`
	StartTime     string  `json:"start_time"`
	PercentCpu    float32 `json:"percent_cpu"`
	PercentMemory float32 `json:"percent_mem"`
	RSS           uint64  `json:"rss"`
	VirtualSize   uint64  `json:"virtual_size"`
	Status        string  `json:"status"`
	RunningTime   string  `json:"running_time"`
	CgroupPath    string  `json:"cgroup_path"`
	Cmd           string  `json:"cmd"`
	FdCount       int     `json:"fd_count"`
	Psr           int     `json:"psr"`
}

// Percentiles holds summary statistics over a collected sample.
type Percentiles struct {
	// Indicates whether the stats are present or not.
	Present bool `json:"present"`
	// Average over the collected sample.
	Mean uint64 `json:"mean"`
	// Standard deviation of the collected sample.
	Std uint64 `json:"std"`
	// Max seen over the collected sample.
	Max uint64 `json:"max"`
	// 50th percentile over the collected sample.
	Fifty uint64 `json:"fifty"`
	// 90th percentile over the collected sample.
	Ninety uint64 `json:"ninety"`
	// 95th percentile over the collected sample.
	NinetyFive uint64 `json:"ninetyfive"`
	// Number of samples used to calculate these percentiles.
	Count uint64 `json:"count"`
}

// Usage holds percentile usage over a window.
type Usage struct {
	// Indicates amount of data available [0-100].
	PercentComplete int32 `json:"percent_complete"`
	// Mean, Max, and 90p cpu rate value in milliCpus/seconds.
	Cpu Percentiles `json:"cpu"`
	// Mean, Max, and 90p memory size in bytes.
	Memory Percentiles `json:"memory"`
}

// InstantUsage is the latest sample collected for a container.
type InstantUsage struct {
	// cpu rate in cpu milliseconds/second.
	Cpu uint64 `json:"cpu"`
	// Memory usage in bytes.
	Memory uint64 `json:"memory"`
}

// DerivedStats are usage stats computed over rolling windows.
type DerivedStats struct {
	// Time of generation of these stats.
	Timestamp time.Time `json:"timestamp"`
	// Latest instantaneous sample.
	LatestUsage InstantUsage `json:"latest_usage"`
	// Percentiles in last observed minute.
	MinuteUsage Usage `json:"minute_usage"`
	// Percentile in last hour.
	HourUsage Usage `json:"hour_usage"`
	// Percentile in last day.
	DayUsage Usage `json:"day_usage"`
}
