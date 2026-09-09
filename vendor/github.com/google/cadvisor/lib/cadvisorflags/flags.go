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

// Package cadvisorflags is the single source of truth for the names of the
// cAdvisor global flags that the Kubernetes kubelet re-registers on its own
// command line (see cmd/kubelet/app/options/globalflags_linux.go).
//
// Background: cAdvisor registers a number of flags on the process-global
// flag.CommandLine via package init(). The kubelet does not want all of them,
// so it looks a chosen subset up BY NAME and re-registers them on its own
// pflag set. Historically those names were string literals in the kubelet, so
// removing or renaming a flag in this library turned into a kubelet STARTUP
// PANIC ("failed to find flag in global flagset") rather than a build error.
//
// Centralizing the names here gives one place to keep in sync, and the
// contract test in this package (flags_contract_test.go) asserts that every
// name below still resolves to a registered flag — so drift becomes a failing
// test in this repo instead of a panic in the consumer. The flag DEFINITIONS
// elsewhere in the tree intentionally keep their string literals; the contract
// test is what binds them to these constants.
package cadvisorflags

// Names of the cAdvisor global flags the kubelet pins. Keep this list in sync
// with the flag definitions across the manager, machine, container/common,
// container/containerd, and storage packages; the contract test enforces it.
const (
	// HousekeepingInterval is kept as a normal kubelet flag (node e2e relies on it).
	HousekeepingInterval = "housekeeping_interval"

	// The remainder are vestigial cAdvisor flags the kubelet exposes only as
	// deprecated, for backwards compatibility with existing command lines.
	ApplicationMetricsCountLimit = "application_metrics_count_limit"
	BootIDFile                   = "boot_id_file"
	ContainerHints               = "container_hints"
	Containerd                   = "containerd"
	EnableLoadReader             = "enable_load_reader"
	EventStorageAgeLimit         = "event_storage_age_limit"
	EventStorageEventLimit       = "event_storage_event_limit"
	GlobalHousekeepingInterval   = "global_housekeeping_interval"
	LogCadvisorUsage             = "log_cadvisor_usage"
	MachineIDFile                = "machine_id_file"
	StorageDriverUser            = "storage_driver_user"
	StorageDriverPassword        = "storage_driver_password"
	StorageDriverHost            = "storage_driver_host"
	StorageDriverDB              = "storage_driver_db"
	StorageDriverTable           = "storage_driver_table"
	StorageDriverSecure          = "storage_driver_secure"
	StorageDriverBufferDuration  = "storage_driver_buffer_duration"
	ContainerdNamespace          = "containerd-namespace"
)

// Kept returns the cAdvisor flag names the kubelet should expose as normal,
// non-deprecated flags.
func Kept() []string {
	return []string{
		HousekeepingInterval,
	}
}

// Deprecated returns the cAdvisor flag names the kubelet should expose but mark
// deprecated. These were historically registered with the kubelet by accident
// and are retained only to avoid breaking existing command lines.
func Deprecated() []string {
	return []string{
		ApplicationMetricsCountLimit,
		BootIDFile,
		ContainerHints,
		Containerd,
		EnableLoadReader,
		EventStorageAgeLimit,
		EventStorageEventLimit,
		GlobalHousekeepingInterval,
		LogCadvisorUsage,
		MachineIDFile,
		StorageDriverUser,
		StorageDriverPassword,
		StorageDriverHost,
		StorageDriverDB,
		StorageDriverTable,
		StorageDriverSecure,
		StorageDriverBufferDuration,
		ContainerdNamespace,
	}
}
