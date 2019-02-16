// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"github.com/spf13/pflag"

	// ensure libs have a chance to globally register their flags
	_ "github.com/google/cadvisor/container/common"
	_ "github.com/google/cadvisor/container/containerd"
	_ "github.com/google/cadvisor/container/docker"
	_ "github.com/google/cadvisor/container/raw"
	_ "github.com/google/cadvisor/machine"
	_ "github.com/google/cadvisor/manager"
	_ "github.com/google/cadvisor/storage"

	"k8s.io/component-base/cli/globalflag"
)

// addCadvisorFlags adds flags from cadvisor
func addCadvisorFlags(fs *pflag.FlagSet) {
	// lookup flags in global flag set and re-register the values with flag.CommandLine.

	// These flags were also implicit from cadvisor, but are actually used by something in the core repo:
	// TODO(mtaufen): This one is stil used by our salt, but for heaven's sake it's even deprecated in cadvisor
	globalflag.Register(fs, "docker_root")
	// e2e node tests rely on this
	globalflag.Register(fs, "housekeeping_interval")

	// These flags were implicit from cadvisor, and are mistakes that should be registered deprecated:
	const deprecated = "This is a cadvisor flag that was mistakenly registered with the Kubelet. Due to legacy concerns, it will follow the standard CLI deprecation timeline before being removed."

	globalflag.RegisterDeprecated(fs, "application_metrics_count_limit", deprecated)
	globalflag.RegisterDeprecated(fs, "boot_id_file", deprecated)
	globalflag.RegisterDeprecated(fs, "container_hints", deprecated)
	globalflag.RegisterDeprecated(fs, "containerd", deprecated)
	globalflag.RegisterDeprecated(fs, "docker", deprecated)
	globalflag.RegisterDeprecated(fs, "docker_env_metadata_whitelist", deprecated)
	globalflag.RegisterDeprecated(fs, "docker_only", deprecated)
	globalflag.RegisterDeprecated(fs, "docker-tls", deprecated)
	globalflag.RegisterDeprecated(fs, "docker-tls-ca", deprecated)
	globalflag.RegisterDeprecated(fs, "docker-tls-cert", deprecated)
	globalflag.RegisterDeprecated(fs, "docker-tls-key", deprecated)
	globalflag.RegisterDeprecated(fs, "enable_load_reader", deprecated)
	globalflag.RegisterDeprecated(fs, "event_storage_age_limit", deprecated)
	globalflag.RegisterDeprecated(fs, "event_storage_event_limit", deprecated)
	globalflag.RegisterDeprecated(fs, "global_housekeeping_interval", deprecated)
	globalflag.RegisterDeprecated(fs, "log_cadvisor_usage", deprecated)
	globalflag.RegisterDeprecated(fs, "machine_id_file", deprecated)
	globalflag.RegisterDeprecated(fs, "storage_driver_user", deprecated)
	globalflag.RegisterDeprecated(fs, "storage_driver_password", deprecated)
	globalflag.RegisterDeprecated(fs, "storage_driver_host", deprecated)
	globalflag.RegisterDeprecated(fs, "storage_driver_db", deprecated)
	globalflag.RegisterDeprecated(fs, "storage_driver_table", deprecated)
	globalflag.RegisterDeprecated(fs, "storage_driver_secure", deprecated)
	globalflag.RegisterDeprecated(fs, "storage_driver_buffer_duration", deprecated)
}
