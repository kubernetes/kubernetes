//go:build linux
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
	"flag"
	"os"

	"github.com/spf13/pflag"

	// ensure libs have a chance to globally register their flags
	_ "github.com/google/cadvisor/container/common"
	_ "github.com/google/cadvisor/container/containerd"
	_ "github.com/google/cadvisor/container/raw"
	_ "github.com/google/cadvisor/machine"
	_ "github.com/google/cadvisor/manager"
	_ "github.com/google/cadvisor/storage"
)

// addCadvisorFlags adds flags from cadvisor
func addCadvisorFlags(fs *pflag.FlagSet) {
	// lookup flags in global flag set and re-register the values with our flagset
	global := flag.CommandLine
	local := pflag.NewFlagSet(os.Args[0], pflag.ExitOnError)

	// e2e node tests rely on this
	register(global, local, "housekeeping_interval")

	// These flags were implicit from cadvisor, and are mistakes that should be registered deprecated:
	const deprecated = "This is a cadvisor flag that was mistakenly registered with the Kubelet. Due to legacy concerns, it will follow the standard CLI deprecation timeline before being removed."

	registerDeprecated(global, local, "application_metrics_count_limit", deprecated)
	registerDeprecated(global, local, "boot_id_file", deprecated)
	registerDeprecated(global, local, "container_hints", deprecated)
	registerDeprecated(global, local, "containerd", deprecated)
	registerDeprecated(global, local, "enable_load_reader", deprecated)
	registerDeprecated(global, local, "event_storage_age_limit", deprecated)
	registerDeprecated(global, local, "event_storage_event_limit", deprecated)
	registerDeprecated(global, local, "global_housekeeping_interval", deprecated)
	registerDeprecated(global, local, "log_cadvisor_usage", deprecated)
	registerDeprecated(global, local, "machine_id_file", deprecated)
	registerDeprecated(global, local, "storage_driver_user", deprecated)
	registerDeprecated(global, local, "storage_driver_password", deprecated)
	registerDeprecated(global, local, "storage_driver_host", deprecated)
	registerDeprecated(global, local, "storage_driver_db", deprecated)
	registerDeprecated(global, local, "storage_driver_table", deprecated)
	registerDeprecated(global, local, "storage_driver_secure", deprecated)
	registerDeprecated(global, local, "storage_driver_buffer_duration", deprecated)
	registerDeprecated(global, local, "containerd-namespace", deprecated)

	// finally, add cadvisor flags to the provided flagset
	fs.AddFlagSet(local)
}
