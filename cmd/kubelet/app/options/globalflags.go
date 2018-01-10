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
	"fmt"
	"os"
	"strings"

	"github.com/spf13/pflag"

	// libs that provide registration functions
	"k8s.io/apiserver/pkg/util/logs"
	"k8s.io/kubernetes/pkg/version/verflag"

	// ensure libs have a chance to globally register their flags
	_ "github.com/golang/glog"
	_ "github.com/google/cadvisor/container/common"
	_ "github.com/google/cadvisor/container/containerd"
	_ "github.com/google/cadvisor/container/docker"
	_ "github.com/google/cadvisor/container/raw"
	_ "github.com/google/cadvisor/machine"
	_ "github.com/google/cadvisor/manager"
	_ "github.com/google/cadvisor/storage"
	_ "k8s.io/kubernetes/pkg/credentialprovider/azure"
	_ "k8s.io/kubernetes/pkg/credentialprovider/gcp"
)

// AddGlobalFlags explicitly registers flags that libraries (glog, verflag, etc.) register
// against the global flagsets from "flag" and "github.com/spf13/pflag".
// We do this in order to prevent unwanted flags from leaking into the Kubelet's flagset.
func AddGlobalFlags(fs *pflag.FlagSet) {
	addGlogFlags(fs)
	addCadvisorFlags(fs)
	verflag.AddFlags(fs)
	logs.AddFlags(fs)
}

// normalize replaces underscores with hyphens
// we should always use hyphens instead of underscores when registering kubelet flags
func normalize(s string) string {
	return strings.Replace(s, "_", "-", -1)
}

// register adds a flag to local that targets the Value associated with the Flag named globalName in global
func register(global *flag.FlagSet, local *pflag.FlagSet, globalName string) {
	if f := global.Lookup(globalName); f != nil {
		f.Name = normalize(f.Name)
		local.AddFlag(pflag.PFlagFromGoFlag(f))
	} else {
		panic(fmt.Sprintf("failed to find flag in global flagset (flag): %s", globalName))
	}
}

// pflagRegister adds a flag to local that targets the Value associated with the Flag named globalName in global
func pflagRegister(global, local *pflag.FlagSet, globalName string) {
	if f := global.Lookup(globalName); f != nil {
		f.Name = normalize(f.Name)
		local.AddFlag(f)
	} else {
		panic(fmt.Sprintf("failed to find flag in global flagset (pflag): %s", globalName))
	}
}

// registerDeprecated registers the flag with register, and then marks it deprecated
func registerDeprecated(global *flag.FlagSet, local *pflag.FlagSet, globalName, deprecated string) {
	register(global, local, globalName)
	local.Lookup(normalize(globalName)).Deprecated = deprecated
}

// pflagRegisterDeprecated registers the flag with pflagRegister, and then marks it deprecated
func pflagRegisterDeprecated(global, local *pflag.FlagSet, globalName, deprecated string) {
	pflagRegister(global, local, globalName)
	local.Lookup(normalize(globalName)).Deprecated = deprecated
}

// addCredentialProviderFlags adds flags from k8s.io/kubernetes/pkg/credentialprovider
func addCredentialProviderFlags(fs *pflag.FlagSet) {
	// lookup flags in global flag set and re-register the values with our flagset
	global := pflag.CommandLine
	local := pflag.NewFlagSet(os.Args[0], pflag.ExitOnError)

	// Note this is deprecated in the library that provides it, so we just allow that deprecation
	// notice to pass through our registration here.
	pflagRegister(global, local, "google-json-key")
	// TODO(#58034): This is not a static file, so it's not quite as straightforward as --google-json-key.
	// We need to figure out how ACR users can dynamically provide pull credentials before we can deprecate this.
	pflagRegister(global, local, "azure-container-registry-config")

	fs.AddFlagSet(local)
}

// addGlogFlags adds flags from github.com/golang/glog
func addGlogFlags(fs *pflag.FlagSet) {
	// lookup flags in global flag set and re-register the values with our flagset
	global := flag.CommandLine
	local := pflag.NewFlagSet(os.Args[0], pflag.ExitOnError)

	register(global, local, "logtostderr")
	register(global, local, "alsologtostderr")
	register(global, local, "v")
	register(global, local, "stderrthreshold")
	register(global, local, "vmodule")
	register(global, local, "log_backtrace_at")
	register(global, local, "log_dir")

	fs.AddFlagSet(local)
}

// addCadvisorFlags adds flags from cadvisor
func addCadvisorFlags(fs *pflag.FlagSet) {
	// lookup flags in global flag set and re-register the values with our flagset
	global := flag.CommandLine
	local := pflag.NewFlagSet(os.Args[0], pflag.ExitOnError)

	// These flags were also implicit from cadvisor, but are actually used by something in the core repo:
	// TODO(mtaufen): This one is stil used by our salt, but for heaven's sake it's even deprecated in cadvisor
	register(global, local, "docker_root")
	// e2e node tests rely on this
	register(global, local, "housekeeping_interval")

	// These flags were implicit from cadvisor, and are mistakes that should be registered deprecated:
	const deprecated = "This is a cadvisor flag that was mistakenly registered with the Kubelet. Due to legacy concerns, it will follow the standard CLI deprecation timeline before being removed."

	registerDeprecated(global, local, "application_metrics_count_limit", deprecated)
	registerDeprecated(global, local, "boot_id_file", deprecated)
	registerDeprecated(global, local, "container_hints", deprecated)
	registerDeprecated(global, local, "containerd", deprecated)
	registerDeprecated(global, local, "docker", deprecated)
	registerDeprecated(global, local, "docker_env_metadata_whitelist", deprecated)
	registerDeprecated(global, local, "docker_only", deprecated)
	registerDeprecated(global, local, "docker-tls", deprecated)
	registerDeprecated(global, local, "docker-tls-ca", deprecated)
	registerDeprecated(global, local, "docker-tls-cert", deprecated)
	registerDeprecated(global, local, "docker-tls-key", deprecated)
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

	// finally, add cadvisor flags to the provided flagset
	fs.AddFlagSet(local)
}
