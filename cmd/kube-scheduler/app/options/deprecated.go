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
	"fmt"
	"net"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/validation/field"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// DeprecatedOptions contains deprecated options and their flags.
// TODO remove these fields once the deprecated flags are removed.
type DeprecatedOptions struct {
	// The fields below here are placeholders for flags that can't be directly
	// mapped into componentconfig.KubeSchedulerConfiguration.
	Port int
}

// TODO: remove these insecure flags in v1.24
func addDummyInsecureFlags(o *DeprecatedOptions, fs *pflag.FlagSet) {
	var (
		bindAddr = net.IPv4(127, 0, 0, 1)
	)
	fs.IPVar(&bindAddr, "address", bindAddr,
		"The IP address on which to serve the insecure --port (set to 0.0.0.0 for all IPv4 interfaces and :: for all IPv6 interfaces).")
	fs.MarkDeprecated("address", "This flag has no effect now and will be removed in v1.24. You can use --bind-address instead.")

	fs.IntVar(&o.Port, "port", o.Port, "The port on which to serve unsecured, unauthenticated access. Set to 0 to disable.")
	fs.MarkDeprecated("port", "This flag has no effect now and will be removed in v1.24. You can use --secure-port instead.")
}

// AddFlags adds flags for the deprecated options.
func (o *DeprecatedOptions) AddFlags(fs *pflag.FlagSet, cfg *kubeschedulerconfig.KubeSchedulerConfiguration) {
	if o == nil {
		return
	}

	addDummyInsecureFlags(o, fs)

	fs.BoolVar(&cfg.EnableProfiling, "profiling", cfg.EnableProfiling, "DEPRECATED: enable profiling via web interface host:port/debug/pprof/. This parameter is ignored if a config file is specified in --config.")
	fs.BoolVar(&cfg.EnableContentionProfiling, "contention-profiling", cfg.EnableContentionProfiling, "DEPRECATED: enable lock contention profiling, if profiling is enabled. This parameter is ignored if a config file is specified in --config.")
	fs.StringVar(&cfg.ClientConnection.Kubeconfig, "kubeconfig", cfg.ClientConnection.Kubeconfig, "DEPRECATED: path to kubeconfig file with authorization and master location information. This parameter is ignored if a config file is specified in --config.")
	fs.StringVar(&cfg.ClientConnection.ContentType, "kube-api-content-type", cfg.ClientConnection.ContentType, "DEPRECATED: content type of requests sent to apiserver. This parameter is ignored if a config file is specified in --config.")
	fs.Float32Var(&cfg.ClientConnection.QPS, "kube-api-qps", cfg.ClientConnection.QPS, "DEPRECATED: QPS to use while talking with kubernetes apiserver. This parameter is ignored if a config file is specified in --config.")
	fs.Int32Var(&cfg.ClientConnection.Burst, "kube-api-burst", cfg.ClientConnection.Burst, "DEPRECATED: burst to use while talking with kubernetes apiserver. This parameter is ignored if a config file is specified in --config.")
	fs.StringVar(&cfg.LeaderElection.ResourceNamespace, "lock-object-namespace", cfg.LeaderElection.ResourceNamespace, "DEPRECATED: define the namespace of the lock object. Will be removed in favor of leader-elect-resource-namespace. This parameter is ignored if a config file is specified in --config.")
	fs.StringVar(&cfg.LeaderElection.ResourceName, "lock-object-name", cfg.LeaderElection.ResourceName, "DEPRECATED: define the name of the lock object. Will be removed in favor of leader-elect-resource-name. This parameter is ignored if a config file is specified in --config.")
}

// Validate validates the deprecated scheduler options.
func (o *DeprecatedOptions) Validate() []error {
	var errs []error

	// TODO: delete this check after insecure flags removed in v1.24
	if o.Port != 0 {
		errs = append(errs, field.Required(field.NewPath("port"), fmt.Sprintf("invalid port value %d: only zero is allowed", o.Port)))
	}
	return errs
}
