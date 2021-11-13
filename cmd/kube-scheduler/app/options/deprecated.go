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

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/validation/field"
	componentbaseconfig "k8s.io/component-base/config"
	schedulerappconfig "k8s.io/kubernetes/cmd/kube-scheduler/app/config"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// DeprecatedOptions contains deprecated options and their flags.
// TODO remove these fields once the deprecated flags are removed.
type DeprecatedOptions struct {
	componentbaseconfig.DebuggingConfiguration
	componentbaseconfig.ClientConnectionConfiguration
	// Note that only the deprecated options (lock-object-name and lock-object-namespace) are populated here.
	componentbaseconfig.LeaderElectionConfiguration
	// The fields below here are placeholders for flags that can't be directly
	// mapped into componentconfig.KubeSchedulerConfiguration.
	PolicyConfigFile         string
	PolicyConfigMapName      string
	PolicyConfigMapNamespace string
	UseLegacyPolicyConfig    bool
}

// AddFlags adds flags for the deprecated options.
func (o *DeprecatedOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.PolicyConfigFile, "policy-config-file", o.PolicyConfigFile, "DEPRECATED: file with scheduler policy configuration. This file is used if policy ConfigMap is not provided or --use-legacy-policy-config=true. Note: The predicates/priorities defined in this file will take precedence over any profiles define in ComponentConfig.")
	usage := fmt.Sprintf("DEPRECATED: name of the ConfigMap object that contains scheduler's policy configuration. It must exist in the system namespace before scheduler initialization if --use-legacy-policy-config=false. The config must be provided as the value of an element in 'Data' map with the key='%v'. Note: The predicates/priorities defined in this file will take precedence over any profiles define in ComponentConfig.", kubeschedulerconfig.SchedulerPolicyConfigMapKey)
	fs.StringVar(&o.PolicyConfigMapName, "policy-configmap", o.PolicyConfigMapName, usage)
	fs.StringVar(&o.PolicyConfigMapNamespace, "policy-configmap-namespace", o.PolicyConfigMapNamespace, "DEPRECATED: the namespace where policy ConfigMap is located. The kube-system namespace will be used if this is not provided or is empty. Note: The predicates/priorities defined in this file will take precedence over any profiles define in ComponentConfig.")
	fs.BoolVar(&o.UseLegacyPolicyConfig, "use-legacy-policy-config", o.UseLegacyPolicyConfig, "DEPRECATED: when set to true, scheduler will ignore policy ConfigMap and uses policy config file. Note: The scheduler will fail if this is combined with Plugin configs")

	fs.BoolVar(&o.EnableProfiling, "profiling", true, "DEPRECATED: enable profiling via web interface host:port/debug/pprof/. This parameter is ignored if a config file is specified in --config.")
	fs.BoolVar(&o.EnableContentionProfiling, "contention-profiling", true, "DEPRECATED: enable lock contention profiling, if profiling is enabled. This parameter is ignored if a config file is specified in --config.")
	fs.StringVar(&o.Kubeconfig, "kubeconfig", "", "DEPRECATED: path to kubeconfig file with authorization and master location information. This parameter is ignored if a config file is specified in --config.")
	fs.StringVar(&o.ContentType, "kube-api-content-type", "application/vnd.kubernetes.protobuf", "DEPRECATED: content type of requests sent to apiserver. This parameter is ignored if a config file is specified in --config.")
	fs.Float32Var(&o.QPS, "kube-api-qps", 50.0, "DEPRECATED: QPS to use while talking with kubernetes apiserver. This parameter is ignored if a config file is specified in --config.")
	fs.Int32Var(&o.Burst, "kube-api-burst", 100, "DEPRECATED: burst to use while talking with kubernetes apiserver. This parameter is ignored if a config file is specified in --config.")
	fs.StringVar(&o.ResourceNamespace, "lock-object-namespace", "kube-system", "DEPRECATED: define the namespace of the lock object. Will be removed in favor of leader-elect-resource-namespace. This parameter is ignored if a config file is specified in --config.")
	fs.StringVar(&o.ResourceName, "lock-object-name", "kube-scheduler", "DEPRECATED: define the name of the lock object. Will be removed in favor of leader-elect-resource-name. This parameter is ignored if a config file is specified in --config.")
}

// Validate validates the deprecated scheduler options.
func (o *DeprecatedOptions) Validate() []error {
	var errs []error

	if o.UseLegacyPolicyConfig && len(o.PolicyConfigFile) == 0 {
		errs = append(errs, field.Required(field.NewPath("policyConfigFile"), "required when --use-legacy-policy-config is true"))
	}

	return errs
}

// ApplyTo sets cfg.PolicySource from flags passed on the command line in the following precedence order:
//
// 1. --use-legacy-policy-config to use a policy file.
// 2. --policy-configmap to use a policy config map value.
func (o *DeprecatedOptions) ApplyTo(c *schedulerappconfig.Config) {
	if o == nil {
		return
	}

	switch {
	case o.UseLegacyPolicyConfig || (len(o.PolicyConfigFile) > 0 && o.PolicyConfigMapName == ""):
		c.LegacyPolicySource = &kubeschedulerconfig.SchedulerPolicySource{
			File: &kubeschedulerconfig.SchedulerPolicyFileSource{
				Path: o.PolicyConfigFile,
			},
		}
	case len(o.PolicyConfigMapName) > 0:
		c.LegacyPolicySource = &kubeschedulerconfig.SchedulerPolicySource{
			ConfigMap: &kubeschedulerconfig.SchedulerPolicyConfigMapSource{
				Name:      o.PolicyConfigMapName,
				Namespace: o.PolicyConfigMapNamespace,
			},
		}
	}
}
