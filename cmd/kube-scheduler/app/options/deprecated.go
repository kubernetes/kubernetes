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
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
)

// DeprecatedOptions contains deprecated options and their flags.
// TODO remove these fields once the deprecated flags are removed.
type DeprecatedOptions struct {
	// The fields below here are placeholders for flags that can't be directly
	// mapped into componentconfig.KubeSchedulerConfiguration.
	PolicyConfigFile               string
	PolicyConfigMapName            string
	PolicyConfigMapNamespace       string
	UseLegacyPolicyConfig          bool
	AlgorithmProvider              string
	HardPodAffinitySymmetricWeight int32
	SchedulerName                  string
}

// AddFlags adds flags for the deprecated options.
func (o *DeprecatedOptions) AddFlags(fs *pflag.FlagSet, cfg *kubeschedulerconfig.KubeSchedulerConfiguration) {
	if o == nil {
		return
	}

	fs.StringVar(&o.AlgorithmProvider, "algorithm-provider", o.AlgorithmProvider, "DEPRECATED: the scheduling algorithm provider to use, one of: "+algorithmprovider.ListAlgorithmProviders())
	fs.StringVar(&o.PolicyConfigFile, "policy-config-file", o.PolicyConfigFile, "DEPRECATED: file with scheduler policy configuration. This file is used if policy ConfigMap is not provided or --use-legacy-policy-config=true")
	usage := fmt.Sprintf("DEPRECATED: name of the ConfigMap object that contains scheduler's policy configuration. It must exist in the system namespace before scheduler initialization if --use-legacy-policy-config=false. The config must be provided as the value of an element in 'Data' map with the key='%v'", kubeschedulerconfig.SchedulerPolicyConfigMapKey)
	fs.StringVar(&o.PolicyConfigMapName, "policy-configmap", o.PolicyConfigMapName, usage)
	fs.StringVar(&o.PolicyConfigMapNamespace, "policy-configmap-namespace", o.PolicyConfigMapNamespace, "DEPRECATED: the namespace where policy ConfigMap is located. The kube-system namespace will be used if this is not provided or is empty.")
	fs.BoolVar(&o.UseLegacyPolicyConfig, "use-legacy-policy-config", o.UseLegacyPolicyConfig, "DEPRECATED: when set to true, scheduler will ignore policy ConfigMap and uses policy config file")

	fs.BoolVar(&cfg.EnableProfiling, "profiling", cfg.EnableProfiling, "DEPRECATED: enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&cfg.EnableContentionProfiling, "contention-profiling", cfg.EnableContentionProfiling, "DEPRECATED: enable lock contention profiling, if profiling is enabled")
	fs.StringVar(&cfg.ClientConnection.Kubeconfig, "kubeconfig", cfg.ClientConnection.Kubeconfig, "DEPRECATED: path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&cfg.ClientConnection.ContentType, "kube-api-content-type", cfg.ClientConnection.ContentType, "DEPRECATED: content type of requests sent to apiserver.")
	fs.Float32Var(&cfg.ClientConnection.QPS, "kube-api-qps", cfg.ClientConnection.QPS, "DEPRECATED: QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&cfg.ClientConnection.Burst, "kube-api-burst", cfg.ClientConnection.Burst, "DEPRECATED: burst to use while talking with kubernetes apiserver")
	fs.StringVar(&cfg.LeaderElection.ResourceNamespace, "lock-object-namespace", cfg.LeaderElection.ResourceNamespace, "DEPRECATED: define the namespace of the lock object. Will be removed in favor of leader-elect-resource-namespace.")
	fs.StringVar(&cfg.LeaderElection.ResourceName, "lock-object-name", cfg.LeaderElection.ResourceName, "DEPRECATED: define the name of the lock object. Will be removed in favor of leader-elect-resource-name")

	fs.Int32Var(&o.HardPodAffinitySymmetricWeight, "hard-pod-affinity-symmetric-weight", o.HardPodAffinitySymmetricWeight,
		"DEPRECATED: RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule corresponding "+
			"to every RequiredDuringScheduling affinity rule. --hard-pod-affinity-symmetric-weight represents the weight of implicit PreferredDuringScheduling affinity rule. Must be in the range 0-100."+
			"This option was moved to the policy configuration file")
	fs.StringVar(&o.SchedulerName, "scheduler-name", o.SchedulerName, "DEPRECATED: name of the scheduler, used to select which pods will be processed by this scheduler, based on pod's \"spec.schedulerName\".")
	// MarkDeprecated hides the flag from the help. We don't want that:
	// fs.MarkDeprecated("hard-pod-affinity-symmetric-weight", "This option was moved to the policy configuration file")
}

// Validate validates the deprecated scheduler options.
func (o *DeprecatedOptions) Validate() []error {
	var errs []error

	if o.UseLegacyPolicyConfig && len(o.PolicyConfigFile) == 0 {
		errs = append(errs, field.Required(field.NewPath("policyConfigFile"), "required when --use-legacy-policy-config is true"))
	}

	if err := interpodaffinity.ValidateHardPodAffinityWeight(field.NewPath("hardPodAffinitySymmetricWeight"), o.HardPodAffinitySymmetricWeight); err != nil {
		errs = append(errs, err)
	}

	return errs
}

// ApplyTo sets cfg.AlgorithmSource from flags passed on the command line in the following precedence order:
//
// 1. --use-legacy-policy-config to use a policy file.
// 2. --policy-configmap to use a policy config map value.
// 3. --algorithm-provider to use a named algorithm provider.
func (o *DeprecatedOptions) ApplyTo(cfg *kubeschedulerconfig.KubeSchedulerConfiguration) error {
	if o == nil {
		return nil
	}

	switch {
	case o.UseLegacyPolicyConfig || (len(o.PolicyConfigFile) > 0 && o.PolicyConfigMapName == ""):
		cfg.AlgorithmSource = kubeschedulerconfig.SchedulerAlgorithmSource{
			Policy: &kubeschedulerconfig.SchedulerPolicySource{
				File: &kubeschedulerconfig.SchedulerPolicyFileSource{
					Path: o.PolicyConfigFile,
				},
			},
		}
	case len(o.PolicyConfigMapName) > 0:
		cfg.AlgorithmSource = kubeschedulerconfig.SchedulerAlgorithmSource{
			Policy: &kubeschedulerconfig.SchedulerPolicySource{
				ConfigMap: &kubeschedulerconfig.SchedulerPolicyConfigMapSource{
					Name:      o.PolicyConfigMapName,
					Namespace: o.PolicyConfigMapNamespace,
				},
			},
		}
	case len(o.AlgorithmProvider) > 0:
		cfg.AlgorithmSource = kubeschedulerconfig.SchedulerAlgorithmSource{
			Provider: &o.AlgorithmProvider,
		}
	}

	// The following deprecated options affect the only existing profile that is
	// added by default.
	profile := &cfg.Profiles[0]
	if len(o.SchedulerName) > 0 {
		profile.SchedulerName = o.SchedulerName
	}
	if o.HardPodAffinitySymmetricWeight != interpodaffinity.DefaultHardPodAffinityWeight {
		args := interpodaffinity.Args{
			HardPodAffinityWeight: &o.HardPodAffinitySymmetricWeight,
		}
		profile.PluginConfig = append(profile.PluginConfig, plugins.NewPluginConfig(interpodaffinity.Name, args))
	}

	return nil
}
