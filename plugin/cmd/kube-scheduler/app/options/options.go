/*
Copyright 2014 The Kubernetes Authors.

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

// Package options provides the scheduler flags
package options

import (
	"fmt"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"
	// install the componentconfig api so we get its defaulting and conversion functions
	_ "k8s.io/kubernetes/pkg/apis/componentconfig/install"

	"github.com/spf13/pflag"
)

// SchedulerPolicyConfigMapKey defines the key of the element in the
// scheduler's policy ConfigMap that contains scheduler's policy config.
const SchedulerPolicyConfigMapKey string = "policy.cfg"

// SchedulerServer has all the context and params needed to run a Scheduler
type SchedulerServer struct {
	componentconfig.KubeSchedulerConfiguration
	// Master is the address of the Kubernetes API server (overrides any
	// value in kubeconfig).
	Master string
	// Kubeconfig is Path to kubeconfig file with authorization and master
	// location information.
	Kubeconfig string
	// Dynamic configuration for scheduler features.
}

// NewSchedulerServer creates a new SchedulerServer with default parameters
func NewSchedulerServer() *SchedulerServer {
	versioned := &v1alpha1.KubeSchedulerConfiguration{}
	legacyscheme.Scheme.Default(versioned)
	cfg := componentconfig.KubeSchedulerConfiguration{}
	legacyscheme.Scheme.Convert(versioned, &cfg, nil)
	cfg.LeaderElection.LeaderElect = true
	s := SchedulerServer{
		KubeSchedulerConfiguration: cfg,
	}
	return &s
}

// AddFlags adds flags for a specific SchedulerServer to the specified FlagSet
func (s *SchedulerServer) AddFlags(fs *pflag.FlagSet) {
	fs.Int32Var(&s.Port, "port", s.Port, "The port that the scheduler's http service runs on")
	fs.StringVar(&s.Address, "address", s.Address, "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&s.AlgorithmProvider, "algorithm-provider", s.AlgorithmProvider, "The scheduling algorithm provider to use, one of: "+factory.ListAlgorithmProviders())
	fs.StringVar(&s.PolicyConfigFile, "policy-config-file", s.PolicyConfigFile, "File with scheduler policy configuration. This file is used if policy ConfigMap is not provided or --use-legacy-policy-config==true")
	usage := fmt.Sprintf("Name of the ConfigMap object that contains scheduler's policy configuration. It must exist in the system namespace before scheduler initialization if --use-legacy-policy-config==false. The config must be provided as the value of an element in 'Data' map with the key='%v'", SchedulerPolicyConfigMapKey)
	fs.StringVar(&s.PolicyConfigMapName, "policy-configmap", s.PolicyConfigMapName, usage)
	fs.StringVar(&s.PolicyConfigMapNamespace, "policy-configmap-namespace", s.PolicyConfigMapNamespace, "The namespace where policy ConfigMap is located. The system namespace will be used if this is not provided or is empty.")
	fs.BoolVar(&s.UseLegacyPolicyConfig, "use-legacy-policy-config", false, "When set to true, scheduler will ignore policy ConfigMap and uses policy config file")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&s.EnableContentionProfiling, "contention-profiling", false, "Enable lock contention profiling, if profiling is enabled")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&s.ContentType, "kube-api-content-type", s.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
	fs.StringVar(&s.SchedulerName, "scheduler-name", s.SchedulerName, "Name of the scheduler, used to select which pods will be processed by this scheduler, based on pod's \"spec.SchedulerName\".")
	fs.StringVar(&s.LockObjectNamespace, "lock-object-namespace", s.LockObjectNamespace, "Define the namespace of the lock object.")
	fs.StringVar(&s.LockObjectName, "lock-object-name", s.LockObjectName, "Define the name of the lock object.")
	fs.IntVar(&s.HardPodAffinitySymmetricWeight, "hard-pod-affinity-symmetric-weight", api.DefaultHardPodAffinitySymmetricWeight,
		"RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule corresponding "+
			"to every RequiredDuringScheduling affinity rule. --hard-pod-affinity-symmetric-weight represents the weight of implicit PreferredDuringScheduling affinity rule.")
	fs.MarkDeprecated("hard-pod-affinity-symmetric-weight", "This option was moved to the policy configuration file")
	fs.StringVar(&s.FailureDomains, "failure-domains", kubeletapis.DefaultFailureDomains, "Indicate the \"all topologies\" set for an empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.")
	fs.MarkDeprecated("failure-domains", "Doesn't have any effect. Will be removed in future version.")
	leaderelectionconfig.BindFlags(&s.LeaderElection, fs)
	utilfeature.DefaultFeatureGate.AddFlag(fs)
}
