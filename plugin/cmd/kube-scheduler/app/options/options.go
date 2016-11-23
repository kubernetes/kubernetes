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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	"k8s.io/kubernetes/pkg/util/config"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/spf13/pflag"
)

// SchedulerServer has all the context and params needed to run a Scheduler
type SchedulerServer struct {
	componentconfig.KubeSchedulerConfiguration
	// Master is the address of the Kubernetes API server (overrides any
	// value in kubeconfig).
	Master string
	// Kubeconfig is Path to kubeconfig file with authorization and master
	// location information.
	Kubeconfig string
	// Dynamic conifguration for scheduler features.
}

// NewSchedulerServer creates a new SchedulerServer with default parameters
func NewSchedulerServer() *SchedulerServer {
	versioned := &v1alpha1.KubeSchedulerConfiguration{}
	api.Scheme.Default(versioned)
	cfg := componentconfig.KubeSchedulerConfiguration{}
	api.Scheme.Convert(versioned, &cfg, nil)
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
	fs.StringVar(&s.PolicyConfigFile, "policy-config-file", s.PolicyConfigFile, "File with scheduler policy configuration")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&s.EnableContentionProfiling, "contention-profiling", false, "Enable lock contention profiling, if profiling is enabled")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&s.ContentType, "kube-api-content-type", s.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
	fs.StringVar(&s.SchedulerName, "scheduler-name", s.SchedulerName, "Name of the scheduler, used to select which pods will be processed by this scheduler, based on pod's annotation with key 'scheduler.alpha.kubernetes.io/name'")
	fs.IntVar(&s.HardPodAffinitySymmetricWeight, "hard-pod-affinity-symmetric-weight", api.DefaultHardPodAffinitySymmetricWeight,
		"RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule corresponding "+
			"to every RequiredDuringScheduling affinity rule. --hard-pod-affinity-symmetric-weight represents the weight of implicit PreferredDuringScheduling affinity rule.")
	fs.StringVar(&s.FailureDomains, "failure-domains", api.DefaultFailureDomains, "Indicate the \"all topologies\" set for an empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.")
	leaderelection.BindFlags(&s.LeaderElection, fs)
	config.DefaultFeatureGate.AddFlag(fs)
}
