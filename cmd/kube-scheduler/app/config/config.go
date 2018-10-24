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

package config

import (
	apiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/record"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// Config has all the context to run a Scheduler
type Config struct {
	// config is the scheduler server's configuration object.
	ComponentConfig kubeschedulerconfig.KubeSchedulerConfiguration

	InsecureServing        *apiserver.DeprecatedInsecureServingInfo // nil will disable serving on an insecure port
	InsecureMetricsServing *apiserver.DeprecatedInsecureServingInfo // non-nil if metrics should be served independently
	Authentication         apiserver.AuthenticationInfo
	Authorization          apiserver.AuthorizationInfo
	SecureServing          *apiserver.SecureServingInfo

	Client          clientset.Interface
	InformerFactory informers.SharedInformerFactory
	PodInformer     coreinformers.PodInformer
	EventClient     v1core.EventsGetter
	Recorder        record.EventRecorder
	Broadcaster     record.EventBroadcaster

	// LeaderElection is optional.
	LeaderElection *leaderelection.LeaderElectionConfig
}

type completedConfig struct {
	*Config
}

// CompletedConfig same as Config, just to swap private object.
type CompletedConfig struct {
	// Embed a private pointer that cannot be instantiated outside of this package.
	*completedConfig
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() CompletedConfig {
	cc := completedConfig{c}

	if c.InsecureServing != nil {
		c.InsecureServing.Name = "healthz"
	}
	if c.InsecureMetricsServing != nil {
		c.InsecureMetricsServing.Name = "metrics"
	}

	return CompletedConfig{&cc}
}
