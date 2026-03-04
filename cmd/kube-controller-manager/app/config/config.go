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
	"k8s.io/apiserver/pkg/server/flagz"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	basecompatibility "k8s.io/component-base/compatibility"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	"time"
)

// Config is the main context object for the controller manager.
type Config struct {
	// Flagz is the Reader interface to get flags for the flagz page.
	Flagz flagz.Reader

	ComponentConfig kubectrlmgrconfig.KubeControllerManagerConfiguration

	SecureServing *apiserver.SecureServingInfo

	Authentication apiserver.AuthenticationInfo
	Authorization  apiserver.AuthorizationInfo

	// the general kube client
	Client *clientset.Clientset

	// the rest config for the master
	Kubeconfig *restclient.Config

	EventBroadcaster record.EventBroadcaster
	EventRecorder    record.EventRecorder

	ControllerShutdownTimeout time.Duration

	// ComponentGlobalsRegistry is the registry where the effective versions and feature gates for all components are stored.
	ComponentGlobalsRegistry basecompatibility.ComponentGlobalsRegistry
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
func (c *Config) Complete() *CompletedConfig {
	cc := completedConfig{c}

	return &CompletedConfig{&cc}
}
