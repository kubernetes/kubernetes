/*
Copyright 2016 The Kubernetes Authors.

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

// Package options contains flags and options for initializing federation-apiserver.
package options

import (
	"time"

	genericoptions "k8s.io/kubernetes/pkg/genericapiserver/options"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"

	"github.com/spf13/pflag"
)

// Runtime options for the federation-apiserver.
type ServerRunOptions struct {
	GenericServerRunOptions *genericoptions.ServerRunOptions
	Etcd                    *genericoptions.EtcdOptions
	SecureServing           *genericoptions.SecureServingOptions
	InsecureServing         *genericoptions.ServingOptions
	Authentication          *kubeoptions.BuiltInAuthenticationOptions
	Authorization           *kubeoptions.BuiltInAuthorizationOptions
	CloudProvider           *kubeoptions.CloudProviderOptions

	EventTTL time.Duration
}

// NewServerRunOptions creates a new ServerRunOptions object with default values.
func NewServerRunOptions() *ServerRunOptions {
	s := ServerRunOptions{
		GenericServerRunOptions: genericoptions.NewServerRunOptions(),
		Etcd:            genericoptions.NewEtcdOptions(),
		SecureServing:   genericoptions.NewSecureServingOptions(),
		InsecureServing: genericoptions.NewInsecureServingOptions(),
		Authentication:  kubeoptions.NewBuiltInAuthenticationOptions().WithAll(),
		Authorization:   kubeoptions.NewBuiltInAuthorizationOptions(),
		CloudProvider:   kubeoptions.NewCloudProviderOptions(),

		EventTTL: 1 * time.Hour,
	}
	return &s
}

// AddFlags adds flags for ServerRunOptions fields to be specified via FlagSet.
func (s *ServerRunOptions) AddFlags(fs *pflag.FlagSet) {
	// Add the generic flags.
	s.GenericServerRunOptions.AddUniversalFlags(fs)
	s.Etcd.AddFlags(fs)
	s.SecureServing.AddFlags(fs)
	s.InsecureServing.AddFlags(fs)
	s.Authentication.AddFlags(fs)
	s.Authorization.AddFlags(fs)
	s.CloudProvider.AddFlags(fs)

	fs.DurationVar(&s.EventTTL, "event-ttl", s.EventTTL,
		"Amount of time to retain events. Default is 1h.")
}
