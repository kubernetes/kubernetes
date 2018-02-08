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

package options

import (
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

// RecommendedOptions contains the recommended options for running an API server.
// If you add something to this list, it should be in a logical grouping.
// Each of them can be nil to leave the feature unconfigured on ApplyTo.
type RecommendedOptions struct {
	Etcd           *EtcdOptions
	SecureServing  *SecureServingOptions
	Authentication *DelegatingAuthenticationOptions
	Authorization  *DelegatingAuthorizationOptions
	Audit          *AuditOptions
	Features       *FeatureOptions
	CoreAPI        *CoreAPIOptions

	// ExtraAdmissionInitializers is called once after all ApplyTo from the options above, to pass the returned
	// admission plugin initializers to Admission.ApplyTo.
	ExtraAdmissionInitializers func(c *server.RecommendedConfig) ([]admission.PluginInitializer, error)
	Admission                  *AdmissionOptions
}

func NewRecommendedOptions(prefix string, codec runtime.Codec) *RecommendedOptions {
	return &RecommendedOptions{
		Etcd:                       NewEtcdOptions(storagebackend.NewDefaultConfig(prefix, codec)),
		SecureServing:              NewSecureServingOptions(),
		Authentication:             NewDelegatingAuthenticationOptions(),
		Authorization:              NewDelegatingAuthorizationOptions(),
		Audit:                      NewAuditOptions(),
		Features:                   NewFeatureOptions(),
		CoreAPI:                    NewCoreAPIOptions(),
		ExtraAdmissionInitializers: func(c *server.RecommendedConfig) ([]admission.PluginInitializer, error) { return nil, nil },
		Admission:                  NewAdmissionOptions(),
	}
}

func (o *RecommendedOptions) AddFlags(fs *pflag.FlagSet) {
	o.Etcd.AddFlags(fs)
	o.SecureServing.AddFlags(fs)
	o.Authentication.AddFlags(fs)
	o.Authorization.AddFlags(fs)
	o.Audit.AddFlags(fs)
	o.Features.AddFlags(fs)
	o.CoreAPI.AddFlags(fs)
	o.Admission.AddFlags(fs)
}

// ApplyTo adds RecommendedOptions to the server configuration.
// scheme is the scheme of the apiserver types that are sent to the admission chain.
// pluginInitializers can be empty, it is only need for additional initializers.
func (o *RecommendedOptions) ApplyTo(config *server.RecommendedConfig, scheme *runtime.Scheme) error {
	if err := o.Etcd.ApplyTo(&config.Config); err != nil {
		return err
	}
	if err := o.SecureServing.ApplyTo(&config.Config); err != nil {
		return err
	}
	if err := o.Authentication.ApplyTo(&config.Config); err != nil {
		return err
	}
	if err := o.Authorization.ApplyTo(&config.Config); err != nil {
		return err
	}
	if err := o.Audit.ApplyTo(&config.Config); err != nil {
		return err
	}
	if err := o.Features.ApplyTo(&config.Config); err != nil {
		return err
	}
	if err := o.CoreAPI.ApplyTo(config); err != nil {
		return err
	}
	if initializers, err := o.ExtraAdmissionInitializers(config); err != nil {
		return err
	} else if err := o.Admission.ApplyTo(&config.Config, config.SharedInformerFactory, config.ClientConfig, scheme, initializers...); err != nil {
		return err
	}

	return nil
}

func (o *RecommendedOptions) Validate() []error {
	errors := []error{}
	errors = append(errors, o.Etcd.Validate()...)
	errors = append(errors, o.SecureServing.Validate()...)
	errors = append(errors, o.Authentication.Validate()...)
	errors = append(errors, o.Authorization.Validate()...)
	errors = append(errors, o.Audit.Validate()...)
	errors = append(errors, o.Features.Validate()...)
	errors = append(errors, o.CoreAPI.Validate()...)
	errors = append(errors, o.Admission.Validate()...)

	return errors
}
