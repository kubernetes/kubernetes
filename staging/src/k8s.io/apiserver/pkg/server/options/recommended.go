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
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

// RecommendedOptions contains the recommended options for running an API server
// If you add something to this list, it should be in a logical grouping
type RecommendedOptions struct {
	Etcd           *EtcdOptions
	SecureServing  *SecureServingOptions
	Authentication *DelegatingAuthenticationOptions
	Authorization  *DelegatingAuthorizationOptions
	Audit          *AuditOptions
	Features       *FeatureOptions
}

func NewRecommendedOptions(prefix string, copier runtime.ObjectCopier, codec runtime.Codec) *RecommendedOptions {
	return &RecommendedOptions{
		Etcd:           NewEtcdOptions(storagebackend.NewDefaultConfig(prefix, copier, codec)),
		SecureServing:  NewSecureServingOptions(),
		Authentication: NewDelegatingAuthenticationOptions(),
		Authorization:  NewDelegatingAuthorizationOptions(),
		Audit:          NewAuditOptions(),
		Features:       NewFeatureOptions(),
	}
}

func (o *RecommendedOptions) AddFlags(fs *pflag.FlagSet) {
	o.Etcd.AddFlags(fs)
	o.SecureServing.AddFlags(fs)
	o.Authentication.AddFlags(fs)
	o.Authorization.AddFlags(fs)
	o.Audit.AddFlags(fs)
	o.Features.AddFlags(fs)
}

func (o *RecommendedOptions) ApplyTo(config *server.Config) error {
	if err := o.Etcd.ApplyTo(config); err != nil {
		return err
	}
	if err := o.SecureServing.ApplyTo(config); err != nil {
		return err
	}
	if err := o.Authentication.ApplyTo(config); err != nil {
		return err
	}
	if err := o.Authorization.ApplyTo(config); err != nil {
		return err
	}
	if err := o.Audit.ApplyTo(config); err != nil {
		return err
	}
	if err := o.Features.ApplyTo(config); err != nil {
		return err
	}

	return nil
}
