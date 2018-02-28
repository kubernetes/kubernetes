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
	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericcontrollermanager "k8s.io/kubernetes/cmd/controller-manager/app"
)

// ResourceQuotaControllerOptions is part of context object for the controller manager.
type ResourceQuotaControllerOptions struct {
	ResourceQuotaSyncPeriod metav1.Duration
}

// AddFlags adds flags related to debugging for controller manager to the specified FlagSet
func (o *ResourceQuotaControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.DurationVar(&o.ResourceQuotaSyncPeriod.Duration, "resource-quota-sync-period", o.ResourceQuotaSyncPeriod.Duration, "The period for syncing quota usage status in the system")
}

// ApplyTo fills up parts of controller manager config with options.
func (o *ResourceQuotaControllerOptions) ApplyTo(c *genericcontrollermanager.Config) error {
	if o == nil {
		return nil
	}

	c.ComponentConfig.ResourceQuotaControllerConfig.ResourceQuotaSyncPeriod = o.ResourceQuotaSyncPeriod

	return nil
}

// Validate checks validation of ResourceQuotaControllerOptions.
func (o *ResourceQuotaControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
