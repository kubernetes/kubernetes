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

	deploymentconfig "k8s.io/kubernetes/pkg/controller/deployment/config"
)

// DeploymentControllerOptions holds the DeploymentController options.
type DeploymentControllerOptions struct {
	*deploymentconfig.DeploymentControllerConfiguration
}

// AddFlags adds flags related to DeploymentController for controller manager to the specified FlagSet.
func (o *DeploymentControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentDeploymentSyncs, "concurrent-deployment-syncs", o.ConcurrentDeploymentSyncs, "The number of deployment objects that are allowed to sync concurrently. Larger number = more responsive deployments, but more CPU (and network) load")
}

// ApplyTo fills up DeploymentController config with options.
func (o *DeploymentControllerOptions) ApplyTo(cfg *deploymentconfig.DeploymentControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentDeploymentSyncs = o.ConcurrentDeploymentSyncs

	return nil
}

// Validate checks validation of DeploymentControllerOptions.
func (o *DeploymentControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
