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
	genericcontrollermanager "k8s.io/kubernetes/cmd/controller-manager/app"
)

// ConcurrentResourcesSyncsOptions is part of context object for the controller manager.
type ConcurrentResourcesSyncsOptions struct {
	ConcurrentJobSyncs           int32
	ConcurrentDaemonSetSyncs     int32
	ConcurrentEndpointSyncs      int32
	ConcurrentRCSyncs            int32
	ConcurrentRSSyncs            int32
	ConcurrentDeploymentSyncs    int32
	ConcurrentSATokenSyncs       int32
	ConcurrentNamespaceSyncs     int32
	ConcurrentResourceQuotaSyncs int32
}

// AddFlags adds flags related to debugging for controller manager to the specified FlagSet
func (o *ConcurrentResourcesSyncsOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentEndpointSyncs, "concurrent-endpoint-syncs", o.ConcurrentEndpointSyncs, "The number of endpoint syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load")
	fs.Int32Var(&o.ConcurrentRCSyncs, "concurrent_rc_syncs", o.ConcurrentRCSyncs, "The number of replication controllers that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load")
	fs.Int32Var(&o.ConcurrentRSSyncs, "concurrent-replicaset-syncs", o.ConcurrentRSSyncs, "The number of replica sets that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load")
	fs.Int32Var(&o.ConcurrentResourceQuotaSyncs, "concurrent-resource-quota-syncs", o.ConcurrentResourceQuotaSyncs, "The number of resource quotas that are allowed to sync concurrently. Larger number = more responsive quota management, but more CPU (and network) load")
	fs.Int32Var(&o.ConcurrentDeploymentSyncs, "concurrent-deployment-syncs", o.ConcurrentDeploymentSyncs, "The number of deployment objects that are allowed to sync concurrently. Larger number = more responsive deployments, but more CPU (and network) load")
	fs.Int32Var(&o.ConcurrentSATokenSyncs, "concurrent-serviceaccount-token-syncs", o.ConcurrentSATokenSyncs, "The number of service account token objects that are allowed to sync concurrently. Larger number = more responsive token generation, but more CPU (and network) load")
	fs.Int32Var(&o.ConcurrentNamespaceSyncs, "concurrent-namespace-syncs", o.ConcurrentNamespaceSyncs, "The number of namespace objects that are allowed to sync concurrently. Larger number = more responsive namespace termination, but more CPU (and network) load")
}

// ApplyTo fills up parts of controller manager config with options.
func (o *ConcurrentResourcesSyncsOptions) ApplyTo(c *genericcontrollermanager.Config) error {
	if o == nil {
		return nil
	}

	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentJobSyncs = o.ConcurrentJobSyncs
	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentDaemonSetSyncs = o.ConcurrentDaemonSetSyncs
	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentEndpointSyncs = o.ConcurrentEndpointSyncs
	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentRCSyncs = o.ConcurrentRCSyncs
	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentRSSyncs = o.ConcurrentRSSyncs
	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentDeploymentSyncs = o.ConcurrentDeploymentSyncs
	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentSATokenSyncs = o.ConcurrentSATokenSyncs
	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentNamespaceSyncs = o.ConcurrentNamespaceSyncs
	c.ComponentConfig.ConcurrentResourcesSyncsConfig.ConcurrentResourceQuotaSyncs = o.ConcurrentResourceQuotaSyncs

	return nil
}

// Validate checks validation of ConcurrentResourcesSyncsOptions.
func (o *ConcurrentResourcesSyncsOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
