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
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

// GarbageCollectorControllerOptions is part of context object for the controller manager.
type GarbageCollectorControllerOptions struct {
	ConcurrentGCSyncs      int32
	GCIgnoredResources     []componentconfig.GroupResource
	EnableGarbageCollector bool
}

// AddFlags adds flags related to debugging for controller manager to the specified FlagSet
func (o *GarbageCollectorControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentGCSyncs, "concurrent-gc-syncs", o.ConcurrentGCSyncs, "The number of garbage collector workers that are allowed to sync concurrently.")
	fs.BoolVar(&o.EnableGarbageCollector, "enable-garbage-collector", o.EnableGarbageCollector, "Enables the generic garbage collector. MUST be synced with the corresponding flag of the kube-apiserver.")
}

// ApplyTo fills up parts of controller manager config with options.
func (o *GarbageCollectorControllerOptions) ApplyTo(c *genericcontrollermanager.Config) error {
	if o == nil {
		return nil
	}

	c.ComponentConfig.GarbageCollectorControllerConfig.ConcurrentGCSyncs = o.ConcurrentGCSyncs
	c.ComponentConfig.GarbageCollectorControllerConfig.EnableGarbageCollector = o.EnableGarbageCollector

	return nil
}

// Validate checks validation of HPAControllerOptions.
func (o *GarbageCollectorControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
