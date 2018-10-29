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

	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
)

// GarbageCollectorControllerOptions holds the GarbageCollectorController options.
type GarbageCollectorControllerOptions struct {
	ConcurrentGCSyncs      int32
	GCIgnoredResources     []kubectrlmgrconfig.GroupResource
	EnableGarbageCollector bool
}

// AddFlags adds flags related to GarbageCollectorController for controller manager to the specified FlagSet.
func (o *GarbageCollectorControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentGCSyncs, "concurrent-gc-syncs", o.ConcurrentGCSyncs, "The number of garbage collector workers that are allowed to sync concurrently.")
	fs.BoolVar(&o.EnableGarbageCollector, "enable-garbage-collector", o.EnableGarbageCollector, "Enables the generic garbage collector. MUST be synced with the corresponding flag of the kube-apiserver.")
}

// ApplyTo fills up GarbageCollectorController config with options.
func (o *GarbageCollectorControllerOptions) ApplyTo(cfg *kubectrlmgrconfig.GarbageCollectorControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentGCSyncs = o.ConcurrentGCSyncs
	cfg.GCIgnoredResources = o.GCIgnoredResources
	cfg.EnableGarbageCollector = o.EnableGarbageCollector

	return nil
}

// Validate checks validation of GarbageCollectorController.
func (o *GarbageCollectorControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
