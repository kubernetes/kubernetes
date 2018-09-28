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

// ReplicaSetControllerOptions holds the ReplicaSetController options.
type ReplicaSetControllerOptions struct {
	ConcurrentRSSyncs int32
}

// AddFlags adds flags related to ReplicaSetController for controller manager to the specified FlagSet.
func (o *ReplicaSetControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentRSSyncs, "concurrent-replicaset-syncs", o.ConcurrentRSSyncs, "The number of replica sets that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load")
}

// ApplyTo fills up ReplicaSetController config with options.
func (o *ReplicaSetControllerOptions) ApplyTo(cfg *kubectrlmgrconfig.ReplicaSetControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentRSSyncs = o.ConcurrentRSSyncs

	return nil
}

// Validate checks validation of ReplicaSetControllerOptions.
func (o *ReplicaSetControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
