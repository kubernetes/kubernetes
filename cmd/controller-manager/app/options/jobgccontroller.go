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
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

// JobGCControllerOptions holds the JobGCController options.
type JobGCControllerOptions struct {
	FinishedJobGCThreshold int32
}

// AddFlags adds flags related to JobGCController for controller manager to the specified FlagSet.
func (o *JobGCControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.FinishedJobGCThreshold, "finished-job-gc-threshold", o.FinishedJobGCThreshold, "Number of finished jobs that can exist before the finished job garbage collector starts deleting finished jobs. If <= 0, the finished job garbage collector is disabled.")
}

// ApplyTo fills up JobGCController config with options.
func (o *JobGCControllerOptions) ApplyTo(cfg *componentconfig.JobGCControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.FinishedJobGCThreshold = o.FinishedJobGCThreshold

	return nil
}

// Validate checks validation of JobGCControllerOptions.
func (o *JobGCControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
