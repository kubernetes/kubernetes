/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"

	"github.com/spf13/pflag"
	"k8s.io/apiserver/pkg/server"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	"k8s.io/utils/path"
)

// PriorityAndFairnessOptions holds the priority and fairness options.
type PriorityAndFairnessOptions struct {
	// ConfigFile is the file path with flow control configuration
	ConfigFile string
}

// NewPriorityAndFairnessOptions creates a new instance of PriorityAndFairnessOptions.
//
// The option is to point to a configuration file for priority and fairness parameters.
func NewPriorityAndFairnessOptions() *PriorityAndFairnessOptions {
	return &PriorityAndFairnessOptions{}
}

// AddFlags adds flags related to priority and fairness
// for a specific APIServer to the specified FlagSet.
func (o *PriorityAndFairnessOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.ConfigFile, "priority-and-fairness-config-file", o.ConfigFile,
		"Config file with apiserver flow control configuration")
}

// ApplyTo overrides default values initialised in PriorityAndFairnessConfiguration
// with values specified by a user in the PriorityAndFairnessOptions config file.
func (o *PriorityAndFairnessOptions) ApplyTo(c *server.Config) error {
	if o == nil {
		return nil
	}

	cfg, err := utilflowcontrol.ApplyConfigFromFileToDefaultConfiguration(o.ConfigFile)
	if err != nil {
		return fmt.Errorf("failed to read flow control config: %v", err)
	}

	if errs := utilflowcontrol.ValidatePriorityAndFairnessConfiguration(cfg); len(errs) > 0 {
		return fmt.Errorf("failed to validate priority and fairness configuration: %v", errs.ToAggregate())
	}

	c.PriorityAndFairnessConfig = *cfg
	return nil
}

// Validate verifies flags passed to PriorityAndFairnessOptions.
func (o *PriorityAndFairnessOptions) Validate() []error {
	if o == nil || o.ConfigFile == "" {
		return nil
	}

	errs := []error{}

	if exists, err := path.Exists(path.CheckFollowSymlink, o.ConfigFile); !exists || err != nil {
		errs = append(errs, fmt.Errorf("priority-and-fairness-config-file %s does not exist", o.ConfigFile))
	}

	return errs
}
