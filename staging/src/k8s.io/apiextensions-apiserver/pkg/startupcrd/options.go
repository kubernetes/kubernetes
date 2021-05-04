/*
Copyright 2021 The Kubernetes Authors.

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

package startupcrd

import (
	"fmt"
	"github.com/spf13/pflag"
	"os"
)

const defaultExtraStartupCRDsDirectory = ""

// Options describes the runtime options of the startup CRD installer.
type Options struct {
	ExtraStartupCRDsDirectory string
}

// NewOptions creates default options of startup CRD installer.
func NewOptions() *Options {
	o := &Options{
		ExtraStartupCRDsDirectory: defaultExtraStartupCRDsDirectory,
	}

	return o
}

// AddFlags adds the startup CRD installer flags to the flagset.
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&o.ExtraStartupCRDsDirectory, "extra-startup-crds-directory",
		defaultExtraStartupCRDsDirectory, "Specify a path to any additional CRDs you want to install at startup")
}

// Validate validates the startup CRD options.
func (o *Options) Validate() []error {
	var errs []error

	// check if the specified directory exists if it is passed and non-empty string
	if _, err := os.Stat(o.ExtraStartupCRDsDirectory); os.IsNotExist(err) && o.ExtraStartupCRDsDirectory != "" {
		errs = append(errs, fmt.Errorf("Specififed ExtraStartupCRDsDirectory %s does not exist.", o.ExtraStartupCRDsDirectory))
	}

	return errs
}

// Complete fills in missing options.
func (o *Options) Complete() error {
	return nil
}
