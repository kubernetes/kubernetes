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

package args

import (
	"fmt"

	"github.com/spf13/pflag"
	"k8s.io/gengo/args"
)

// CustomArgs is used by the gengo framework to pass args specific to this generator.
type CustomArgs struct {
	VersionedClientSetPackage        string
	ExternalVersionsInformersPackage string
	ListersPackage                   string
	ForceKinds                       string
}

// NewDefaults returns default arguments for the generator.
func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	customArgs := &CustomArgs{}
	genericArgs.CustomArgs = customArgs
	return genericArgs, customArgs
}

// AddFlags add the generator flags to the flag set.
func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&ca.VersionedClientSetPackage, "versioned-clientset-package", ca.VersionedClientSetPackage, "the full package name for the versioned injection clientset to use")
	fs.StringVar(&ca.ExternalVersionsInformersPackage, "external-versions-informers-package", ca.ExternalVersionsInformersPackage, "the full package name for the external versions injection informer to use")
	fs.StringVar(&ca.ListersPackage, "listers-package", ca.ListersPackage, "the full package name for client listers to use")
	fs.StringVar(&ca.ForceKinds, "force-genreconciler-kinds", ca.ForceKinds, `force kinds will override the genreconciler tag setting for the given set of kinds, comma separated: "Foo,Bar,Baz"`)
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	customArgs := genericArgs.CustomArgs.(*CustomArgs)

	if len(genericArgs.OutputPackagePath) == 0 {
		return fmt.Errorf("output package cannot be empty")
	}
	if len(customArgs.VersionedClientSetPackage) == 0 {
		return fmt.Errorf("versioned clientset package cannot be empty")
	}
	if len(customArgs.ExternalVersionsInformersPackage) == 0 {
		return fmt.Errorf("external versions informers package cannot be empty")
	}

	return nil
}
