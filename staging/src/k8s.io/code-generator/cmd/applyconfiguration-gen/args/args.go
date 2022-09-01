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
	"path"

	"github.com/spf13/pflag"
	"k8s.io/gengo/args"
	"k8s.io/gengo/types"

	codegenutil "k8s.io/code-generator/pkg/util"
)

// CustomArgs is a wrapper for arguments to applyconfiguration-gen.
type CustomArgs struct {
	// ExternalApplyConfigurations provides the locations of externally generated
	// apply configuration types for types referenced by the go structs provided as input.
	// Locations are provided as a comma separated list of <package>.<typeName>:<applyconfiguration-package>
	// entries.
	//
	// E.g. if a type references appsv1.Deployment, the location of its apply configuration should
	// be provided:
	//   k8s.io/api/apps/v1.Deployment:k8s.io/client-go/applyconfigurations/apps/v1
	//
	// meta/v1 types (TypeMeta and ObjectMeta) are always included and do not need to be passed in.
	ExternalApplyConfigurations map[types.Name]string

	OpenAPISchemaFilePath string
}

// NewDefaults returns default arguments for the generator.
func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	customArgs := &CustomArgs{
		ExternalApplyConfigurations: map[types.Name]string{
			// Always include TypeMeta and ObjectMeta. They are sufficient for the vast majority of use cases.
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "TypeMeta"}:       "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ObjectMeta"}:     "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "OwnerReference"}: "k8s.io/client-go/applyconfigurations/meta/v1",
		},
	}
	genericArgs.CustomArgs = customArgs

	if pkg := codegenutil.CurrentPackage(); len(pkg) != 0 {
		genericArgs.OutputPackagePath = path.Join(pkg, "pkg/client/applyconfigurations")
	}

	return genericArgs, customArgs
}

func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet, inputBase string) {
	pflag.Var(NewExternalApplyConfigurationValue(&ca.ExternalApplyConfigurations, nil), "external-applyconfigurations",
		"list of comma separated external apply configurations locations in <type-package>.<type-name>:<applyconfiguration-package> form."+
			"For example: k8s.io/api/apps/v1.Deployment:k8s.io/client-go/applyconfigurations/apps/v1")
	pflag.StringVar(&ca.OpenAPISchemaFilePath, "openapi-schema", "",
		"path to the openapi schema containing all the types that apply configurations will be generated for")
}

// Validate checks the given arguments.
func Validate(genericArgs *args.GeneratorArgs) error {
	if len(genericArgs.OutputPackagePath) == 0 {
		return fmt.Errorf("output package cannot be empty")
	}
	return nil
}
