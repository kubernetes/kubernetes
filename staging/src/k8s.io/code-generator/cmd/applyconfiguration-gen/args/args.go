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
	"k8s.io/gengo/v2/types"
)

// Args is a wrapper for arguments to applyconfiguration-gen.
type Args struct {
	OutputDir string // must be a directory path
	OutputPkg string // must be a Go import-path

	GoHeaderFile string

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

// New returns default arguments for the generator.
func New() *Args {
	return &Args{
		ExternalApplyConfigurations: map[types.Name]string{
			// Always include the applyconfigurations we've generated in client-go. They are sufficient for the vast majority of use cases.
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "Condition"}:                "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "DeleteOptions"}:            "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "LabelSelector"}:            "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "LabelSelectorRequirement"}: "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ManagedFieldsEntry"}:       "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "ObjectMeta"}:               "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "OwnerReference"}:           "k8s.io/client-go/applyconfigurations/meta/v1",
			{Package: "k8s.io/apimachinery/pkg/apis/meta/v1", Name: "TypeMeta"}:                 "k8s.io/client-go/applyconfigurations/meta/v1",
		},
	}
}

func (args *Args) AddFlags(fs *pflag.FlagSet, inputBase string) {
	fs.StringVar(&args.OutputDir, "output-dir", "",
		"the base directory under which to generate results")
	fs.StringVar(&args.OutputPkg, "output-pkg", args.OutputPkg,
		"the Go import-path of the generated results")
	fs.StringVar(&args.GoHeaderFile, "go-header-file", "",
		"the path to a file containing boilerplate header text; the string \"YEAR\" will be replaced with the current 4-digit year")
	fs.Var(NewExternalApplyConfigurationValue(&args.ExternalApplyConfigurations, nil), "external-applyconfigurations",
		"list of comma separated external apply configurations locations in <type-package>.<type-name>:<applyconfiguration-package> form."+
			"For example: k8s.io/api/apps/v1.Deployment:k8s.io/client-go/applyconfigurations/apps/v1")
	fs.StringVar(&args.OpenAPISchemaFilePath, "openapi-schema", "",
		"path to the openapi schema containing all the types that apply configurations will be generated for")
}

// Validate checks the given arguments.
func (args *Args) Validate() error {
	if len(args.OutputDir) == 0 {
		return fmt.Errorf("--output-dir must be specified")
	}
	if len(args.OutputPkg) == 0 {
		return fmt.Errorf("--output-pkg must be specified")
	}
	return nil
}
