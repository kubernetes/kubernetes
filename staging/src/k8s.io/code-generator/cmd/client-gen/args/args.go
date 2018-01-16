/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/code-generator/cmd/client-gen/types"
	codegenutil "k8s.io/code-generator/pkg/util"
)

var DefaultInputDirs = []string{
	"k8s.io/apimachinery/pkg/fields",
	"k8s.io/apimachinery/pkg/labels",
	"k8s.io/apimachinery/pkg/watch",
	"k8s.io/apimachinery/pkg/apimachinery/registered",
}

// ClientGenArgs is a wrapper for arguments to client-gen.
type CustomArgs struct {
	// A sorted list of group versions to generate. For each of them the package path is found
	// in GroupVersionToInputPath.
	Groups []types.GroupVersions

	// Overrides for which types should be included in the client.
	IncludedTypesOverrides map[types.GroupVersion][]string

	// ClientsetName is the name of the clientset to be generated. It's
	// populated from command-line arguments.
	ClientsetName string
	// ClientsetAPIPath is the default API HTTP path for generated clients.
	ClientsetAPIPath string
	// ClientsetOnly determines if we should generate the clients for groups and
	// types along with the clientset. It's populated from command-line
	// arguments.
	ClientsetOnly bool
	// FakeClient determines if client-gen generates the fake clients.
	FakeClient bool
}

func NewDefaults() (*args.GeneratorArgs, *CustomArgs) {
	genericArgs := args.Default().WithoutDefaultFlagParsing()
	customArgs := &CustomArgs{
		ClientsetName:    "internalclientset",
		ClientsetAPIPath: "/apis",
		ClientsetOnly:    false,
		FakeClient:       true,
	}
	genericArgs.CustomArgs = customArgs
	genericArgs.InputDirs = DefaultInputDirs

	if pkg := codegenutil.CurrentPackage(); len(pkg) != 0 {
		genericArgs.OutputPackagePath = path.Join(pkg, "pkg/client/clientset")
	}

	return genericArgs, customArgs
}

func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet, inputBase string) {
	gvsBuilder := NewGroupVersionsBuilder(&ca.Groups)
	pflag.Var(NewGVPackagesValue(gvsBuilder, nil), "input", "group/versions that client-gen will generate clients for. At most one version per group is allowed. Specified in the format \"group1/version1,group2/version2...\".")
	pflag.Var(NewGVTypesValue(&ca.IncludedTypesOverrides, []string{}), "included-types-overrides", "list of group/version/type for which client should be generated. By default, client is generated for all types which have genclient in types.go. This overrides that. For each groupVersion in this list, only the types mentioned here will be included. The default check of genclient will be used for other group versions.")
	pflag.Var(NewInputBasePathValue(gvsBuilder, inputBase), "input-base", "base path to look for the api group.")
	pflag.StringVarP(&ca.ClientsetName, "clientset-name", "n", ca.ClientsetName, "the name of the generated clientset package.")
	pflag.StringVarP(&ca.ClientsetAPIPath, "clientset-api-path", "", ca.ClientsetAPIPath, "the value of default API HTTP path, starting with / and without trailing /.")
	pflag.BoolVar(&ca.ClientsetOnly, "clientset-only", ca.ClientsetOnly, "when set, client-gen only generates the clientset shell, without generating the individual typed clients")
	pflag.BoolVar(&ca.FakeClient, "fake-clientset", ca.FakeClient, "when set, client-gen will generate the fake clientset that can be used in tests")

	// support old flags
	fs.SetNormalizeFunc(mapFlagName("clientset-path", "output-package", fs.GetNormalizeFunc()))
}

func Validate(genericArgs *args.GeneratorArgs) error {
	customArgs := genericArgs.CustomArgs.(*CustomArgs)

	if len(genericArgs.OutputPackagePath) == 0 {
		return fmt.Errorf("output package cannot be empty")
	}
	if len(customArgs.ClientsetName) == 0 {
		return fmt.Errorf("clientset name cannot be empty")
	}
	if len(customArgs.ClientsetAPIPath) == 0 {
		return fmt.Errorf("clientset API path cannot be empty")
	}

	return nil
}

// GroupVersionPackages returns a map from GroupVersion to the package with the types.go.
func (ca *CustomArgs) GroupVersionPackages() map[types.GroupVersion]string {
	res := map[types.GroupVersion]string{}
	for _, pkg := range ca.Groups {
		for _, v := range pkg.Versions {
			res[types.GroupVersion{Group: pkg.Group, Version: v.Version}] = v.Package
		}
	}
	return res
}

func mapFlagName(from, to string, old func(fs *pflag.FlagSet, name string) pflag.NormalizedName) func(fs *pflag.FlagSet, name string) pflag.NormalizedName {
	return func(fs *pflag.FlagSet, name string) pflag.NormalizedName {
		if name == from {
			name = to
		}
		return old(fs, name)
	}
}
