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
	"github.com/spf13/pflag"

	"k8s.io/code-generator/cmd/client-gen/types"
)

// ClientGenArgs is a wrapper for arguments to client-gen.
type CustomArgs struct {
	// A sorted list of group versions to generate. For each of them the package path is found
	// in GroupVersionToInputPath.
	Groups []types.GroupVersions
	// GroupVersionToInputPath is a map between GroupVersion and the path to the respective
	// types.go, relative to InputBasePath. We still need GroupVersions in the
	// struct because we need an order.
	GroupVersionToInputPath map[types.GroupVersion]string
	// The base for the path of GroupVersionToInputPath.
	InputBasePath string

	// Overrides for which types should be included in the client.
	IncludedTypesOverrides map[types.GroupVersion][]string

	// ClientsetName is the name of the clientset to be generated. It's
	// populated from command-line arguments.
	ClientsetName string
	// ClientsetOutputPath is the path the clientset will be generated at. It's
	// populated from command-line arguments.
	ClientsetOutputPath string
	// ClientsetAPIPath is the default API path for generated clients.
	ClientsetAPIPath string
	// ClientsetOnly determines if we should generate the clients for groups and
	// types along with the clientset. It's populated from command-line
	// arguments.
	ClientsetOnly bool
	// FakeClient determines if client-gen generates the fake clients.
	FakeClient bool
}

func (ca *CustomArgs) AddFlags(fs *pflag.FlagSet) {
	pflag.Var(NewGVPackagesValue(&ca.GroupVersionToInputPath, &ca.Groups, nil), "input", "group/versions that client-gen will generate clients for. At most one version per group is allowed. Specified in the format \"group1/version1,group2/version2...\".")
	pflag.Var(NewGVTypesValue(&ca.IncludedTypesOverrides, []string{}), "included-types-overrides", "list of group/version/type for which client should be generated. By default, client is generated for all types which have genclient in types.go. This overrides that. For each groupVersion in this list, only the types mentioned here will be included. The default check of genclient will be used for other group versions.")
	pflag.StringVar(&ca.InputBasePath, "input-base", "k8s.io/kubernetes/pkg/apis", "base path to look for the api group.")
	pflag.StringVarP(&ca.ClientsetName, "clientset-name", "n", "internalclientset", "the name of the generated clientset package.")
	pflag.StringVarP(&ca.ClientsetAPIPath, "clientset-api-path", "", "", "the value of default API path.")
	pflag.StringVar(&ca.ClientsetOutputPath, "clientset-path", "k8s.io/kubernetes/pkg/client/clientset_generated/", "the generated clientset will be output to <clientset-path>/<clientset-name>.")
	pflag.BoolVar(&ca.ClientsetOnly, "clientset-only", false, "when set, client-gen only generates the clientset shell, without generating the individual typed clients")
	pflag.BoolVar(&ca.FakeClient, "fake-clientset", true, "when set, client-gen will generate the fake clientset that can be used in tests")
}
