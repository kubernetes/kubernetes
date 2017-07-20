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

import "k8s.io/kube-gen/cmd/client-gen/types"

// ClientGenArgs is a wrapper for arguments to client-gen.
type Args struct {
	Groups []types.GroupVersions

	// GroupVersionToInputPath is a map between GroupVersion and the path to
	// the respective types.go. We still need GroupVersions in the struct because
	// we need an order.
	GroupVersionToInputPath map[types.GroupVersion]string

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
	// CmdArgs is the command line arguments supplied when the client-gen is called.
	CmdArgs string
}
