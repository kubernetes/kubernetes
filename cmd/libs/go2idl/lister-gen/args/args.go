package args

import "k8s.io/kubernetes/pkg/api/unversioned"

type Args struct {
	// TODO: we should make another type declaration of GroupVersion out of the
	// unversioned package, which is part of our API. Tools like client-gen
	// shouldn't depend on an API.
	GroupVersions []unversioned.GroupVersion

	// GroupVersionToInputPath is a map between GroupVersion and the path to
	// the respective types.go. We still need GroupVersions in the struct because
	// we need an order.
	GroupVersionToInputPath map[unversioned.GroupVersion]string

	// Overrides for which types should be included in the client.
	IncludedTypesOverrides map[unversioned.GroupVersion][]string

	// CmdArgs is the command line arguments supplied when lister-gen is called.
	CmdArgs string
}
