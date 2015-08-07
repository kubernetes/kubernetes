/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package latest

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/registered"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	_ "k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/v1alpha1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

var (
	// GroupVersion represents the current external default version of the group. It
	// is in the form of "group/version".
	GroupVersion string

	// LatestVersion represents the current external default version of the group.
	// It equals to the "version" part of GroupVersion.
	Version string

	// Group represents the name of the group
	Group string

	// GroupVersions is the list of groupVersions in the form of "group/version"
	// that are recognized in code. The order provided may be assumed to be least
	// feature rich to most feature rich, and clients may choose to prefer the
	// latter items in the list over the former items when presented with a set of
	// versions to choose.
	GroupVersions []string

	// Versions is the "version" part of GroupVersions
	Versions []string

	accessor   = meta.NewAccessor()
	Codec      runtime.Codec
	SelfLinker = runtime.SelfLinker(accessor)
	RESTMapper meta.RESTMapper
)

const importPrefix = "k8s.io/kubernetes/pkg/expapi"

func init() {
	expGroupVersions := registered.GroupVersionsForGroup("experimental")
	if len(expGroupVersions) == 0 {
		return
	}
	GroupVersion = expGroupVersions[0]
	Group = apiutil.GetGroup(GroupVersion)
	Version = apiutil.GetVersion(GroupVersion)
	Codec = runtime.CodecFor(api.Scheme, GroupVersion)
	// Put the registered versions in Versions in reverse order.
	for i := len(expGroupVersions) - 1; i >= 0; i-- {
		GroupVersions = append(GroupVersions, expGroupVersions[i])
		Versions = append(Versions, apiutil.GetVersion(expGroupVersions[i]))
	}

	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := util.NewStringSet()

	ignoredKinds := util.NewStringSet()

	RESTMapper = api.NewDefaultRESTMapper(Group, GroupVersions, InterfacesFor, importPrefix, ignoredKinds, rootScoped)
	api.RegisterRESTMapper(RESTMapper)
}

// InterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func InterfacesFor(version string) (*meta.VersionInterfaces, error) {
	switch version {
	case "v1alpha1":
		return &meta.VersionInterfaces{
			Codec:            v1alpha1.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: accessor,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %s)", version, strings.Join(GroupVersions, ", "))
	}
}
