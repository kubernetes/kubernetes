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

package api

import (
	"strings"

	"k8s.io/kubernetes/pkg/api/rest/restmapper"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/sets"
)

var RESTMapper restmapper.RESTMapper

func init() {
	RESTMapper = restmapper.MultiRESTMapper{}
}

func RegisterRESTMapper(m restmapper.RESTMapper) {
	RESTMapper = append(RESTMapper.(restmapper.MultiRESTMapper), m)
}

func NewDefaultRESTMapper(defaultGroupVersions []unversioned.GroupVersion, interfacesFunc restmapper.VersionInterfacesFunc,
	importPathPrefix string, ignoredKinds, rootScoped sets.String) *restmapper.DefaultRESTMapper {

	mapper := restmapper.NewDefaultRESTMapper(defaultGroupVersions, interfacesFunc)
	// enumerate all supported versions, get the kinds, and register with the mapper how to address
	// our resources.
	for _, gv := range defaultGroupVersions {
		for kind, oType := range Scheme.KnownTypes(gv) {
			gvk := gv.WithKind(kind)
			// TODO: Remove import path prefix check.
			// We check the import path prefix because we currently stuff both "api" and "extensions" objects
			// into the same group within Scheme since Scheme has no notion of groups yet.
			if !strings.HasPrefix(oType.PkgPath(), importPathPrefix) || ignoredKinds.Has(kind) {
				continue
			}
			scope := restmapper.RESTScopeNamespace
			if rootScoped.Has(kind) {
				scope = restmapper.RESTScopeRoot
			}
			mapper.Add(gvk, scope, false)
		}
	}
	return mapper
}
