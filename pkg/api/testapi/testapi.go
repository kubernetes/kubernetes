/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Package testapi provides a helper for retrieving the KUBE_TEST_API environment variable.
package testapi

import (
	"fmt"
	"os"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"

	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/meta"
	apiutil "k8s.io/kubernetes/pkg/api/util"
	"k8s.io/kubernetes/pkg/runtime"
)

var (
	Groups     = make(map[string]TestGroup)
	Default    TestGroup
	Extensions TestGroup
)

type TestGroup struct {
	// Name of the group
	Group string
	// Version of the group Group under test
	VersionUnderTest string
	// Group and Version. In most cases equals to Group + "/" + VersionUnverTest
	GroupVersionUnderTest string
}

func init() {
	kubeTestAPI := os.Getenv("KUBE_TEST_API")
	if kubeTestAPI != "" {
		testGroupVersions := strings.Split(kubeTestAPI, ",")
		for _, groupVersion := range testGroupVersions {
			// TODO: caesarxuchao: the apiutil package is hacky, it will be replaced
			// by a following PR.
			Groups[apiutil.GetGroup(groupVersion)] =
				TestGroup{apiutil.GetGroup(groupVersion), apiutil.GetVersion(groupVersion), groupVersion}
		}
	}

	// TODO: caesarxuchao: we need a central place to store all available API
	// groups and their metadata.
	if _, ok := Groups[""]; !ok {
		// TODO: The second latest.GroupOrDie("").Version will be latest.GroupVersion after we
		// have multiple group support
		Groups[""] = TestGroup{"", latest.GroupOrDie("").Version, latest.GroupOrDie("").GroupVersion}
	}
	if _, ok := Groups["extensions"]; !ok {
		Groups["extensions"] = TestGroup{"extensions", latest.GroupOrDie("extensions").Version, latest.GroupOrDie("extensions").GroupVersion}
	}

	Default = Groups[""]
	Extensions = Groups["extensions"]
}

// Version returns the API version to test against, as set by the KUBE_TEST_API env var.
func (g TestGroup) Version() string {
	return g.VersionUnderTest
}

// GroupAndVersion returns the API version to test against for a group, as set
// by the KUBE_TEST_API env var.
// Return value is in the form of "group/version".
func (g TestGroup) GroupAndVersion() string {
	return g.GroupVersionUnderTest
}

// Codec returns the codec for the API version to test against, as set by the
// KUBE_TEST_API env var.
func (g TestGroup) Codec() runtime.Codec {
	// TODO: caesarxuchao: Restructure the body once we have a central `latest`.
	if g.Group == "" {
		interfaces, err := latest.GroupOrDie("").InterfacesFor(g.GroupVersionUnderTest)
		if err != nil {
			panic(err)
		}
		return interfaces.Codec
	}
	if g.Group == "extensions" {
		interfaces, err := latest.GroupOrDie("extensions").InterfacesFor(g.GroupVersionUnderTest)
		if err != nil {
			panic(err)
		}
		return interfaces.Codec
	}
	panic(fmt.Errorf("cannot test group %s", g.Group))
}

// Converter returns the api.Scheme for the API version to test against, as set by the
// KUBE_TEST_API env var.
func (g TestGroup) Converter() runtime.ObjectConvertor {
	// TODO: caesarxuchao: Restructure the body once we have a central `latest`.
	if g.Group == "" {
		interfaces, err := latest.GroupOrDie("").InterfacesFor(g.VersionUnderTest)
		if err != nil {
			panic(err)
		}
		return interfaces.ObjectConvertor
	}
	if g.Group == "extensions" {
		interfaces, err := latest.GroupOrDie("extensions").InterfacesFor(g.VersionUnderTest)
		if err != nil {
			panic(err)
		}
		return interfaces.ObjectConvertor
	}
	panic(fmt.Errorf("cannot test group %s", g.Group))

}

// MetadataAccessor returns the MetadataAccessor for the API version to test against,
// as set by the KUBE_TEST_API env var.
func (g TestGroup) MetadataAccessor() meta.MetadataAccessor {
	// TODO: caesarxuchao: Restructure the body once we have a central `latest`.
	if g.Group == "" {
		interfaces, err := latest.GroupOrDie("").InterfacesFor(g.VersionUnderTest)
		if err != nil {
			panic(err)
		}
		return interfaces.MetadataAccessor
	}
	if g.Group == "extensions" {
		interfaces, err := latest.GroupOrDie("extensions").InterfacesFor(g.VersionUnderTest)
		if err != nil {
			panic(err)
		}
		return interfaces.MetadataAccessor
	}
	panic(fmt.Errorf("cannot test group %s", g.Group))
}

// SelfLink returns a self link that will appear to be for the version Version().
// 'resource' should be the resource path, e.g. "pods" for the Pod type. 'name' should be
// empty for lists.
func (g TestGroup) SelfLink(resource, name string) string {
	if g.Group == "" {
		if name == "" {
			return fmt.Sprintf("/api/%s/%s", g.Version(), resource)
		}
		return fmt.Sprintf("/api/%s/%s/%s", g.Version(), resource, name)
	} else {
		// TODO: will need a /apis prefix once we have proper multi-group
		// support
		if name == "" {
			return fmt.Sprintf("/apis/%s/%s/%s", g.Group, g.Version(), resource)
		}
		return fmt.Sprintf("/apis/%s/%s/%s/%s", g.Group, g.Version(), resource, name)
	}
}

// Returns the appropriate path for the given prefix (watch, proxy, redirect, etc), resource, namespace and name.
// For ex, this is of the form:
// /api/v1/watch/namespaces/foo/pods/pod0 for v1.
func (g TestGroup) ResourcePathWithPrefix(prefix, resource, namespace, name string) string {
	var path string
	if len(g.Group) == 0 {
		path = "/api/" + g.Version()
	} else {
		// TODO: switch back once we have proper multiple group support
		// path = "/apis/" + g.Group + "/" + Version(group...)
		path = "/apis/" + g.Group + "/" + g.Version()
	}

	if prefix != "" {
		path = path + "/" + prefix
	}
	if namespace != "" {
		path = path + "/namespaces/" + namespace
	}
	// Resource names are lower case.
	resource = strings.ToLower(resource)
	if resource != "" {
		path = path + "/" + resource
	}
	if name != "" {
		path = path + "/" + name
	}
	return path
}

// Returns the appropriate path for the given resource, namespace and name.
// For example, this is of the form:
// /api/v1/namespaces/foo/pods/pod0 for v1.
func (g TestGroup) ResourcePath(resource, namespace, name string) string {
	return g.ResourcePathWithPrefix("", resource, namespace, name)
}

func (g TestGroup) RESTMapper() meta.RESTMapper {
	return latest.GroupOrDie(g.Group).RESTMapper
}

// Get codec based on runtime.Object
func GetCodecForObject(obj runtime.Object) (runtime.Codec, error) {
	_, kind, err := api.Scheme.ObjectVersionAndKind(obj)
	if err != nil {
		return nil, fmt.Errorf("unexpected encoding error: %v", err)
	}
	// TODO: caesarxuchao: we should detect which group an object belongs to
	// by using the version returned by Schem.ObjectVersionAndKind() once we
	// split the schemes for internal objects.
	// TODO: caesarxuchao: we should add a map from kind to group in Scheme.
	for _, group := range Groups {
		if api.Scheme.Recognizes(group.GroupAndVersion(), kind) {
			return group.Codec(), nil
		}
	}
	return nil, fmt.Errorf("unexpected kind: %v", kind)
}
