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
	"reflect"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/runtime"

	_ "k8s.io/kubernetes/pkg/api/install"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/componentconfig/install"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	_ "k8s.io/kubernetes/pkg/apis/metrics/install"
)

var (
	Groups      = make(map[string]TestGroup)
	Default     TestGroup
	Autoscaling TestGroup
	Batch       TestGroup
	Extensions  TestGroup
)

type TestGroup struct {
	externalGroupVersion unversioned.GroupVersion
	internalGroupVersion unversioned.GroupVersion
	internalTypes        map[string]reflect.Type
}

func init() {
	kubeTestAPI := os.Getenv("KUBE_TEST_API")
	if len(kubeTestAPI) != 0 {
		testGroupVersions := strings.Split(kubeTestAPI, ",")
		for _, gvString := range testGroupVersions {
			groupVersion, err := unversioned.ParseGroupVersion(gvString)
			if err != nil {
				panic(fmt.Sprintf("Error parsing groupversion %v: %v", gvString, err))
			}

			internalGroupVersion := unversioned.GroupVersion{Group: groupVersion.Group, Version: runtime.APIVersionInternal}
			Groups[groupVersion.Group] = TestGroup{
				externalGroupVersion: groupVersion,
				internalGroupVersion: internalGroupVersion,
				internalTypes:        api.Scheme.KnownTypes(internalGroupVersion),
			}
		}
	}

	if _, ok := Groups[api.GroupName]; !ok {
		Groups[api.GroupName] = TestGroup{
			externalGroupVersion: unversioned.GroupVersion{Group: api.GroupName, Version: registered.GroupOrDie(api.GroupName).GroupVersion.Version},
			internalGroupVersion: api.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(api.SchemeGroupVersion),
		}
	}
	if _, ok := Groups[extensions.GroupName]; !ok {
		Groups[extensions.GroupName] = TestGroup{
			externalGroupVersion: unversioned.GroupVersion{Group: extensions.GroupName, Version: registered.GroupOrDie(extensions.GroupName).GroupVersion.Version},
			internalGroupVersion: extensions.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(extensions.SchemeGroupVersion),
		}
	}
	if _, ok := Groups[autoscaling.GroupName]; !ok {
		internalTypes := make(map[string]reflect.Type)
		for k, t := range api.Scheme.KnownTypes(extensions.SchemeGroupVersion) {
			if k == "Scale" {
				continue
			}
			internalTypes[k] = t
		}
		Groups[autoscaling.GroupName] = TestGroup{
			externalGroupVersion: unversioned.GroupVersion{Group: autoscaling.GroupName, Version: registered.GroupOrDie(autoscaling.GroupName).GroupVersion.Version},
			internalGroupVersion: extensions.SchemeGroupVersion,
			internalTypes:        internalTypes,
		}
	}
	if _, ok := Groups[autoscaling.GroupName+"IntraGroup"]; !ok {
		internalTypes := make(map[string]reflect.Type)
		for k, t := range api.Scheme.KnownTypes(extensions.SchemeGroupVersion) {
			if k == "Scale" {
				internalTypes[k] = t
				break
			}
		}
		Groups[autoscaling.GroupName] = TestGroup{
			externalGroupVersion: unversioned.GroupVersion{Group: autoscaling.GroupName, Version: registered.GroupOrDie(autoscaling.GroupName).GroupVersion.Version},
			internalGroupVersion: autoscaling.SchemeGroupVersion,
			internalTypes:        internalTypes,
		}
	}
	if _, ok := Groups[batch.GroupName]; !ok {
		Groups[batch.GroupName] = TestGroup{
			externalGroupVersion: unversioned.GroupVersion{Group: batch.GroupName, Version: registered.GroupOrDie(batch.GroupName).GroupVersion.Version},
			internalGroupVersion: extensions.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(extensions.SchemeGroupVersion),
		}
	}

	Default = Groups[api.GroupName]
	Autoscaling = Groups[autoscaling.GroupName]
	Batch = Groups[batch.GroupName]
	Extensions = Groups[extensions.GroupName]
}

func (g TestGroup) ContentConfig() (string, *unversioned.GroupVersion, runtime.Codec) {
	return "application/json", g.GroupVersion(), g.Codec()
}

func (g TestGroup) GroupVersion() *unversioned.GroupVersion {
	copyOfGroupVersion := g.externalGroupVersion
	return &copyOfGroupVersion
}

// InternalGroupVersion returns the group,version used to identify the internal
// types for this API
func (g TestGroup) InternalGroupVersion() unversioned.GroupVersion {
	return g.internalGroupVersion
}

// InternalTypes returns a map of internal API types' kind names to their Go types.
func (g TestGroup) InternalTypes() map[string]reflect.Type {
	return g.internalTypes
}

// Codec returns the codec for the API version to test against, as set by the
// KUBE_TEST_API env var.
func (g TestGroup) Codec() runtime.Codec {
	return api.Codecs.LegacyCodec(g.externalGroupVersion)
}

// Converter returns the api.Scheme for the API version to test against, as set by the
// KUBE_TEST_API env var.
func (g TestGroup) Converter() runtime.ObjectConvertor {
	interfaces, err := registered.GroupOrDie(g.externalGroupVersion.Group).InterfacesFor(g.externalGroupVersion)
	if err != nil {
		panic(err)
	}
	return interfaces.ObjectConvertor
}

// MetadataAccessor returns the MetadataAccessor for the API version to test against,
// as set by the KUBE_TEST_API env var.
func (g TestGroup) MetadataAccessor() meta.MetadataAccessor {
	interfaces, err := registered.GroupOrDie(g.externalGroupVersion.Group).InterfacesFor(g.externalGroupVersion)
	if err != nil {
		panic(err)
	}
	return interfaces.MetadataAccessor
}

// SelfLink returns a self link that will appear to be for the version Version().
// 'resource' should be the resource path, e.g. "pods" for the Pod type. 'name' should be
// empty for lists.
func (g TestGroup) SelfLink(resource, name string) string {
	if g.externalGroupVersion.Group == api.GroupName {
		if name == "" {
			return fmt.Sprintf("/api/%s/%s", g.externalGroupVersion.Version, resource)
		}
		return fmt.Sprintf("/api/%s/%s/%s", g.externalGroupVersion.Version, resource, name)
	} else {
		// TODO: will need a /apis prefix once we have proper multi-group
		// support
		if name == "" {
			return fmt.Sprintf("/apis/%s/%s/%s", g.externalGroupVersion.Group, g.externalGroupVersion.Version, resource)
		}
		return fmt.Sprintf("/apis/%s/%s/%s/%s", g.externalGroupVersion.Group, g.externalGroupVersion.Version, resource, name)
	}
}

// Returns the appropriate path for the given prefix (watch, proxy, redirect, etc), resource, namespace and name.
// For ex, this is of the form:
// /api/v1/watch/namespaces/foo/pods/pod0 for v1.
func (g TestGroup) ResourcePathWithPrefix(prefix, resource, namespace, name string) string {
	var path string
	if g.externalGroupVersion.Group == api.GroupName {
		path = "/api/" + g.externalGroupVersion.Version
	} else {
		// TODO: switch back once we have proper multiple group support
		// path = "/apis/" + g.Group + "/" + Version(group...)
		path = "/apis/" + g.externalGroupVersion.Group + "/" + g.externalGroupVersion.Version
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
	return registered.RESTMapper()
}

// Get codec based on runtime.Object
func GetCodecForObject(obj runtime.Object) (runtime.Codec, error) {
	kind, err := api.Scheme.ObjectKind(obj)
	if err != nil {
		return nil, fmt.Errorf("unexpected encoding error: %v", err)
	}

	for _, group := range Groups {
		if group.GroupVersion().Group != kind.Group {
			continue
		}

		if api.Scheme.Recognizes(kind) {
			return group.Codec(), nil
		}
	}
	// Codec used for unversioned types
	if api.Scheme.Recognizes(kind) {
		serializer, ok := api.Codecs.SerializerForFileExtension("json")
		if !ok {
			return nil, fmt.Errorf("no serializer registered for json")
		}
		return serializer, nil
	}
	return nil, fmt.Errorf("unexpected kind: %v", kind)
}

func NewTestGroup(external, internal unversioned.GroupVersion, internalTypes map[string]reflect.Type) TestGroup {
	return TestGroup{external, internal, internalTypes}
}
