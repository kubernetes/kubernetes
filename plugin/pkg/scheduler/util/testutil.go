/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"mime"
	"os"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"

	_ "k8s.io/kubernetes/pkg/api/install"
)

type TestGroup struct {
	externalGroupVersion schema.GroupVersion
	internalGroupVersion schema.GroupVersion
	internalTypes        map[string]reflect.Type
	externalTypes        map[string]reflect.Type
}

var (
	Groups = make(map[string]TestGroup)
	Test   TestGroup

	serializer runtime.SerializerInfo
)

func init() {
	if apiMediaType := os.Getenv("KUBE_TEST_API_TYPE"); len(apiMediaType) > 0 {
		var ok bool
		mediaType, _, err := mime.ParseMediaType(apiMediaType)
		if err != nil {
			panic(err)
		}
		serializer, ok = runtime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), mediaType)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", apiMediaType))
		}
	}

	kubeTestAPI := os.Getenv("KUBE_TEST_API")
	if len(kubeTestAPI) != 0 {
		// priority is "first in list preferred", so this has to run in reverse order
		testGroupVersions := strings.Split(kubeTestAPI, ",")
		for i := len(testGroupVersions) - 1; i >= 0; i-- {
			gvString := testGroupVersions[i]
			groupVersion, err := schema.ParseGroupVersion(gvString)
			if err != nil {
				panic(fmt.Sprintf("Error parsing groupversion %v: %v", gvString, err))
			}

			internalGroupVersion := schema.GroupVersion{Group: groupVersion.Group, Version: runtime.APIVersionInternal}
			Groups[groupVersion.Group] = TestGroup{
				externalGroupVersion: groupVersion,
				internalGroupVersion: internalGroupVersion,
				internalTypes:        legacyscheme.Scheme.KnownTypes(internalGroupVersion),
				externalTypes:        legacyscheme.Scheme.KnownTypes(groupVersion),
			}
		}
	}

	if _, ok := Groups[api.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: api.GroupName, Version: legacyscheme.Registry.GroupOrDie(api.GroupName).GroupVersion.Version}
		Groups[api.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: api.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(api.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}

	Test = Groups[api.GroupName]
}

// Codec returns the codec for the API version to test against, as set by the
// KUBE_TEST_API_TYPE env var.
func (g TestGroup) Codec() runtime.Codec {
	if serializer.Serializer == nil {
		return legacyscheme.Codecs.LegacyCodec(g.externalGroupVersion)
	}
	return legacyscheme.Codecs.CodecForVersions(serializer.Serializer, legacyscheme.Codecs.UniversalDeserializer(), schema.GroupVersions{g.externalGroupVersion}, nil)
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
	}

	// TODO: will need a /apis prefix once we have proper multi-group
	// support
	if name == "" {
		return fmt.Sprintf("/apis/%s/%s/%s", g.externalGroupVersion.Group, g.externalGroupVersion.Version, resource)
	}
	return fmt.Sprintf("/apis/%s/%s/%s/%s", g.externalGroupVersion.Group, g.externalGroupVersion.Version, resource, name)
}

// ResourcePathWithPrefix returns the appropriate path for the given prefix (watch, proxy, redirect, etc), resource, namespace and name.
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

// ResourcePath returns the appropriate path for the given resource, namespace and name.
// For example, this is of the form:
// /api/v1/namespaces/foo/pods/pod0 for v1.
func (g TestGroup) ResourcePath(resource, namespace, name string) string {
	return g.ResourcePathWithPrefix("", resource, namespace, name)
}

// SubResourcePath returns the appropriate path for the given resource, namespace,
// name and subresource.
func (g TestGroup) SubResourcePath(resource, namespace, name, sub string) string {
	path := g.ResourcePathWithPrefix("", resource, namespace, name)
	if sub != "" {
		path = path + "/" + sub
	}

	return path
}
