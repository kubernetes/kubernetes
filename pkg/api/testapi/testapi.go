/*
Copyright 2014 The Kubernetes Authors.

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
	"mime"
	"os"
	"reflect"
	"strings"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/imagepolicy"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/recognizer"

	_ "k8s.io/kubernetes/federation/apis/federation/install"
	_ "k8s.io/kubernetes/pkg/api/install"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/authentication/install"
	_ "k8s.io/kubernetes/pkg/apis/authorization/install"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	_ "k8s.io/kubernetes/pkg/apis/componentconfig/install"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	_ "k8s.io/kubernetes/pkg/apis/imagepolicy/install"
	_ "k8s.io/kubernetes/pkg/apis/policy/install"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	_ "k8s.io/kubernetes/pkg/apis/storage/install"
)

var (
	Groups       = make(map[string]TestGroup)
	Default      TestGroup
	Autoscaling  TestGroup
	Batch        TestGroup
	Extensions   TestGroup
	Apps         TestGroup
	Policy       TestGroup
	Federation   TestGroup
	Rbac         TestGroup
	Certificates TestGroup
	Storage      TestGroup
	ImagePolicy  TestGroup

	serializer        runtime.SerializerInfo
	storageSerializer runtime.SerializerInfo
)

type TestGroup struct {
	externalGroupVersion unversioned.GroupVersion
	internalGroupVersion unversioned.GroupVersion
	internalTypes        map[string]reflect.Type
	externalTypes        map[string]reflect.Type
}

func init() {
	if apiMediaType := os.Getenv("KUBE_TEST_API_TYPE"); len(apiMediaType) > 0 {
		var ok bool
		mediaType, options, err := mime.ParseMediaType(apiMediaType)
		if err != nil {
			panic(err)
		}
		serializer, ok = api.Codecs.SerializerForMediaType(mediaType, options)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", apiMediaType))
		}
	}

	if storageMediaType := StorageMediaType(); len(storageMediaType) > 0 {
		var ok bool
		mediaType, options, err := mime.ParseMediaType(storageMediaType)
		if err != nil {
			panic(err)
		}
		storageSerializer, ok = api.Codecs.SerializerForMediaType(mediaType, options)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", storageMediaType))
		}
	}

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
				externalTypes:        api.Scheme.KnownTypes(groupVersion),
			}
		}
	}

	if _, ok := Groups[api.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: api.GroupName, Version: registered.GroupOrDie(api.GroupName).GroupVersion.Version}
		Groups[api.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: api.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(api.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[extensions.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: extensions.GroupName, Version: registered.GroupOrDie(extensions.GroupName).GroupVersion.Version}
		Groups[extensions.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: extensions.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(extensions.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
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
		externalGroupVersion := unversioned.GroupVersion{Group: autoscaling.GroupName, Version: registered.GroupOrDie(autoscaling.GroupName).GroupVersion.Version}
		Groups[autoscaling.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: extensions.SchemeGroupVersion,
			internalTypes:        internalTypes,
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
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
		externalGroupVersion := unversioned.GroupVersion{Group: autoscaling.GroupName, Version: registered.GroupOrDie(autoscaling.GroupName).GroupVersion.Version}
		Groups[autoscaling.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: autoscaling.SchemeGroupVersion,
			internalTypes:        internalTypes,
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[batch.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: batch.GroupName, Version: registered.GroupOrDie(batch.GroupName).GroupVersion.Version}
		Groups[batch.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: batch.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(batch.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[apps.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: apps.GroupName, Version: registered.GroupOrDie(apps.GroupName).GroupVersion.Version}
		Groups[apps.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: extensions.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(extensions.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[policy.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: policy.GroupName, Version: registered.GroupOrDie(policy.GroupName).GroupVersion.Version}
		Groups[policy.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: policy.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(policy.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[federation.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: federation.GroupName, Version: registered.GroupOrDie(federation.GroupName).GroupVersion.Version}
		Groups[federation.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: federation.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(federation.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[rbac.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: rbac.GroupName, Version: registered.GroupOrDie(rbac.GroupName).GroupVersion.Version}
		Groups[rbac.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: rbac.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(rbac.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[storage.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: storage.GroupName, Version: registered.GroupOrDie(storage.GroupName).GroupVersion.Version}
		Groups[storage.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: storage.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(storage.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[certificates.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: certificates.GroupName, Version: registered.GroupOrDie(certificates.GroupName).GroupVersion.Version}
		Groups[certificates.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: certificates.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(certificates.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}

	if _, ok := Groups[imagepolicy.GroupName]; !ok {
		externalGroupVersion := unversioned.GroupVersion{Group: imagepolicy.GroupName, Version: registered.GroupOrDie(imagepolicy.GroupName).GroupVersion.Version}
		Groups[imagepolicy.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: imagepolicy.SchemeGroupVersion,
			internalTypes:        api.Scheme.KnownTypes(imagepolicy.SchemeGroupVersion),
			externalTypes:        api.Scheme.KnownTypes(externalGroupVersion),
		}
	}

	Default = Groups[api.GroupName]
	Autoscaling = Groups[autoscaling.GroupName]
	Batch = Groups[batch.GroupName]
	Apps = Groups[apps.GroupName]
	Policy = Groups[policy.GroupName]
	Certificates = Groups[certificates.GroupName]
	Extensions = Groups[extensions.GroupName]
	Federation = Groups[federation.GroupName]
	Rbac = Groups[rbac.GroupName]
	Storage = Groups[storage.GroupName]
	ImagePolicy = Groups[imagepolicy.GroupName]
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

// ExternalTypes returns a map of external API types' kind names to their Go types.
func (g TestGroup) ExternalTypes() map[string]reflect.Type {
	return g.externalTypes
}

// Codec returns the codec for the API version to test against, as set by the
// KUBE_TEST_API_TYPE env var.
func (g TestGroup) Codec() runtime.Codec {
	if serializer.Serializer == nil {
		return api.Codecs.LegacyCodec(g.externalGroupVersion)
	}
	return api.Codecs.CodecForVersions(serializer, api.Codecs.UniversalDeserializer(), unversioned.GroupVersions{g.externalGroupVersion}, nil)
}

// NegotiatedSerializer returns the negotiated serializer for the server.
func (g TestGroup) NegotiatedSerializer() runtime.NegotiatedSerializer {
	return api.Codecs
}

func StorageMediaType() string {
	return os.Getenv("KUBE_TEST_API_STORAGE_TYPE")
}

// StorageCodec returns the codec for the API version to store in etcd, as set by the
// KUBE_TEST_API_STORAGE_TYPE env var.
func (g TestGroup) StorageCodec() runtime.Codec {
	s := storageSerializer.Serializer

	if s == nil {
		return api.Codecs.LegacyCodec(g.externalGroupVersion)
	}

	// etcd2 only supports string data - we must wrap any result before returning
	// TODO: remove for etcd3 / make parameterizable
	if !storageSerializer.EncodesAsText {
		s = runtime.NewBase64Serializer(s)
	}
	ds := recognizer.NewDecoder(s, api.Codecs.UniversalDeserializer())

	return api.Codecs.CodecForVersions(s, ds, unversioned.GroupVersions{g.externalGroupVersion}, nil)
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

// ExternalGroupVersions returns all external group versions allowed for the server.
func ExternalGroupVersions() unversioned.GroupVersions {
	versions := []unversioned.GroupVersion{}
	for _, g := range Groups {
		gv := g.GroupVersion()
		versions = append(versions, *gv)
	}
	return versions
}

// Get codec based on runtime.Object
func GetCodecForObject(obj runtime.Object) (runtime.Codec, error) {
	kinds, _, err := api.Scheme.ObjectKinds(obj)
	if err != nil {
		return nil, fmt.Errorf("unexpected encoding error: %v", err)
	}
	kind := kinds[0]

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

func NewTestGroup(external, internal unversioned.GroupVersion, internalTypes map[string]reflect.Type, externalTypes map[string]reflect.Type) TestGroup {
	return TestGroup{external, internal, internalTypes, externalTypes}
}
