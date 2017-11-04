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
//
// TODO(lavalamp): this package is a huge disaster at the moment. I intend to
// refactor. All code currently using this package should change:
// 1. Declare your own api.Registry.APIGroupRegistrationManager in your own test code.
// 2. Import the relevant install packages.
// 3. Register the types you need, from the announced.APIGroupAnnouncementManager.
package testapi

import (
	"fmt"
	"mime"
	"os"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/admission"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/authorization"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/imagepolicy"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/apis/settings"
	"k8s.io/kubernetes/pkg/apis/storage"

	_ "k8s.io/kubernetes/pkg/api/install"
	_ "k8s.io/kubernetes/pkg/apis/admission/install"
	_ "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	_ "k8s.io/kubernetes/pkg/apis/authentication/install"
	_ "k8s.io/kubernetes/pkg/apis/authorization/install"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	_ "k8s.io/kubernetes/pkg/apis/componentconfig/install"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	_ "k8s.io/kubernetes/pkg/apis/imagepolicy/install"
	_ "k8s.io/kubernetes/pkg/apis/networking/install"
	_ "k8s.io/kubernetes/pkg/apis/policy/install"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
	_ "k8s.io/kubernetes/pkg/apis/settings/install"
	_ "k8s.io/kubernetes/pkg/apis/storage/install"
)

var (
	Groups        = make(map[string]TestGroup)
	Default       TestGroup
	Authorization TestGroup
	Autoscaling   TestGroup
	Batch         TestGroup
	Extensions    TestGroup
	Apps          TestGroup
	Policy        TestGroup
	Rbac          TestGroup
	Certificates  TestGroup
	Scheduling    TestGroup
	Settings      TestGroup
	Storage       TestGroup
	ImagePolicy   TestGroup
	Admission     TestGroup
	Networking    TestGroup

	serializer        runtime.SerializerInfo
	storageSerializer runtime.SerializerInfo
)

type TestGroup struct {
	externalGroupVersion schema.GroupVersion
	internalGroupVersion schema.GroupVersion
	internalTypes        map[string]reflect.Type
	externalTypes        map[string]reflect.Type
}

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

	if storageMediaType := StorageMediaType(); len(storageMediaType) > 0 {
		var ok bool
		mediaType, _, err := mime.ParseMediaType(storageMediaType)
		if err != nil {
			panic(err)
		}
		storageSerializer, ok = runtime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), mediaType)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", storageMediaType))
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
	if _, ok := Groups[extensions.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: extensions.GroupName, Version: legacyscheme.Registry.GroupOrDie(extensions.GroupName).GroupVersion.Version}
		Groups[extensions.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: extensions.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(extensions.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[autoscaling.GroupName]; !ok {
		internalTypes := make(map[string]reflect.Type)
		for k, t := range legacyscheme.Scheme.KnownTypes(extensions.SchemeGroupVersion) {
			if k == "Scale" {
				continue
			}
			internalTypes[k] = t
		}
		externalGroupVersion := schema.GroupVersion{Group: autoscaling.GroupName, Version: legacyscheme.Registry.GroupOrDie(autoscaling.GroupName).GroupVersion.Version}
		Groups[autoscaling.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: extensions.SchemeGroupVersion,
			internalTypes:        internalTypes,
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[autoscaling.GroupName+"IntraGroup"]; !ok {
		internalTypes := make(map[string]reflect.Type)
		for k, t := range legacyscheme.Scheme.KnownTypes(extensions.SchemeGroupVersion) {
			if k == "Scale" {
				internalTypes[k] = t
				break
			}
		}
		externalGroupVersion := schema.GroupVersion{Group: autoscaling.GroupName, Version: legacyscheme.Registry.GroupOrDie(autoscaling.GroupName).GroupVersion.Version}
		Groups[autoscaling.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: autoscaling.SchemeGroupVersion,
			internalTypes:        internalTypes,
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[batch.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: batch.GroupName, Version: legacyscheme.Registry.GroupOrDie(batch.GroupName).GroupVersion.Version}
		Groups[batch.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: batch.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(batch.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[apps.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: apps.GroupName, Version: legacyscheme.Registry.GroupOrDie(apps.GroupName).GroupVersion.Version}
		Groups[apps.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: apps.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(apps.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[policy.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: policy.GroupName, Version: legacyscheme.Registry.GroupOrDie(policy.GroupName).GroupVersion.Version}
		Groups[policy.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: policy.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(policy.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[rbac.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: rbac.GroupName, Version: legacyscheme.Registry.GroupOrDie(rbac.GroupName).GroupVersion.Version}
		Groups[rbac.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: rbac.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(rbac.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[scheduling.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: scheduling.GroupName, Version: legacyscheme.Registry.GroupOrDie(scheduling.GroupName).GroupVersion.Version}
		Groups[scheduling.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: scheduling.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(scheduling.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[settings.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: settings.GroupName, Version: legacyscheme.Registry.GroupOrDie(settings.GroupName).GroupVersion.Version}
		Groups[settings.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: settings.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(settings.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[storage.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: storage.GroupName, Version: legacyscheme.Registry.GroupOrDie(storage.GroupName).GroupVersion.Version}
		Groups[storage.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: storage.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(storage.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[certificates.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: certificates.GroupName, Version: legacyscheme.Registry.GroupOrDie(certificates.GroupName).GroupVersion.Version}
		Groups[certificates.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: certificates.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(certificates.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[imagepolicy.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: imagepolicy.GroupName, Version: legacyscheme.Registry.GroupOrDie(imagepolicy.GroupName).GroupVersion.Version}
		Groups[imagepolicy.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: imagepolicy.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(imagepolicy.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[authorization.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: authorization.GroupName, Version: legacyscheme.Registry.GroupOrDie(authorization.GroupName).GroupVersion.Version}
		Groups[authorization.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: authorization.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(authorization.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[admissionregistration.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: admissionregistration.GroupName, Version: legacyscheme.Registry.GroupOrDie(admissionregistration.GroupName).GroupVersion.Version}
		Groups[admissionregistration.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: admissionregistration.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(admissionregistration.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[admission.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: admission.GroupName, Version: legacyscheme.Registry.GroupOrDie(admission.GroupName).GroupVersion.Version}
		Groups[admission.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: admission.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(admission.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}
	if _, ok := Groups[networking.GroupName]; !ok {
		externalGroupVersion := schema.GroupVersion{Group: networking.GroupName, Version: legacyscheme.Registry.GroupOrDie(networking.GroupName).GroupVersion.Version}
		Groups[networking.GroupName] = TestGroup{
			externalGroupVersion: externalGroupVersion,
			internalGroupVersion: networking.SchemeGroupVersion,
			internalTypes:        legacyscheme.Scheme.KnownTypes(networking.SchemeGroupVersion),
			externalTypes:        legacyscheme.Scheme.KnownTypes(externalGroupVersion),
		}
	}

	Default = Groups[api.GroupName]
	Autoscaling = Groups[autoscaling.GroupName]
	Batch = Groups[batch.GroupName]
	Apps = Groups[apps.GroupName]
	Policy = Groups[policy.GroupName]
	Certificates = Groups[certificates.GroupName]
	Extensions = Groups[extensions.GroupName]
	Rbac = Groups[rbac.GroupName]
	Scheduling = Groups[scheduling.GroupName]
	Settings = Groups[settings.GroupName]
	Storage = Groups[storage.GroupName]
	ImagePolicy = Groups[imagepolicy.GroupName]
	Authorization = Groups[authorization.GroupName]
	Admission = Groups[admission.GroupName]
	Networking = Groups[networking.GroupName]
}

func (g TestGroup) ContentConfig() (string, *schema.GroupVersion, runtime.Codec) {
	return "application/json", g.GroupVersion(), g.Codec()
}

func (g TestGroup) GroupVersion() *schema.GroupVersion {
	copyOfGroupVersion := g.externalGroupVersion
	return &copyOfGroupVersion
}

// InternalGroupVersion returns the group,version used to identify the internal
// types for this API
func (g TestGroup) InternalGroupVersion() schema.GroupVersion {
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
		return legacyscheme.Codecs.LegacyCodec(g.externalGroupVersion)
	}
	return legacyscheme.Codecs.CodecForVersions(serializer.Serializer, legacyscheme.Codecs.UniversalDeserializer(), schema.GroupVersions{g.externalGroupVersion}, nil)
}

// NegotiatedSerializer returns the negotiated serializer for the server.
func (g TestGroup) NegotiatedSerializer() runtime.NegotiatedSerializer {
	return legacyscheme.Codecs
}

func StorageMediaType() string {
	return os.Getenv("KUBE_TEST_API_STORAGE_TYPE")
}

// StorageCodec returns the codec for the API version to store in etcd, as set by the
// KUBE_TEST_API_STORAGE_TYPE env var.
func (g TestGroup) StorageCodec() runtime.Codec {
	s := storageSerializer.Serializer

	if s == nil {
		return legacyscheme.Codecs.LegacyCodec(g.externalGroupVersion)
	}

	// etcd2 only supports string data - we must wrap any result before returning
	// TODO: remove for etcd3 / make parameterizable
	if !storageSerializer.EncodesAsText {
		s = runtime.NewBase64Serializer(s, s)
	}
	ds := recognizer.NewDecoder(s, legacyscheme.Codecs.UniversalDeserializer())

	return legacyscheme.Codecs.CodecForVersions(s, ds, schema.GroupVersions{g.externalGroupVersion}, nil)
}

// Converter returns the legacyscheme.Scheme for the API version to test against, as set by the
// KUBE_TEST_API env var.
func (g TestGroup) Converter() runtime.ObjectConvertor {
	interfaces, err := legacyscheme.Registry.GroupOrDie(g.externalGroupVersion.Group).InterfacesFor(g.externalGroupVersion)
	if err != nil {
		panic(err)
	}
	return interfaces.ObjectConvertor
}

// MetadataAccessor returns the MetadataAccessor for the API version to test against,
// as set by the KUBE_TEST_API env var.
func (g TestGroup) MetadataAccessor() meta.MetadataAccessor {
	interfaces, err := legacyscheme.Registry.GroupOrDie(g.externalGroupVersion.Group).InterfacesFor(g.externalGroupVersion)
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

// RESTMapper returns RESTMapper in legacyscheme.Registry.
func (g TestGroup) RESTMapper() meta.RESTMapper {
	return legacyscheme.Registry.RESTMapper()
}

// ExternalGroupVersions returns all external group versions allowed for the server.
func ExternalGroupVersions() schema.GroupVersions {
	versions := []schema.GroupVersion{}
	for _, g := range Groups {
		gv := g.GroupVersion()
		versions = append(versions, *gv)
	}
	return versions
}

// GetCodecForObject gets codec based on runtime.Object
func GetCodecForObject(obj runtime.Object) (runtime.Codec, error) {
	kinds, _, err := legacyscheme.Scheme.ObjectKinds(obj)
	if err != nil {
		return nil, fmt.Errorf("unexpected encoding error: %v", err)
	}
	kind := kinds[0]

	for _, group := range Groups {
		if group.GroupVersion().Group != kind.Group {
			continue
		}

		if legacyscheme.Scheme.Recognizes(kind) {
			return group.Codec(), nil
		}
	}
	// Codec used for unversioned types
	if legacyscheme.Scheme.Recognizes(kind) {
		serializer, ok := runtime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
		if !ok {
			return nil, fmt.Errorf("no serializer registered for json")
		}
		return serializer.Serializer, nil
	}
	return nil, fmt.Errorf("unexpected kind: %v", kind)
}

// NewTestGroup creates a new TestGroup.
func NewTestGroup(external, internal schema.GroupVersion, internalTypes map[string]reflect.Type, externalTypes map[string]reflect.Type) TestGroup {
	return TestGroup{external, internal, internalTypes, externalTypes}
}
