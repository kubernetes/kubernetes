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
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"

	"k8s.io/kubernetes/pkg/apis/admission"
	admissioninstall "k8s.io/kubernetes/pkg/apis/admission/install"
	admissionregistrationinstall "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
	"k8s.io/kubernetes/pkg/apis/apps"
	appsinstall "k8s.io/kubernetes/pkg/apis/apps/install"
	authenticationinstall "k8s.io/kubernetes/pkg/apis/authentication/install"
	"k8s.io/kubernetes/pkg/apis/authorization"
	authorizationinstall "k8s.io/kubernetes/pkg/apis/authorization/install"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalinginstall "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchinstall "k8s.io/kubernetes/pkg/apis/batch/install"
	"k8s.io/kubernetes/pkg/apis/certificates"
	certificatesinstall "k8s.io/kubernetes/pkg/apis/certificates/install"
	"k8s.io/kubernetes/pkg/apis/core"
	coreinstall "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/apis/events"
	eventsinstall "k8s.io/kubernetes/pkg/apis/events/install"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsinstall "k8s.io/kubernetes/pkg/apis/extensions/install"
	"k8s.io/kubernetes/pkg/apis/imagepolicy"
	imagepolicyinstall "k8s.io/kubernetes/pkg/apis/imagepolicy/install"
	"k8s.io/kubernetes/pkg/apis/networking"
	networkinginstall "k8s.io/kubernetes/pkg/apis/networking/install"
	"k8s.io/kubernetes/pkg/apis/policy"
	policyinstall "k8s.io/kubernetes/pkg/apis/policy/install"
	"k8s.io/kubernetes/pkg/apis/rbac"
	rbacinstall "k8s.io/kubernetes/pkg/apis/rbac/install"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	schedulinginstall "k8s.io/kubernetes/pkg/apis/scheduling/install"
	"k8s.io/kubernetes/pkg/apis/settings"
	settingsinstall "k8s.io/kubernetes/pkg/apis/settings/install"
	"k8s.io/kubernetes/pkg/apis/storage"
	storageinstall "k8s.io/kubernetes/pkg/apis/storage/install"
)

type TestAPI struct {
	// GroupFactoryRegistry is the APIGroupFactoryRegistry (overlaps a bit with
	// Registry, see comments in package for details)
	GroupFactoryRegistry announced.APIGroupFactoryRegistry

	// Registry is an instance of an API registry.
	Registry *registered.APIRegistrationManager

	// Scheme is an instance of runtime.Scheme to which types in the Kubernetes
	// API are already registered.
	Scheme *runtime.Scheme

	// Codecs provides access to encoding and decoding for the scheme
	Codecs serializer.CodecFactory

	// ParameterCodec handles versioning of objects that are converted to query parameters.
	ParameterCodec runtime.ParameterCodec

	Groups map[string]TestGroup

	apiMediaType      string
	apiSerializer     runtime.SerializerInfo
	storageMediaType  string
	storageSerializer runtime.SerializerInfo
}

type CompleteTestAPI struct {
	*TestAPI

	Core          TestGroup
	Authorization TestGroup
	Autoscaling   TestGroup
	Batch         TestGroup
	Extensions    TestGroup
	Events        TestGroup
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
}

type TestGroup struct {
	tapi                 *TestAPI
	externalGroupVersion schema.GroupVersion
	internalGroupVersion schema.GroupVersion
	internalTypes        map[string]reflect.Type
	externalTypes        map[string]reflect.Type
}

type APIInstaller func(announced.APIGroupFactoryRegistry, *registered.APIRegistrationManager, *runtime.Scheme)

func NewTestAPI(apiMediaType, storageMediaType string, installers ...APIInstaller) *TestAPI {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)

	tapi := &TestAPI{
		GroupFactoryRegistry: make(announced.APIGroupFactoryRegistry),
		Registry:             registered.NewOrDie(""),
		Scheme:               scheme,
		Codecs:               codecs,
		ParameterCodec:       runtime.NewParameterCodec(scheme),
		Groups:               make(map[string]TestGroup),
	}

	if apiMediaType != "" {
		mediaType, _, err := mime.ParseMediaType(apiMediaType)
		if err != nil {
			panic(err)
		}
		apiSerializer, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", apiMediaType))
		}

		tapi.apiMediaType = mediaType
		tapi.apiSerializer = apiSerializer
	}

	if storageMediaType != "" {
		mediaType, _, err := mime.ParseMediaType(storageMediaType)
		if err != nil {
			panic(err)
		}
		storageSerializer, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
		if !ok {
			panic(fmt.Sprintf("no serializer for %s", storageMediaType))
		}

		tapi.storageMediaType = mediaType
		tapi.storageSerializer = storageSerializer
	}

	for _, install := range installers {
		install(tapi.GroupFactoryRegistry, tapi.Registry, tapi.Scheme)
	}

	for name, _ := range tapi.GroupFactoryRegistry {
		egv := schema.GroupVersion{
			Group:   name,
			Version: tapi.Registry.GroupOrDie(name).GroupVersion.Version,
		}
		igv := schema.GroupVersion{
			Group:   name,
			Version: runtime.APIVersionInternal,
		}
		tapi.Groups[name] = TestGroup{
			tapi:                 tapi,
			internalGroupVersion: igv,
			externalGroupVersion: egv,
			internalTypes:        tapi.Scheme.KnownTypes(igv),
			externalTypes:        tapi.Scheme.KnownTypes(egv),
		}
	}

	return tapi
}

func NewCompleteTestAPI() *CompleteTestAPI {
	tapi := &CompleteTestAPI{
		TestAPI: NewTestAPI("", "",
			admissionregistrationinstall.Install,
			admissioninstall.Install,
			appsinstall.Install,
			authenticationinstall.Install,
			authorizationinstall.Install,
			autoscalinginstall.Install,
			batchinstall.Install,
			certificatesinstall.Install,
			coreinstall.Install,
			eventsinstall.Install,
			extensionsinstall.Install,
			imagepolicyinstall.Install,
			networkinginstall.Install,
			policyinstall.Install,
			rbacinstall.Install,
			schedulinginstall.Install,
			settingsinstall.Install,
			storageinstall.Install,
		),
	}

	tapi.Core = tapi.Groups[core.GroupName]
	tapi.Autoscaling = tapi.Groups[autoscaling.GroupName]
	tapi.Batch = tapi.Groups[batch.GroupName]
	tapi.Apps = tapi.Groups[apps.GroupName]
	tapi.Policy = tapi.Groups[policy.GroupName]
	tapi.Certificates = tapi.Groups[certificates.GroupName]
	tapi.Extensions = tapi.Groups[extensions.GroupName]
	tapi.Events = tapi.Groups[events.GroupName]
	tapi.Rbac = tapi.Groups[rbac.GroupName]
	tapi.Scheduling = tapi.Groups[scheduling.GroupName]
	tapi.Settings = tapi.Groups[settings.GroupName]
	tapi.Storage = tapi.Groups[storage.GroupName]
	tapi.ImagePolicy = tapi.Groups[imagepolicy.GroupName]
	tapi.Authorization = tapi.Groups[authorization.GroupName]
	tapi.Admission = tapi.Groups[admission.GroupName]
	tapi.Networking = tapi.Groups[networking.GroupName]

	return tapi
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

// Codec returns the codec for the API version to test against.
func (g TestGroup) Codec() runtime.Codec {
	if g.tapi.apiSerializer.Serializer == nil {
		return g.tapi.Codecs.LegacyCodec(g.externalGroupVersion)
	}
	return g.tapi.Codecs.CodecForVersions(g.tapi.apiSerializer.Serializer, g.tapi.Codecs.UniversalDeserializer(), schema.GroupVersions{g.externalGroupVersion}, nil)
}

// NegotiatedSerializer returns the negotiated serializer for the server.
func (g TestGroup) NegotiatedSerializer() runtime.NegotiatedSerializer {
	return g.tapi.Codecs
}

func (t *TestAPI) StorageMediaType() string {
	return t.storageMediaType
}

// StorageCodec returns the codec for the API version to store in etcd, as set by the
// KUBE_TEST_API_STORAGE_TYPE env var.
func (g TestGroup) StorageCodec() runtime.Codec {
	s := g.tapi.storageSerializer.Serializer

	if s == nil {
		return g.tapi.Codecs.LegacyCodec(g.externalGroupVersion)
	}

	// etcd2 only supports string data - we must wrap any result before returning
	// TODO: remove for etcd3 / make parameterizable
	if !g.tapi.storageSerializer.EncodesAsText {
		s = runtime.NewBase64Serializer(s, s)
	}
	ds := recognizer.NewDecoder(s, g.tapi.Codecs.UniversalDeserializer())

	return g.tapi.Codecs.CodecForVersions(s, ds, schema.GroupVersions{g.externalGroupVersion}, nil)
}

// Converter returns the legacyscheme.Scheme for the API version to test against, as set by the
// KUBE_TEST_API env var.
func (g TestGroup) Converter() runtime.ObjectConvertor {
	interfaces, err := g.tapi.Registry.GroupOrDie(g.externalGroupVersion.Group).InterfacesFor(g.externalGroupVersion)
	if err != nil {
		panic(err)
	}
	return interfaces.ObjectConvertor
}

// MetadataAccessor returns the MetadataAccessor for the API version to test against,
// as set by the KUBE_TEST_API env var.
func (g TestGroup) MetadataAccessor() meta.MetadataAccessor {
	interfaces, err := g.tapi.Registry.GroupOrDie(g.externalGroupVersion.Group).InterfacesFor(g.externalGroupVersion)
	if err != nil {
		panic(err)
	}
	return interfaces.MetadataAccessor
}

// SelfLink returns a self link that will appear to be for the version Version().
// 'resource' should be the resource path, e.g. "pods" for the Pod type. 'name' should be
// empty for lists.
func (g TestGroup) SelfLink(resource, name string) string {
	if g.externalGroupVersion.Group == core.GroupName {
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
	if g.externalGroupVersion.Group == core.GroupName {
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
	return g.tapi.Registry.RESTMapper()
}

// ExternalGroupVersions returns all external group versions allowed for the server.
func (t *TestAPI) ExternalGroupVersions() schema.GroupVersions {
	versions := []schema.GroupVersion{}
	for _, g := range t.Groups {
		gv := g.GroupVersion()
		versions = append(versions, *gv)
	}
	return versions
}

// GetCodecForObject gets codec based on runtime.Object
func (t *TestAPI) GetCodecForObject(obj runtime.Object) (runtime.Codec, error) {
	kinds, _, err := t.Scheme.ObjectKinds(obj)
	if err != nil {
		return nil, fmt.Errorf("unexpected encoding error: %v", err)
	}
	kind := kinds[0]

	for _, group := range t.Groups {
		if group.GroupVersion().Group != kind.Group {
			continue
		}

		if t.Scheme.Recognizes(kind) {
			return group.Codec(), nil
		}
	}
	// Codec used for unversioned types
	if t.Scheme.Recognizes(kind) {
		serializer, ok := runtime.SerializerInfoForMediaType(t.Codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
		if !ok {
			return nil, fmt.Errorf("no serializer registered for json")
		}
		return serializer.Serializer, nil
	}
	return nil, fmt.Errorf("unexpected kind: %v", kind)
}

// NewTestGroup creates a new TestGroup.
func (t *TestAPI) NewTestGroup(external, internal schema.GroupVersion, internalTypes map[string]reflect.Type, externalTypes map[string]reflect.Type) TestGroup {
	return TestGroup{
		tapi:                 t,
		externalGroupVersion: external,
		internalGroupVersion: internal,
		internalTypes:        internalTypes,
		externalTypes:        externalTypes,
	}
}
