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

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	goruntime "runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	openapi "github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	genericapifilters "k8s.io/apiserver/pkg/endpoints/filters"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	"k8s.io/apiserver/pkg/registry/rest"
	genericfilters "k8s.io/apiserver/pkg/server/filters"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	kubeopenapi "k8s.io/kube-openapi/pkg/common"
)

const (
	extensionsGroupName = "extensions"
)

var (
	v1GroupVersion = schema.GroupVersion{Group: "", Version: "v1"}

	scheme         = runtime.NewScheme()
	codecs         = serializer.NewCodecFactory(scheme)
	parameterCodec = runtime.NewParameterCodec(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	scheme.AddUnversionedTypes(v1GroupVersion,
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
}

func buildTestOpenAPIDefinition() kubeopenapi.OpenAPIDefinition {
	return kubeopenapi.OpenAPIDefinition{
		Schema: openapi.Schema{
			SchemaProps: openapi.SchemaProps{
				Description: "Description",
				Properties:  map[string]openapi.Schema{},
			},
			VendorExtensible: openapi.VendorExtensible{
				Extensions: openapi.Extensions{
					"x-kubernetes-group-version-kind": []interface{}{
						map[string]interface{}{
							"group":   "",
							"version": "v1",
							"kind":    "Getter",
						},
						map[string]interface{}{
							"group":   "batch",
							"version": "v1",
							"kind":    "Getter",
						},
						map[string]interface{}{
							"group":   "extensions",
							"version": "v1",
							"kind":    "Getter",
						},
					},
				},
			},
		},
	}
}

func testGetOpenAPIDefinitions(_ kubeopenapi.ReferenceCallback) map[string]kubeopenapi.OpenAPIDefinition {
	return map[string]kubeopenapi.OpenAPIDefinition{
		"k8s.io/apimachinery/pkg/apis/meta/v1.Status":          {},
		"k8s.io/apimachinery/pkg/apis/meta/v1.APIVersions":     {},
		"k8s.io/apimachinery/pkg/apis/meta/v1.APIGroupList":    {},
		"k8s.io/apimachinery/pkg/apis/meta/v1.APIGroup":        buildTestOpenAPIDefinition(),
		"k8s.io/apimachinery/pkg/apis/meta/v1.APIResourceList": {},
	}
}

// setUp is a convience function for setting up for (most) tests.
func setUp(t *testing.T) (Config, *assert.Assertions) {
	config := NewConfig(codecs)
	config.ExternalAddress = "192.168.10.4:443"
	config.PublicAddress = net.ParseIP("192.168.10.4")
	config.LegacyAPIGroupPrefixes = sets.NewString("/api")
	config.LoopbackClientConfig = &restclient.Config{}

	clientset := fake.NewSimpleClientset()
	if clientset == nil {
		t.Fatal("unable to create fake client set")
	}

	config.OpenAPIConfig = DefaultOpenAPIConfig(testGetOpenAPIDefinitions, openapinamer.NewDefinitionNamer(runtime.NewScheme()))
	config.OpenAPIConfig.Info.Version = "unversioned"
	sharedInformers := informers.NewSharedInformerFactory(clientset, config.LoopbackClientConfig.Timeout)
	config.Complete(sharedInformers)

	return *config, assert.New(t)
}

func newMaster(t *testing.T) (*GenericAPIServer, Config, *assert.Assertions) {
	config, assert := setUp(t)

	s, err := config.Complete(nil).New("test", NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}
	return s, config, assert
}

// TestNew verifies that the New function returns a GenericAPIServer
// using the configuration properly.
func TestNew(t *testing.T) {
	s, config, assert := newMaster(t)

	// Verify many of the variables match their config counterparts
	assert.Equal(s.legacyAPIGroupPrefixes, config.LegacyAPIGroupPrefixes)
	assert.Equal(s.admissionControl, config.AdmissionControl)
}

// Verifies that AddGroupVersions works as expected.
func TestInstallAPIGroups(t *testing.T) {
	config, assert := setUp(t)

	config.LegacyAPIGroupPrefixes = sets.NewString("/apiPrefix")
	config.DiscoveryAddresses = discovery.DefaultAddresses{DefaultAddress: "ExternalAddress"}

	s, err := config.Complete(nil).New("test", NewEmptyDelegate())
	if err != nil {
		t.Fatalf("Error in bringing up the server: %v", err)
	}

	testAPI := func(gv schema.GroupVersion) APIGroupInfo {
		getter, noVerbs := testGetterStorage{}, testNoVerbsStorage{}

		scheme := runtime.NewScheme()
		scheme.AddKnownTypeWithName(gv.WithKind("Getter"), getter.New())
		scheme.AddKnownTypeWithName(gv.WithKind("NoVerb"), noVerbs.New())
		scheme.AddKnownTypes(v1GroupVersion, &metav1.Status{})
		metav1.AddToGroupVersion(scheme, v1GroupVersion)

		return APIGroupInfo{
			PrioritizedVersions: []schema.GroupVersion{gv},
			VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
				gv.Version: {
					"getter":  &testGetterStorage{Version: gv.Version},
					"noverbs": &testNoVerbsStorage{Version: gv.Version},
				},
			},
			OptionsExternalVersion: &schema.GroupVersion{Version: "v1"},
			ParameterCodec:         parameterCodec,
			NegotiatedSerializer:   codecs,
			Scheme:                 scheme,
		}
	}

	apis := []APIGroupInfo{
		testAPI(schema.GroupVersion{Group: "", Version: "v1"}),
		testAPI(schema.GroupVersion{Group: extensionsGroupName, Versi