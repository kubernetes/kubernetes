/*
Copyright 2016 The Kubernetes Authors.

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

package v2_test

import (
	"encoding/json"
	"math/rand"
	"net/http"
	"net/http/httptest"

	"strconv"
	"strings"
	"sync"
	"testing"

	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/v2"
	k8sscheme "k8s.io/client-go/kubernetes/scheme"
)

var scheme = runtime.NewScheme()
var codecs = runtimeserializer.NewCodecFactory(scheme)
var serializer runtime.NegotiatedSerializer

const discoveryPath = "/discovery/v2"

func init() {
	// Add all builtin types to scheme
	k8sscheme.AddToScheme(scheme)
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok {
		panic("failed to create serializer info")
	}

	serializer = runtime.NewSimpleNegotiatedSerializer(info)
}

func fuzzAPIGroups(atLeastNumGroups, maxNumGroups int, seed int64) metav1.DiscoveryAPIGroupList {
	fuzzer := fuzz.NewWithSeed(seed)
	fuzzer.NumElements(atLeastNumGroups, maxNumGroups)
	fuzzer.NilChance(0)
	fuzzer.Funcs(func(o *metav1.DiscoveryAPIGroup, c fuzz.Continue) {
		c.FuzzNoCustom(o)

		// The ResourceManager will just not serve the grouop if its versions
		// list is empty
		atLeastOne := metav1.DiscoveryGroupVersion{}
		c.Fuzz(&atLeastOne)
		o.Versions = append(o.Versions, atLeastOne)

		o.TypeMeta = metav1.TypeMeta{
			Kind:       "DiscoveryAPIGroup",
			APIVersion: "v1",
		}
	})

	var apis []metav1.DiscoveryAPIGroup
	fuzzer.Fuzz(&apis)

	return metav1.DiscoveryAPIGroupList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "DiscoveryAPIGroupList",
			APIVersion: "v1",
		},
		Groups: apis,
	}
}

func fetchPath(handler http.Handler, path string, etag string) (*http.Response, []byte, *metav1.DiscoveryAPIGroupList) {
	// Expect json-formatted apis group list
	w := httptest.NewRecorder()
	req := httptest.NewRequest("GET", discoveryPath, nil)

	// Ask for JSON response
	req.Header.Set("Accept", "application/json")

	if etag != "" {
		// Quote provided etag if unquoted
		quoted := etag
		if !strings.HasPrefix(etag, "\"") {
			quoted = strconv.Quote(etag)
		}
		req.Header.Set("If-None-Match", quoted)
	}

	handler.ServeHTTP(w, req)

	bytes := w.Body.Bytes()
	var decoded *metav1.DiscoveryAPIGroupList
	if len(bytes) > 0 {
		decoded = &metav1.DiscoveryAPIGroupList{}
		runtime.DecodeInto(codecs.UniversalDecoder(), bytes, decoded)
	}

	return w.Result(), bytes, decoded
}

// Add all builtin APIServices to the manager and check the output
func TestBasicResponse(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager(serializer)

	apis := fuzzAPIGroups(1, 3, 10)
	manager.SetGroups(apis.Groups)

	response, body, decoded := fetchPath(manager, discoveryPath, "")

	jsonFormatted, err := json.Marshal(&apis)
	require.NoError(t, err, "json marshal should always succeed")

	assert.Equal(t, http.StatusOK, response.StatusCode, "response should be 200 OK")
	assert.Equal(t, "application/json", response.Header.Get("Content-Type"), "Content-Type response header should be as requested in Accept header if supported")
	assert.NotEmpty(t, response.Header.Get("ETag"), "E-Tag should be set")

	assert.NoError(t, err, "decode should always succeed")
	assert.EqualValues(t, &apis, decoded, "decoded value should equal input")
	assert.Equal(t, string(jsonFormatted)+"\n", string(body), "response should be the api group list")
}

// Test that an etag associated with the service only depends on the apiresources
// e.g.: Multiple services with the same contents should have the same etag.
func TestEtagConsistent(t *testing.T) {
	// Create 2 managers, add a bunch of services to each
	manager1 := discoveryendpoint.NewResourceManager(serializer)
	manager2 := discoveryendpoint.NewResourceManager(serializer)

	apis := fuzzAPIGroups(1, 3, 11)
	manager1.SetGroups(apis.Groups)
	manager2.SetGroups(apis.Groups)

	// Make sure etag of each is the same
	res1_initial, _, _ := fetchPath(manager1, discoveryPath, "")
	res2_initial, _, _ := fetchPath(manager2, discoveryPath, "")

	assert.NotEmpty(t, res1_initial.Header.Get("ETag"), "Etag should be populated")
	assert.NotEmpty(t, res2_initial.Header.Get("ETag"), "Etag should be populated")
	assert.Equal(t, res1_initial.Header.Get("ETag"), res2_initial.Header.Get("ETag"), "etag should be deterministic")

	// Then add one service to only one.
	// Make sure etag is changed, but other is the same
	apis = fuzzAPIGroups(1, 1, 11)
	for _, group := range apis.Groups {
		for _, version := range group.Versions {
			manager1.AddGroupVersion(group.Name, version)
		}
	}

	res1_addedToOne, _, _ := fetchPath(manager1, discoveryPath, "")
	res2_addedToOne, _, _ := fetchPath(manager2, discoveryPath, "")

	assert.NotEmpty(t, res1_addedToOne.Header.Get("ETag"), "Etag should be populated")
	assert.NotEmpty(t, res2_addedToOne.Header.Get("ETag"), "Etag should be populated")
	assert.NotEqual(t, res1_initial.Header.Get("ETag"), res1_addedToOne.Header.Get("ETag"), "ETag should be changed since version was added")
	assert.Equal(t, res2_initial.Header.Get("ETag"), res2_addedToOne.Header.Get("ETag"), "ETag should be unchanged since data was unchanged")

	// Then add service to other one
	// Make sure etag is the same
	for _, group := range apis.Groups {
		for _, version := range group.Versions {
			manager2.AddGroupVersion(group.Name, version)
		}
	}

	res1_addedToBoth, _, _ := fetchPath(manager1, discoveryPath, "")
	res2_addedToBoth, _, _ := fetchPath(manager2, discoveryPath, "")

	assert.NotEmpty(t, res1_addedToOne.Header.Get("ETag"), "Etag should be populated")
	assert.NotEmpty(t, res2_addedToOne.Header.Get("ETag"), "Etag should be populated")
	assert.Equal(t, res1_addedToBoth.Header.Get("ETag"), res2_addedToBoth.Header.Get("ETag"), "ETags should be equal since content is equal")
	assert.NotEqual(t, res2_initial.Header.Get("ETag"), res2_addedToBoth.Header.Get("ETag"), "ETag should be changed since data was changed")

	// Remove the group version from both. Initial E-Tag should be restored
	for _, group := range apis.Groups {
		for _, version := range group.Versions {
			manager1.RemoveGroupVersion(metav1.GroupVersion{
				Group:   group.Name,
				Version: version.Version,
			})
			manager2.RemoveGroupVersion(metav1.GroupVersion{
				Group:   group.Name,
				Version: version.Version,
			})
		}
	}

	res1_removeFromBoth, _, _ := fetchPath(manager1, discoveryPath, "")
	res2_removeFromBoth, _, _ := fetchPath(manager2, discoveryPath, "")

	assert.NotEmpty(t, res1_addedToOne.Header.Get("ETag"), "Etag should be populated")
	assert.NotEmpty(t, res2_addedToOne.Header.Get("ETag"), "Etag should be populated")
	assert.Equal(t, res1_removeFromBoth.Header.Get("ETag"), res2_removeFromBoth.Header.Get("ETag"), "ETags should be equal since content is equal")
	assert.Equal(t, res1_initial.Header.Get("ETag"), res1_removeFromBoth.Header.Get("ETag"), "ETag should be equal to initial value since added content was removed")
}

// Test that if a request comes in with an If-None-Match header with an incorrect
// E-Tag, that fresh content is returned.
func TestEtagNonMatching(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager(serializer)
	apis := fuzzAPIGroups(1, 3, 12)
	manager.SetGroups(apis.Groups)

	// fetch the document once
	initial, _, _ := fetchPath(manager, discoveryPath, "")
	assert.NotEmpty(t, initial.Header.Get("ETag"), "ETag should be populated")

	// Send another request with a wrong e-tag. The same response should
	// get sent again
	second, _, _ := fetchPath(manager, discoveryPath, "wrongetag")

	assert.Equal(t, http.StatusOK, initial.StatusCode, "response should be 200 OK")
	assert.Equal(t, http.StatusOK, second.StatusCode, "response should be 200 OK")
	assert.Equal(t, initial.Header.Get("ETag"), second.Header.Get("ETag"), "ETag of both requests should be equal")
}

// Test that if a request comes in with an If-None-Match header with a correct
// E-Tag, that 304 Not Modified is returned
func TestEtagMatching(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager(serializer)
	apis := fuzzAPIGroups(1, 3, 12)
	manager.SetGroups(apis.Groups)

	// fetch the document once
	initial, initialBody, _ := fetchPath(manager, discoveryPath, "")
	assert.NotEmpty(t, initial.Header.Get("ETag"), "ETag should be populated")
	assert.NotEmpty(t, initialBody, "body should not be empty")

	// Send another request with a wrong e-tag. The same response should
	// get sent again
	second, secondBody, _ := fetchPath(manager, discoveryPath, initial.Header.Get("ETag"))

	assert.Equal(t, http.StatusOK, initial.StatusCode, "initial response should be 200 OK")
	assert.Equal(t, http.StatusNotModified, second.StatusCode, "second response should be 304 Not Modified")
	assert.Equal(t, initial.Header.Get("ETag"), second.Header.Get("ETag"), "ETag of both requests should be equal")
	assert.Empty(t, secondBody, "body should be empty when returning 304 Not Modified")
}

// Test that if a request comes in with an If-None-Match header with an old
// E-Tag, that fresh content is returned
func TestEtagOutdated(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager(serializer)
	apis := fuzzAPIGroups(1, 3, 15)
	manager.SetGroups(apis.Groups)

	// fetch the document once
	initial, initialBody, _ := fetchPath(manager, discoveryPath, "")
	assert.NotEmpty(t, initial.Header.Get("ETag"), "ETag should be populated")
	assert.NotEmpty(t, initialBody, "body should not be empty")

	// Then add some services so the etag changes
	apis = fuzzAPIGroups(1, 3, 14)
	for _, group := range apis.Groups {
		for _, version := range group.Versions {
			manager.AddGroupVersion(group.Name, version)
		}
	}

	// Send another request with the old e-tag. Response should not be 304 Not Modified
	second, secondBody, _ := fetchPath(manager, discoveryPath, initial.Header.Get("ETag"))

	assert.Equal(t, http.StatusOK, initial.StatusCode, "initial response should be 200 OK")
	assert.Equal(t, http.StatusOK, second.StatusCode, "second response should be 304 Not Modified")
	assert.NotEqual(t, initial.Header.Get("ETag"), second.Header.Get("ETag"), "ETag of both requests should be unequal since contents differ")
	assert.NotEmpty(t, secondBody, "body should be not empty when returning 304 Not Modified")
}

// Test that an api service can be added or removed
func TestAddRemove(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager(serializer)
	apis := fuzzAPIGroups(1, 3, 15)
	for _, group := range apis.Groups {
		for _, version := range group.Versions {
			manager.AddGroupVersion(group.Name, version)
		}
	}

	_, _, initialDocument := fetchPath(manager, discoveryPath, "")

	for _, group := range apis.Groups {
		for _, version := range group.Versions {
			manager.RemoveGroupVersion(metav1.GroupVersion{
				Group:   group.Name,
				Version: version.Version,
			})
		}
	}

	_, _, secondDocument := fetchPath(manager, discoveryPath, "")

	require.NotNil(t, initialDocument, "initial document should parse")
	require.NotNil(t, secondDocument, "second document should parse")
	assert.Len(t, initialDocument.Groups, len(apis.Groups), "initial document should have set number of groups")
	assert.Len(t, secondDocument.Groups, 0, "second document should have no groups")
}

// Show that updating an existing service replaces and does not add the entry
// and instead replaces it
func TestUpdateService(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager(serializer)
	apis := fuzzAPIGroups(1, 3, 15)
	for _, group := range apis.Groups {
		for _, version := range group.Versions {
			manager.AddGroupVersion(group.Name, version)
		}
	}

	_, _, initialDocument := fetchPath(manager, discoveryPath, "")

	assert.Equal(t, initialDocument, &apis, "should have returned expected document")

	apis.Groups[0].Versions[0].APIResources[0].Name = "changed a resource name!"

	for _, group := range apis.Groups {
		for _, version := range group.Versions {
			manager.AddGroupVersion(group.Name, version)
		}
	}

	_, _, secondDocument := fetchPath(manager, discoveryPath, "")
	assert.Equal(t, secondDocument, &apis, "should have returned expected document")
	assert.NotEqual(t, secondDocument, initialDocument, "should have returned expected document")
}

// Show the discovery manager is capable of serving requests to multiple users
// with unchanging data
func TestConcurrentRequests(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager(serializer)
	apis := fuzzAPIGroups(1, 3, 15)
	manager.SetGroups(apis.Groups)

	waitGroup := sync.WaitGroup{}

	numReaders := 100
	numRequestsPerReader := 100

	// Spawn a bunch of readers that will keep sending requests to the server
	for i := 0; i < numReaders; i++ {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			etag := ""
			for j := 0; j < numRequestsPerReader; j++ {
				usedEtag := etag
				if j%2 == 0 {
					// Disable use of etag for every second request
					usedEtag = ""
				}
				response, body, document := fetchPath(manager, discoveryPath, usedEtag)

				if usedEtag != "" {
					assert.Equal(t, http.StatusNotModified, response.StatusCode, "response should be Not Modified if etag was used")
					assert.Empty(t, body, "body should be empty if etag used")
				} else {
					assert.Equal(t, http.StatusOK, response.StatusCode, "response should be OK if etag was unused")
					assert.Equal(t, &apis, document, "document should be equal")
				}

				etag = response.Header.Get("ETag")
			}
		}()
	}
	waitGroup.Wait()
}

// Show the handler is capable of serving many concurrent readers and many
// concurrent writers without tripping up. Good to run with go '-race' detector
// since there are not many "correctness" checks
func TestAbuse(t *testing.T) {
	manager := discoveryendpoint.NewResourceManager(serializer)

	numReaders := 100
	numRequestsPerReader := 1000

	numWriters := 10
	numWritesPerWriter := 1000

	waitGroup := sync.WaitGroup{}

	// Spawn a bunch of writers that randomly add groups, remove groups, and
	// reset the list of groups
	for i := 0; i < numWriters; i++ {
		source := rand.NewSource(int64(i))

		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()

			// track list of groups we've added so that we can remove them
			// randomly
			var addedGroups []metav1.GroupVersion

			for j := 0; j < numWritesPerWriter; j++ {
				switch source.Int63() % 3 {
				case 0:
					// Add a fuzzed group
					apis := fuzzAPIGroups(1, 2, 15)
					for _, group := range apis.Groups {
						for _, version := range group.Versions {
							manager.AddGroupVersion(group.Name, version)
							addedGroups = append(addedGroups, metav1.GroupVersion{
								Group:   group.Name,
								Version: version.Version,
							})
						}
					}
				case 1:
					// Remove a group that we have added
					if len(addedGroups) > 0 {
						manager.RemoveGroupVersion(addedGroups[0])
						addedGroups = addedGroups[1:]
					} else {
						// Send a request and try to remove a group someone else
						// might have added
						_, _, document := fetchPath(manager, discoveryPath, "")
						assert.NotNil(t, document, "manager should always succeed in returning a document")

						if len(document.Groups) > 0 {
							manager.RemoveGroupVersion(metav1.GroupVersion{
								Group:   document.Groups[0].Name,
								Version: document.Groups[0].Versions[0].Version,
							})
						}

					}
				case 2:
					manager.SetGroups(nil)
					addedGroups = nil
				default:
					panic("unreachable")
				}
			}
		}()
	}

	// Spawn a bunch of readers that will keep sending requests to the server
	// and making sure the response makes sense
	for i := 0; i < numReaders; i++ {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()

			etag := ""
			for j := 0; j < numRequestsPerReader; j++ {
				response, body, document := fetchPath(manager, discoveryPath, etag)

				if response.StatusCode == http.StatusNotModified {
					assert.Equal(t, etag, response.Header.Get("ETag"))
					assert.Empty(t, body, "body should be empty if etag used")
					assert.Nil(t, document)
				} else {
					assert.Equal(t, http.StatusOK, response.StatusCode, "response should be OK if etag was unused")
					assert.NotNil(t, document)
				}

				etag = response.Header.Get("ETag")
			}
		}()
	}

	waitGroup.Wait()
}
