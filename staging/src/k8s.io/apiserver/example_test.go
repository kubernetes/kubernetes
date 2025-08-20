/*
Copyright 2025 The Kubernetes Authors.

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

package apiserver_test

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/server"
)

func Example_instantiation() {
	// 1. Create a scheme and a codec factory.
	// In a real application, you would register your API types with the scheme.
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)

	// 2. Create a new server configuration.
	// This creates a recommended configuration with default values.
	serverConfig := server.NewRecommendedConfig(codecs)

	// 3. Create a new GenericAPIServer.
	// The "delegate" is used to chain to another API server. For a standalone
	// server, we can use an empty delegate.
	genericServer, err := serverConfig.Complete().New("my-api-server", server.NewEmptyDelegate())
	if err != nil {
		fmt.Printf("Error creating generic API server: %v", err)
		return
	}

	// 4. Install an API group.
	// This is where you would define your API resources and their storage.
	// For this example, we'll use an empty APIGroupInfo.
	apiGroupInfo := &server.APIGroupInfo{
		PrioritizedVersions:          []schema.GroupVersion{{Group: "mygroup.example.com", Version: "v1"}},
		VersionedResourcesStorageMap: make(map[string]map[string]rest.Storage),
		Scheme:                       scheme,
		ParameterCodec:               runtime.NewParameterCodec(scheme),
		NegotiatedSerializer:         codecs,
	}
	if err := genericServer.InstallAPIGroup(apiGroupInfo); err != nil {
		fmt.Printf("Error installing API group: %v", err)
		return
	}

	fmt.Println("GenericAPIServer created successfully.")

	// Output:
	// GenericAPIServer created successfully.
}
