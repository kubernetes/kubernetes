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

// Package cloud implements a more golang friendly interface to the GCE compute
// API. The code in this package is generated automatically via the generator
// implemented in "gen/main.go".  The code generator creates the basic CRUD
// actions for the given resource: "Insert", "Get", "List" and "Delete".
// Additional methods by customizing the ServiceInfo object (see below).
// Generated code includes a full mock of the GCE compute API.
//
// Usage
//
// The root of the GCE compute API is the interface "Cloud". Code written using
// Cloud can be used against the actual implementation "GCE" or "MockGCE".
//
//  func foo(cloud Cloud) {
//    igs, err := cloud.InstanceGroups().List(ctx, "us-central1-b", filter.None)
//    ...
//  }
//  // Run foo against the actual cloud.
//  foo(NewGCE(&Service{...}))
//  // Run foo with a mock.
//  foo(NewMockGCE())
//
// Rate limiting and routing
//
// The generated code allows for custom policies for operation rate limiting
// and GCE project routing. See RateLimiter and ProjectRouter for more details.
//
// Mocks
//
// Mocks are automatically generated for each type implementing basic logic for
// resource manipulation.  This eliminates the boilerplate required to mock GCE
// functionality.  Each method will also have a corresponding "xxxHook"
// function generated in the mock structure where unit test code can hook the
// execution of the method.
//
// Mocks for different versions of the same service will share the same set of
// objects, i.e. an alpha object will be visible with beta and GA methods.
// Note that translation is done with JSON serialization between the API versions.
//
// Changing service code generation
//
// The list of services to generate is contained in "meta/meta.go". To add a
// service, add an entry to the list "meta.AllServices". An example entry:
//
//  &ServiceInfo{
//    Object:      "InstanceGroup",   // Name of the object type.
//    Service:     "InstanceGroups",  // Name of the service.
//    Resource:    "instanceGroups",  // Lowercase resource name (as appears in the URL).
//    version:     meta.VersionAlpha, // API version (one entry per version is needed).
//    keyType:     Zonal,             // What kind of resource this is.
//    serviceType: reflect.TypeOf(&alpha.InstanceGroupsService{}), // Associated golang type.
//    additionalMethods: []string{    // Additional methods to generate code for.
//      "SetNamedPorts",
//    },
//    options: <options>              // Or'd ("|") together.
//  }
//
// Read-only objects
//
// Services such as Regions and Zones do not allow for mutations. Specify
// "ReadOnly" in ServiceInfo.options to omit the mutation methods.
//
// Adding custom methods
//
// Some methods that may not be properly handled by the generated code. To enable
// addition of custom code to the generated mocks, set the "CustomOps" option
// in "meta.ServiceInfo" entry. This will make the generated service interface
// embed a "<ServiceName>Ops" interface. This interface MUST be written by hand
// and contain the custom method logic. Corresponding methods must be added to
// the corresponding Mockxxx and GCExxx struct types.
//
//  // In "meta/meta.go":
//  &ServiceInfo{
//    Object: "InstanceGroup",
//    ...
//    options: CustomOps,
//  }
//
//  // In the generated code "gen.go":
//  type InstanceGroups interface {
//    InstanceGroupsOps // Added by CustomOps option.
//    ...
//  }
//
//  // In hand written file:
//  type InstanceGroupsOps interface {
//    MyMethod()
//  }
//
//  func (mock *MockInstanceGroups) MyMethod() {
//    // Custom mock implementation.
//  }
//
//  func (gce *GCEInstanceGroups) MyMethod() {
//    // Custom implementation.
//  }
//
// Update generated codes
//
// Run hack/update-cloudprovider-gce.sh to update the generated codes.
//
package cloud
