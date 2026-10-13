/*
Copyright 2020 The Kubernetes Authors.

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

// The external controller manager is responsible for running controller loops that
// are cloud provider dependent. It uses the API to listen to new events on resources.

package main

// NOTE: Importing all in-tree cloud-providers is not required when
// implementing an out-of-tree cloud-provider. Leaving this empty file
// here as a reference.

// Here is how you would inject a cloud provider, first
// you would use a init() method in say "k8s.io/legacy-cloud-providers/gce"
// package that calls `cloudprovider.RegisterCloudProvider()`
// and then here in this file you would add an import.
//
// import _ "k8s.io/legacy-cloud-providers/gce"
