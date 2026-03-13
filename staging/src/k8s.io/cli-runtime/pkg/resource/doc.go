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

// Package resource assists clients in dealing with RESTful objects that match the
// Kubernetes API conventions. The Helper object provides simple CRUD operations
// on resources. The Visitor interface makes it easy to deal with multiple resources
// in bulk for retrieval and operation. The Builder object simplifies converting
// standard command line arguments and parameters into a Visitor that can iterate
// over all of the identified resources, whether on the server or on the local
// filesystem.
package resource
