/*
Copyright 2018 The Kubernetes Authors.

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

// Package schema defines a targeted schema language which allows one to
// represent all the schema information necessary to perform "structured"
// merges and diffs.
//
// Due to the targeted nature of the data model, the schema language can fit in
// just a few hundred lines of go code, making it much more understandable and
// concise than e.g. OpenAPI.
//
// This schema was derived by observing the API objects used by Kubernetes, and
// formalizing a model which allows certain operations ("apply") to be more
// well defined. It is currently missing one feature: one-of ("unions").
package schema
