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

// Package namer has support for making different type naming systems.
//
// This is because sometimes you want to refer to the literal type, sometimes
// you want to make a name for the thing you're generating, and you want to
// make the name based on the type. For example, if you have `type foo string`,
// you want to be able to generate something like `func FooPrinter(f *foo) {
// Print(string(*f)) }`; that is, you want to refer to a public name, a literal
// name, and the underlying literal name.
//
// This package supports the idea of a "Namer" and a set of "NameSystems" to
// support these use cases.
//
// Additionally, a "RawNamer" can optionally keep track of what needs to be
// imported.
package namer
