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

// Package generator defines an interface for code generators to implement.
//
// To use this package, you'll implement the "Package" and "Generator"
// interfaces; you'll call NewContext to load up the types you want to work
// with, and then you'll call one or more of the Execute methods. See the
// interface definitions for explanations. All output will have gofmt called on
// it automatically, so you do not need to worry about generating correct
// indentation.
//
// This package also exposes SnippetWriter. SnippetWriter reduces to a minimum
// the boilerplate involved in setting up a template from go's text/template
// package. Additionally, all naming systems in the Context will be added as
// functions to the parsed template, so that they can be called directly from
// your templates!
package generator // import "k8s.io/gengo/v2/generator"
