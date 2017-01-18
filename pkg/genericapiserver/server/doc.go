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

// Package genericapiserver contains code to setup a generic kubernetes-like API server.
// This does not contain any kubernetes API specific code.
// Note that this is a work in progress. We are pulling out generic code (specifically from
// pkg/master) here.
// We plan to move this package into a separate repo on github once it is done.
// For more details: https://github.com/kubernetes/kubernetes/issues/2742
package genericapiserver // import "k8s.io/kubernetes/pkg/genericapiserver"
