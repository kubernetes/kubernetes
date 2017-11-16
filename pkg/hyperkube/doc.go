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

// Package hyperkube is a framework for kubernetes server components.  It
// allows us to combine all of the kubernetes server components into a single
// binary where the user selects which components to run in any individual
// process.
//
// Currently, only one server component can be run at once.  As such there is
// no need to harmonize flags or identify logs across the various servers.  In
// the future we will support launching and running many servers -- either by
// managing processes or running in-proc.
//
// This package is inspired by https://github.com/spf13/cobra.  However, as
// the eventual goal is to run *multiple* servers from one call, a new package
// was needed.
package hyperkube // import "k8s.io/kubernetes/pkg/hyperkube"
