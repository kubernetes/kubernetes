/*
Copyright 2021 The Kubernetes Authors.

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

// Package init registers the command line flags for k8s.io/klogr/testing in
// the flag.CommandLine. This is done during initialization, so merely
// importing it is enough.
//
// Experimental
//
// Notice: This package is EXPERIMENTAL and may be changed or removed in a
// later release.
package init

import (
	"flag"

	"k8s.io/klog/v2/ktesting"
)

func init() {
	ktesting.DefaultConfig.AddFlags(flag.CommandLine)
}
