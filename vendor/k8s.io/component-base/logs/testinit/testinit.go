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

// Package testinit adds logging flags to a Ginkgo or Go test program during
// initialization, something that the logs package itself no longer does.
//
// Normal commands should not use this and instead manage logging flags with
// logs.Options and/or cli.Run.
package testinit

import (
	"flag"

	"k8s.io/component-base/logs"
)

func init() {
	logs.AddGoFlags(flag.CommandLine)
}
