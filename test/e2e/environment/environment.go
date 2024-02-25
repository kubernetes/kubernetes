/*
Copyright 2023 The Kubernetes Authors.

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

// Package environment contains pre-defined environments used by test/e2e
// and/or test/e2e_node.
package environment

import (
	"k8s.io/kubernetes/test/e2e/framework"
)

var (
	// Please keep the list in alphabetical order.

	// The test does not work in UserNS (for example, `open /proc/sys/kernel/shm_rmid_forced: permission denied`).
	NotInUserNS = framework.WithEnvironment(framework.ValidEnvironments.Add("NotInUserNS"))

	// Please keep the list in alphabetical order.
)

func init() {
	// This prevents adding additional ad-hoc environments in tests.
	framework.ValidEnvironments.Freeze()
}
