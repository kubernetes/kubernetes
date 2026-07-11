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

package framework

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
)

// PerTestConfig represents parameters that control test execution.
// One instance gets allocated for each test and is then passed
// via pointer to functions involved in the test.
type PerTestConfig struct {
	// The test driver for the test.
	Driver TestDriver

	// Some short word that gets inserted into dynamically
	// generated entities (pods, paths) as first part of the name
	// to make debugging easier. Can be the same for different
	// tests inside the test suite.
	Prefix string

	// The framework instance allocated for the current test.
	Framework *framework.Framework

	// If non-empty, Pods using a volume will be scheduled
	// according to the NodeSelection. Otherwise Kubernetes will
	// pick a node.
	ClientNodeSelection e2epod.NodeSelection

	// Some test drivers initialize a storage server. This is
	// the configuration that then has to be used to run tests.
	// The values above are ignored for such tests.
	ServerConfig *e2evolume.TestConfig

	// Some drivers run in their own namespace
	DriverNamespace *v1.Namespace
}

// GetUniqueDriverName returns unique driver name that can be used parallelly in tests
func (config *PerTestConfig) GetUniqueDriverName() string {
	return config.Driver.GetDriverInfo().Name + "-" + config.Framework.UniqueName
}

// ConvertTestConfig returns a framework test config with the
// parameters specified for the testsuite or (if available) the
// dynamically created config for the volume server.
//
// This is done because TestConfig is the public API for
// the testsuites package whereas volume.TestConfig is merely
// an implementation detail. It contains fields that have no effect.
func ConvertTestConfig(in *PerTestConfig) e2evolume.TestConfig {
	if in.ServerConfig != nil {
		return *in.ServerConfig
	}

	return e2evolume.TestConfig{
		Namespace:           in.Framework.Namespace.Name,
		Prefix:              in.Prefix,
		ClientNodeSelection: in.ClientNodeSelection,
	}
}
