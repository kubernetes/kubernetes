/*
Copyright 2016 The Kubernetes Authors.

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
	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	e2eframework "k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Framework supports common operations used by e2e tests; it will keep a client & a namespace for you.
// Eventual goal is to merge this with integration test framework.
type Framework struct {
	BaseName string

	// CRI client
	CRIClient *InternalApiClient

	// To make sure that this framework cleans up after itself, no matter what,
	// we install a Cleanup action before each test and clear it after.  If we
	// should abort, the AfterSuite hook should run all Cleanup actions.
	cleanupHandle e2eframework.CleanupActionHandle
}

type InternalApiClient struct {
	CRIRuntimeClient internalapi.RuntimeService
	CRIImageClient   internalapi.ImageManagerService
}

// NewDefaultCRIFramework makes a new framework and sets up a BeforeEach/AfterEach for
// you (you can write additional before/after each functions).
func NewDefaultCRIFramework(baseName string) *Framework {
	return NewCRIFramework(baseName, nil)
}

func NewCRIFramework(baseName string, client *InternalApiClient) *Framework {
	f := &Framework{
		BaseName:  baseName,
		CRIClient: client,
	}

	BeforeEach(f.BeforeEach)
	AfterEach(f.AfterEach)

	return f
}

// BeforeEach gets a client
func (f *Framework) BeforeEach() {
	// The fact that we need this feels like a bug in ginkgo.
	// https://github.com/onsi/ginkgo/issues/222
	f.cleanupHandle = e2eframework.AddCleanupAction(f.AfterEach)

	if f.CRIClient == nil {
		c, err := LoadCRIClient()
		Expect(err).NotTo(HaveOccurred())
		f.CRIClient = c
	}
}

// AfterEach clean resourses and print summaries
func (f *Framework) AfterEach() {
	// remove the cleanup handle.
	e2eframework.RemoveCleanupAction(f.cleanupHandle)

	f.CRIClient = nil
}

func KubeDescribe(text string, body func()) bool {
	return Describe("[k8s.io] "+text, body)
}
