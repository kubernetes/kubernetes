// Copyright 2016 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"

	"testing"
)

func TestNoop(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "No-op plugin Suite")
}

const packagePath = "github.com/containernetworking/cni/plugins/test/noop"

var pathToPlugin string

var _ = SynchronizedBeforeSuite(func() []byte {
	var err error
	pathToPlugin, err = gexec.Build(packagePath)
	Expect(err).NotTo(HaveOccurred())
	return []byte(pathToPlugin)
}, func(crossNodeData []byte) {
	pathToPlugin = string(crossNodeData)
})

var _ = SynchronizedAfterSuite(func() {}, func() {
	gexec.CleanupBuildArtifacts()
})
