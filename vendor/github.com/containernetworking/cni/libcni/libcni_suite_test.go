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

package libcni_test

import (
	"fmt"
	"path/filepath"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"

	"testing"
)

func TestLibcni(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Libcni Suite")
}

var plugins = map[string]string{
	"noop": "github.com/containernetworking/cni/plugins/test/noop",
}

var pluginPaths map[string]string
var pluginDirs []string // array of plugin dirs

var _ = SynchronizedBeforeSuite(func() []byte {
	dirs := make([]string, 0, len(plugins))

	for name, packagePath := range plugins {
		execPath, err := gexec.Build(packagePath)
		Expect(err).NotTo(HaveOccurred())
		dirs = append(dirs, fmt.Sprintf("%s=%s", name, execPath))
	}

	return []byte(strings.Join(dirs, ":"))
}, func(crossNodeData []byte) {
	pluginPaths = make(map[string]string)
	for _, str := range strings.Split(string(crossNodeData), ":") {
		kvs := strings.SplitN(str, "=", 2)
		if len(kvs) != 2 {
			Fail("Invalid inter-node data...")
		}
		pluginPaths[kvs[0]] = kvs[1]
		pluginDirs = append(pluginDirs, filepath.Dir(kvs[1]))
	}
})

var _ = SynchronizedAfterSuite(func() {}, func() {
	gexec.CleanupBuildArtifacts()
})
