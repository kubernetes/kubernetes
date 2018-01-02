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
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/containernetworking/cni/libcni"
	"github.com/containernetworking/cni/pkg/version/legacy_examples"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("Backwards compatibility", func() {
	It("correctly handles the response from a legacy plugin", func() {
		example := legacy_examples.V010
		pluginPath, err := example.Build()
		Expect(err).NotTo(HaveOccurred())

		netConf, err := libcni.ConfFromBytes([]byte(fmt.Sprintf(
			`{ "name": "old-thing", "type": "%s" }`, example.Name)))
		Expect(err).NotTo(HaveOccurred())

		runtimeConf := &libcni.RuntimeConf{
			ContainerID: "some-container-id",
			NetNS:       "/some/netns/path",
			IfName:      "eth0",
		}

		cniConfig := &libcni.CNIConfig{Path: []string{filepath.Dir(pluginPath)}}

		result, err := cniConfig.AddNetwork(netConf, runtimeConf)
		Expect(err).NotTo(HaveOccurred())

		Expect(result).To(Equal(legacy_examples.ExpectedResult))

		Expect(os.RemoveAll(pluginPath)).To(Succeed())
	})

	It("correctly handles the request from a runtime with an older libcni", func() {
		example := legacy_examples.V010_Runtime

		binPath, err := example.Build()
		Expect(err).NotTo(HaveOccurred())

		for _, configName := range example.NetConfs {
			conf, err := example.GenerateNetConf(configName)
			if err != nil {
				Fail("Failed to generate config name " + configName + ": " + err.Error())
			}
			defer conf.Cleanup()

			cmd := exec.Command(binPath, pluginDirs...)
			cmd.Stdin = strings.NewReader(conf.Config)

			session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
			Expect(err).NotTo(HaveOccurred())

			Eventually(session).Should(gexec.Exit(0))
		}
	})
})
