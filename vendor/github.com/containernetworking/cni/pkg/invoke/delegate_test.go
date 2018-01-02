// Copyright 2017 CNI authors
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

package invoke_test

import (
	"encoding/json"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"

	"github.com/containernetworking/cni/pkg/invoke"
	"github.com/containernetworking/cni/pkg/types/current"
	"github.com/containernetworking/cni/plugins/test/noop/debug"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Delegate", func() {
	var (
		pluginName     string
		netConf        []byte
		debugFileName  string
		debugBehavior  *debug.Debug
		expectedResult *current.Result
	)

	BeforeEach(func() {
		netConf, _ = json.Marshal(map[string]string{
			"cniVersion": "0.3.1",
		})

		expectedResult = &current.Result{
			CNIVersion: "0.3.1",
			IPs: []*current.IPConfig{
				&current.IPConfig{
					Version: "4",
					Address: net.IPNet{
						IP:   net.ParseIP("10.1.2.3"),
						Mask: net.CIDRMask(24, 32),
					},
				},
			},
		}
		expectedResultBytes, _ := json.Marshal(expectedResult)

		debugFile, err := ioutil.TempFile("", "cni_debug")
		Expect(err).NotTo(HaveOccurred())
		Expect(debugFile.Close()).To(Succeed())
		debugFileName = debugFile.Name()
		debugBehavior = &debug.Debug{
			ReportResult: string(expectedResultBytes),
		}
		Expect(debugBehavior.WriteDebug(debugFileName)).To(Succeed())
		pluginName = "noop"

		os.Setenv("CNI_ARGS", "DEBUG="+debugFileName)
		os.Setenv("CNI_PATH", filepath.Dir(pathToPlugin))
		os.Setenv("CNI_NETNS", "/tmp/some/netns/path")
		os.Setenv("CNI_IFNAME", "eth7")
	})

	AfterEach(func() {
		os.RemoveAll(debugFileName)
	})

	Describe("DelegateAdd", func() {
		BeforeEach(func() {
			os.Setenv("CNI_COMMAND", "ADD")
		})

		It("finds and execs the named plugin", func() {
			result, err := invoke.DelegateAdd(pluginName, netConf)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(expectedResult))

			pluginInvocation, err := debug.ReadDebug(debugFileName)
			Expect(err).NotTo(HaveOccurred())
			Expect(pluginInvocation.Command).To(Equal("ADD"))
			Expect(pluginInvocation.CmdArgs.IfName).To(Equal("eth7"))
		})

		Context("if the delegation isn't part of an existing ADD command", func() {
			BeforeEach(func() {
				os.Setenv("CNI_COMMAND", "NOPE")
			})

			It("aborts and returns a useful error", func() {
				_, err := invoke.DelegateAdd(pluginName, netConf)
				Expect(err).To(MatchError("CNI_COMMAND is not ADD"))
			})
		})

		Context("when the plugin cannot be found", func() {
			BeforeEach(func() {
				pluginName = "non-existent-plugin"
			})

			It("returns a useful error", func() {
				_, err := invoke.DelegateAdd(pluginName, netConf)
				Expect(err).To(MatchError(HavePrefix("failed to find plugin")))
			})
		})
	})

	Describe("DelegateDel", func() {
		BeforeEach(func() {
			os.Setenv("CNI_COMMAND", "DEL")
		})

		It("finds and execs the named plugin", func() {
			err := invoke.DelegateDel(pluginName, netConf)
			Expect(err).NotTo(HaveOccurred())

			pluginInvocation, err := debug.ReadDebug(debugFileName)
			Expect(err).NotTo(HaveOccurred())
			Expect(pluginInvocation.Command).To(Equal("DEL"))
			Expect(pluginInvocation.CmdArgs.IfName).To(Equal("eth7"))
		})

		Context("if the delegation isn't part of an existing DEL command", func() {
			BeforeEach(func() {
				os.Setenv("CNI_COMMAND", "NOPE")
			})

			It("aborts and returns a useful error", func() {
				err := invoke.DelegateDel(pluginName, netConf)
				Expect(err).To(MatchError("CNI_COMMAND is not DEL"))
			})
		})

		Context("when the plugin cannot be found", func() {
			BeforeEach(func() {
				pluginName = "non-existent-plugin"
			})

			It("returns a useful error", func() {
				err := invoke.DelegateDel(pluginName, netConf)
				Expect(err).To(MatchError(HavePrefix("failed to find plugin")))
			})
		})
	})
})
