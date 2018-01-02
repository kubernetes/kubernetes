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

package invoke_test

import (
	"encoding/json"
	"errors"

	"github.com/containernetworking/cni/pkg/invoke"
	"github.com/containernetworking/cni/pkg/invoke/fakes"
	"github.com/containernetworking/cni/pkg/types/current"
	"github.com/containernetworking/cni/pkg/version"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Executing a plugin, unit tests", func() {
	var (
		pluginExec     *invoke.PluginExec
		rawExec        *fakes.RawExec
		versionDecoder *fakes.VersionDecoder

		pluginPath string
		netconf    []byte
		cniargs    *fakes.CNIArgs
	)

	BeforeEach(func() {
		rawExec = &fakes.RawExec{}
		rawExec.ExecPluginCall.Returns.ResultBytes = []byte(`{ "ips": [ { "version": "4", "address": "1.2.3.4/24" } ] }`)

		versionDecoder = &fakes.VersionDecoder{}
		versionDecoder.DecodeCall.Returns.PluginInfo = version.PluginSupports("0.42.0")

		pluginExec = &invoke.PluginExec{
			RawExec:        rawExec,
			VersionDecoder: versionDecoder,
		}
		pluginPath = "/some/plugin/path"
		netconf = []byte(`{ "some": "stdin", "cniVersion": "0.3.1" }`)
		cniargs = &fakes.CNIArgs{}
		cniargs.AsEnvCall.Returns.Env = []string{"SOME=ENV"}
	})

	Describe("returning a result", func() {
		It("unmarshals the result bytes into the Result type", func() {
			r, err := pluginExec.WithResult(pluginPath, netconf, cniargs)
			Expect(err).NotTo(HaveOccurred())

			result, err := current.GetResult(r)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(result.IPs)).To(Equal(1))
			Expect(result.IPs[0].Address.IP.String()).To(Equal("1.2.3.4"))
		})

		It("passes its arguments through to the rawExec", func() {
			pluginExec.WithResult(pluginPath, netconf, cniargs)
			Expect(rawExec.ExecPluginCall.Received.PluginPath).To(Equal(pluginPath))
			Expect(rawExec.ExecPluginCall.Received.StdinData).To(Equal(netconf))
			Expect(rawExec.ExecPluginCall.Received.Environ).To(Equal([]string{"SOME=ENV"}))
		})

		Context("when the rawExec fails", func() {
			BeforeEach(func() {
				rawExec.ExecPluginCall.Returns.Error = errors.New("banana")
			})
			It("returns the error", func() {
				_, err := pluginExec.WithResult(pluginPath, netconf, cniargs)
				Expect(err).To(MatchError("banana"))
			})
		})
	})

	Describe("without returning a result", func() {
		It("passes its arguments through to the rawExec", func() {
			pluginExec.WithoutResult(pluginPath, netconf, cniargs)
			Expect(rawExec.ExecPluginCall.Received.PluginPath).To(Equal(pluginPath))
			Expect(rawExec.ExecPluginCall.Received.StdinData).To(Equal(netconf))
			Expect(rawExec.ExecPluginCall.Received.Environ).To(Equal([]string{"SOME=ENV"}))
		})

		Context("when the rawExec fails", func() {
			BeforeEach(func() {
				rawExec.ExecPluginCall.Returns.Error = errors.New("banana")
			})
			It("returns the error", func() {
				err := pluginExec.WithoutResult(pluginPath, netconf, cniargs)
				Expect(err).To(MatchError("banana"))
			})
		})
	})

	Describe("discovering the plugin version", func() {
		BeforeEach(func() {
			rawExec.ExecPluginCall.Returns.ResultBytes = []byte(`{ "some": "version-info" }`)
		})

		It("execs the plugin with the command VERSION", func() {
			pluginExec.GetVersionInfo(pluginPath)
			Expect(rawExec.ExecPluginCall.Received.PluginPath).To(Equal(pluginPath))
			Expect(rawExec.ExecPluginCall.Received.Environ).To(ContainElement("CNI_COMMAND=VERSION"))
			expectedStdin, _ := json.Marshal(map[string]string{"cniVersion": version.Current()})
			Expect(rawExec.ExecPluginCall.Received.StdinData).To(MatchJSON(expectedStdin))
		})

		It("decodes and returns the version info", func() {
			versionInfo, err := pluginExec.GetVersionInfo(pluginPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(versionInfo.SupportedVersions()).To(Equal([]string{"0.42.0"}))
			Expect(versionDecoder.DecodeCall.Received.JSONBytes).To(MatchJSON(`{ "some": "version-info" }`))
		})

		Context("when the rawExec fails", func() {
			BeforeEach(func() {
				rawExec.ExecPluginCall.Returns.Error = errors.New("banana")
			})
			It("returns the error", func() {
				_, err := pluginExec.GetVersionInfo(pluginPath)
				Expect(err).To(MatchError("banana"))
			})
		})

		Context("when the plugin is too old to recognize the VERSION command", func() {
			BeforeEach(func() {
				rawExec.ExecPluginCall.Returns.Error = errors.New("unknown CNI_COMMAND: VERSION")
			})

			It("interprets the error as a 0.1.0 version", func() {
				versionInfo, err := pluginExec.GetVersionInfo(pluginPath)
				Expect(err).NotTo(HaveOccurred())
				Expect(versionInfo.SupportedVersions()).To(ConsistOf("0.1.0"))
			})

			It("sets dummy values for env vars required by very old plugins", func() {
				pluginExec.GetVersionInfo(pluginPath)

				env := rawExec.ExecPluginCall.Received.Environ
				Expect(env).To(ContainElement("CNI_NETNS=dummy"))
				Expect(env).To(ContainElement("CNI_IFNAME=dummy"))
				Expect(env).To(ContainElement("CNI_PATH=dummy"))
			})
		})
	})
})
