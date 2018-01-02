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
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"

	"github.com/containernetworking/cni/pkg/skel"
	"github.com/containernetworking/cni/pkg/version"
	noop_debug "github.com/containernetworking/cni/plugins/test/noop/debug"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

var _ = Describe("No-op plugin", func() {
	var (
		cmd             *exec.Cmd
		debugFileName   string
		debug           *noop_debug.Debug
		expectedCmdArgs skel.CmdArgs
	)

	const reportResult = `{ "ips": [{ "version": "4", "address": "10.1.2.3/24" }], "dns": {} }`

	BeforeEach(func() {
		debug = &noop_debug.Debug{
			ReportResult:         reportResult,
			ReportVersionSupport: []string{"0.1.0", "0.2.0", "0.3.0", "0.3.1"},
		}

		debugFile, err := ioutil.TempFile("", "cni_debug")
		Expect(err).NotTo(HaveOccurred())
		Expect(debugFile.Close()).To(Succeed())
		debugFileName = debugFile.Name()

		Expect(debug.WriteDebug(debugFileName)).To(Succeed())

		cmd = exec.Command(pathToPlugin)

		args := fmt.Sprintf("DEBUG=%s;FOO=BAR", debugFileName)
		cmd.Env = []string{
			"CNI_COMMAND=ADD",
			"CNI_CONTAINERID=some-container-id",
			"CNI_NETNS=/some/netns/path",
			"CNI_IFNAME=some-eth0",
			"CNI_PATH=/some/bin/path",
			// Keep this last
			"CNI_ARGS=" + args,
		}
		cmd.Stdin = strings.NewReader(`{"some":"stdin-json", "cniVersion": "0.3.1"}`)
		expectedCmdArgs = skel.CmdArgs{
			ContainerID: "some-container-id",
			Netns:       "/some/netns/path",
			IfName:      "some-eth0",
			Args:        args,
			Path:        "/some/bin/path",
			StdinData:   []byte(`{"some":"stdin-json", "cniVersion": "0.3.1"}`),
		}
	})

	AfterEach(func() {
		os.Remove(debugFileName)
	})

	It("responds to ADD using the ReportResult debug field", func() {
		session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
		Expect(err).NotTo(HaveOccurred())
		Eventually(session).Should(gexec.Exit(0))
		Expect(session.Out.Contents()).To(MatchJSON(reportResult))
	})

	It("panics when no debug file is given", func() {
		// Remove the DEBUG option from CNI_ARGS and regular args
		cmd.Env[len(cmd.Env)-1] = "CNI_ARGS=FOO=BAR"
		expectedCmdArgs.Args = "FOO=BAR"

		session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
		Expect(err).NotTo(HaveOccurred())
		Eventually(session).Should(gexec.Exit(2))
	})

	It("pass previous result through when ReportResult is PASSTHROUGH", func() {
		debug = &noop_debug.Debug{ReportResult: "PASSTHROUGH"}
		Expect(debug.WriteDebug(debugFileName)).To(Succeed())

		cmd.Stdin = strings.NewReader(`{
	"some":"stdin-json",
	"cniVersion": "0.3.1",
	"prevResult": {
		"ips": [{"version": "4", "address": "10.1.2.15/24"}]
	}
}`)
		session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
		Expect(err).NotTo(HaveOccurred())
		Eventually(session).Should(gexec.Exit(0))
		Expect(session.Out.Contents()).To(MatchJSON(`{"ips": [{"version": "4", "address": "10.1.2.15/24"}], "dns": {}}`))
	})

	It("injects DNS into previous result when ReportResult is INJECT-DNS", func() {
		debug = &noop_debug.Debug{ReportResult: "INJECT-DNS"}
		Expect(debug.WriteDebug(debugFileName)).To(Succeed())

		cmd.Stdin = strings.NewReader(`{
	"some":"stdin-json",
	"cniVersion": "0.3.1",
	"prevResult": {
		"ips": [{"version": "4", "address": "10.1.2.3/24"}],
		"dns": {}
	}
}`)

		session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
		Expect(err).NotTo(HaveOccurred())
		Eventually(session).Should(gexec.Exit(0))
		Expect(session.Out.Contents()).To(MatchJSON(`{
  "cniVersion": "0.3.1",
	"ips": [{"version": "4", "address": "10.1.2.3/24"}],
	"dns": {"nameservers": ["1.2.3.4"]}
}`))
	})

	It("allows passing debug file in config JSON", func() {
		// Remove the DEBUG option from CNI_ARGS and regular args
		newArgs := "FOO=BAR"
		cmd.Env[len(cmd.Env)-1] = "CNI_ARGS=" + newArgs
		newStdin := fmt.Sprintf(`{"some":"stdin-json", "cniVersion": "0.3.1", "debugFile": "%s"}`, debugFileName)
		cmd.Stdin = strings.NewReader(newStdin)
		expectedCmdArgs.Args = newArgs
		expectedCmdArgs.StdinData = []byte(newStdin)

		session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
		Expect(err).NotTo(HaveOccurred())
		Eventually(session).Should(gexec.Exit(0))
		Expect(session.Out.Contents()).To(MatchJSON(reportResult))

		debug, err := noop_debug.ReadDebug(debugFileName)
		Expect(err).NotTo(HaveOccurred())
		Expect(debug.Command).To(Equal("ADD"))
		Expect(debug.CmdArgs).To(Equal(expectedCmdArgs))
	})

	It("records all the args provided by skel.PluginMain", func() {
		session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
		Expect(err).NotTo(HaveOccurred())
		Eventually(session).Should(gexec.Exit(0))

		debug, err := noop_debug.ReadDebug(debugFileName)
		Expect(err).NotTo(HaveOccurred())
		Expect(debug.Command).To(Equal("ADD"))
		Expect(debug.CmdArgs).To(Equal(expectedCmdArgs))
	})

	Context("when the ReportResult debug field is empty", func() {
		BeforeEach(func() {
			debug.ReportResult = ""
			Expect(debug.WriteDebug(debugFileName)).To(Succeed())
		})

		It("substitutes a helpful message for the test author", func() {
			expectedResultString := fmt.Sprintf(` { "result": %q }`, noop_debug.EmptyReportResultMessage)

			session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
			Expect(err).NotTo(HaveOccurred())
			Eventually(session).Should(gexec.Exit(0))
			Expect(session.Out.Contents()).To(MatchJSON(expectedResultString))

			debug, err := noop_debug.ReadDebug(debugFileName)
			Expect(err).NotTo(HaveOccurred())
			Expect(debug.ReportResult).To(MatchJSON(expectedResultString))
		})
	})

	Context("when the ReportError debug field is set", func() {
		BeforeEach(func() {
			debug.ReportError = "banana"
			Expect(debug.WriteDebug(debugFileName)).To(Succeed())
		})

		It("returns an error to skel.PluginMain, causing the process to exit code 1", func() {
			session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
			Expect(err).NotTo(HaveOccurred())
			Eventually(session).Should(gexec.Exit(1))
			Expect(session.Out.Contents()).To(MatchJSON(`{ "code": 100, "msg": "banana" }`))
		})
	})

	Context("when the CNI_COMMAND is DEL", func() {
		BeforeEach(func() {
			cmd.Env[0] = "CNI_COMMAND=DEL"
			debug.ReportResult = `{ "some": "delete-data" }`
			Expect(debug.WriteDebug(debugFileName)).To(Succeed())
		})

		It("still does all the debug behavior", func() {
			session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
			Expect(err).NotTo(HaveOccurred())
			Eventually(session).Should(gexec.Exit(0))
			Expect(session.Out.Contents()).To(MatchJSON(`{
				"some": "delete-data"
      }`))
			debug, err := noop_debug.ReadDebug(debugFileName)
			Expect(err).NotTo(HaveOccurred())
			Expect(debug.Command).To(Equal("DEL"))
			Expect(debug.CmdArgs).To(Equal(expectedCmdArgs))
		})
	})

	Context("when the CNI_COMMAND is VERSION", func() {
		BeforeEach(func() {
			cmd.Env[0] = "CNI_COMMAND=VERSION"
			debug.ReportVersionSupport = []string{"0.123.0", "0.2.0"}

			Expect(debug.WriteDebug(debugFileName)).To(Succeed())
		})

		It("claims to support the specified versions", func() {
			session, err := gexec.Start(cmd, GinkgoWriter, GinkgoWriter)
			Expect(err).NotTo(HaveOccurred())
			Eventually(session).Should(gexec.Exit(0))
			decoder := &version.PluginDecoder{}
			pluginInfo, err := decoder.Decode(session.Out.Contents())
			Expect(err).NotTo(HaveOccurred())
			Expect(pluginInfo.SupportedVersions()).To(ConsistOf(
				"0.123.0", "0.2.0"))
		})
	})
})
