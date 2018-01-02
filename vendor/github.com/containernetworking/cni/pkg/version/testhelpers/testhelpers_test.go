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
package testhelpers_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/containernetworking/cni/pkg/version/testhelpers"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("BuildAt", func() {
	var (
		gitRef         string
		outputFilePath string
		outputDir      string
		programSource  []byte
	)
	BeforeEach(func() {
		programSource = []byte(`package main

import "github.com/containernetworking/cni/pkg/skel"

func c(_ *skel.CmdArgs) error { return nil }

func main() { skel.PluginMain(c, c) }
`)
		gitRef = "f4364185253"

		var err error
		outputDir, err = ioutil.TempDir("", "bin")
		Expect(err).NotTo(HaveOccurred())
		outputFilePath = filepath.Join(outputDir, "some-binary")
	})

	AfterEach(func() {
		Expect(os.RemoveAll(outputDir)).To(Succeed())
	})

	It("builds the provided source code using the CNI library at the given git ref", func() {
		Expect(outputFilePath).NotTo(BeAnExistingFile())

		err := testhelpers.BuildAt(programSource, gitRef, outputFilePath)
		Expect(err).NotTo(HaveOccurred())

		Expect(outputFilePath).To(BeAnExistingFile())

		cmd := exec.Command(outputFilePath)
		cmd.Env = []string{"CNI_COMMAND=VERSION"}
		output, err := cmd.CombinedOutput()
		Expect(err).To(BeAssignableToTypeOf(&exec.ExitError{}))
		Expect(output).To(ContainSubstring("unknown CNI_COMMAND: VERSION"))
	})
})

var _ = Describe("LocateCurrentGitRepo", func() {
	It("returns the path to the root of the CNI git repo", func() {
		path, err := testhelpers.LocateCurrentGitRepo()
		Expect(err).NotTo(HaveOccurred())

		AssertItIsTheCNIRepoRoot(path)
	})

	Context("when run from a different directory", func() {
		BeforeEach(func() {
			os.Chdir("..")
		})

		It("still finds the CNI repo root", func() {
			path, err := testhelpers.LocateCurrentGitRepo()
			Expect(err).NotTo(HaveOccurred())

			AssertItIsTheCNIRepoRoot(path)
		})
	})
})

func AssertItIsTheCNIRepoRoot(path string) {
	Expect(path).To(BeADirectory())
	files, err := ioutil.ReadDir(path)
	Expect(err).NotTo(HaveOccurred())

	names := []string{}
	for _, file := range files {
		names = append(names, file.Name())
	}

	Expect(names).To(ContainElement("SPEC.md"))
	Expect(names).To(ContainElement("libcni"))
	Expect(names).To(ContainElement("cnitool"))
}
