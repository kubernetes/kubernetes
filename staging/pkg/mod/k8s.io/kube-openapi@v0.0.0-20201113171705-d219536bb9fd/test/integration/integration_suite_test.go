/*
Copyright 2018 The Kubernetes Authors.

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

package integration

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

const (
	testdataDir = "./testdata"
	testPkgDir  = "k8s.io/kube-openapi/test/integration/testdata"
	inputDir    = testPkgDir + "/listtype" +
		"," + testPkgDir + "/maptype" +
		"," + testPkgDir + "/structtype" +
		"," + testPkgDir + "/dummytype" +
		"," + testPkgDir + "/uniontype" +
		"," + testPkgDir + "/custom" +
		"," + testPkgDir + "/defaults"
	outputBase               = "pkg"
	outputPackage            = "generated"
	outputBaseFileName       = "openapi_generated"
	generatedSwaggerFileName = "generated.v2.json"
	generatedReportFileName  = "generated.v2.report"
	goldenSwaggerFileName    = "golden.v2.json"
	goldenReportFileName     = "golden.v2.report"
	timeoutSeconds           = 5.0
)

func TestGenerators(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Integration Test Suite")
}

var _ = Describe("Open API Definitions Generation", func() {

	var (
		workingDirectory string
		tempDir          string
		terr             error
		openAPIGenPath   string
	)

	testdataFile := func(filename string) string { return filepath.Join(testdataDir, filename) }
	generatedFile := func(filename string) string { return filepath.Join(tempDir, filename) }

	BeforeSuite(func() {
		// Explicitly manage working directory
		abs, err := filepath.Abs("")
		Expect(err).ShouldNot(HaveOccurred())
		workingDirectory = abs

		// Create a temporary directory for generated swagger files.
		tempDir, terr = ioutil.TempDir("./", "openapi")
		Expect(terr).ShouldNot(HaveOccurred())

		// Build the OpenAPI code generator.
		By("building openapi-gen")
		binary_path, berr := gexec.Build("../../cmd/openapi-gen/openapi-gen.go")
		Expect(berr).ShouldNot(HaveOccurred())
		openAPIGenPath = binary_path

		// Run the OpenAPI code generator, creating OpenAPIDefinition code
		// to be compiled into builder.
		By("processing go idl with openapi-gen")
		gr := generatedFile(generatedReportFileName)
		command := exec.Command(openAPIGenPath,
			"-i", inputDir,
			"-o", outputBase,
			"-p", outputPackage,
			"-O", outputBaseFileName,
			"-r", gr,
		)
		command.Dir = workingDirectory
		session, err := gexec.Start(command, GinkgoWriter, GinkgoWriter)
		Expect(err).ShouldNot(HaveOccurred())
		Eventually(session, timeoutSeconds).Should(gexec.Exit(0))

		By("writing swagger")
		// Create the OpenAPI swagger builder.
		binary_path, berr = gexec.Build("./builder/main.go")
		Expect(berr).ShouldNot(HaveOccurred())

		// Execute the builder, generating an OpenAPI swagger file with definitions.
		gs := generatedFile(generatedSwaggerFileName)
		By("writing swagger to " + gs)
		command = exec.Command(binary_path, gs)
		command.Dir = workingDirectory
		session, err = gexec.Start(command, GinkgoWriter, GinkgoWriter)
		Expect(err).ShouldNot(HaveOccurred())
		Eventually(session, timeoutSeconds).Should(gexec.Exit(0))
	})

	AfterSuite(func() {
		os.RemoveAll(tempDir)
		gexec.CleanupBuildArtifacts()
	})

	Describe("openapi-gen --verify", func() {
		It("Verifies that the existing files are correct", func() {
			command := exec.Command(openAPIGenPath,
				"-i", inputDir,
				"-o", outputBase,
				"-p", outputPackage,
				"-O", outputBaseFileName,
				"-r", testdataFile(goldenReportFileName),
				"--verify-only",
			)
			command.Dir = workingDirectory
			session, err := gexec.Start(command, GinkgoWriter, GinkgoWriter)
			Expect(err).ShouldNot(HaveOccurred())
			Eventually(session, timeoutSeconds).Should(gexec.Exit(0))
		})
	})

	Describe("Validating OpenAPI Definition Generation", func() {
		It("Generated OpenAPI swagger definitions should match golden files", func() {
			// Diff the generated swagger against the golden swagger. Exit code should be zero.
			command := exec.Command(
				"diff",
				testdataFile(goldenSwaggerFileName),
				generatedFile(generatedSwaggerFileName),
			)
			command.Dir = workingDirectory
			session, err := gexec.Start(command, GinkgoWriter, GinkgoWriter)
			Expect(err).ShouldNot(HaveOccurred())
			Eventually(session, timeoutSeconds).Should(gexec.Exit(0))
		})
	})

	Describe("Validating API Rule Violation Reporting", func() {
		It("Generated API rule violations should match golden report files", func() {
			// Diff the generated report against the golden report. Exit code should be zero.
			command := exec.Command(
				"diff",
				testdataFile(goldenReportFileName),
				generatedFile(generatedReportFileName),
			)
			command.Dir = workingDirectory
			session, err := gexec.Start(command, GinkgoWriter, GinkgoWriter)
			Expect(err).ShouldNot(HaveOccurred())
			Eventually(session, timeoutSeconds).Should(gexec.Exit(0))
		})
	})
})
