package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"

	"go/build"

	"github.com/onsi/ginkgo/ginkgo/nodot"
)

func BuildBootstrapCommand() *Command {
	var agouti, noDot bool
	flagSet := flag.NewFlagSet("bootstrap", flag.ExitOnError)
	flagSet.BoolVar(&agouti, "agouti", false, "If set, bootstrap will generate a bootstrap file for writing Agouti tests")
	flagSet.BoolVar(&noDot, "nodot", false, "If set, bootstrap will generate a bootstrap file that does not . import ginkgo and gomega")

	return &Command{
		Name:         "bootstrap",
		FlagSet:      flagSet,
		UsageCommand: "ginkgo bootstrap <FLAGS>",
		Usage: []string{
			"Bootstrap a test suite for the current package",
			"Accepts the following flags:",
		},
		Command: func(args []string, additionalArgs []string) {
			generateBootstrap(agouti, noDot)
		},
	}
}

var bootstrapText = `package {{.Package}}_test

import (
	{{.GinkgoImport}}
	{{.GomegaImport}}

	"testing"
)

func Test{{.FormattedName}}(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "{{.FormattedName}} Suite")
}
`

var agoutiBootstrapText = `package {{.Package}}_test

import (
	{{.GinkgoImport}}
	{{.GomegaImport}}
	"github.com/sclevine/agouti"

	"testing"
)

func Test{{.FormattedName}}(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "{{.FormattedName}} Suite")
}

var agoutiDriver *agouti.WebDriver

var _ = BeforeSuite(func() {
	// Choose a WebDriver:

	agoutiDriver = agouti.PhantomJS()
	// agoutiDriver = agouti.Selenium()
	// agoutiDriver = agouti.ChromeDriver()

	Expect(agoutiDriver.Start()).To(Succeed())
})

var _ = AfterSuite(func() {
	Expect(agoutiDriver.Stop()).To(Succeed())
})
`

type bootstrapData struct {
	Package       string
	FormattedName string
	GinkgoImport  string
	GomegaImport  string
}

func getPackageAndFormattedName() (string, string, string) {
	path, err := os.Getwd()
	if err != nil {
		complainAndQuit("Could not get current working directory: \n" + err.Error())
	}

	dirName := strings.Replace(filepath.Base(path), "-", "_", -1)
	dirName = strings.Replace(dirName, " ", "_", -1)

	pkg, err := build.ImportDir(path, 0)
	packageName := pkg.Name
	if err != nil {
		packageName = dirName
	}

	formattedName := prettifyPackageName(filepath.Base(path))
	return packageName, dirName, formattedName
}

func prettifyPackageName(name string) string {
	name = strings.Replace(name, "-", " ", -1)
	name = strings.Replace(name, "_", " ", -1)
	name = strings.Title(name)
	name = strings.Replace(name, " ", "", -1)
	return name
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true
	}
	return false
}

func generateBootstrap(agouti bool, noDot bool) {
	packageName, bootstrapFilePrefix, formattedName := getPackageAndFormattedName()
	data := bootstrapData{
		Package:       packageName,
		FormattedName: formattedName,
		GinkgoImport:  `. "github.com/onsi/ginkgo"`,
		GomegaImport:  `. "github.com/onsi/gomega"`,
	}

	if noDot {
		data.GinkgoImport = `"github.com/onsi/ginkgo"`
		data.GomegaImport = `"github.com/onsi/gomega"`
	}

	targetFile := fmt.Sprintf("%s_suite_test.go", bootstrapFilePrefix)
	if fileExists(targetFile) {
		fmt.Printf("%s already exists.\n\n", targetFile)
		os.Exit(1)
	} else {
		fmt.Printf("Generating ginkgo test suite bootstrap for %s in:\n\t%s\n", packageName, targetFile)
	}

	f, err := os.Create(targetFile)
	if err != nil {
		complainAndQuit("Could not create file: " + err.Error())
		panic(err.Error())
	}
	defer f.Close()

	var templateText string
	if agouti {
		templateText = agoutiBootstrapText
	} else {
		templateText = bootstrapText
	}

	bootstrapTemplate, err := template.New("bootstrap").Parse(templateText)
	if err != nil {
		panic(err.Error())
	}

	buf := &bytes.Buffer{}
	bootstrapTemplate.Execute(buf, data)

	if noDot {
		contents, err := nodot.ApplyNoDot(buf.Bytes())
		if err != nil {
			complainAndQuit("Failed to import nodot declarations: " + err.Error())
		}
		fmt.Println("To update the nodot declarations in the future, switch to this directory and run:\n\tginkgo nodot")
		buf = bytes.NewBuffer(contents)
	}

	buf.WriteTo(f)

	goFmt(targetFile)
}
