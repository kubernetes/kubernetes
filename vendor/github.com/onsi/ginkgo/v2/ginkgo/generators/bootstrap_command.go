package generators

import (
	"bytes"
	"fmt"
	"os"
	"text/template"

	sprig "github.com/go-task/slim-sprig"
	"github.com/onsi/ginkgo/v2/ginkgo/command"
	"github.com/onsi/ginkgo/v2/ginkgo/internal"
	"github.com/onsi/ginkgo/v2/types"
)

func BuildBootstrapCommand() command.Command {
	conf := GeneratorsConfig{}
	flags, err := types.NewGinkgoFlagSet(
		types.GinkgoFlags{
			{Name: "agouti", KeyPath: "Agouti",
				Usage: "If set, bootstrap will generate a bootstrap file for writing Agouti tests"},
			{Name: "nodot", KeyPath: "NoDot",
				Usage: "If set, bootstrap will generate a bootstrap test file that does not dot-import ginkgo and gomega"},
			{Name: "internal", KeyPath: "Internal",
				Usage: "If set, bootstrap will generate a bootstrap test file that uses the regular package name (i.e. `package X`, not `package X_test`)"},
			{Name: "template", KeyPath: "CustomTemplate",
				UsageArgument: "template-file",
				Usage:         "If specified, generate will use the contents of the file passed as the bootstrap template"},
		},
		&conf,
		types.GinkgoFlagSections{},
	)

	if err != nil {
		panic(err)
	}

	return command.Command{
		Name:     "bootstrap",
		Usage:    "ginkgo bootstrap",
		ShortDoc: "Bootstrap a test suite for the current package",
		Documentation: `Tests written in Ginkgo and Gomega require a small amount of boilerplate to hook into Go's testing infrastructure.

{{bold}}ginkgo bootstrap{{/}} generates this boilerplate for you in a file named X_suite_test.go where X is the name of the package under test.`,
		DocLink: "generators",
		Flags:   flags,
		Command: func(_ []string, _ []string) {
			generateBootstrap(conf)
		},
	}
}

type bootstrapData struct {
	Package       string
	FormattedName string

	GinkgoImport  string
	GomegaImport  string
	GinkgoPackage string
	GomegaPackage string
}

func generateBootstrap(conf GeneratorsConfig) {
	packageName, bootstrapFilePrefix, formattedName := getPackageAndFormattedName()

	data := bootstrapData{
		Package:       determinePackageName(packageName, conf.Internal),
		FormattedName: formattedName,

		GinkgoImport:  `. "github.com/onsi/ginkgo/v2"`,
		GomegaImport:  `. "github.com/onsi/gomega"`,
		GinkgoPackage: "",
		GomegaPackage: "",
	}

	if conf.NoDot {
		data.GinkgoImport = `"github.com/onsi/ginkgo/v2"`
		data.GomegaImport = `"github.com/onsi/gomega"`
		data.GinkgoPackage = `ginkgo.`
		data.GomegaPackage = `gomega.`
	}

	targetFile := fmt.Sprintf("%s_suite_test.go", bootstrapFilePrefix)
	if internal.FileExists(targetFile) {
		command.AbortWith("{{bold}}%s{{/}} already exists", targetFile)
	} else {
		fmt.Printf("Generating ginkgo test suite bootstrap for %s in:\n\t%s\n", packageName, targetFile)
	}

	f, err := os.Create(targetFile)
	command.AbortIfError("Failed to create file:", err)
	defer f.Close()

	var templateText string
	if conf.CustomTemplate != "" {
		tpl, err := os.ReadFile(conf.CustomTemplate)
		command.AbortIfError("Failed to read custom bootstrap file:", err)
		templateText = string(tpl)
	} else if conf.Agouti {
		templateText = agoutiBootstrapText
	} else {
		templateText = bootstrapText
	}

	bootstrapTemplate, err := template.New("bootstrap").Funcs(sprig.TxtFuncMap()).Parse(templateText)
	command.AbortIfError("Failed to parse bootstrap template:", err)

	buf := &bytes.Buffer{}
	bootstrapTemplate.Execute(buf, data)

	buf.WriteTo(f)

	internal.GoFmt(targetFile)
}
