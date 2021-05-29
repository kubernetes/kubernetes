package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/template"
)

func BuildGenerateCommand() *Command {
	var agouti, noDot, internal bool
	flagSet := flag.NewFlagSet("generate", flag.ExitOnError)
	flagSet.BoolVar(&agouti, "agouti", false, "If set, generate will generate a test file for writing Agouti tests")
	flagSet.BoolVar(&noDot, "nodot", false, "If set, generate will generate a test file that does not . import ginkgo and gomega")
	flagSet.BoolVar(&internal, "internal", false, "If set, generate will generate a test file that uses the regular package name")

	return &Command{
		Name:         "generate",
		FlagSet:      flagSet,
		UsageCommand: "ginkgo generate <filename(s)>",
		Usage: []string{
			"Generate a test file named filename_test.go",
			"If the optional <filenames> argument is omitted, a file named after the package in the current directory will be created.",
			"Accepts the following flags:",
		},
		Command: func(args []string, additionalArgs []string) {
			generateSpec(args, agouti, noDot, internal)
		},
	}
}

var specText = `package {{.Package}}

import (
	{{if .IncludeImports}}. "github.com/onsi/ginkgo"{{end}}
	{{if .IncludeImports}}. "github.com/onsi/gomega"{{end}}

	{{if .ImportPackage}}"{{.PackageImportPath}}"{{end}}
)

var _ = Describe("{{.Subject}}", func() {

})
`

var agoutiSpecText = `package {{.Package}}

import (
	{{if .IncludeImports}}. "github.com/onsi/ginkgo"{{end}}
	{{if .IncludeImports}}. "github.com/onsi/gomega"{{end}}
	"github.com/sclevine/agouti"
	. "github.com/sclevine/agouti/matchers"

	{{if .ImportPackage}}"{{.PackageImportPath}}"{{end}}
)

var _ = Describe("{{.Subject}}", func() {
	var page *agouti.Page

	BeforeEach(func() {
		var err error
		page, err = agoutiDriver.NewPage()
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		Expect(page.Destroy()).To(Succeed())
	})
})
`

type specData struct {
	Package           string
	Subject           string
	PackageImportPath string
	IncludeImports    bool
	ImportPackage     bool
}

func generateSpec(args []string, agouti, noDot, internal bool) {
	if len(args) == 0 {
		err := generateSpecForSubject("", agouti, noDot, internal)
		if err != nil {
			fmt.Println(err.Error())
			fmt.Println("")
			os.Exit(1)
		}
		fmt.Println("")
		return
	}

	var failed bool
	for _, arg := range args {
		err := generateSpecForSubject(arg, agouti, noDot, internal)
		if err != nil {
			failed = true
			fmt.Println(err.Error())
		}
	}
	fmt.Println("")
	if failed {
		os.Exit(1)
	}
}

func generateSpecForSubject(subject string, agouti, noDot, internal bool) error {
	packageName, specFilePrefix, formattedName := getPackageAndFormattedName()
	if subject != "" {
		specFilePrefix = formatSubject(subject)
		formattedName = prettifyPackageName(specFilePrefix)
	}

	data := specData{
		Package:           determinePackageName(packageName, internal),
		Subject:           formattedName,
		PackageImportPath: getPackageImportPath(),
		IncludeImports:    !noDot,
		ImportPackage:     !internal,
	}

	targetFile := fmt.Sprintf("%s_test.go", specFilePrefix)
	if fileExists(targetFile) {
		return fmt.Errorf("%s already exists.", targetFile)
	} else {
		fmt.Printf("Generating ginkgo test for %s in:\n  %s\n", data.Subject, targetFile)
	}

	f, err := os.Create(targetFile)
	if err != nil {
		return err
	}
	defer f.Close()

	var templateText string
	if agouti {
		templateText = agoutiSpecText
	} else {
		templateText = specText
	}

	specTemplate, err := template.New("spec").Parse(templateText)
	if err != nil {
		return err
	}

	specTemplate.Execute(f, data)
	goFmt(targetFile)
	return nil
}

func formatSubject(name string) string {
	name = strings.Replace(name, "-", "_", -1)
	name = strings.Replace(name, " ", "_", -1)
	name = strings.Split(name, ".go")[0]
	name = strings.Split(name, "_test")[0]
	return name
}

// moduleName returns module name from go.mod from given module root directory
func moduleName(modRoot string) string {
	modFile, err := os.Open(filepath.Join(modRoot, "go.mod"))
	if err != nil {
		return ""
	}

	mod := make([]byte, 128)
	_, err = modFile.Read(mod)
	if err != nil {
		return ""
	}

	slashSlash := []byte("//")
	moduleStr := []byte("module")

	for len(mod) > 0 {
		line := mod
		mod = nil
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, mod = line[:i], line[i+1:]
		}
		if i := bytes.Index(line, slashSlash); i >= 0 {
			line = line[:i]
		}
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, moduleStr) {
			continue
		}
		line = line[len(moduleStr):]
		n := len(line)
		line = bytes.TrimSpace(line)
		if len(line) == n || len(line) == 0 {
			continue
		}

		if line[0] == '"' || line[0] == '`' {
			p, err := strconv.Unquote(string(line))
			if err != nil {
				return "" // malformed quoted string or multiline module path
			}
			return p
		}

		return string(line)
	}

	return "" // missing module path
}

func findModuleRoot(dir string) (root string) {
	dir = filepath.Clean(dir)

	// Look for enclosing go.mod.
	for {
		if fi, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !fi.IsDir() {
			return dir
		}
		d := filepath.Dir(dir)
		if d == dir {
			break
		}
		dir = d
	}
	return ""
}

func getPackageImportPath() string {
	workingDir, err := os.Getwd()
	if err != nil {
		panic(err.Error())
	}

	// Try go.mod file first
	modRoot := findModuleRoot(workingDir)
	if modRoot != "" {
		modName := moduleName(modRoot)
		if modName != "" {
			cd := strings.Replace(workingDir, modRoot, "", -1)
			return modName + cd
		}
	}

	// Fallback to GOPATH structure
	sep := string(filepath.Separator)
	paths := strings.Split(workingDir, sep+"src"+sep)
	if len(paths) == 1 {
		fmt.Printf("\nCouldn't identify package import path.\n\n\tginkgo generate\n\nMust be run within a package directory under $GOPATH/src/...\nYou're going to have to change UNKNOWN_PACKAGE_PATH in the generated file...\n\n")
		return "UNKNOWN_PACKAGE_PATH"
	}
	return filepath.ToSlash(paths[len(paths)-1])
}
