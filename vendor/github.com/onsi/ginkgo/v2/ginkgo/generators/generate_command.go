package generators

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/template"

	sprig "github.com/go-task/slim-sprig"
	"github.com/onsi/ginkgo/v2/ginkgo/command"
	"github.com/onsi/ginkgo/v2/ginkgo/internal"
	"github.com/onsi/ginkgo/v2/types"
)

func BuildGenerateCommand() command.Command {
	conf := GeneratorsConfig{}
	flags, err := types.NewGinkgoFlagSet(
		types.GinkgoFlags{
			{Name: "agouti", KeyPath: "Agouti",
				Usage: "If set, generate will create a test file for writing Agouti tests"},
			{Name: "nodot", KeyPath: "NoDot",
				Usage: "If set, generate will create a test file that does not dot-import ginkgo and gomega"},
			{Name: "internal", KeyPath: "Internal",
				Usage: "If set, generate will create a test file that uses the regular package name (i.e. `package X`, not `package X_test`)"},
			{Name: "template", KeyPath: "CustomTemplate",
				UsageArgument: "template-file",
				Usage:         "If specified, generate will use the contents of the file passed as the test file template"},
			{Name: "template-data", KeyPath: "CustomTemplateData",
				UsageArgument: "template-data-file",
				Usage:         "If specified, generate will use the contents of the file passed as data to be rendered in the test file template"},
			{Name: "tags", KeyPath: "Tags",
				UsageArgument: "build-tags",
				Usage:         "If specified, generate will create a test file that uses the given build tags (i.e. `--tags e2e,!unit` will add `//go:build e2e,!unit`)"},
		},
		&conf,
		types.GinkgoFlagSections{},
	)

	if err != nil {
		panic(err)
	}

	return command.Command{
		Name:     "generate",
		Usage:    "ginkgo generate <filename(s)>",
		ShortDoc: "Generate a test file named <filename>_test.go",
		Documentation: `If the optional <filename> argument is omitted, a file named after the package in the current directory will be created.

You can pass multiple <filename(s)> to generate multiple files simultaneously.  The resulting files are named <filename>_test.go.

You can also pass a <filename> of the form "file.go" and generate will emit "file_test.go".`,
		DocLink: "generators",
		Flags:   flags,
		Command: func(args []string, _ []string) {
			generateTestFiles(conf, args)
		},
	}
}

type specData struct {
	BuildTags         string
	Package           string
	Subject           string
	PackageImportPath string
	ImportPackage     bool

	GinkgoImport  string
	GomegaImport  string
	GinkgoPackage string
	GomegaPackage string
	CustomData    map[string]any
}

func generateTestFiles(conf GeneratorsConfig, args []string) {
	subjects := args
	if len(subjects) == 0 {
		subjects = []string{""}
	}
	for _, subject := range subjects {
		generateTestFileForSubject(subject, conf)
	}
}

func generateTestFileForSubject(subject string, conf GeneratorsConfig) {
	packageName, specFilePrefix, formattedName := getPackageAndFormattedName()
	if subject != "" {
		specFilePrefix = formatSubject(subject)
		formattedName = prettifyName(specFilePrefix)
	}

	if conf.Internal {
		specFilePrefix = specFilePrefix + "_internal"
	}

	data := specData{
		BuildTags:         getBuildTags(conf.Tags),
		Package:           determinePackageName(packageName, conf.Internal),
		Subject:           formattedName,
		PackageImportPath: getPackageImportPath(),
		ImportPackage:     !conf.Internal,

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

	targetFile := fmt.Sprintf("%s_test.go", specFilePrefix)
	if internal.FileExists(targetFile) {
		command.AbortWith("{{bold}}%s{{/}} already exists", targetFile)
	} else {
		fmt.Printf("Generating ginkgo test for %s in:\n  %s\n", data.Subject, targetFile)
	}

	f, err := os.Create(targetFile)
	command.AbortIfError("Failed to create test file:", err)
	defer f.Close()

	var templateText string
	if conf.CustomTemplate != "" {
		tpl, err := os.ReadFile(conf.CustomTemplate)
		command.AbortIfError("Failed to read custom template file:", err)
		templateText = string(tpl)
		if conf.CustomTemplateData != "" {
			var tplCustomDataMap map[string]any
			tplCustomData, err := os.ReadFile(conf.CustomTemplateData)
			command.AbortIfError("Failed to read custom template data file:", err)
			if !json.Valid([]byte(tplCustomData)) {
				command.AbortWith("Invalid JSON object in custom data file.")
			}
			//create map from the custom template data
			json.Unmarshal(tplCustomData, &tplCustomDataMap)
			data.CustomData = tplCustomDataMap
		}
	} else if conf.Agouti {
		templateText = agoutiSpecText
	} else {
		templateText = specText
	}

	//Setting the option to explicitly fail if template is rendered trying to access missing key
	specTemplate, err := template.New("spec").Funcs(sprig.TxtFuncMap()).Option("missingkey=error").Parse(templateText)
	command.AbortIfError("Failed to read parse test template:", err)

	//Being explicit about failing sooner during template rendering
	//when accessing custom data rather than during the go fmt command
	err = specTemplate.Execute(f, data)
	command.AbortIfError("Failed to render bootstrap template:", err)
	internal.GoFmt(targetFile)
}

func formatSubject(name string) string {
	name = strings.ReplaceAll(name, "-", "_")
	name = strings.ReplaceAll(name, " ", "_")
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

	sep := string(filepath.Separator)

	// Try go.mod file first
	modRoot := findModuleRoot(workingDir)
	if modRoot != "" {
		modName := moduleName(modRoot)
		if modName != "" {
			cd := strings.ReplaceAll(workingDir, modRoot, "")
			cd = strings.ReplaceAll(cd, sep, "/")
			return modName + cd
		}
	}

	// Fallback to GOPATH structure
	paths := strings.Split(workingDir, sep+"src"+sep)
	if len(paths) == 1 {
		fmt.Printf("\nCouldn't identify package import path.\n\n\tginkgo generate\n\nMust be run within a package directory under $GOPATH/src/...\nYou're going to have to change UNKNOWN_PACKAGE_PATH in the generated file...\n\n")
		return "UNKNOWN_PACKAGE_PATH"
	}
	return filepath.ToSlash(paths[len(paths)-1])
}
