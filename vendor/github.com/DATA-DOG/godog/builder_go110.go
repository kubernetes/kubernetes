// +build go1.10

package godog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/build"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"text/template"
	"time"
	"unicode"
)

var (
	tooldir         = findToolDir()
	compiler        = filepath.Join(tooldir, "compile")
	linker          = filepath.Join(tooldir, "link")
	gopaths         = filepath.SplitList(build.Default.GOPATH)
	godogImportPath = "github.com/DATA-DOG/godog"

	// godep
	runnerTemplate = template.Must(template.New("testmain").Parse(`package main

import (
	"github.com/DATA-DOG/godog"
	{{if .Contexts}}_test "{{.ImportPath}}"{{end}}
	{{if .XContexts}}_xtest "{{.ImportPath}}_test"{{end}}
	{{if .XContexts}}"testing/internal/testdeps"{{end}}
	"os"
)

{{if .XContexts}}
func init() {
	testdeps.ImportPath = "{{.ImportPath}}"
}
{{end}}

func main() {
	status := godog.Run("{{ .Name }}", func (suite *godog.Suite) {
		os.Setenv("GODOG_TESTED_PACKAGE", "{{.ImportPath}}")
		{{range .Contexts}}
			_test.{{ . }}(suite)
		{{end}}
		{{range .XContexts}}
			_xtest.{{ . }}(suite)
		{{end}}
	})
	os.Exit(status)
}`))

	// temp file for import
	tempFileTemplate = template.Must(template.New("temp").Parse(`package {{.Name}}

import "github.com/DATA-DOG/godog"

var _ = godog.Version
`))
)

// Build creates a test package like go test command at given target path.
// If there are no go files in tested directory, then
// it simply builds a godog executable to scan features.
//
// If there are go test files, it first builds a test
// package with standard go test command.
//
// Finally it generates godog suite executable which
// registers exported godog contexts from the test files
// of tested package.
//
// Returns the path to generated executable
func Build(bin string) error {
	abs, err := filepath.Abs(".")
	if err != nil {
		return err
	}

	// we allow package to be nil, if godog is run only when
	// there is a feature file in empty directory
	pkg := importPackage(abs)
	src, err := buildTestMain(pkg)
	if err != nil {
		return err
	}

	// may need to produce temp file for godog dependency
	srcTemp, err := buildTempFile(pkg)
	if err != nil {
		return err
	}

	if srcTemp != nil {
		// @TODO: in case of modules we cannot build it our selves, we need to have this hacky option
		pathTemp := filepath.Join(abs, "godog_dependency_file_test.go")
		err = ioutil.WriteFile(pathTemp, srcTemp, 0644)
		if err != nil {
			return err
		}
		defer os.Remove(pathTemp)
	}

	workdir := ""
	testdir := workdir

	// build and compile the tested package.
	// generated test executable will be removed
	// since we do not need it for godog suite.
	// we also print back the temp WORK directory
	// go has built. We will reuse it for our suite workdir.
	temp := fmt.Sprintf(filepath.Join("%s", "temp-%d.test"), os.TempDir(), time.Now().UnixNano())
	testOutput, err := exec.Command("go", "test", "-c", "-work", "-o", temp).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to compile tested package: %s, reason: %v, output: %s", abs, err, string(testOutput))
	}
	defer os.Remove(temp)

	// extract go-build temporary directory as our workdir
	linesOut := strings.Split(strings.TrimSpace(string(testOutput)), "\n")
	// it may have some compilation warnings, in the output, but these are not
	// considered to be errors, since command exit status is 0
	for _, ln := range linesOut {
		if !strings.HasPrefix(ln, "WORK=") {
			continue
		}
		workdir = strings.Replace(ln, "WORK=", "", 1)
		break
	}

	// may not locate it in output
	if workdir == testdir {
		return fmt.Errorf("expected WORK dir path to be present in output: %s", string(testOutput))
	}

	// check whether workdir exists
	stats, err := os.Stat(workdir)
	if os.IsNotExist(err) {
		return fmt.Errorf("expected WORK dir: %s to be available", workdir)
	}

	if !stats.IsDir() {
		return fmt.Errorf("expected WORK dir: %s to be directory", workdir)
	}
	testdir = filepath.Join(workdir, "b001")
	defer os.RemoveAll(workdir)

	// replace _testmain.go file with our own
	testmain := filepath.Join(testdir, "_testmain.go")
	err = ioutil.WriteFile(testmain, src, 0644)
	if err != nil {
		return err
	}

	// godog package may be vendored and may need importmap
	vendored := maybeVendoredGodog()

	// compile godog testmain package archive
	// we do not depend on CGO so a lot of checks are not necessary
	linkerCfg := filepath.Join(testdir, "importcfg.link")
	compilerCfg := linkerCfg
	if vendored != nil {
		data, err := ioutil.ReadFile(linkerCfg)
		if err != nil {
			return err
		}

		data = append(data, []byte(fmt.Sprintf("importmap %s=%s\n", godogImportPath, vendored.ImportPath))...)
		compilerCfg = filepath.Join(testdir, "importcfg")

		err = ioutil.WriteFile(compilerCfg, data, 0644)
		if err != nil {
			return err
		}
	}

	testMainPkgOut := filepath.Join(testdir, "main.a")
	args := []string{
		"-o", testMainPkgOut,
		"-importcfg", compilerCfg,
		"-p", "main",
		"-complete",
	}

	args = append(args, "-pack", testmain)
	cmd := exec.Command(compiler, args...)
	cmd.Env = os.Environ()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to compile testmain package: %v - output: %s", err, string(out))
	}

	// link test suite executable
	args = []string{
		"-o", bin,
		"-importcfg", linkerCfg,
		"-buildmode=exe",
	}
	args = append(args, testMainPkgOut)
	cmd = exec.Command(linker, args...)
	cmd.Env = os.Environ()

	out, err = cmd.CombinedOutput()
	if err != nil {
		msg := `failed to link test executable:
	reason: %s
	command: %s`
		return fmt.Errorf(msg, string(out), linker+" '"+strings.Join(args, "' '")+"'")
	}

	return nil
}

func maybeVendoredGodog() *build.Package {
	dir, err := filepath.Abs(".")
	if err != nil {
		return nil
	}

	for _, gopath := range gopaths {
		gopath = filepath.Join(gopath, "src")
		for strings.HasPrefix(dir, gopath) && dir != gopath {
			pkg, err := build.ImportDir(filepath.Join(dir, "vendor", godogImportPath), 0)
			if err != nil {
				dir = filepath.Dir(dir)
				continue
			}
			return pkg
		}
	}
	return nil
}

func importPackage(dir string) *build.Package {
	pkg, _ := build.ImportDir(dir, 0)

	// normalize import path for local import packages
	// taken from go source code
	// see: https://github.com/golang/go/blob/go1.7rc5/src/cmd/go/pkg.go#L279
	if pkg != nil && pkg.ImportPath == "." {
		pkg.ImportPath = path.Join("_", strings.Map(makeImportValid, filepath.ToSlash(dir)))
	}

	return pkg
}

// from go src
func makeImportValid(r rune) rune {
	// Should match Go spec, compilers, and ../../go/parser/parser.go:/isValidImport.
	const illegalChars = `!"#$%&'()*,:;<=>?[\]^{|}` + "`\uFFFD"
	if !unicode.IsGraphic(r) || unicode.IsSpace(r) || strings.ContainsRune(illegalChars, r) {
		return '_'
	}
	return r
}

// build temporary file content if godog
// package is not present in currently tested package
func buildTempFile(pkg *build.Package) ([]byte, error) {
	shouldBuild := true
	var name string
	if pkg != nil {
		name = pkg.Name
		all := pkg.Imports
		all = append(all, pkg.TestImports...)
		all = append(all, pkg.XTestImports...)
		for _, imp := range all {
			if imp == godogImportPath {
				shouldBuild = false
				break
			}
		}

		// maybe we are testing the godog package on it's own
		if name == "godog" {
			if parseImport(pkg.ImportPath, pkg.Root) == godogImportPath {
				shouldBuild = false
			}
		}
	}

	if name == "" {
		name = "main"
	}

	if !shouldBuild {
		return nil, nil
	}

	data := struct{ Name string }{name}
	var buf bytes.Buffer
	if err := tempFileTemplate.Execute(&buf, data); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// buildTestMain if given package is valid
// it scans test files for contexts
// and produces a testmain source code.
func buildTestMain(pkg *build.Package) ([]byte, error) {
	var (
		contexts         []string
		xcontexts        []string
		err              error
		name, importPath string
	)
	if nil != pkg {
		contexts, err = processPackageTestFiles(pkg.TestGoFiles)
		if err != nil {
			return nil, err
		}
		xcontexts, err = processPackageTestFiles(pkg.XTestGoFiles)
		if err != nil {
			return nil, err
		}
		importPath = parseImport(pkg.ImportPath, pkg.Root)
		name = pkg.Name
	} else {
		name = "main"
	}
	data := struct {
		Name       string
		Contexts   []string
		XContexts  []string
		ImportPath string
	}{
		Name:       name,
		Contexts:   contexts,
		XContexts:  xcontexts,
		ImportPath: importPath,
	}

	var buf bytes.Buffer
	if err = runnerTemplate.Execute(&buf, data); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// parseImport parses the import path to deal with go module.
func parseImport(rawPath, rootPath string) string {
	// with go > 1.11 and go module enabled out of the GOPATH,
	// the import path begins with an underscore and the GOPATH is unknown on build.
	if rootPath != "" {
		// go < 1.11 or it's a module inside the GOPATH
		return rawPath
	}
	// for module support, query the module import path
	cmd := exec.Command("go", "list", "-m", "-json")
	out, err := cmd.StdoutPipe()
	if err != nil {
		// Unable to read stdout
		return rawPath
	}
	if cmd.Start() != nil {
		// Does not using modules
		return rawPath
	}
	var mod struct {
		Dir  string `json:"Dir"`
		Path string `json:"Path"`
	}
	if json.NewDecoder(out).Decode(&mod) != nil {
		// Unexpected result
		return rawPath
	}
	if cmd.Wait() != nil {
		return rawPath
	}
	// Concatenates the module path with the current sub-folders if needed
	return mod.Path + filepath.ToSlash(strings.TrimPrefix(strings.TrimPrefix(rawPath, "_"), mod.Dir))
}

// processPackageTestFiles runs through ast of each test
// file pack and looks for godog suite contexts to register
// on run
func processPackageTestFiles(packs ...[]string) ([]string, error) {
	var ctxs []string
	fset := token.NewFileSet()
	for _, pack := range packs {
		for _, testFile := range pack {
			node, err := parser.ParseFile(fset, testFile, nil, 0)
			if err != nil {
				return ctxs, err
			}

			ctxs = append(ctxs, astContexts(node)...)
		}
	}
	var failed []string
	for _, ctx := range ctxs {
		runes := []rune(ctx)
		if unicode.IsLower(runes[0]) {
			expected := append([]rune{unicode.ToUpper(runes[0])}, runes[1:]...)
			failed = append(failed, fmt.Sprintf("%s - should be: %s", ctx, string(expected)))
		}
	}
	if len(failed) > 0 {
		return ctxs, fmt.Errorf("godog contexts must be exported:\n\t%s", strings.Join(failed, "\n\t"))
	}
	return ctxs, nil
}

func findToolDir() string {
	if out, err := exec.Command("go", "env", "GOTOOLDIR").Output(); err != nil {
		return filepath.Clean(strings.TrimSpace(string(out)))
	}
	return filepath.Clean(build.ToolDir)
}
