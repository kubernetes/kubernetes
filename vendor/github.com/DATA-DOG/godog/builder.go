// +build !go1.10

package godog

import (
	"bytes"
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

var tooldir = findToolDir()
var compiler = filepath.Join(tooldir, "compile")
var linker = filepath.Join(tooldir, "link")
var gopaths = filepath.SplitList(build.Default.GOPATH)
var goarch = build.Default.GOARCH
var goos = build.Default.GOOS

var godogImportPath = "github.com/DATA-DOG/godog"
var runnerTemplate = template.Must(template.New("testmain").Parse(`package main

import (
	"github.com/DATA-DOG/godog"
	{{if .Contexts}}_test "{{.ImportPath}}"{{end}}
	"os"
)

func main() {
	status := godog.Run("{{ .Name }}", func (suite *godog.Suite) {
		os.Setenv("GODOG_TESTED_PACKAGE", "{{.ImportPath}}")
		{{range .Contexts}}
			_test.{{ . }}(suite)
		{{end}}
	})
	os.Exit(status)
}`))

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
	src, anyContexts, err := buildTestMain(pkg)
	if err != nil {
		return err
	}

	workdir := fmt.Sprintf(filepath.Join("%s", "godog-%d"), os.TempDir(), time.Now().UnixNano())
	testdir := workdir

	// if none of test files exist, or there are no contexts found
	// we will skip test package compilation, since it is useless
	if anyContexts {
		// first of all compile test package dependencies
		// that will save was many compilations for dependencies
		// go does it better
		out, err := exec.Command("go", "test", "-i").CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to compile package: %s, reason: %v, output: %s", pkg.Name, err, string(out))
		}

		// build and compile the tested package.
		// generated test executable will be removed
		// since we do not need it for godog suite.
		// we also print back the temp WORK directory
		// go has built. We will reuse it for our suite workdir.
		// go1.5 does not support os.DevNull as void output
		temp := fmt.Sprintf(filepath.Join("%s", "temp-%d.test"), os.TempDir(), time.Now().UnixNano())
		out, err = exec.Command("go", "test", "-c", "-work", "-o", temp).CombinedOutput()
		if err != nil {
			return fmt.Errorf("failed to compile tested package: %s, reason: %v, output: %s", pkg.Name, err, string(out))
		}
		defer os.Remove(temp)

		// extract go-build temporary directory as our workdir
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		// it may have some compilation warnings, in the output, but these are not
		// considered to be errors, since command exit status is 0
		for _, ln := range lines {
			if !strings.HasPrefix(ln, "WORK=") {
				continue
			}
			workdir = strings.Replace(ln, "WORK=", "", 1)
			break
		}

		// may not locate it in output
		if workdir == testdir {
			return fmt.Errorf("expected WORK dir path to be present in output: %s", string(out))
		}

		// check whether workdir exists
		stats, err := os.Stat(workdir)
		if os.IsNotExist(err) {
			return fmt.Errorf("expected WORK dir: %s to be available", workdir)
		}

		if !stats.IsDir() {
			return fmt.Errorf("expected WORK dir: %s to be directory", workdir)
		}
		testdir = filepath.Join(workdir, pkg.ImportPath, "_test")
	} else {
		// still need to create temporary workdir
		if err = os.MkdirAll(testdir, 0755); err != nil {
			return err
		}
	}
	defer os.RemoveAll(workdir)

	// replace _testmain.go file with our own
	testmain := filepath.Join(testdir, "_testmain.go")
	err = ioutil.WriteFile(testmain, src, 0644)
	if err != nil {
		return err
	}

	// godog library may not be imported in tested package
	// but we need it for our testmain package.
	// So we look it up in available source paths
	// including vendor directory, supported since 1.5.
	try := maybeVendorPaths(abs)
	for _, d := range build.Default.SrcDirs() {
		try = append(try, filepath.Join(d, godogImportPath))
	}
	godogPkg, err := locatePackage(try)
	if err != nil {
		return err
	}

	// make sure godog package archive is installed, gherkin
	// will be installed as dependency of godog
	cmd := exec.Command("go", "install", godogPkg.ImportPath)
	cmd.Env = os.Environ()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to install godog package: %s, reason: %v", string(out), err)
	}

	// collect all possible package dirs, will be
	// used for includes and linker
	pkgDirs := []string{workdir, testdir}
	for _, gopath := range gopaths {
		pkgDirs = append(pkgDirs, filepath.Join(gopath, "pkg", goos+"_"+goarch))
	}
	pkgDirs = uniqStringList(pkgDirs)

	// compile godog testmain package archive
	// we do not depend on CGO so a lot of checks are not necessary
	testMainPkgOut := filepath.Join(testdir, "main.a")
	args := []string{
		"-o", testMainPkgOut,
		// "-trimpath", workdir,
		"-p", "main",
		"-complete",
	}
	// if godog library is in vendor directory
	// link it with import map
	if i := strings.LastIndex(godogPkg.ImportPath, "vendor/"); i != -1 {
		args = append(args, "-importmap", godogImportPath+"="+godogPkg.ImportPath)
	}
	for _, inc := range pkgDirs {
		args = append(args, "-I", inc)
	}
	args = append(args, "-pack", testmain)
	cmd = exec.Command(compiler, args...)
	cmd.Env = os.Environ()
	out, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to compile testmain package: %v - output: %s", err, string(out))
	}

	// link test suite executable
	args = []string{
		"-o", bin,
		"-buildmode=exe",
	}
	for _, link := range pkgDirs {
		args = append(args, "-L", link)
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

func locatePackage(try []string) (*build.Package, error) {
	for _, p := range try {
		abs, err := filepath.Abs(p)
		if err != nil {
			continue
		}
		pkg, err := build.ImportDir(abs, 0)
		if err != nil {
			continue
		}
		return pkg, nil
	}
	return nil, fmt.Errorf("failed to find godog package in any of:\n%s", strings.Join(try, "\n"))
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

type void struct{}

func uniqStringList(strs []string) (unique []string) {
	uniq := make(map[string]void, len(strs))
	for _, s := range strs {
		if _, ok := uniq[s]; !ok {
			uniq[s] = void{}
			unique = append(unique, s)
		}
	}
	return
}

// buildTestMain if given package is valid
// it scans test files for contexts
// and produces a testmain source code.
func buildTestMain(pkg *build.Package) ([]byte, bool, error) {
	var contexts []string
	var importPath string
	name := "main"
	if nil != pkg {
		ctxs, err := processPackageTestFiles(
			pkg.TestGoFiles,
			pkg.XTestGoFiles,
		)
		if err != nil {
			return nil, false, err
		}
		contexts = ctxs
		importPath = pkg.ImportPath
		name = pkg.Name
	}

	data := struct {
		Name       string
		Contexts   []string
		ImportPath string
	}{name, contexts, importPath}

	var buf bytes.Buffer
	if err := runnerTemplate.Execute(&buf, data); err != nil {
		return nil, len(contexts) > 0, err
	}
	return buf.Bytes(), len(contexts) > 0, nil
}

// maybeVendorPaths determines possible vendor paths
// which goes levels down from given directory
// until it reaches GOPATH source dir
func maybeVendorPaths(dir string) (paths []string) {
	for _, gopath := range gopaths {
		gopath = filepath.Join(gopath, "src")
		for strings.HasPrefix(dir, gopath) && dir != gopath {
			paths = append(paths, filepath.Join(dir, "vendor", godogImportPath))
			dir = filepath.Dir(dir)
		}
	}
	return
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
