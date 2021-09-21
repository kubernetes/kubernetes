package convert

import (
	"fmt"
	"go/build"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
)

/*
 * RewritePackage takes a name (eg: my-package/tools), finds its test files using
 * Go's build package, and then rewrites them. A ginkgo test suite file will
 * also be added for this package, and all of its child packages.
 */
func RewritePackage(packageName string) {
	pkg, err := packageWithName(packageName)
	if err != nil {
		panic(fmt.Sprintf("unexpected error reading package: '%s'\n%s\n", packageName, err.Error()))
	}

	for _, filename := range findTestsInPackage(pkg) {
		rewriteTestsInFile(filename)
	}
}

/*
 * Given a package, findTestsInPackage reads the test files in the directory,
 * and then recurses on each child package, returning a slice of all test files
 * found in this process.
 */
func findTestsInPackage(pkg *build.Package) (testfiles []string) {
	for _, file := range append(pkg.TestGoFiles, pkg.XTestGoFiles...) {
		testfile, _ := filepath.Abs(filepath.Join(pkg.Dir, file))
		testfiles = append(testfiles, testfile)
	}

	dirFiles, err := ioutil.ReadDir(pkg.Dir)
	if err != nil {
		panic(fmt.Sprintf("unexpected error reading dir: '%s'\n%s\n", pkg.Dir, err.Error()))
	}

	re := regexp.MustCompile(`^[._]`)

	for _, file := range dirFiles {
		if !file.IsDir() {
			continue
		}

		if re.Match([]byte(file.Name())) {
			continue
		}

		packageName := filepath.Join(pkg.ImportPath, file.Name())
		subPackage, err := packageWithName(packageName)
		if err != nil {
			panic(fmt.Sprintf("unexpected error reading package: '%s'\n%s\n", packageName, err.Error()))
		}

		testfiles = append(testfiles, findTestsInPackage(subPackage)...)
	}

	addGinkgoSuiteForPackage(pkg)
	goFmtPackage(pkg)
	return
}

/*
 * Shells out to `ginkgo bootstrap` to create a test suite file
 */
func addGinkgoSuiteForPackage(pkg *build.Package) {
	originalDir, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	suite_test_file := filepath.Join(pkg.Dir, pkg.Name+"_suite_test.go")

	_, err = os.Stat(suite_test_file)
	if err == nil {
		return // test file already exists, this should be a no-op
	}

	err = os.Chdir(pkg.Dir)
	if err != nil {
		panic(err)
	}

	output, err := exec.Command("ginkgo", "bootstrap").Output()

	if err != nil {
		panic(fmt.Sprintf("error running 'ginkgo bootstrap'.\nstdout: %s\n%s\n", output, err.Error()))
	}

	err = os.Chdir(originalDir)
	if err != nil {
		panic(err)
	}
}

/*
 * Shells out to `go fmt` to format the package
 */
func goFmtPackage(pkg *build.Package) {
	path, _ := filepath.Abs(pkg.ImportPath)
	output, err := exec.Command("go", "fmt", path).CombinedOutput()

	if err != nil {
		fmt.Printf("Warning: Error running 'go fmt %s'.\nstdout: %s\n%s\n", path, output, err.Error())
	}
}

/*
 * Attempts to return a package with its test files already read.
 * The ImportMode arg to build.Import lets you specify if you want go to read the
 * buildable go files inside the package, but it fails if the package has no go files
 */
func packageWithName(name string) (pkg *build.Package, err error) {
	pkg, err = build.Default.Import(name, ".", build.ImportMode(0))
	if err == nil {
		return
	}

	pkg, err = build.Default.Import(name, ".", build.ImportMode(1))
	return
}
