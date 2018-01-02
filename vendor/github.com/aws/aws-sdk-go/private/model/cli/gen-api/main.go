// +build codegen

// Command aws-gen-gocli parses a JSON description of an AWS API and generates a
// Go file containing a client for the API.
//
//     aws-gen-gocli apis/s3/2006-03-03/api-2.json
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go/private/model/api"
	"github.com/aws/aws-sdk-go/private/util"
)

type generateInfo struct {
	*api.API
	PackageDir string
}

var excludeServices = map[string]struct{}{
	"importexport": {},
}

// newGenerateInfo initializes the service API's folder structure for a specific service.
// If the SERVICES environment variable is set, and this service is not apart of the list
// this service will be skipped.
func newGenerateInfo(modelFile, svcPath, svcImportPath string) *generateInfo {
	g := &generateInfo{API: &api.API{SvcClientImportPath: svcImportPath, BaseCrosslinkURL: "https://docs.aws.amazon.com"}}
	g.API.Attach(modelFile)

	if _, ok := excludeServices[g.API.PackageName()]; ok {
		return nil
	}

	paginatorsFile := strings.Replace(modelFile, "api-2.json", "paginators-1.json", -1)
	if _, err := os.Stat(paginatorsFile); err == nil {
		g.API.AttachPaginators(paginatorsFile)
	} else if !os.IsNotExist(err) {
		fmt.Println("api-2.json error:", err)
	}

	docsFile := strings.Replace(modelFile, "api-2.json", "docs-2.json", -1)
	if _, err := os.Stat(docsFile); err == nil {
		g.API.AttachDocs(docsFile)
	} else {
		fmt.Println("docs-2.json error:", err)
	}

	waitersFile := strings.Replace(modelFile, "api-2.json", "waiters-2.json", -1)
	if _, err := os.Stat(waitersFile); err == nil {
		g.API.AttachWaiters(waitersFile)
	} else if !os.IsNotExist(err) {
		fmt.Println("waiters-2.json error:", err)
	}

	examplesFile := strings.Replace(modelFile, "api-2.json", "examples-1.json", -1)
	if _, err := os.Stat(examplesFile); err == nil {
		g.API.AttachExamples(examplesFile)
	} else if !os.IsNotExist(err) {
		fmt.Println("examples-1.json error:", err)
	}

	//	pkgDocAddonsFile := strings.Replace(modelFile, "api-2.json", "go-pkg-doc.gotmpl", -1)
	//	if _, err := os.Stat(pkgDocAddonsFile); err == nil {
	//		g.API.AttachPackageDocAddons(pkgDocAddonsFile)
	//	} else if !os.IsNotExist(err) {
	//		fmt.Println("go-pkg-doc.gotmpl error:", err)
	//	}

	g.API.Setup()

	if svc := os.Getenv("SERVICES"); svc != "" {
		svcs := strings.Split(svc, ",")

		included := false
		for _, s := range svcs {
			if s == g.API.PackageName() {
				included = true
				break
			}
		}
		if !included {
			// skip this non-included service
			return nil
		}
	}

	// ensure the directory exists
	pkgDir := filepath.Join(svcPath, g.API.PackageName())
	os.MkdirAll(pkgDir, 0775)
	os.MkdirAll(filepath.Join(pkgDir, g.API.InterfacePackageName()), 0775)

	g.PackageDir = pkgDir

	return g
}

// Generates service api, examples, and interface from api json definition files.
//
// Flags:
// -path alternative service path to write generated files to for each service.
//
// Env:
//  SERVICES comma separated list of services to generate.
func main() {
	var svcPath, sessionPath, svcImportPath string
	flag.StringVar(&svcPath, "path", "service", "directory to generate service clients in")
	flag.StringVar(&sessionPath, "sessionPath", filepath.Join("aws", "session"), "generate session service client factories")
	flag.StringVar(&svcImportPath, "svc-import-path", "github.com/aws/aws-sdk-go/service", "namespace to generate service client Go code import path under")
	flag.Parse()
	api.Bootstrap()

	files := []string{}
	for i := 0; i < flag.NArg(); i++ {
		file := flag.Arg(i)
		if strings.Contains(file, "*") {
			paths, _ := filepath.Glob(file)
			files = append(files, paths...)
		} else {
			files = append(files, file)
		}
	}

	for svcName := range excludeServices {
		if strings.Contains(os.Getenv("SERVICES"), svcName) {
			fmt.Printf("Service %s is not supported\n", svcName)
			os.Exit(1)
		}
	}

	sort.Strings(files)

	// Remove old API versions from list
	m := map[string]bool{}
	for i := range files {
		idx := len(files) - 1 - i
		parts := strings.Split(files[idx], string(filepath.Separator))
		svc := parts[len(parts)-3] // service name is 2nd-to-last component

		if m[svc] {
			files[idx] = "" // wipe this one out if we already saw the service
		}
		m[svc] = true
	}

	wg := sync.WaitGroup{}
	for i := range files {
		filename := files[i]
		if filename == "" { // empty file
			continue
		}

		genInfo := newGenerateInfo(filename, svcPath, svcImportPath)
		if genInfo == nil {
			continue
		}
		if _, ok := excludeServices[genInfo.API.PackageName()]; ok {
			// Skip services not yet supported.
			continue
		}

		wg.Add(1)
		go func(g *generateInfo, filename string) {
			defer wg.Done()
			writeServiceFiles(g, filename)
		}(genInfo, filename)
	}

	wg.Wait()
}

func writeServiceFiles(g *generateInfo, filename string) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "Error generating %s\n%s\n%s\n",
				filename, r, debug.Stack())
		}
	}()

	fmt.Printf("Generating %s (%s)...\n",
		g.API.PackageName(), g.API.Metadata.APIVersion)

	// write files for service client and API
	Must(writeServiceDocFile(g))
	Must(writeAPIFile(g))
	Must(writeServiceFile(g))
	Must(writeInterfaceFile(g))
	Must(writeWaitersFile(g))
	Must(writeAPIErrorsFile(g))
	Must(writeExamplesFile(g))
}

// Must will panic if the error passed in is not nil.
func Must(err error) {
	if err != nil {
		panic(err)
	}
}

const codeLayout = `// Code generated by private/model/cli/gen-api/main.go. DO NOT EDIT.

%s
package %s

%s
`

func writeGoFile(file string, layout string, args ...interface{}) error {
	return ioutil.WriteFile(file, []byte(util.GoFmt(fmt.Sprintf(layout, args...))), 0664)
}

// writeServiceDocFile generates the documentation for service package.
func writeServiceDocFile(g *generateInfo) error {
	return writeGoFile(filepath.Join(g.PackageDir, "doc.go"),
		codeLayout,
		strings.TrimSpace(g.API.ServicePackageDoc()),
		g.API.PackageName(),
		"",
	)
}

// writeExamplesFile writes out the service example file.
func writeExamplesFile(g *generateInfo) error {
	code := g.API.ExamplesGoCode()
	if len(code) > 0 {
		return writeGoFile(filepath.Join(g.PackageDir, "examples_test.go"),
			codeLayout,
			"",
			g.API.PackageName()+"_test",
			code,
		)
	}
	return nil
}

// writeServiceFile writes out the service initialization file.
func writeServiceFile(g *generateInfo) error {
	return writeGoFile(filepath.Join(g.PackageDir, "service.go"),
		codeLayout,
		"",
		g.API.PackageName(),
		g.API.ServiceGoCode(),
	)
}

// writeInterfaceFile writes out the service interface file.
func writeInterfaceFile(g *generateInfo) error {
	const pkgDoc = `
// Package %s provides an interface to enable mocking the %s service client
// for testing your code.
//
// It is important to note that this interface will have breaking changes
// when the service model is updated and adds new API operations, paginators,
// and waiters.`
	return writeGoFile(filepath.Join(g.PackageDir, g.API.InterfacePackageName(), "interface.go"),
		codeLayout,
		fmt.Sprintf(pkgDoc, g.API.InterfacePackageName(), g.API.Metadata.ServiceFullName),
		g.API.InterfacePackageName(),
		g.API.InterfaceGoCode(),
	)
}

func writeWaitersFile(g *generateInfo) error {
	if len(g.API.Waiters) == 0 {
		return nil
	}

	return writeGoFile(filepath.Join(g.PackageDir, "waiters.go"),
		codeLayout,
		"",
		g.API.PackageName(),
		g.API.WaitersGoCode(),
	)
}

// writeAPIFile writes out the service API file.
func writeAPIFile(g *generateInfo) error {
	return writeGoFile(filepath.Join(g.PackageDir, "api.go"),
		codeLayout,
		"",
		g.API.PackageName(),
		g.API.APIGoCode(),
	)
}

// writeAPIErrorsFile writes out the service API errors file.
func writeAPIErrorsFile(g *generateInfo) error {
	return writeGoFile(filepath.Join(g.PackageDir, "errors.go"),
		codeLayout,
		"",
		g.API.PackageName(),
		g.API.APIErrorsGoCode(),
	)
}
