// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Program aebundler turns a Go app into a fully self-contained tar file.
// The app and its subdirectories (if any) are placed under "."
// and the dependencies from $GOPATH are placed under ./_gopath/src.
// A main func is synthesized if one does not exist.
//
// A sample Dockerfile to be used with this bundler could look like this:
//     FROM gcr.io/google_appengine/go-compat
//     ADD . /app
//     RUN GOPATH=/app/_gopath go build -tags appenginevm -o /app/_ah/exe
package main

import (
	"archive/tar"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

var (
	output  = flag.String("o", "", "name of output tar file or '-' for stdout")
	rootDir = flag.String("root", ".", "directory name of application root")
	vm      = flag.Bool("vm", true, `bundle an app for App Engine "flexible environment"`)

	skipFiles = map[string]bool{
		".git":        true,
		".gitconfig":  true,
		".hg":         true,
		".travis.yml": true,
	}
)

const (
	newMain = `package main
import "google.golang.org/appengine"
func main() {
	appengine.Main()
}
`
)

func usage() {
	fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "\t%s -o <file.tar|->\tBundle app to named tar file or stdout\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "\noptional arguments:\n")
	flag.PrintDefaults()
}

func main() {
	flag.Usage = usage
	flag.Parse()

	var tags []string
	if *vm {
		tags = append(tags, "appenginevm")
	} else {
		tags = append(tags, "appengine")
	}

	tarFile := *output
	if tarFile == "" {
		usage()
		errorf("Required -o flag not specified.")
	}

	app, err := analyze(tags)
	if err != nil {
		errorf("Error analyzing app: %v", err)
	}
	if err := app.bundle(tarFile); err != nil {
		errorf("Unable to bundle app: %v", err)
	}
}

// errorf prints the error message and exits.
func errorf(format string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, "aebundler: "+format+"\n", a...)
	os.Exit(1)
}

type app struct {
	hasMain  bool
	appFiles []string
	imports  map[string]string
}

// analyze checks the app for building with the given build tags and returns hasMain,
// app files, and a map of full directory import names to original import names.
func analyze(tags []string) (*app, error) {
	ctxt := buildContext(tags)
	hasMain, appFiles, err := checkMain(ctxt)
	if err != nil {
		return nil, err
	}
	gopath := filepath.SplitList(ctxt.GOPATH)
	im, err := imports(ctxt, *rootDir, gopath)
	return &app{
		hasMain:  hasMain,
		appFiles: appFiles,
		imports:  im,
	}, err
}

// buildContext returns the context for building the source.
func buildContext(tags []string) *build.Context {
	return &build.Context{
		GOARCH:    build.Default.GOARCH,
		GOOS:      build.Default.GOOS,
		GOROOT:    build.Default.GOROOT,
		GOPATH:    build.Default.GOPATH,
		Compiler:  build.Default.Compiler,
		BuildTags: append(build.Default.BuildTags, tags...),
	}
}

// bundle bundles the app into the named tarFile ("-"==stdout).
func (s *app) bundle(tarFile string) (err error) {
	var out io.Writer
	if tarFile == "-" {
		out = os.Stdout
	} else {
		f, err := os.Create(tarFile)
		if err != nil {
			return err
		}
		defer func() {
			if cerr := f.Close(); err == nil {
				err = cerr
			}
		}()
		out = f
	}
	tw := tar.NewWriter(out)

	for srcDir, importName := range s.imports {
		dstDir := "_gopath/src/" + importName
		if err = copyTree(tw, dstDir, srcDir); err != nil {
			return fmt.Errorf("unable to copy directory %v to %v: %v", srcDir, dstDir, err)
		}
	}
	if err := copyTree(tw, ".", *rootDir); err != nil {
		return fmt.Errorf("unable to copy root directory to /app: %v", err)
	}
	if !s.hasMain {
		if err := synthesizeMain(tw, s.appFiles); err != nil {
			return fmt.Errorf("unable to synthesize new main func: %v", err)
		}
	}

	if err := tw.Close(); err != nil {
		return fmt.Errorf("unable to close tar file %v: %v", tarFile, err)
	}
	return nil
}

// synthesizeMain generates a new main func and writes it to the tarball.
func synthesizeMain(tw *tar.Writer, appFiles []string) error {
	appMap := make(map[string]bool)
	for _, f := range appFiles {
		appMap[f] = true
	}
	var f string
	for i := 0; i < 100; i++ {
		f = fmt.Sprintf("app_main%d.go", i)
		if !appMap[filepath.Join(*rootDir, f)] {
			break
		}
	}
	if appMap[filepath.Join(*rootDir, f)] {
		return fmt.Errorf("unable to find unique name for %v", f)
	}
	hdr := &tar.Header{
		Name: f,
		Mode: 0644,
		Size: int64(len(newMain)),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return fmt.Errorf("unable to write header for %v: %v", f, err)
	}
	if _, err := tw.Write([]byte(newMain)); err != nil {
		return fmt.Errorf("unable to write %v to tar file: %v", f, err)
	}
	return nil
}

// imports returns a map of all import directories (recursively) used by the app.
// The return value maps full directory names to original import names.
func imports(ctxt *build.Context, srcDir string, gopath []string) (map[string]string, error) {
	pkg, err := ctxt.ImportDir(srcDir, 0)
	if err != nil {
		return nil, fmt.Errorf("unable to analyze source: %v", err)
	}

	// Resolve all non-standard-library imports
	result := make(map[string]string)
	for _, v := range pkg.Imports {
		if !strings.Contains(v, ".") {
			continue
		}
		src, err := findInGopath(v, gopath)
		if err != nil {
			return nil, fmt.Errorf("unable to find import %v in gopath %v: %v", v, gopath, err)
		}
		result[src] = v
		im, err := imports(ctxt, src, gopath)
		if err != nil {
			return nil, fmt.Errorf("unable to parse package %v: %v", src, err)
		}
		for k, v := range im {
			result[k] = v
		}
	}
	return result, nil
}

// findInGopath searches the gopath for the named import directory.
func findInGopath(dir string, gopath []string) (string, error) {
	for _, v := range gopath {
		dst := filepath.Join(v, "src", dir)
		if _, err := os.Stat(dst); err == nil {
			return dst, nil
		}
	}
	return "", fmt.Errorf("unable to find package %v in gopath %v", dir, gopath)
}

// copyTree copies srcDir to tar file dstDir, ignoring skipFiles.
func copyTree(tw *tar.Writer, dstDir, srcDir string) error {
	entries, err := ioutil.ReadDir(srcDir)
	if err != nil {
		return fmt.Errorf("unable to read dir %v: %v", srcDir, err)
	}
	for _, entry := range entries {
		n := entry.Name()
		if skipFiles[n] {
			continue
		}
		s := filepath.Join(srcDir, n)
		d := filepath.Join(dstDir, n)
		if entry.IsDir() {
			if err := copyTree(tw, d, s); err != nil {
				return fmt.Errorf("unable to copy dir %v to %v: %v", s, d, err)
			}
			continue
		}
		if err := copyFile(tw, d, s); err != nil {
			return fmt.Errorf("unable to copy dir %v to %v: %v", s, d, err)
		}
	}
	return nil
}

// copyFile copies src to tar file dst.
func copyFile(tw *tar.Writer, dst, src string) error {
	s, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("unable to open %v: %v", src, err)
	}
	defer s.Close()
	fi, err := s.Stat()
	if err != nil {
		return fmt.Errorf("unable to stat %v: %v", src, err)
	}

	hdr, err := tar.FileInfoHeader(fi, dst)
	if err != nil {
		return fmt.Errorf("unable to create tar header for %v: %v", dst, err)
	}
	hdr.Name = dst
	if err := tw.WriteHeader(hdr); err != nil {
		return fmt.Errorf("unable to write header for %v: %v", dst, err)
	}
	_, err = io.Copy(tw, s)
	if err != nil {
		return fmt.Errorf("unable to copy %v to %v: %v", src, dst, err)
	}
	return nil
}

// checkMain verifies that there is a single "main" function.
// It also returns a list of all Go source files in the app.
func checkMain(ctxt *build.Context) (bool, []string, error) {
	pkg, err := ctxt.ImportDir(*rootDir, 0)
	if err != nil {
		return false, nil, fmt.Errorf("unable to analyze source: %v", err)
	}
	if !pkg.IsCommand() {
		errorf("Your app's package needs to be changed from %q to \"main\".\n", pkg.Name)
	}
	// Search for a "func main"
	var hasMain bool
	var appFiles []string
	for _, f := range pkg.GoFiles {
		n := filepath.Join(*rootDir, f)
		appFiles = append(appFiles, n)
		if hasMain, err = readFile(n); err != nil {
			return false, nil, fmt.Errorf("error parsing %q: %v", n, err)
		}
	}
	return hasMain, appFiles, nil
}

// isMain returns whether the given function declaration is a main function.
// Such a function must be called "main", not have a receiver, and have no arguments or return types.
func isMain(f *ast.FuncDecl) bool {
	ft := f.Type
	return f.Name.Name == "main" && f.Recv == nil && ft.Params.NumFields() == 0 && ft.Results.NumFields() == 0
}

// readFile reads and parses the Go source code file and returns whether it has a main function.
func readFile(filename string) (hasMain bool, err error) {
	var src []byte
	src, err = ioutil.ReadFile(filename)
	if err != nil {
		return
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, src, 0)
	for _, decl := range file.Decls {
		funcDecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if !isMain(funcDecl) {
			continue
		}
		hasMain = true
		break
	}
	return
}
