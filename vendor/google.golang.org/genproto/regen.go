// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build ignore

// Regen.go regenerates the genproto repository.
//
// Regen.go recursively walks through each directory named by given arguments,
// looking for all .proto files. (Symlinks are not followed.)
// If the pkg_prefix flag is not an empty string,
// any proto file without `go_package` option
// or whose option does not begin with the prefix is ignored.
// Protoc is executed on remaining files,
// one invocation per set of files declaring the same Go package.
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var goPkgOptRe = regexp.MustCompile(`(?m)^option go_package = (.*);`)

func usage() {
	fmt.Fprintln(os.Stderr, `usage: go run regen.go -go_out=path/to/output [-pkg_prefix=pkg/prefix] roots...

Most users will not need to run this file directly.
To regenerate this repository, run regen.sh instead.`)
	flag.PrintDefaults()
}

func main() {
	goOutDir := flag.String("go_out", "", "go_out argument to pass to protoc-gen-go")
	pkgPrefix := flag.String("pkg_prefix", "", "only include proto files with go_package starting with this prefix")
	flag.Usage = usage
	flag.Parse()

	if *goOutDir == "" {
		log.Fatal("need go_out flag")
	}

	pkgFiles := make(map[string][]string)
	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || !strings.HasSuffix(path, ".proto") {
			return nil
		}
		pkg, err := goPkg(path)
		if err != nil {
			return err
		}
		pkgFiles[pkg] = append(pkgFiles[pkg], path)
		return nil
	}
	for _, root := range flag.Args() {
		if err := filepath.Walk(root, walkFn); err != nil {
			log.Fatal(err)
		}
	}
	for pkg, fnames := range pkgFiles {
		if !strings.HasPrefix(pkg, *pkgPrefix) {
			continue
		}
		if out, err := protoc(*goOutDir, flag.Args(), fnames); err != nil {
			log.Fatalf("error executing protoc: %s\n%s", err, out)
		}
	}
}

// goPkg reports the import path declared in the given file's
// `go_package` option. If the option is missing, goPkg returns empty string.
func goPkg(fname string) (string, error) {
	content, err := ioutil.ReadFile(fname)
	if err != nil {
		return "", err
	}

	var pkgName string
	if match := goPkgOptRe.FindSubmatch(content); len(match) > 0 {
		pn, err := strconv.Unquote(string(match[1]))
		if err != nil {
			return "", err
		}
		pkgName = pn
	}
	if p := strings.IndexRune(pkgName, ';'); p > 0 {
		pkgName = pkgName[:p]
	}
	return pkgName, nil
}

// protoc executes the "protoc" command on files named in fnames,
// passing go_out and include flags specified in goOut and includes respectively.
// protoc returns combined output from stdout and stderr.
func protoc(goOut string, includes, fnames []string) ([]byte, error) {
	args := []string{"--go_out=plugins=grpc:" + goOut}
	for _, inc := range includes {
		args = append(args, "-I", inc)
	}
	args = append(args, fnames...)
	return exec.Command("protoc", args...).CombinedOutput()
}
