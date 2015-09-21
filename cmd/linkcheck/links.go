/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//This command checks if the hyperlinks in files are valid. It checks the files
//with 'fileSuffix' in 'rootDir' for URLs that match 'prefix'. It trims the
//'prefix' from the URL, uses what's left as the relative path to repoRoot to
//verify if the link is valid. For example:
//$ linkcheck --root-dir=${TYPEROOT} --repo-root=${KUBE_ROOT} \
//  --file-suffix=types.go --prefix=http://releases.k8s.io/HEAD

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	flag "github.com/spf13/pflag"
)

var (
	httpRE *regexp.Regexp

	rootDir    = flag.String("root-dir", "", "Root directory containing documents to be processed.")
	repoRoot   = flag.String("repo-root", "", `Root directory of k8s repository.`)
	fileSuffix = flag.String("file-suffix", "", "suffix of files to be checked")
	prefix     = flag.String("prefix", "", "Longest common prefix of the link URL, e.g., http://release.k8s.io/HEAD/ for links in pkg/api/types.go")
)

func newWalkFunc(invalidLink *bool) filepath.WalkFunc {
	return func(filePath string, info os.FileInfo, err error) error {
		if !strings.HasSuffix(info.Name(), *fileSuffix) {
			return nil
		}
		fileBytes, err := ioutil.ReadFile(filePath)
		if err != nil {
			return err
		}
		foundInvalid := false
		matches := httpRE.FindAllSubmatch(fileBytes, -1)
		for _, match := range matches {
			//match[1] should look like docs/devel/api-conventions.md
			if _, err := os.Stat(path.Join(*repoRoot, string(match[1]))); err != nil {
				fmt.Fprintf(os.Stderr, "Link is not valid: %s\n", string(match[0]))
				foundInvalid = true
			}
		}
		if foundInvalid {
			fmt.Fprintf(os.Stderr, "Found invalid links in %s\n", filePath)
			*invalidLink = true
		}
		return nil
	}
}

func main() {
	flag.Parse()
	httpRE = regexp.MustCompile(*prefix + `(.*\.md)`)

	if *rootDir == "" || *repoRoot == "" || *prefix == "" {
		flag.Usage()
		os.Exit(2)
	}
	invalidLink := false
	if err := filepath.Walk(*rootDir, newWalkFunc(&invalidLink)); err != nil {
		fmt.Fprintf(os.Stderr, "Fail: %v.\n", err)
		os.Exit(2)
	}
	if invalidLink {
		os.Exit(1)
	}
}
