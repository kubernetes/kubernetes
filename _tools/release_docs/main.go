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

// Import docs from a git branch and format them for gh-pages.
// usage: go run _tools/release_docs/main.go --branch release-1.0 --output-dir v1.0
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	// Finds markdown links of the form [foo](bar "alt-text").
	linkRE = regexp.MustCompile(`\[([^]]*)\]\(([^)]*)\)`)
	// Splits the link target into link target and alt-text.
	altTextRE = regexp.MustCompile(`(.*)( ".*")`)

	branch    = flag.String("branch", "", "The git branch from which to pull docs. (e.g. release-1.0, master).")
	outputDir = flag.String("output-dir", "", "The directory in which to save results.")
	remote    = flag.String("remote", "upstream", "The name of the remote repo from which to pull docs.")
)

func fixURL(u *url.URL) (modified bool) {
	if u.Host != "" {
		return
	}

	if strings.HasSuffix(u.Path, ".md") {
		u.Path = u.Path[:len(u.Path)-3] + ".html"
		modified = true
	}

	if strings.HasSuffix(u.Path, "/") {
		u.Path = u.Path + "README.html"
		modified = true
	}
	return
}

func processFile(filename string) error {
	fileBytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	output := linkRE.ReplaceAllFunc(fileBytes, func(in []byte) (out []byte) {
		match := linkRE.FindSubmatch(in)

		visibleText := string(match[1])
		linkText := string(match[2])
		altText := ""

		if parts := altTextRE.FindStringSubmatch(linkText); parts != nil {
			linkText = parts[1]
			altText = parts[2]
		}

		u, err := url.Parse(linkText)
		if err != nil {
			return in
		}

		if !fixURL(u) {
			return in
		}
		return []byte(fmt.Sprintf("[%s](%s)", visibleText, u.String()+altText))
	})

	f, err := os.Create(filename)
	if err != nil {
		return err
	}

	_, err = f.WriteString("---\nlayout: docwithnav\n---\n")
	if err != nil {
		return err
	}

	_, err = f.Write(output)
	return err
}

func runGitUpdate(remote string) error {
	cmd := exec.Command("git", "fetch", remote)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("git fetch failed: %v\n%s", err, out)
	}
	return err
}

func copyFiles(remoteRepo, directory, branch string) error {
	if err := runGitUpdate(remoteRepo); err != nil {
		return err
	}

	if !strings.HasSuffix(directory, "/") {
		directory += "/"
	}
	prefix := fmt.Sprintf("--prefix=%s", directory)
	tagRef := fmt.Sprintf("%s/%s", remoteRepo, branch)
	gitCmd := exec.Command("git", "archive", "--format=tar", prefix, tagRef, "docs", "examples")
	tarCmd := exec.Command("tar", "-x")

	var err error
	tarCmd.Stdin, err = gitCmd.StdoutPipe()
	if err != nil {
		return err
	}

	gitStderr, err := gitCmd.StderrPipe()
	if err != nil {
		return err
	}
	gitErrs := bytes.Buffer{}

	fmt.Printf("Copying docs and examples from %s to %s\n", tagRef, directory)

	if err = tarCmd.Start(); err != nil {
		return fmt.Errorf("tar command failed: %v", err)
	}

	if err = gitCmd.Run(); err != nil {
		gitErrs.ReadFrom(gitStderr)
		return fmt.Errorf("git command failed: %v\n%s", err, gitErrs.String())
	}

	return tarCmd.Wait()
}

func main() {
	flag.Parse()
	if len(*branch) == 0 {
		fmt.Println("You must specify a branch with --branch.")
		os.Exit(1)
	}
	if len(*outputDir) == 0 {
		fmt.Println("You must specify an output dir with --output-dir.")
		os.Exit(1)
	}

	if err := checkCWD(); err != nil {
		fmt.Printf("Could not find the kubernetes root: %v\n", err)
		os.Exit(1)
	}

	if err := copyFiles(*remote, *outputDir, *branch); err != nil {
		fmt.Printf("Error copying files: %v\n", err)
		os.Exit(1)
	}

	err := filepath.Walk(*outputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() && strings.HasSuffix(info.Name(), ".md") {
			fmt.Printf("Processing %s\n", path)
			return processFile(path)
		}
		return nil
	})

	if err != nil {
		fmt.Printf("Error while processing markdown files: %v\n", err)
		os.Exit(1)
	}
}

func checkCWD() error {
	dir, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return err
	}
	return os.Chdir(strings.TrimSpace(string(dir)))
}
