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
// usage: go run _tools/release_docs/main.go _tools/release_docs/api-reference-process.go --branch release-1.0 --output-dir v1.0
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
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

	branch       = flag.String("branch", "", "The git branch from which to pull docs. (e.g. release-1.0, master).")
	outputDir    = flag.String("output-dir", "", "The directory in which to save results.")
	remote       = flag.String("remote", "upstream", "The name of the remote repo from which to pull docs.")
	apiReference = flag.Bool("apiReference", true, "Whether update api reference")
)

func fixURL(u *url.URL) bool {
	if u.Host != "" && strings.HasSuffix(u.Path, ".md") {
		u.Path = u.Path[:len(u.Path)-3] + ".html"
		return true
	}

	return false
}

func processFile(filename string) error {
	fileBytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	output := rewriteLinks(fileBytes)
	output = rewriteCodeBlocks(output)

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

func rewriteLinks(fileBytes []byte) []byte {
	return linkRE.ReplaceAllFunc(fileBytes, func(in []byte) (out []byte) {
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
}

// Allow more than 3 tick because people write this stuff.
var ticticticRE = regexp.MustCompile("^`{3,}\\s*(.*)$")
var notTicticticRE = regexp.MustCompile("^```(.*)```")
var languageFixups = map[string]string{
	"shell": "sh",
}

func rewriteCodeBlocks(fileBytes []byte) []byte {
	lines := strings.Split(string(fileBytes), "\n")
	inside := false
	highlight := false
	for i := range lines {
		trimmed := []byte(strings.TrimLeft(lines[i], " "))
		if !ticticticRE.Match(trimmed) || notTicticticRE.Match(trimmed) {
			continue
		}
		if !inside {
			out := ticticticRE.FindSubmatch(trimmed)
			lang := strings.ToLower(string(out[1]))
			// Can't syntax highlight unknown language.
			if fixedLang := languageFixups[lang]; fixedLang != "" {
				lang = fixedLang
			}
			if lang != "" {
				// The "redcarpet" markdown renderer will accept ```lang, but
				// "kramdown" will not.  They both accept this, format, and we
				// need a hook to fixup language codes anyway (until we have a
				// munger in master).
				lines[i] = fmt.Sprintf("{%% highlight %s %%}", lang)
				highlight = true
			}
		} else if highlight {
			lines[i] = `{% endhighlight %}`
			highlight = false
		}
		inside = !inside
	}
	return []byte(strings.Join(lines, "\n") + "\n")
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

func copySingleFile(src, dst string) (err error) {
	_, err = os.Stat(dst)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}

	defer func() {
		cerr := out.Close()
		if err == nil {
			err = cerr
		}
	}()

	_, err = io.Copy(out, in)
	return
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
			if err = processFile(path); err != nil {
				return err
			}

			if strings.ToLower(info.Name()) == "readme.md" {
				newpath := path[0:len(path)-len("readme.md")] + "index.md"
				fmt.Printf("Copying %s to %s\n", path, newpath)
				if err = copySingleFile(path, newpath); err != nil {
					return err
				}
			}
		}

		if *apiReference && !info.IsDir() && (info.Name() == "definitions.html" || info.Name() == "operations.html") {
			fmt.Printf("Processing %s\n", path)
			err := addHeader(path)
			if err != nil {
				return err
			}
			return fixHeadAlign(path)
		}
		return nil

	})

	if err != nil {
		fmt.Printf("Error while processing markdown and html files: %v\n", err)
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
