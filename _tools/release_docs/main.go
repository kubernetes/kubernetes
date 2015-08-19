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
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	branch       = flag.String("branch", "", "The git branch from which to pull docs. (e.g. release-1.0, master).")
	outputDir    = flag.String("output-dir", "", "The directory in which to save results.")
	remote       = flag.String("remote", "upstream", "The name of the remote repo from which to pull docs.")
	apiReference = flag.Bool("apiReference", true, "Whether update api reference")

	subdirs = []string{"docs", "examples"}
)

func fileExistsInBranch(path string) bool {
	out, err := exec.Command("git", "ls-tree", fmt.Sprintf("%s/%s", *remote, *branch), path).Output()
	return err == nil && len(out) != 0
}

func fixURL(filename string, u *url.URL) bool {
	if u.Host != "" || u.Path == "" {
		return false
	}

	target := filepath.Join(filepath.Dir(filename), u.Path)
	if fi, err := os.Stat(target); os.IsNotExist(err) {
		// We're linking to something we didn't copy over. Send
		// it through the redirector.
		rel, err := filepath.Rel(*outputDir, target)
		if err != nil {
			return false
		}

		if fileExistsInBranch(rel) {
			u.Path = filepath.Join("HEAD", rel)
			u.Host = "releases.k8s.io"
			u.Scheme = "https"
			return true
		}
	} else if fi.IsDir() {
		// If there's no README.md in the directory, redirect to github
		// for the directory view.
		files, err := filepath.Glob(target + "/*")
		if err != nil {
			return false
		}

		hasReadme := false
		for _, f := range files {
			if strings.ToLower(filepath.Base(f)) == "readme.md" {
				hasReadme = true
			}
		}
		if !hasReadme {
			rel, err := filepath.Rel(*outputDir, target)
			if err != nil {
				return false
			}
			u.Path = filepath.Join("HEAD", rel)
			u.Host = "releases.k8s.io"
			u.Scheme = "https"
			return true
		}
	} else if strings.HasSuffix(u.Path, ".md") {
		u.Path = u.Path[:len(u.Path)-3] + ".html"
		return true
	}

	return false
}

func processFile(prefix, filename string) error {
	fileBytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	title := getTitle(fileBytes)
	if len(title) == 0 {
		title = filename[len(prefix)+1 : len(filename)-len(".md")]
	}

	output := rewriteLinks(filename, fileBytes)
	output = rewriteCodeBlocks(output)

	f, err := os.Create(filename)
	if err != nil {
		return err
	}

	_, err = f.WriteString(fmt.Sprintf("---\nlayout: docwithnav\ntitle: %q\n---\n", title))
	if err != nil {
		return err
	}

	_, err = f.Write(output)
	return err
}

var (
	// Finds markdown links of the form [foo](bar "alt-text").
	linkRE = regexp.MustCompile(`\[([^]]*)\]\(([^)]*)\)`)
	// Finds markdown refence style link definitions
	refRE = regexp.MustCompile(`(?m)^\[([^]]*)\]:\s+(.*)$`)
	// Splits the link target into link target and alt-text.
	altTextRE = regexp.MustCompile(`(.*)( ".*")`)
)

func rewriteLinks(filename string, fileBytes []byte) []byte {
	getParts := func(re *regexp.Regexp, in []byte) (text, link, caption string, changed bool) {
		match := re.FindSubmatch(in)
		text = string(match[1])
		link = strings.TrimSpace(string(match[2]))

		if parts := altTextRE.FindStringSubmatch(link); parts != nil {
			link = parts[1]
			caption = parts[2]
		}

		u, err := url.Parse(link)
		if err != nil || !fixURL(filename, u) {
			return "", "", "", false
		}

		return text, u.String(), caption, true
	}

	for _, conversion := range []struct {
		re     *regexp.Regexp
		format string
	}{
		{linkRE, "[%s](%s)"},
		{refRE, "[%s]: %s"},
	} {

		fileBytes = conversion.re.ReplaceAllFunc(fileBytes, func(in []byte) (out []byte) {
			text, link, caption, changed := getParts(conversion.re, in)
			if !changed {
				return in
			}

			return []byte(fmt.Sprintf(conversion.format, text, link+caption))
		})

	}
	return fileBytes
}

var (
	// Allow more than 3 tick because people write this stuff.
	ticticticRE    = regexp.MustCompile("^`{3,}\\s*(.*)$")
	notTicticticRE = regexp.MustCompile("^```(.*)```")
	languageFixups = map[string]string{
		"shell": "sh",
	}
)

func rewriteCodeBlocks(fileBytes []byte) []byte {
	lines := strings.Split(string(fileBytes), "\n")
	inside := false
	highlight := false
	output := []string{}
	for i := range lines {
		trimmed := []byte(strings.TrimLeft(lines[i], " "))
		if !ticticticRE.Match(trimmed) || notTicticticRE.Match(trimmed) {
			output = append(output, lines[i])
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
				output = append(output, fmt.Sprintf("{%% highlight %s %%}", lang))
				highlight = true
			} else {
				output = append(output, lines[i])
			}
			output = append(output, `{% raw %}`)
		} else {
			output = append(output, `{% endraw %}`)
			if highlight {
				output = append(output, `{% endhighlight %}`)
				highlight = false
			} else {
				output = append(output, lines[i])
			}
		}
		inside = !inside
	}
	return []byte(strings.Join(output, "\n") + "\n")
}

var (
	// matches "# headers" and "## headers"
	atxTitleRE = regexp.MustCompile(`(?m)^\s*##?\s+(?P<title>.*)$`)
	// matches
	//  Headers
	//  =======
	// and
	//  Headers
	//  --
	setextTitleRE = regexp.MustCompile("(?m)^(?P<title>.+)\n((=+)|(-+))$")
	ignoredRE     = regexp.MustCompile(`[*_\\]`)
)

// removeLinks removes markdown links from the input leaving only the
// display text.
func removeLinks(input []byte) []byte {
	indices := linkRE.FindAllSubmatchIndex(input, -1)
	if len(indices) == 0 {
		return input
	}

	out := make([]byte, 0, len(input))
	cur := 0
	for _, index := range indices {
		linkStart, linkEnd, textStart, textEnd := index[0], index[1], index[2], index[3]
		// append bytes between previous match and this one
		out = append(out, input[cur:linkStart]...)
		// extract and append link text
		out = append(out, input[textStart:textEnd]...)
		// update cur
		cur = linkEnd
	}
	// pick up the remaining and return it
	return append(out, input[cur:len(input)]...)
}

// findTitleMatch returns the start of the match and the "title" subgroup of
// bytes. If the regexp doesn't match, it will return -1 and nil.
func findTitleMatch(titleRE *regexp.Regexp, input []byte) (start int, title []byte) {
	indices := titleRE.FindSubmatchIndex(input)
	if len(indices) == 0 {
		return -1, nil
	}

	for i, name := range titleRE.SubexpNames() {
		if name == "title" {
			start, end := indices[2*i], indices[2*i+1]
			return indices[0], input[start:end]
		}
	}

	// there was no grouped named title
	return -1, nil
}

func getTitle(fileBytes []byte) string {
	atxStart, atxMatch := findTitleMatch(atxTitleRE, fileBytes)
	setextStart, setextMatch := findTitleMatch(setextTitleRE, fileBytes)

	var title []byte

	switch {
	case atxStart == -1 && setextStart == -1:
		return ""
	case atxStart == -1:
		title = setextMatch
	case setextStart == -1:
		title = atxMatch
	case setextStart < atxStart:
		title = setextMatch
	default:
		title = atxMatch
	}

	// Handle the case where there's a link in the header.
	title = removeLinks(title)

	// Take out all markdown stylings.
	return string(ignoredRE.ReplaceAll(title, nil))
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
	gitArgs := append([]string{"archive", "--format=tar", prefix, tagRef}, subdirs...)
	gitCmd := exec.Command("git", gitArgs...)
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

	for _, subDir := range subdirs {
		prefix := path.Join(*outputDir, subDir)
		err := filepath.Walk(prefix, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if !info.IsDir() && strings.HasSuffix(info.Name(), ".md") {
				fmt.Printf("Processing %s\n", path)
				if err = processFile(prefix, path); err != nil {
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
				err := processHTML(path, info.Name(), *outputDir)
				if err != nil {
					return err
				}
			}
			return nil

		})

		if err != nil {
			fmt.Printf("Error while processing markdown and html files: %v\n", err)
			os.Exit(1)
		}
	}
}

func checkCWD() error {
	dir, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return err
	}
	return os.Chdir(strings.TrimSpace(string(dir)))
}
