package main

import (
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

	version = flag.String("version", "", "A version tag to process docs. (e.g. 1.0).")
	remote  = flag.String("remote", "upstream", "The name of the remote repo from which to pull docs.")
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

func copyFiles(remoteRepo, directory, releaseTag string) error {
	prefix := fmt.Sprintf("--prefix=%s", directory)
	tagRef := fmt.Sprintf("%s/%s", remoteRepo, releaseTag)
	gitCmd := exec.Command("git", "archive", "--format=tar", prefix, tagRef, "docs", "examples")
	tarCmd := exec.Command("tar", "-x")

	var err error
	tarCmd.Stdin, err = gitCmd.StdoutPipe()
	if err != nil {
		return err
	}

	fmt.Printf("Copying docs and examples from %s to %s\n", tagRef, directory)

	if err = tarCmd.Start(); err != nil {
		return err
	}

	if err = gitCmd.Run(); err != nil {
		return err
	}

	return tarCmd.Wait()
}

func main() {
	flag.Parse()
	if len(*version) == 0 {
		fmt.Println("You must specify a version with --version.")
		os.Exit(1)
	}

	if err := checkCWD(); err != nil {
		fmt.Printf("Could not find the kubernetes root: %v\n", err)
		os.Exit(1)
	}

	dir := fmt.Sprintf("v%s/", *version)
	releaseTag := fmt.Sprintf("release-%s", *version)
	if err := copyFiles(*remote, dir, releaseTag); err != nil {
		fmt.Printf("Error copying files: %v\n", err)
		os.Exit(1)
	}

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
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
