package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

var cmdGo = &Command{
	Name:  "go",
	Args:  "command [arguments]",
	Short: "run the go tool with saved dependencies",
	Long: `
Go runs the go tool with a modified GOPATH giving access to
dependencies saved in Godeps.

Any go tool command can run this way, but "godep go get"
is unnecessary and has been disabled. Instead, use
"godep go install".
`,
	Run:          runGo,
	OnlyInGOPATH: true,
}

// Find the godep GOPATH for this file tree and run the go tool.
func runGo(cmd *Command, args []string) {
	gopath := prepareGopath()
	if s := os.Getenv("GOPATH"); s != "" {
		gopath += string(os.PathListSeparator) + os.Getenv("GOPATH")
	}
	if len(args) > 0 && args[0] == "get" {
		log.Printf("invalid subcommand: %q", "go get")
		fmt.Fprintln(os.Stderr, "Use 'godep go install' instead.")
		fmt.Fprintln(os.Stderr, "Run 'godep help go' for usage.")
		os.Exit(2)
	}
	c := exec.Command("go", args...)
	c.Env = append(envNoGopath(), "GOPATH="+gopath)
	c.Stdin = os.Stdin
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	err := c.Run()
	if err != nil {
		log.Fatalln("go", err)
	}
}

// prepareGopath reads dependency information from the filesystem
// entry name, fetches any necessary code, and returns a gopath
// causing the specified dependencies to be used.
func prepareGopath() (gopath string) {
	dir, isDir := findGodeps()
	if dir == "" {
		log.Fatalln("No Godeps found (or in any parent directory)")
	}
	if !isDir {
		log.Fatalln(strings.TrimSpace(needSource))
	}
	return filepath.Join(dir, "Godeps", "_workspace")
}

// findGodeps looks for a directory entry "Godeps" in the
// current directory or any parent, and returns the containing
// directory and whether the entry itself is a directory.
// If Godeps can't be found, findGodeps returns "".
// For any other error, it exits the program.
func findGodeps() (dir string, isDir bool) {
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalln(err)
	}
	return findInParents(wd, "Godeps")
}

// isRoot returns true iff a path is a root.
// On Unix: "/".
// On Windows: "C:\", "D:\", ...
func isRoot(p string) bool {
	p = filepath.Clean(p)
	volume := filepath.VolumeName(p)

	p = strings.TrimPrefix(p, volume)
	p = filepath.ToSlash(p)

	return p == "/"
}

// findInParents returns the path to the directory containing name
// in dir or any ancestor, and whether name itself is a directory.
// If name cannot be found, findInParents returns the empty string.
func findInParents(dir, name string) (container string, isDir bool) {
	for {
		fi, err := os.Stat(filepath.Join(dir, name))
		if os.IsNotExist(err) && isRoot(dir) {
			return "", false
		}
		if os.IsNotExist(err) {
			dir = filepath.Dir(dir)
			continue
		}
		if err != nil {
			log.Fatalln(err)
		}
		return dir, fi.IsDir()
	}
}

func envNoGopath() (a []string) {
	for _, s := range os.Environ() {
		if !strings.HasPrefix(s, "GOPATH=") {
			a = append(a, s)
		}
	}
	return a
}

const needSource = `
outdated Godeps missing source code

This dependency list was created with an old version of godep.

To work around this, you have two options:
1. Run 'godep restore', and try again.
2. Ask the maintainer to switch to a newer version of godep,
then try again with the updated package.
`
