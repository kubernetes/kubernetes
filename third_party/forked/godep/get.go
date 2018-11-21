package main

import (
	"log"
	"os"
	"os/exec"
)

var cmdGet = &Command{
	Name:  "get",
	Args:  "[-t] [packages]",
	Short: "download and install packages with specified dependencies",
	Long: `
Get downloads to GOPATH the packages named by the import paths, and installs
them with the dependencies specified in their Godeps files.

If any of the packages do not have Godeps files, those are installed
as if by go get.

If -t is given, dependencies of test files are also downloaded and installed.

For more about specifying packages, see 'go help packages'.
`,
	Run:          runGet,
	OnlyInGOPATH: true,
}

var getT bool

func init() {
	cmdGet.Flag.BoolVar(&getT, "t", false, "get test dependencies")
}

func runGet(cmd *Command, args []string) {
	if len(args) == 0 {
		args = []string{"."}
	}

	cmdArgs := []interface{}{"get", "-d"}
	if verbose {
		cmdArgs = append(cmdArgs, "-v")
	}

	if getT {
		cmdArgs = append(cmdArgs, "-t")
	}

	err := command("go", append(cmdArgs, args)...).Run()
	if err != nil {
		log.Fatalln(err)
	}

	// group import paths by Godeps location
	groups := make(map[string][]string)
	ps, err := LoadPackages(args...)
	if err != nil {
		log.Fatalln(err)
	}
	for _, pkg := range ps {
		if pkg.Error.Err != "" {
			log.Fatalln(pkg.Error.Err)
		}
		dir, _ := findInParents(pkg.Dir, "Godeps")
		groups[dir] = append(groups[dir], pkg.ImportPath)
	}
	for dir, packages := range groups {
		var c *exec.Cmd
		if dir == "" {
			c = command("go", "install", packages)
		} else {
			c = command("godep", "go", "install", packages)
			c.Dir = dir
		}
		if err := c.Run(); err != nil {
			log.Fatalln(err)
		}
	}
}

// command is like exec.Command, but the returned
// Cmd inherits stderr from the current process, and
// elements of args may be either string or []string.
func command(name string, args ...interface{}) *exec.Cmd {
	var a []string
	for _, arg := range args {
		switch v := arg.(type) {
		case string:
			a = append(a, v)
		case []string:
			a = append(a, v...)
		}
	}
	c := exec.Command(name, a...)
	c.Stderr = os.Stderr
	return c
}
