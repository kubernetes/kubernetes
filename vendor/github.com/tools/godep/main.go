package main

import (
	"flag"
	"fmt"
	"go/build"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"text/template"
)

var (
	cpuprofile       string
	verbose          bool // Verbose flag for commands that support it
	debug            bool // Debug flag for commands that support it
	majorGoVersion   string
	VendorExperiment bool
	sep              string
)

// Command is an implementation of a godep command
// like godep save or godep go.
type Command struct {
	// Run runs the command.
	// The args are the arguments after the command name.
	Run func(cmd *Command, args []string)

	// Name of the command
	Name string

	// Args the command would expect
	Args string

	// Short is the short description shown in the 'godep help' output.
	Short string

	// Long is the long message shown in the
	// 'godep help <this-command>' output.
	Long string

	// Flag is a set of flags specific to this command.
	Flag flag.FlagSet

	// OnlyInGOPATH limits this command to being run only while inside of a GOPATH
	OnlyInGOPATH bool
}

// UsageExit prints usage information and exits.
func (c *Command) UsageExit() {
	fmt.Fprintf(os.Stderr, "Args: godep %s [-v] [-d] %s\n\n", c.Name, c.Args)
	fmt.Fprintf(os.Stderr, "Run 'godep help %s' for help.\n", c.Name)
	os.Exit(2)
}

// Commands lists the available commands and help topics.
// The order here is the order in which they are printed
// by 'godep help'.
var commands = []*Command{
	cmdSave,
	cmdGo,
	cmdGet,
	cmdPath,
	cmdRestore,
	cmdUpdate,
	cmdDiff,
	cmdVersion,
}

// VendorExperiment is the Go 1.5 vendor directory experiment flag, see
// https://github.com/golang/go/commit/183cc0cd41f06f83cb7a2490a499e3f9101befff
// Honor the env var unless the project already has an old school godep workspace
func determineVendor(v string) bool {
	go15ve := os.Getenv("GO15VENDOREXPERIMENT")
	var ev bool
	switch v {
	case "go1", "go1.1", "go1.2", "go1.3", "go1.4":
		ev = false
	case "go1.5":
		ev = go15ve == "1"
	case "go1.6":
		ev = go15ve != "0"
	default: //go1.7+, devel*
		ev = true
	}

	ws := filepath.Join("Godeps", "_workspace")
	s, err := os.Stat(ws)
	if err == nil && s.IsDir() {
		log.Printf("WARNING: Godep workspaces (./Godeps/_workspace) are deprecated and support for them will be removed when go1.8 is released.")
		if ev {
			log.Printf("WARNING: Go version (%s) & $GO15VENDOREXPERIMENT=%s wants to enable the vendor experiment, but disabling because a Godep workspace (%s) exists\n", v, go15ve, ws)
		}
		return false
	}

	return ev
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("godep: ")
	log.SetOutput(os.Stderr)

	flag.Usage = usageExit
	flag.Parse()
	args := flag.Args()
	if len(args) < 1 {
		usageExit()
	}

	if args[0] == "help" {
		help(args[1:])
		return
	}

	var err error
	majorGoVersion, err = goVersion()
	if err != nil {
		log.Fatal(err)
	}

	for _, cmd := range commands {
		if cmd.Name == args[0] {
			if cmd.OnlyInGOPATH {
				checkInGOPATH()
			}

			VendorExperiment = determineVendor(majorGoVersion)
			// sep is the signature set of path elements that
			// precede the original path of an imported package.
			sep = defaultSep(VendorExperiment)

			cmd.Flag.BoolVar(&verbose, "v", false, "enable verbose output")
			cmd.Flag.BoolVar(&debug, "d", false, "enable debug output")
			cmd.Flag.StringVar(&cpuprofile, "cpuprofile", "", "Write cpu profile to this file")
			cmd.Flag.Usage = func() { cmd.UsageExit() }
			cmd.Flag.Parse(args[1:])

			debugln("versionString()", versionString())
			debugln("majorGoVersion", majorGoVersion)
			debugln("VendorExperiment", VendorExperiment)
			debugln("sep", sep)

			if cpuprofile != "" {
				f, err := os.Create(cpuprofile)
				if err != nil {
					log.Fatal(err)
				}
				pprof.StartCPUProfile(f)
				defer pprof.StopCPUProfile()
			}
			cmd.Run(cmd, cmd.Flag.Args())
			return
		}
	}

	fmt.Fprintf(os.Stderr, "godep: unknown command %q\n", args[0])
	fmt.Fprintf(os.Stderr, "Run 'godep help' for usage.\n")
	os.Exit(2)
}

func subPath(sub, path string) bool {
	ls := strings.ToLower(sub)
	lp := strings.ToLower(path)
	if ls == lp {
		return false
	}
	return strings.HasPrefix(ls, lp)
}

func checkInGOPATH() {
	pwd, err := os.Getwd()
	if err != nil {
		log.Fatal("Unable to determine current working directory", err)
	}
	dirs := build.Default.SrcDirs()
	for _, p := range dirs {
		if ok := subPath(pwd, p); ok {
			return
		}
	}

	log.Println("[WARNING]: godep should only be used inside a valid go package directory and")
	log.Println("[WARNING]: may not function correctly. You are probably outside of your $GOPATH.")
	log.Printf("[WARNING]:\tCurrent Directory: %s\n", pwd)
	log.Printf("[WARNING]:\t$GOPATH: %s\n", os.Getenv("GOPATH"))
}

var usageTemplate = `
Godep is a tool for managing Go package dependencies.

Usage:

	godep command [arguments]

The commands are:
{{range .}}
    {{.Name | printf "%-8s"}} {{.Short}}{{end}}

Use "godep help [command]" for more information about a command.
`

var helpTemplate = `
Args: godep {{.Name}} [-v] [-d] {{.Args}}

{{.Long | trim}}

If -v is given, verbose output is enabled.

If -d is given, debug output is enabled (you probably don't want this, see -v).

`

func help(args []string) {
	if len(args) == 0 {
		printUsage(os.Stdout)
		return
	}
	if len(args) != 1 {
		fmt.Fprintf(os.Stderr, "usage: godep help command\n\n")
		fmt.Fprintf(os.Stderr, "Too many arguments given.\n")
		os.Exit(2)
	}
	for _, cmd := range commands {
		if cmd.Name == args[0] {
			tmpl(os.Stdout, helpTemplate, cmd)
			return
		}
	}
}

func usageExit() {
	printUsage(os.Stderr)
	os.Exit(2)
}

func printUsage(w io.Writer) {
	tmpl(w, usageTemplate, commands)
}

// tmpl executes the given template text on data, writing the result to w.
func tmpl(w io.Writer, text string, data interface{}) {
	t := template.New("top")
	t.Funcs(template.FuncMap{
		"trim": strings.TrimSpace,
	})
	template.Must(t.Parse(strings.TrimSpace(text) + "\n\n"))
	if err := t.Execute(w, data); err != nil {
		panic(err)
	}
}
