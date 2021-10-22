// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go build -o gotext.latest
//go:generate ./gotext.latest help gendocumentation
//go:generate rm gotext.latest

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/build"
	"go/format"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"sync"
	"text/template"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/message/pipeline"

	"golang.org/x/text/language"
	"golang.org/x/tools/go/buildutil"
)

func init() {
	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)
}

var (
	srcLang = flag.String("srclang", "en-US", "the source-code language")
	dir     = flag.String("dir", "locales", "default subdirectory to store translation files")
)

func config() (*pipeline.Config, error) {
	tag, err := language.Parse(*srcLang)
	if err != nil {
		return nil, wrap(err, "invalid srclang")
	}
	return &pipeline.Config{
		SourceLanguage:      tag,
		Supported:           getLangs(),
		TranslationsPattern: `messages\.(.*)\.json`,
		GenFile:             *out,
	}, nil
}

// NOTE: the Command struct is copied from the go tool in core.

// A Command is an implementation of a go command
// like go build or go fix.
type Command struct {
	// Run runs the command.
	// The args are the arguments after the command name.
	Run func(cmd *Command, c *pipeline.Config, args []string) error

	// UsageLine is the one-line usage message.
	// The first word in the line is taken to be the command name.
	UsageLine string

	// Short is the short description shown in the 'go help' output.
	Short string

	// Long is the long message shown in the 'go help <this-command>' output.
	Long string

	// Flag is a set of flags specific to this command.
	Flag flag.FlagSet
}

// Name returns the command's name: the first word in the usage line.
func (c *Command) Name() string {
	name := c.UsageLine
	i := strings.Index(name, " ")
	if i >= 0 {
		name = name[:i]
	}
	return name
}

func (c *Command) Usage() {
	fmt.Fprintf(os.Stderr, "usage: %s\n\n", c.UsageLine)
	fmt.Fprintf(os.Stderr, "%s\n", strings.TrimSpace(c.Long))
	os.Exit(2)
}

// Runnable reports whether the command can be run; otherwise
// it is a documentation pseudo-command such as importpath.
func (c *Command) Runnable() bool {
	return c.Run != nil
}

// Commands lists the available commands and help topics.
// The order here is the order in which they are printed by 'go help'.
var commands = []*Command{
	cmdUpdate,
	cmdExtract,
	cmdRewrite,
	cmdGenerate,
	// TODO:
	// - update: full-cycle update of extraction, sending, and integration
	// - report: report of freshness of translations
}

var exitStatus = 0
var exitMu sync.Mutex

func setExitStatus(n int) {
	exitMu.Lock()
	if exitStatus < n {
		exitStatus = n
	}
	exitMu.Unlock()
}

var origEnv []string

func main() {
	flag.Usage = usage
	flag.Parse()
	log.SetFlags(0)

	args := flag.Args()
	if len(args) < 1 {
		usage()
	}

	if args[0] == "help" {
		help(args[1:])
		return
	}

	for _, cmd := range commands {
		if cmd.Name() == args[0] && cmd.Runnable() {
			cmd.Flag.Usage = func() { cmd.Usage() }
			cmd.Flag.Parse(args[1:])
			args = cmd.Flag.Args()
			config, err := config()
			if err != nil {
				fatalf("gotext: %+v", err)
			}
			if err := cmd.Run(cmd, config, args); err != nil {
				fatalf("gotext: %+v", err)
			}
			exit()
			return
		}
	}

	fmt.Fprintf(os.Stderr, "gotext: unknown subcommand %q\nRun 'go help' for usage.\n", args[0])
	setExitStatus(2)
	exit()
}

var usageTemplate = `gotext is a tool for managing text in Go source code.

Usage:

	gotext command [arguments]

The commands are:
{{range .}}{{if .Runnable}}
	{{.Name | printf "%-11s"}} {{.Short}}{{end}}{{end}}

Use "gotext help [command]" for more information about a command.

Additional help topics:
{{range .}}{{if not .Runnable}}
	{{.Name | printf "%-11s"}} {{.Short}}{{end}}{{end}}

Use "gotext help [topic]" for more information about that topic.

`

var helpTemplate = `{{if .Runnable}}usage: gotext {{.UsageLine}}

{{end}}{{.Long | trim}}
`

var documentationTemplate = `{{range .}}{{if .Short}}{{.Short | capitalize}}

{{end}}{{if .Runnable}}Usage:

	gotext {{.UsageLine}}

{{end}}{{.Long | trim}}


{{end}}`

// commentWriter writes a Go comment to the underlying io.Writer,
// using line comment form (//).
type commentWriter struct {
	W            io.Writer
	wroteSlashes bool // Wrote "//" at the beginning of the current line.
}

func (c *commentWriter) Write(p []byte) (int, error) {
	var n int
	for i, b := range p {
		if !c.wroteSlashes {
			s := "//"
			if b != '\n' {
				s = "// "
			}
			if _, err := io.WriteString(c.W, s); err != nil {
				return n, err
			}
			c.wroteSlashes = true
		}
		n0, err := c.W.Write(p[i : i+1])
		n += n0
		if err != nil {
			return n, err
		}
		if b == '\n' {
			c.wroteSlashes = false
		}
	}
	return len(p), nil
}

// An errWriter wraps a writer, recording whether a write error occurred.
type errWriter struct {
	w   io.Writer
	err error
}

func (w *errWriter) Write(b []byte) (int, error) {
	n, err := w.w.Write(b)
	if err != nil {
		w.err = err
	}
	return n, err
}

// tmpl executes the given template text on data, writing the result to w.
func tmpl(w io.Writer, text string, data interface{}) {
	t := template.New("top")
	t.Funcs(template.FuncMap{"trim": strings.TrimSpace, "capitalize": capitalize})
	template.Must(t.Parse(text))
	ew := &errWriter{w: w}
	err := t.Execute(ew, data)
	if ew.err != nil {
		// I/O error writing. Ignore write on closed pipe.
		if strings.Contains(ew.err.Error(), "pipe") {
			os.Exit(1)
		}
		fatalf("writing output: %v", ew.err)
	}
	if err != nil {
		panic(err)
	}
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	r, n := utf8.DecodeRuneInString(s)
	return string(unicode.ToTitle(r)) + s[n:]
}

func printUsage(w io.Writer) {
	bw := bufio.NewWriter(w)
	tmpl(bw, usageTemplate, commands)
	bw.Flush()
}

func usage() {
	printUsage(os.Stderr)
	os.Exit(2)
}

// help implements the 'help' command.
func help(args []string) {
	if len(args) == 0 {
		printUsage(os.Stdout)
		// not exit 2: succeeded at 'go help'.
		return
	}
	if len(args) != 1 {
		fmt.Fprintf(os.Stderr, "usage: go help command\n\nToo many arguments given.\n")
		os.Exit(2) // failed at 'go help'
	}

	arg := args[0]

	// 'go help documentation' generates doc.go.
	if strings.HasSuffix(arg, "documentation") {
		w := &bytes.Buffer{}

		fmt.Fprintln(w, "// Code generated by go generate. DO NOT EDIT.")
		fmt.Fprintln(w)
		buf := new(bytes.Buffer)
		printUsage(buf)
		usage := &Command{Long: buf.String()}
		tmpl(&commentWriter{W: w}, documentationTemplate, append([]*Command{usage}, commands...))
		fmt.Fprintln(w, "package main")
		if arg == "gendocumentation" {
			b, err := format.Source(w.Bytes())
			if err != nil {
				logf("Could not format generated docs: %v\n", err)
			}
			if err := ioutil.WriteFile("doc.go", b, 0666); err != nil {
				logf("Could not create file alldocs.go: %v\n", err)
			}
		} else {
			fmt.Println(w.String())
		}
		return
	}

	for _, cmd := range commands {
		if cmd.Name() == arg {
			tmpl(os.Stdout, helpTemplate, cmd)
			// not exit 2: succeeded at 'go help cmd'.
			return
		}
	}

	fmt.Fprintf(os.Stderr, "Unknown help topic %#q.  Run 'gotext help'.\n", arg)
	os.Exit(2) // failed at 'go help cmd'
}

func getLangs() (tags []language.Tag) {
	for _, t := range strings.Split(*lang, ",") {
		if t == "" {
			continue
		}
		tag, err := language.Parse(t)
		if err != nil {
			fatalf("gotext: could not parse language %q: %v", t, err)
		}
		tags = append(tags, tag)
	}
	return tags
}

var atexitFuncs []func()

func atexit(f func()) {
	atexitFuncs = append(atexitFuncs, f)
}

func exit() {
	for _, f := range atexitFuncs {
		f()
	}
	os.Exit(exitStatus)
}

func fatalf(format string, args ...interface{}) {
	logf(format, args...)
	exit()
}

func logf(format string, args ...interface{}) {
	log.Printf(format, args...)
	setExitStatus(1)
}

func exitIfErrors() {
	if exitStatus != 0 {
		exit()
	}
}
