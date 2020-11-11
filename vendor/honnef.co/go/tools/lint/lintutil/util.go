// Copyright (c) 2013 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd.

// Package lintutil provides helpers for writing linter command lines.
package lintutil // import "honnef.co/go/tools/lint/lintutil"

import (
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"go/token"
	"io"
	"log"
	"os"
	"os/signal"
	"regexp"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"honnef.co/go/tools/config"
	"honnef.co/go/tools/internal/cache"
	"honnef.co/go/tools/lint"
	"honnef.co/go/tools/lint/lintutil/format"
	"honnef.co/go/tools/version"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/packages"
)

func NewVersionFlag() flag.Getter {
	tags := build.Default.ReleaseTags
	v := tags[len(tags)-1][2:]
	version := new(VersionFlag)
	if err := version.Set(v); err != nil {
		panic(fmt.Sprintf("internal error: %s", err))
	}
	return version
}

type VersionFlag int

func (v *VersionFlag) String() string {
	return fmt.Sprintf("1.%d", *v)

}

func (v *VersionFlag) Set(s string) error {
	if len(s) < 3 {
		return errors.New("invalid Go version")
	}
	if s[0] != '1' {
		return errors.New("invalid Go version")
	}
	if s[1] != '.' {
		return errors.New("invalid Go version")
	}
	i, err := strconv.Atoi(s[2:])
	*v = VersionFlag(i)
	return err
}

func (v *VersionFlag) Get() interface{} {
	return int(*v)
}

func usage(name string, flags *flag.FlagSet) func() {
	return func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", name)
		fmt.Fprintf(os.Stderr, "\t%s [flags] # runs on package in current directory\n", name)
		fmt.Fprintf(os.Stderr, "\t%s [flags] packages\n", name)
		fmt.Fprintf(os.Stderr, "\t%s [flags] directory\n", name)
		fmt.Fprintf(os.Stderr, "\t%s [flags] files... # must be a single package\n", name)
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flags.PrintDefaults()
	}
}

type list []string

func (list *list) String() string {
	return `"` + strings.Join(*list, ",") + `"`
}

func (list *list) Set(s string) error {
	if s == "" {
		*list = nil
		return nil
	}

	*list = strings.Split(s, ",")
	return nil
}

func FlagSet(name string) *flag.FlagSet {
	flags := flag.NewFlagSet("", flag.ExitOnError)
	flags.Usage = usage(name, flags)
	flags.String("tags", "", "List of `build tags`")
	flags.Bool("tests", true, "Include tests")
	flags.Bool("version", false, "Print version and exit")
	flags.Bool("show-ignored", false, "Don't filter ignored problems")
	flags.String("f", "text", "Output `format` (valid choices are 'stylish', 'text' and 'json')")
	flags.String("explain", "", "Print description of `check`")

	flags.String("debug.cpuprofile", "", "Write CPU profile to `file`")
	flags.String("debug.memprofile", "", "Write memory profile to `file`")
	flags.Bool("debug.version", false, "Print detailed version information about this program")
	flags.Bool("debug.no-compile-errors", false, "Don't print compile errors")
	flags.String("debug.measure-analyzers", "", "Write analysis measurements to `file`. `file` will be opened for appending if it already exists.")
	flags.Uint("debug.repeat-analyzers", 0, "Run analyzers `num` times")

	checks := list{"inherit"}
	fail := list{"all"}
	flags.Var(&checks, "checks", "Comma-separated list of `checks` to enable.")
	flags.Var(&fail, "fail", "Comma-separated list of `checks` that can cause a non-zero exit status.")

	tags := build.Default.ReleaseTags
	v := tags[len(tags)-1][2:]
	version := new(VersionFlag)
	if err := version.Set(v); err != nil {
		panic(fmt.Sprintf("internal error: %s", err))
	}

	flags.Var(version, "go", "Target Go `version` in the format '1.x'")
	return flags
}

func findCheck(cs []*analysis.Analyzer, check string) (*analysis.Analyzer, bool) {
	for _, c := range cs {
		if c.Name == check {
			return c, true
		}
	}
	return nil, false
}

func ProcessFlagSet(cs []*analysis.Analyzer, cums []lint.CumulativeChecker, fs *flag.FlagSet) {
	tags := fs.Lookup("tags").Value.(flag.Getter).Get().(string)
	tests := fs.Lookup("tests").Value.(flag.Getter).Get().(bool)
	goVersion := fs.Lookup("go").Value.(flag.Getter).Get().(int)
	formatter := fs.Lookup("f").Value.(flag.Getter).Get().(string)
	printVersion := fs.Lookup("version").Value.(flag.Getter).Get().(bool)
	showIgnored := fs.Lookup("show-ignored").Value.(flag.Getter).Get().(bool)
	explain := fs.Lookup("explain").Value.(flag.Getter).Get().(string)

	cpuProfile := fs.Lookup("debug.cpuprofile").Value.(flag.Getter).Get().(string)
	memProfile := fs.Lookup("debug.memprofile").Value.(flag.Getter).Get().(string)
	debugVersion := fs.Lookup("debug.version").Value.(flag.Getter).Get().(bool)
	debugNoCompile := fs.Lookup("debug.no-compile-errors").Value.(flag.Getter).Get().(bool)
	debugRepeat := fs.Lookup("debug.repeat-analyzers").Value.(flag.Getter).Get().(uint)

	var measureAnalyzers func(analysis *analysis.Analyzer, pkg *lint.Package, d time.Duration)
	if path := fs.Lookup("debug.measure-analyzers").Value.(flag.Getter).Get().(string); path != "" {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			log.Fatal(err)
		}

		mu := &sync.Mutex{}
		measureAnalyzers = func(analysis *analysis.Analyzer, pkg *lint.Package, d time.Duration) {
			mu.Lock()
			defer mu.Unlock()
			if _, err := fmt.Fprintf(f, "%s\t%s\t%d\n", analysis.Name, pkg.ID, d.Nanoseconds()); err != nil {
				log.Println("error writing analysis measurements:", err)
			}
		}
	}

	cfg := config.Config{}
	cfg.Checks = *fs.Lookup("checks").Value.(*list)

	exit := func(code int) {
		if cpuProfile != "" {
			pprof.StopCPUProfile()
		}
		if memProfile != "" {
			f, err := os.Create(memProfile)
			if err != nil {
				panic(err)
			}
			runtime.GC()
			pprof.WriteHeapProfile(f)
		}
		os.Exit(code)
	}
	if cpuProfile != "" {
		f, err := os.Create(cpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
	}

	if debugVersion {
		version.Verbose()
		exit(0)
	}

	if printVersion {
		version.Print()
		exit(0)
	}

	// Validate that the tags argument is well-formed. go/packages
	// doesn't detect malformed build flags and returns unhelpful
	// errors.
	tf := buildutil.TagsFlag{}
	if err := tf.Set(tags); err != nil {
		fmt.Fprintln(os.Stderr, fmt.Errorf("invalid value %q for flag -tags: %s", tags, err))
		exit(1)
	}

	if explain != "" {
		var haystack []*analysis.Analyzer
		haystack = append(haystack, cs...)
		for _, cum := range cums {
			haystack = append(haystack, cum.Analyzer())
		}
		check, ok := findCheck(haystack, explain)
		if !ok {
			fmt.Fprintln(os.Stderr, "Couldn't find check", explain)
			exit(1)
		}
		if check.Doc == "" {
			fmt.Fprintln(os.Stderr, explain, "has no documentation")
			exit(1)
		}
		fmt.Println(check.Doc)
		exit(0)
	}

	ps, err := Lint(cs, cums, fs.Args(), &Options{
		Tags:                     tags,
		LintTests:                tests,
		GoVersion:                goVersion,
		Config:                   cfg,
		PrintAnalyzerMeasurement: measureAnalyzers,
		RepeatAnalyzers:          debugRepeat,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		exit(1)
	}

	var f format.Formatter
	switch formatter {
	case "text":
		f = format.Text{W: os.Stdout}
	case "stylish":
		f = &format.Stylish{W: os.Stdout}
	case "json":
		f = format.JSON{W: os.Stdout}
	default:
		fmt.Fprintf(os.Stderr, "unsupported output format %q\n", formatter)
		exit(2)
	}

	var (
		total    int
		errors   int
		warnings int
		ignored  int
	)

	fail := *fs.Lookup("fail").Value.(*list)
	analyzers := make([]*analysis.Analyzer, len(cs), len(cs)+len(cums))
	copy(analyzers, cs)
	for _, cum := range cums {
		analyzers = append(analyzers, cum.Analyzer())
	}
	shouldExit := lint.FilterChecks(analyzers, fail)
	shouldExit["compile"] = true

	total = len(ps)
	for _, p := range ps {
		if p.Check == "compile" && debugNoCompile {
			continue
		}
		if p.Severity == lint.Ignored && !showIgnored {
			ignored++
			continue
		}
		if shouldExit[p.Check] {
			errors++
		} else {
			p.Severity = lint.Warning
			warnings++
		}
		f.Format(p)
	}
	if f, ok := f.(format.Statter); ok {
		f.Stats(total, errors, warnings, ignored)
	}
	if errors > 0 {
		exit(1)
	}
	exit(0)
}

type Options struct {
	Config config.Config

	Tags                     string
	LintTests                bool
	GoVersion                int
	PrintAnalyzerMeasurement func(analysis *analysis.Analyzer, pkg *lint.Package, d time.Duration)
	RepeatAnalyzers          uint
}

func computeSalt() ([]byte, error) {
	if version.Version != "devel" {
		return []byte(version.Version), nil
	}
	p, err := os.Executable()
	if err != nil {
		return nil, err
	}
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nil, err
	}
	return h.Sum(nil), nil
}

func Lint(cs []*analysis.Analyzer, cums []lint.CumulativeChecker, paths []string, opt *Options) ([]lint.Problem, error) {
	salt, err := computeSalt()
	if err != nil {
		return nil, fmt.Errorf("could not compute salt for cache: %s", err)
	}
	cache.SetSalt(salt)

	if opt == nil {
		opt = &Options{}
	}

	l := &lint.Linter{
		Checkers:           cs,
		CumulativeCheckers: cums,
		GoVersion:          opt.GoVersion,
		Config:             opt.Config,
		RepeatAnalyzers:    opt.RepeatAnalyzers,
	}
	l.Stats.PrintAnalyzerMeasurement = opt.PrintAnalyzerMeasurement
	cfg := &packages.Config{}
	if opt.LintTests {
		cfg.Tests = true
	}
	if opt.Tags != "" {
		cfg.BuildFlags = append(cfg.BuildFlags, "-tags", opt.Tags)
	}

	printStats := func() {
		// Individual stats are read atomically, but overall there
		// is no synchronisation. For printing rough progress
		// information, this doesn't matter.
		switch atomic.LoadUint32(&l.Stats.State) {
		case lint.StateInitializing:
			fmt.Fprintln(os.Stderr, "Status: initializing")
		case lint.StateGraph:
			fmt.Fprintln(os.Stderr, "Status: loading package graph")
		case lint.StateProcessing:
			fmt.Fprintf(os.Stderr, "Packages: %d/%d initial, %d/%d total; Workers: %d/%d; Problems: %d\n",
				atomic.LoadUint32(&l.Stats.ProcessedInitialPackages),
				atomic.LoadUint32(&l.Stats.InitialPackages),
				atomic.LoadUint32(&l.Stats.ProcessedPackages),
				atomic.LoadUint32(&l.Stats.TotalPackages),
				atomic.LoadUint32(&l.Stats.ActiveWorkers),
				atomic.LoadUint32(&l.Stats.TotalWorkers),
				atomic.LoadUint32(&l.Stats.Problems),
			)
		case lint.StateCumulative:
			fmt.Fprintln(os.Stderr, "Status: processing cumulative checkers")
		}
	}
	if len(infoSignals) > 0 {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, infoSignals...)
		defer signal.Stop(ch)
		go func() {
			for range ch {
				printStats()
			}
		}()
	}

	ps, err := l.Lint(cfg, paths)
	return ps, err
}

var posRe = regexp.MustCompile(`^(.+?):(\d+)(?::(\d+)?)?$`)

func parsePos(pos string) token.Position {
	if pos == "-" || pos == "" {
		return token.Position{}
	}
	parts := posRe.FindStringSubmatch(pos)
	if parts == nil {
		panic(fmt.Sprintf("internal error: malformed position %q", pos))
	}
	file := parts[1]
	line, _ := strconv.Atoi(parts[2])
	col, _ := strconv.Atoi(parts[3])
	return token.Position{
		Filename: file,
		Line:     line,
		Column:   col,
	}
}

func InitializeAnalyzers(docs map[string]*lint.Documentation, analyzers map[string]*analysis.Analyzer) map[string]*analysis.Analyzer {
	out := make(map[string]*analysis.Analyzer, len(analyzers))
	for k, v := range analyzers {
		vc := *v
		out[k] = &vc

		vc.Name = k
		doc, ok := docs[k]
		if !ok {
			panic(fmt.Sprintf("missing documentation for check %s", k))
		}
		vc.Doc = doc.String()
		if vc.Flags.Usage == nil {
			fs := flag.NewFlagSet("", flag.PanicOnError)
			fs.Var(NewVersionFlag(), "go", "Target Go version")
			vc.Flags = *fs
		}
	}
	return out
}
