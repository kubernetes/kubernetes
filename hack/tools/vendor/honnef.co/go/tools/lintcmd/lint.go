package lintcmd

import (
	"crypto/sha256"
	"fmt"
	"go/build"
	"go/token"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"honnef.co/go/tools/analysis/lint"
	"honnef.co/go/tools/config"
	"honnef.co/go/tools/go/loader"
	"honnef.co/go/tools/internal/cache"
	"honnef.co/go/tools/lintcmd/runner"
	"honnef.co/go/tools/unused"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
)

// A linter lints Go source code.
type linter struct {
	Checkers []*lint.Analyzer
	Config   config.Config
	Runner   *runner.Runner
}

func computeSalt() ([]byte, error) {
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

func newLinter(cfg config.Config) (*linter, error) {
	c, err := cache.Default()
	if err != nil {
		return nil, err
	}
	salt, err := computeSalt()
	if err != nil {
		return nil, fmt.Errorf("could not compute salt for cache: %s", err)
	}
	c.SetSalt(salt)
	r, err := runner.New(cfg, c)
	if err != nil {
		return nil, err
	}
	r.FallbackGoVersion = defaultGoVersion()
	return &linter{
		Config: cfg,
		Runner: r,
	}, nil
}

func (l *linter) Lint(cfg *packages.Config, patterns []string) (problems []problem, warnings []string, err error) {
	cs := make([]*analysis.Analyzer, len(l.Checkers))
	for i, a := range l.Checkers {
		cs[i] = a.Analyzer
	}
	results, err := l.Runner.Run(cfg, cs, patterns)
	if err != nil {
		return nil, nil, err
	}

	if len(results) == 0 && err == nil {
		// TODO(dh): emulate Go's behavior more closely once we have
		// access to go list's Match field.
		for _, pattern := range patterns {
			fmt.Fprintf(os.Stderr, "warning: %q matched no packages\n", pattern)
		}
	}

	analyzerNames := make([]string, len(l.Checkers))
	for i, a := range l.Checkers {
		analyzerNames[i] = a.Analyzer.Name
	}

	used := map[unusedKey]bool{}
	var unuseds []unusedPair
	for _, res := range results {
		if len(res.Errors) > 0 && !res.Failed {
			panic("package has errors but isn't marked as failed")
		}
		if res.Failed {
			problems = append(problems, failed(res)...)
		} else {
			if res.Skipped {
				warnings = append(warnings, fmt.Sprintf("skipped package %s because it is too large", res.Package))
				continue
			}

			if !res.Initial {
				continue
			}

			allowedAnalyzers := filterAnalyzerNames(analyzerNames, res.Config.Checks)
			resd, err := res.Load()
			if err != nil {
				return nil, nil, err
			}
			ps := success(allowedAnalyzers, resd)
			filtered, err := filterIgnored(ps, resd, allowedAnalyzers)
			if err != nil {
				return nil, nil, err
			}
			problems = append(problems, filtered...)

			for _, obj := range resd.Unused.Used {
				// FIXME(dh): pick the object whose filename does not include $GOROOT
				key := unusedKey{
					pkgPath: res.Package.PkgPath,
					base:    filepath.Base(obj.Position.Filename),
					line:    obj.Position.Line,
					name:    obj.Name,
				}
				used[key] = true
			}

			if allowedAnalyzers["U1000"] {
				for _, obj := range resd.Unused.Unused {
					key := unusedKey{
						pkgPath: res.Package.PkgPath,
						base:    filepath.Base(obj.Position.Filename),
						line:    obj.Position.Line,
						name:    obj.Name,
					}
					unuseds = append(unuseds, unusedPair{key, obj})
					if _, ok := used[key]; !ok {
						used[key] = false
					}
				}
			}
		}
	}

	for _, uo := range unuseds {
		if used[uo.key] {
			continue
		}
		if uo.obj.InGenerated {
			continue
		}
		problems = append(problems, problem{
			Diagnostic: runner.Diagnostic{
				Position: uo.obj.DisplayPosition,
				Message:  fmt.Sprintf("%s %s is unused", uo.obj.Kind, uo.obj.Name),
				Category: "U1000",
			},
		})
	}

	if len(problems) == 0 {
		return nil, warnings, nil
	}

	sort.Slice(problems, func(i, j int) bool {
		pi := problems[i].Position
		pj := problems[j].Position

		if pi.Filename != pj.Filename {
			return pi.Filename < pj.Filename
		}
		if pi.Line != pj.Line {
			return pi.Line < pj.Line
		}
		if pi.Column != pj.Column {
			return pi.Column < pj.Column
		}

		return problems[i].Message < problems[j].Message
	})

	var out []problem
	out = append(out, problems[0])
	for i, p := range problems[1:] {
		// We may encounter duplicate problems because one file
		// can be part of many packages.
		if !problems[i].equal(p) {
			out = append(out, p)
		}
	}
	return out, warnings, nil
}

func filterIgnored(problems []problem, res runner.ResultData, allowedAnalyzers map[string]bool) ([]problem, error) {
	couldHaveMatched := func(ig *lineIgnore) bool {
		for _, c := range ig.Checks {
			if c == "U1000" {
				// We never want to flag ignores for U1000,
				// because U1000 isn't local to a single
				// package. For example, an identifier may
				// only be used by tests, in which case an
				// ignore would only fire when not analyzing
				// tests. To avoid spurious "useless ignore"
				// warnings, just never flag U1000.
				return false
			}

			// Even though the runner always runs all analyzers, we
			// still only flag unmatched ignores for the set of
			// analyzers the user has expressed interest in. That way,
			// `staticcheck -checks=SA1000` won't complain about an
			// unmatched ignore for an unrelated check.
			if allowedAnalyzers[c] {
				return true
			}
		}

		return false
	}

	ignores, moreProblems := parseDirectives(res.Directives)

	for _, ig := range ignores {
		for i := range problems {
			p := &problems[i]
			if ig.Match(*p) {
				p.Severity = severityIgnored
			}
		}

		if ig, ok := ig.(*lineIgnore); ok && !ig.Matched && couldHaveMatched(ig) {
			p := problem{
				Diagnostic: runner.Diagnostic{
					Position: ig.Pos,
					Message:  "this linter directive didn't match anything; should it be removed?",
					Category: "staticcheck",
				},
			}
			moreProblems = append(moreProblems, p)
		}
	}

	return append(problems, moreProblems...), nil
}

type ignore interface {
	Match(p problem) bool
}

type lineIgnore struct {
	File    string
	Line    int
	Checks  []string
	Matched bool
	Pos     token.Position
}

func (li *lineIgnore) Match(p problem) bool {
	pos := p.Position
	if pos.Filename != li.File || pos.Line != li.Line {
		return false
	}
	for _, c := range li.Checks {
		if m, _ := filepath.Match(c, p.Category); m {
			li.Matched = true
			return true
		}
	}
	return false
}

func (li *lineIgnore) String() string {
	matched := "not matched"
	if li.Matched {
		matched = "matched"
	}
	return fmt.Sprintf("%s:%d %s (%s)", li.File, li.Line, strings.Join(li.Checks, ", "), matched)
}

type fileIgnore struct {
	File   string
	Checks []string
}

func (fi *fileIgnore) Match(p problem) bool {
	if p.Position.Filename != fi.File {
		return false
	}
	for _, c := range fi.Checks {
		if m, _ := filepath.Match(c, p.Category); m {
			return true
		}
	}
	return false
}

type severity uint8

const (
	severityError severity = iota
	severityWarning
	severityIgnored
)

func (s severity) String() string {
	switch s {
	case severityError:
		return "error"
	case severityWarning:
		return "warning"
	case severityIgnored:
		return "ignored"
	default:
		return fmt.Sprintf("Severity(%d)", s)
	}
}

// problem represents a problem in some source code.
type problem struct {
	runner.Diagnostic
	Severity severity
}

func (p problem) equal(o problem) bool {
	return p.Position == o.Position &&
		p.End == o.End &&
		p.Message == o.Message &&
		p.Category == o.Category &&
		p.Severity == o.Severity
}

func (p *problem) String() string {
	return fmt.Sprintf("%s (%s)", p.Message, p.Category)
}

func failed(res runner.Result) []problem {
	var problems []problem

	for _, e := range res.Errors {
		switch e := e.(type) {
		case packages.Error:
			msg := e.Msg
			if len(msg) != 0 && msg[0] == '\n' {
				// TODO(dh): See https://github.com/golang/go/issues/32363
				msg = msg[1:]
			}

			var posn token.Position
			if e.Pos == "" {
				// Under certain conditions (malformed package
				// declarations, multiple packages in the same
				// directory), go list emits an error on stderr
				// instead of JSON. Those errors do not have
				// associated position information in
				// go/packages.Error, even though the output on
				// stderr may contain it.
				if p, n, err := parsePos(msg); err == nil {
					if abs, err := filepath.Abs(p.Filename); err == nil {
						p.Filename = abs
					}
					posn = p
					msg = msg[n+2:]
				}
			} else {
				var err error
				posn, _, err = parsePos(e.Pos)
				if err != nil {
					panic(fmt.Sprintf("internal error: %s", e))
				}
			}
			p := problem{
				Diagnostic: runner.Diagnostic{
					Position: posn,
					Message:  msg,
					Category: "compile",
				},
				Severity: severityError,
			}
			problems = append(problems, p)
		case error:
			p := problem{
				Diagnostic: runner.Diagnostic{
					Position: token.Position{},
					Message:  e.Error(),
					Category: "compile",
				},
				Severity: severityError,
			}
			problems = append(problems, p)
		}
	}

	return problems
}

type unusedKey struct {
	pkgPath string
	base    string
	line    int
	name    string
}

type unusedPair struct {
	key unusedKey
	obj unused.SerializedObject
}

func success(allowedChecks map[string]bool, res runner.ResultData) []problem {
	diags := res.Diagnostics
	var problems []problem
	for _, diag := range diags {
		if !allowedChecks[diag.Category] {
			continue
		}
		problems = append(problems, problem{Diagnostic: diag})
	}
	return problems
}

func defaultGoVersion() string {
	tags := build.Default.ReleaseTags
	v := tags[len(tags)-1][2:]
	return v
}

func filterAnalyzerNames(analyzers []string, checks []string) map[string]bool {
	allowedChecks := map[string]bool{}

	for _, check := range checks {
		b := true
		if len(check) > 1 && check[0] == '-' {
			b = false
			check = check[1:]
		}
		if check == "*" || check == "all" {
			// Match all
			for _, c := range analyzers {
				allowedChecks[c] = b
			}
		} else if strings.HasSuffix(check, "*") {
			// Glob
			prefix := check[:len(check)-1]
			isCat := strings.IndexFunc(prefix, func(r rune) bool { return unicode.IsNumber(r) }) == -1

			for _, a := range analyzers {
				idx := strings.IndexFunc(a, func(r rune) bool { return unicode.IsNumber(r) })
				if isCat {
					// Glob is S*, which should match S1000 but not SA1000
					cat := a[:idx]
					if prefix == cat {
						allowedChecks[a] = b
					}
				} else {
					// Glob is S1*
					if strings.HasPrefix(a, prefix) {
						allowedChecks[a] = b
					}
				}
			}
		} else {
			// Literal check name
			allowedChecks[check] = b
		}
	}
	return allowedChecks
}

var posRe = regexp.MustCompile(`^(.+?):(\d+)(?::(\d+)?)?`)

func parsePos(pos string) (token.Position, int, error) {
	if pos == "-" || pos == "" {
		return token.Position{}, 0, nil
	}
	parts := posRe.FindStringSubmatch(pos)
	if parts == nil {
		return token.Position{}, 0, fmt.Errorf("internal error: malformed position %q", pos)
	}
	file := parts[1]
	line, _ := strconv.Atoi(parts[2])
	col, _ := strconv.Atoi(parts[3])
	return token.Position{
		Filename: file,
		Line:     line,
		Column:   col,
	}, len(parts[0]), nil
}

type options struct {
	Config config.Config

	Tags                     string
	LintTests                bool
	GoVersion                string
	PrintAnalyzerMeasurement func(analysis *analysis.Analyzer, pkg *loader.PackageSpec, d time.Duration)
}

func doLint(cs []*lint.Analyzer, paths []string, opt *options) ([]problem, []string, error) {
	if opt == nil {
		opt = &options{}
	}

	l, err := newLinter(opt.Config)
	if err != nil {
		return nil, nil, err
	}
	l.Checkers = cs
	l.Runner.GoVersion = opt.GoVersion
	l.Runner.Stats.PrintAnalyzerMeasurement = opt.PrintAnalyzerMeasurement

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
		switch l.Runner.Stats.State() {
		case runner.StateInitializing:
			fmt.Fprintln(os.Stderr, "Status: initializing")
		case runner.StateLoadPackageGraph:
			fmt.Fprintln(os.Stderr, "Status: loading package graph")
		case runner.StateBuildActionGraph:
			fmt.Fprintln(os.Stderr, "Status: building action graph")
		case runner.StateProcessing:
			fmt.Fprintf(os.Stderr, "Packages: %d/%d initial, %d/%d total; Workers: %d/%d\n",
				l.Runner.Stats.ProcessedInitialPackages(),
				l.Runner.Stats.InitialPackages(),
				l.Runner.Stats.ProcessedPackages(),
				l.Runner.Stats.TotalPackages(),
				l.Runner.ActiveWorkers(),
				l.Runner.TotalWorkers(),
			)
		case runner.StateFinalizing:
			fmt.Fprintln(os.Stderr, "Status: finalizing")
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
	return l.Lint(cfg, paths)
}
