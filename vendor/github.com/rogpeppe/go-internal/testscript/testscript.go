// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Script-driven tests.
// See testdata/script/README for an overview.

package testscript

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/rogpeppe/go-internal/imports"
	"github.com/rogpeppe/go-internal/internal/misspell"
	"github.com/rogpeppe/go-internal/internal/os/execpath"
	"github.com/rogpeppe/go-internal/par"
	"github.com/rogpeppe/go-internal/testenv"
	"github.com/rogpeppe/go-internal/testscript/internal/pty"
	"github.com/rogpeppe/go-internal/txtar"
)

var goVersionRegex = regexp.MustCompile(`^go([1-9][0-9]*)\.([1-9][0-9]*)$`)

var execCache par.Cache

// If -testwork is specified, the test prints the name of the temp directory
// and does not remove it when done, so that a programmer can
// poke at the test file tree afterward.
var testWork = flag.Bool("testwork", false, "")

// timeSince is defined as a variable so that it can be overridden
// for the local testscript tests so that we can test against predictable
// output.
var timeSince = time.Since

// showVerboseEnv specifies whether the environment should be displayed
// automatically when in verbose mode. This is set to false for the local testscript tests so we
// can test against predictable output.
var showVerboseEnv = true

// Env holds the environment to use at the start of a test script invocation.
type Env struct {
	// WorkDir holds the path to the root directory of the
	// extracted files.
	WorkDir string
	// Vars holds the initial set environment variables that will be passed to the
	// testscript commands.
	Vars []string
	// Cd holds the initial current working directory.
	Cd string
	// Values holds a map of arbitrary values for use by custom
	// testscript commands. This enables Setup to pass arbitrary
	// values (not just strings) through to custom commands.
	Values map[any]any

	ts *TestScript
}

// Value returns a value from Env.Values, or nil if no
// value was set by Setup.
func (ts *TestScript) Value(key any) any {
	return ts.values[key]
}

// Defer arranges for f to be called at the end
// of the test. If Defer is called multiple times, the
// defers are executed in reverse order (similar
// to Go's defer statement)
func (e *Env) Defer(f func()) {
	e.ts.Defer(f)
}

// Getenv retrieves the value of the environment variable named by the key. It
// returns the value, which will be empty if the variable is not present.
func (e *Env) Getenv(key string) string {
	key = envvarname(key)
	for i := len(e.Vars) - 1; i >= 0; i-- {
		if pair := strings.SplitN(e.Vars[i], "=", 2); len(pair) == 2 && envvarname(pair[0]) == key {
			return pair[1]
		}
	}
	return ""
}

func envvarname(k string) string {
	if runtime.GOOS == "windows" {
		return strings.ToLower(k)
	}
	return k
}

// Setenv sets the value of the environment variable named by the key. It
// panics if key is invalid.
func (e *Env) Setenv(key, value string) {
	if key == "" || strings.IndexByte(key, '=') != -1 {
		panic(fmt.Errorf("invalid environment variable key %q", key))
	}
	e.Vars = append(e.Vars, key+"="+value)
}

// T returns the t argument passed to the current test by the T.Run method.
// Note that if the tests were started by calling Run,
// the returned value will implement testing.TB.
// Note that, despite that, the underlying value will not be of type
// *testing.T because *testing.T does not implement T.
//
// If Cleanup is called on the returned value, the function will run
// after any functions passed to Env.Defer.
func (e *Env) T() T {
	return e.ts.t
}

// Params holds parameters for a call to Run.
type Params struct {
	// Dir holds the name of the directory holding the scripts.
	// All files in the directory with a .txtar or .txt suffix will be
	// considered as test scripts. By default the current directory is used.
	// Dir is interpreted relative to the current test directory.
	Dir string

	// Files holds a set of script filenames. If Dir is empty and this
	// is non-nil, these files will be used instead of reading
	// a directory.
	Files []string

	// Setup is called, if not nil, to complete any setup required
	// for a test. The WorkDir and Vars fields will have already
	// been initialized and all the files extracted into WorkDir,
	// and Cd will be the same as WorkDir.
	// The Setup function may modify Vars and Cd as it wishes.
	Setup func(*Env) error

	// Condition is called, if not nil, to determine whether a particular
	// condition is true. It's called only for conditions not in the
	// standard set, and may be nil.
	Condition func(cond string) (bool, error)

	// Cmds holds a map of commands available to the script.
	// It will only be consulted for commands not part of the standard set.
	Cmds map[string]func(ts *TestScript, neg bool, args []string)

	// TestWork specifies that working directories should be
	// left intact for later inspection.
	TestWork bool

	// WorkdirRoot specifies the directory within which scripts' work
	// directories will be created. Setting WorkdirRoot implies TestWork=true.
	// If empty, the work directories will be created inside
	// $GOTMPDIR/go-test-script*, where $GOTMPDIR defaults to os.TempDir().
	WorkdirRoot string

	// Deprecated: this option is no longer used.
	IgnoreMissedCoverage bool

	// UpdateScripts specifies that if a `cmp` command fails and its second
	// argument refers to a file inside the testscript file, the command will
	// succeed and the testscript file will be updated to reflect the actual
	// content (which could be stdout, stderr or a real file).
	//
	// The content will be quoted with txtar.Quote if needed;
	// a manual change will be needed if it is not unquoted in the
	// script.
	UpdateScripts bool

	// RequireExplicitExec requires that commands passed to [Main] must be used
	// in test scripts via `exec cmd` and not simply `cmd`. This can help keep
	// consistency across test scripts as well as keep separate process
	// executions explicit.
	RequireExplicitExec bool

	// RequireUniqueNames requires that names in the txtar archive are unique.
	// By default, later entries silently overwrite earlier ones.
	RequireUniqueNames bool

	// ContinueOnError causes a testscript to try to continue in
	// the face of errors. Once an error has occurred, the script
	// will continue as if in verbose mode.
	ContinueOnError bool

	// Deadline, if not zero, specifies the time at which the test run will have
	// exceeded the timeout. It is equivalent to testing.T's Deadline method,
	// and Run will set it to the method's return value if this field is zero.
	Deadline time.Time
}

// RunDir runs the tests in the given directory. All files in dir with a ".txt"
// or ".txtar" extension are considered to be test files.
func Run(t *testing.T, p Params) {
	if deadline, ok := t.Deadline(); ok && p.Deadline.IsZero() {
		p.Deadline = deadline
	}
	RunT(tshim{t}, p)
}

// T holds all the methods of the *testing.T type that
// are used by testscript.
type T interface {
	Skip(...any)
	Fatal(...any)
	Parallel()
	Log(...any)
	FailNow()
	Run(string, func(T))
	// Verbose is usually implemented by the testing package
	// directly rather than on the *testing.T type.
	Verbose() bool
}

// Deprecated: this type is unused.
type TFailed interface {
	Failed() bool
}

type tshim struct {
	*testing.T
}

func (t tshim) Run(name string, f func(T)) {
	t.T.Run(name, func(t *testing.T) {
		f(tshim{t})
	})
}

func (t tshim) Verbose() bool {
	return testing.Verbose()
}

// RunT is like Run but uses an interface type instead of the concrete *testing.T
// type to make it possible to use testscript functionality outside of go test.
func RunT(t T, p Params) {
	var files []string
	if p.Dir == "" && p.Files != nil {
		files = p.Files
	} else {
		entries, err := os.ReadDir(p.Dir)
		if os.IsNotExist(err) {
			// Continue so we give a helpful error on len(files)==0 below.
		} else if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			name := entry.Name()
			if strings.HasSuffix(name, ".txtar") || strings.HasSuffix(name, ".txt") {
				files = append(files, filepath.Join(p.Dir, name))
			}
		}

		if len(files) == 0 {
			t.Fatal(fmt.Sprintf("no txtar nor txt scripts found in dir %s", p.Dir))
		}
	}
	testTempDir := p.WorkdirRoot
	var err error
	if testTempDir == "" {
		testTempDir, err = os.MkdirTemp(os.Getenv("GOTMPDIR"), "go-test-script")
		if err != nil {
			t.Fatal(err)
		}
	} else {
		p.TestWork = true
	}
	// The temp dir returned by os.MkdirTemp might be a sym linked dir (default
	// behaviour in macOS). That could mess up matching that includes $WORK if,
	// for example, an external program outputs resolved paths. Evaluating the
	// dir here will ensure consistency.
	testTempDir, err = filepath.EvalSymlinks(testTempDir)
	if err != nil {
		t.Fatal(err)
	}

	var (
		ctx         = context.Background()
		gracePeriod = 100 * time.Millisecond
		cancel      context.CancelFunc
	)
	if !p.Deadline.IsZero() {
		timeout := time.Until(p.Deadline)

		// If time allows, increase the termination grace period to 5% of the
		// remaining time.
		if gp := timeout / 20; gp > gracePeriod {
			gracePeriod = gp
		}

		// When we run commands that execute subprocesses, we want to reserve two
		// grace periods to clean up. We will send the first termination signal when
		// the context expires, then wait one grace period for the process to
		// produce whatever useful output it can (such as a stack trace). After the
		// first grace period expires, we'll escalate to os.Kill, leaving the second
		// grace period for the test function to record its output before the test
		// process itself terminates.
		timeout -= 2 * gracePeriod

		ctx, cancel = context.WithTimeout(ctx, timeout)
		// We don't defer cancel() because RunT returns before the sub-tests,
		// and we don't have access to Cleanup due to the T interface. Instead,
		// we call it after the refCount goes to zero below.
		_ = cancel
	}

	refCount := int32(len(files))
	names := make(map[string]bool)
	for _, file := range files {
		name := filepath.Base(file)
		if name1, ok := strings.CutSuffix(name, ".txt"); ok {
			name = name1
		} else if name1, ok := strings.CutSuffix(name, ".txtar"); ok {
			name = name1
		}
		// We can have duplicate names when files are passed explicitly,
		// so disambiguate by adding a counter.
		// Take care to handle the situation where a name with a counter-like
		// suffix already exists, for example:
		//	a/foo.txt
		//	b/foo.txtar
		//	c/foo#1.txt
		prefix := name
		for i := 1; names[name]; i++ {
			name = prefix + "#" + strconv.Itoa(i)
		}
		names[name] = true
		t.Run(name, func(t T) {
			t.Parallel()
			ts := &TestScript{
				t:             t,
				testTempDir:   testTempDir,
				name:          name,
				file:          file,
				params:        p,
				ctxt:          ctx,
				gracePeriod:   gracePeriod,
				deferred:      func() {},
				scriptFiles:   make(map[string]string),
				scriptUpdates: make(map[string]string),
			}
			defer func() {
				if p.TestWork || *testWork {
					return
				}
				removeAll(ts.workdir)
				if atomic.AddInt32(&refCount, -1) == 0 {
					// This is the last subtest to finish. Remove the
					// parent directory too, and cancel the context.
					os.Remove(testTempDir)
					if cancel != nil {
						cancel()
					}
				}
			}()
			ts.run()
		})
	}
}

// A TestScript holds execution state for a single test script.
type TestScript struct {
	params        Params
	t             T
	testTempDir   string
	workdir       string            // temporary work dir ($WORK)
	log           bytes.Buffer      // test execution log (printed at end of test)
	mark          int               // offset of next log truncation
	cd            string            // current directory during test execution; initially $WORK/gopath/src
	name          string            // short name of test ("foo")
	file          string            // full file name ("testdata/script/foo.txt")
	lineno        int               // line number currently executing
	line          string            // line currently executing
	env           []string          // environment list (for os/exec)
	envMap        map[string]string // environment mapping (matches env; on Windows keys are lowercase)
	values        map[any]any       // values for custom commands
	stdin         string            // standard input to next 'go' command; set by 'stdin' command.
	stdout        string            // standard output from last 'go' command; for 'stdout' command
	stderr        string            // standard error from last 'go' command; for 'stderr' command
	ttyin         string            // terminal input; set by 'ttyin' command
	stdinPty      bool              // connect pty to standard input; set by 'ttyin -stdin' command
	ttyout        string            // terminal output; for 'ttyout' command
	stopped       bool              // test wants to stop early
	start         time.Time         // time phase started
	background    []backgroundCmd   // backgrounded 'exec' and 'go' commands
	deferred      func()            // deferred cleanup actions.
	archive       *txtar.Archive    // the testscript being run.
	scriptFiles   map[string]string // files stored in the txtar archive (absolute paths -> path in script)
	scriptUpdates map[string]string // updates to testscript files via UpdateScripts.

	// runningBuiltin indicates if we are running a user-supplied builtin
	// command. These commands are specified via Params.Cmds.
	runningBuiltin bool

	// builtinStd(out|err) are established if a user-supplied builtin command
	// requests Stdout() or Stderr(). Either both are non-nil, or both are nil.
	// This invariant is maintained by both setBuiltinStd() and
	// clearBuiltinStd().
	builtinStdout *strings.Builder
	builtinStderr *strings.Builder

	ctxt        context.Context // per TestScript context
	gracePeriod time.Duration   // time between SIGQUIT and SIGKILL
}

type backgroundCmd struct {
	name string
	cmd  *exec.Cmd
	wait <-chan struct{}
	neg  bool // if true, cmd should fail
}

func writeFile(name string, data []byte, perm fs.FileMode, excl bool) error {
	oflags := os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	if excl {
		oflags |= os.O_EXCL
	}
	f, err := os.OpenFile(name, oflags, perm)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(data); err != nil {
		return fmt.Errorf("cannot write file contents: %v", err)
	}
	return nil
}

// Name returns the short name or basename of the test script.
func (ts *TestScript) Name() string { return ts.name }

// setup sets up the test execution temporary directory and environment.
// It returns the comment section of the txtar archive.
func (ts *TestScript) setup() string {
	defer catchFailNow(func() {
		// There's been a failure in setup; fail immediately regardless
		// of the ContinueOnError flag.
		ts.t.FailNow()
	})
	ts.workdir = filepath.Join(ts.testTempDir, "script-"+ts.name)

	// Establish a temporary directory in workdir, but use a prefix that ensures
	// this directory will not be walked when resolving the ./... pattern from
	// workdir. This is important because when resolving a ./... pattern, cmd/go
	// (which is used by go/packages) creates temporary build files and
	// directories. This can, and does, therefore interfere with the ./...
	// pattern when used from workdir and can lead to race conditions within
	// cmd/go as it walks directories to match the ./... pattern.
	tmpDir := filepath.Join(ts.workdir, ".tmp")

	ts.Check(os.MkdirAll(tmpDir, 0o777))
	env := &Env{
		Vars: []string{
			"WORK=" + ts.workdir, // must be first for ts.abbrev
			"PATH=" + os.Getenv("PATH"),
			"GOTRACEBACK=system",
			homeEnvName() + "=/no-home",
			tempEnvName() + "=" + tmpDir,
			"devnull=" + os.DevNull,
			"/=" + string(os.PathSeparator),
			":=" + string(os.PathListSeparator),
			"$=$",
		},
		WorkDir: ts.workdir,
		Values:  make(map[any]any),
		Cd:      ts.workdir,
		ts:      ts,
	}

	// These env vars affect how a Go program behaves at run-time;
	// If the user or `go test` wrapper set them, we should propagate them
	// so that sub-process commands run via the test binary see them as well.
	for _, name := range []string{
		// If we are collecting coverage profiles, e.g. `go test -coverprofile`.
		"GOCOVERDIR",
		// If the user set GORACE when running a command like `go test -race`,
		// such as GORACE=atexit_sleep_ms=10 to avoid the default 1s sleeps.
		"GORACE",
	} {
		if val := os.Getenv(name); val != "" {
			env.Vars = append(env.Vars, name+"="+val)
		}
	}
	// Must preserve SYSTEMROOT on Windows: https://github.com/golang/go/issues/25513 et al
	if runtime.GOOS == "windows" {
		env.Vars = append(env.Vars,
			"SYSTEMROOT="+os.Getenv("SYSTEMROOT"),
			"exe=.exe",
		)
	} else {
		env.Vars = append(env.Vars,
			"exe=",
		)
	}
	ts.cd = env.Cd
	// Unpack archive.
	a, err := txtar.ParseFile(ts.file)
	ts.Check(err)
	ts.archive = a
	for _, f := range a.Files {
		name := ts.MkAbs(ts.expand(f.Name))
		ts.scriptFiles[name] = f.Name
		ts.Check(os.MkdirAll(filepath.Dir(name), 0o777))
		switch err := writeFile(name, f.Data, 0o666, ts.params.RequireUniqueNames); {
		case ts.params.RequireUniqueNames && errors.Is(err, fs.ErrExist):
			ts.Check(fmt.Errorf("%s would overwrite %s (because RequireUniqueNames is enabled)", f.Name, name))
		default:
			ts.Check(err)
		}
	}
	// Run any user-defined setup.
	if ts.params.Setup != nil {
		ts.Check(ts.params.Setup(env))
	}
	ts.cd = env.Cd
	ts.env = env.Vars
	ts.values = env.Values

	ts.envMap = make(map[string]string)
	for _, kv := range ts.env {
		if i := strings.Index(kv, "="); i >= 0 {
			ts.envMap[envvarname(kv[:i])] = kv[i+1:]
		}
	}
	return string(a.Comment)
}

// run runs the test script.
func (ts *TestScript) run() {
	// Truncate log at end of last phase marker,
	// discarding details of successful phase.
	verbose := ts.t.Verbose()
	rewind := func() {
		if !verbose {
			ts.log.Truncate(ts.mark)
		}
	}

	// Insert elapsed time for phase at end of phase marker
	markTime := func() {
		if ts.mark > 0 && !ts.start.IsZero() {
			afterMark := slices.Clone(ts.log.Bytes()[ts.mark:])
			ts.log.Truncate(ts.mark - 1) // cut \n and afterMark
			fmt.Fprintf(&ts.log, " (%.3fs)\n", timeSince(ts.start).Seconds())
			ts.log.Write(afterMark)
		}
		ts.start = time.Time{}
	}

	failed := false
	defer func() {
		// On a normal exit from the test loop, background processes are cleaned up
		// before we print PASS. If we return early (e.g., due to a test failure),
		// don't print anything about the processes that were still running.
		for _, bg := range ts.background {
			interruptProcess(bg.cmd.Process)
		}
		if ts.t.Verbose() || failed {
			// In verbose mode or on test failure, we want to see what happened in the background
			// processes too.
			ts.waitBackground(false)
		} else {
			for _, bg := range ts.background {
				<-bg.wait
			}
			ts.background = nil
		}

		markTime()
		// Flush testScript log to testing.T log.
		ts.t.Log(ts.abbrev(ts.log.String()))
	}()
	defer func() {
		ts.deferred()
	}()
	script := ts.setup()

	// With -v or -testwork, start log with full environment.
	if *testWork || (showVerboseEnv && ts.t.Verbose()) {
		// Display environment.
		ts.cmdEnv(false, nil)
		fmt.Fprintf(&ts.log, "\n")
		ts.mark = ts.log.Len()
	}
	defer ts.applyScriptUpdates()

	// Run script.
	// See testdata/script/README for documentation of script form.
	for script != "" {
		// Extract next line.
		ts.lineno++
		var line string
		if i := strings.Index(script, "\n"); i >= 0 {
			line, script = script[:i], script[i+1:]
		} else {
			line, script = script, ""
		}

		// # is a comment indicating the start of new phase.
		if strings.HasPrefix(line, "#") {
			// If there was a previous phase, it succeeded,
			// so rewind the log to delete its details (unless -v is in use or
			// ContinueOnError was enabled and there was a previous error,
			// causing verbose to be set to true).
			// If nothing has happened at all since the mark,
			// rewinding is a no-op and adding elapsed time
			// for doing nothing is meaningless, so don't.
			if ts.log.Len() > ts.mark {
				rewind()
				markTime()
			}
			// Print phase heading and mark start of phase output.
			fmt.Fprintf(&ts.log, "%s\n", line)
			ts.mark = ts.log.Len()
			ts.start = time.Now()
			continue
		}

		ok := ts.runLine(line)
		if !ok {
			failed = true
			if ts.params.ContinueOnError {
				verbose = true
			} else {
				ts.t.FailNow()
			}
		}

		// Command can ask script to stop early.
		if ts.stopped {
			// Break instead of returning, so that we check the status of any
			// background processes and print PASS.
			break
		}
	}

	for _, bg := range ts.background {
		interruptProcess(bg.cmd.Process)
	}
	// On some platforms like Windows, we kill background commands directly
	// as we can't send them an interrupt signal, so they always fail.
	// Moreover, it's relatively common for a process to fail when interrupted.
	// Once we've reached the end of the script, ignore the status of background commands.
	ts.waitBackground(false)

	// If we reached here but we've failed (probably because ContinueOnError
	// was set), don't wipe the log and print "PASS".
	if failed {
		ts.t.FailNow()
	}

	// Final phase ended.
	rewind()
	markTime()
	if !ts.stopped {
		fmt.Fprintf(&ts.log, "PASS\n")
	}
}

func (ts *TestScript) runLine(line string) (runOK bool) {
	defer catchFailNow(func() {
		runOK = false
	})

	// Parse input line. Ignore blanks entirely.
	args := ts.parse(line)
	if len(args) == 0 {
		return true
	}

	// Echo command to log.
	fmt.Fprintf(&ts.log, "> %s\n", line)

	// Command prefix [cond] means only run this command if cond is satisfied.
	for strings.HasPrefix(args[0], "[") && strings.HasSuffix(args[0], "]") {
		cond := args[0]
		cond = cond[1 : len(cond)-1]
		cond = strings.TrimSpace(cond)
		args = args[1:]
		if len(args) == 0 {
			ts.Fatalf("missing command after condition")
		}
		want := true
		if strings.HasPrefix(cond, "!") {
			want = false
			cond = strings.TrimSpace(cond[1:])
		}
		ok, err := ts.condition(cond)
		if err != nil {
			ts.Fatalf("bad condition %q: %v", cond, err)
		}
		if ok != want {
			// Don't run rest of line.
			return true
		}
	}

	// Command prefix ! means negate the expectations about this command:
	// go command should fail, match should not be found, etc.
	neg := false
	if args[0] == "!" {
		neg = true
		args = args[1:]
		if len(args) == 0 {
			ts.Fatalf("! on line by itself")
		}
	}

	// Run command.
	cmd := scriptCmds[args[0]]
	if cmd == nil {
		cmd = ts.params.Cmds[args[0]]
	}
	if cmd == nil {
		// try to find spelling corrections. We arbitrarily limit the number of
		// corrections, to not be too noisy.
		switch c := ts.cmdSuggestions(args[0]); len(c) {
		case 1:
			ts.Fatalf("unknown command %q (did you mean %q?)", args[0], c[0])
		case 2, 3, 4:
			ts.Fatalf("unknown command %q (did you mean one of %q?)", args[0], c)
		default:
			ts.Fatalf("unknown command %q", args[0])
		}
	}
	ts.callBuiltinCmd(func() {
		cmd(ts, neg, args[1:])
	})
	return true
}

func (ts *TestScript) callBuiltinCmd(runCmd func()) {
	ts.runningBuiltin = true
	defer func() {
		r := recover()
		ts.runningBuiltin = false
		ts.clearBuiltinStd()
		switch r {
		case nil:
			// we did not panic
		default:
			// re-"throw" the panic
			panic(r)
		}
	}()
	runCmd()
}

func (ts *TestScript) cmdSuggestions(name string) []string {
	// special case: spell-correct `!cmd` to `! cmd`
	if strings.HasPrefix(name, "!") {
		if _, ok := scriptCmds[name[1:]]; ok {
			return []string{"! " + name[1:]}
		}
		if _, ok := ts.params.Cmds[name[1:]]; ok {
			return []string{"! " + name[1:]}
		}
	}
	var candidates []string
	for c := range scriptCmds {
		if misspell.AlmostEqual(name, c) {
			candidates = append(candidates, c)
		}
	}
	for c := range ts.params.Cmds {
		if misspell.AlmostEqual(name, c) {
			candidates = append(candidates, c)
		}
	}
	if len(candidates) == 0 {
		return nil
	}
	// deduplicate candidates
	slices.Sort(candidates)
	return slices.Compact(candidates)
}

func (ts *TestScript) applyScriptUpdates() {
	if len(ts.scriptUpdates) == 0 {
		return
	}
	for name, content := range ts.scriptUpdates {
		found := false
		for i := range ts.archive.Files {
			f := &ts.archive.Files[i]
			if f.Name != name {
				continue
			}
			data := []byte(content)
			if txtar.NeedsQuote(data) {
				data1, err := txtar.Quote(data)
				if err != nil {
					ts.Fatalf("cannot update script file %q: %v", f.Name, err)
					continue
				}
				data = data1
			}
			f.Data = data
			found = true
		}
		// Sanity check.
		if !found {
			panic("script update file not found")
		}
	}
	if err := os.WriteFile(ts.file, txtar.Format(ts.archive), 0o666); err != nil {
		ts.t.Fatal("cannot update script: ", err)
	}
	ts.Logf("%s updated", ts.file)
}

var failNow = errors.New("fail now!")

// catchFailNow catches any panic from Fatalf and calls
// f if it did so. It must be called in a defer.
func catchFailNow(f func()) {
	e := recover()
	if e == nil {
		return
	}
	if e != failNow {
		panic(e)
	}
	f()
}

// condition reports whether the given condition is satisfied.
func (ts *TestScript) condition(cond string) (bool, error) {
	switch {
	case cond == "short":
		return testing.Short(), nil
	case cond == "net":
		return testenv.HasExternalNetwork(), nil
	case cond == "link":
		return testenv.HasLink(), nil
	case cond == "symlink":
		return testenv.HasSymlink(), nil
	case imports.KnownOS[cond]:
		return cond == runtime.GOOS, nil
	case cond == "unix":
		return imports.UnixOS[runtime.GOOS], nil
	case imports.KnownArch[cond]:
		return cond == runtime.GOARCH, nil
	case strings.HasPrefix(cond, "exec:"):
		prog := cond[len("exec:"):]
		ok := execCache.Do(prog, func() any {
			_, err := execpath.Look(prog, ts.Getenv)
			return err == nil
		}).(bool)
		return ok, nil
	case cond == "gc" || cond == "gccgo":
		// TODO this reflects the compiler that the current
		// binary was built with but not necessarily the compiler
		// that will be used.
		return cond == runtime.Compiler, nil
	case goVersionRegex.MatchString(cond):
		if slices.Contains(build.Default.ReleaseTags, cond) {
			return true, nil
		}
		return false, nil
	case ts.params.Condition != nil:
		return ts.params.Condition(cond)
	default:
		ts.Fatalf("unknown condition %q", cond)
		panic("unreachable")
	}
}

// Helpers for command implementations.

// abbrev abbreviates the actual work directory in the string s to the literal string "$WORK".
func (ts *TestScript) abbrev(s string) string {
	s = strings.Replace(s, ts.workdir, "$WORK", -1)
	if *testWork || ts.params.TestWork {
		// Expose actual $WORK value in environment dump on first line of work script,
		// so that the user can find out what directory -testwork left behind.
		s = "WORK=" + ts.workdir + "\n" + strings.TrimPrefix(s, "WORK=$WORK\n")
	}
	return s
}

// Defer arranges for f to be called at the end
// of the test. If Defer is called multiple times, the
// defers are executed in reverse order (similar
// to Go's defer statement)
func (ts *TestScript) Defer(f func()) {
	old := ts.deferred
	ts.deferred = func() {
		defer old()
		f()
	}
}

// Check calls ts.Fatalf if err != nil.
func (ts *TestScript) Check(err error) {
	if err != nil {
		ts.Fatalf("%v", err)
	}
}

// Stdout returns an io.Writer that can be used by a user-supplied builtin
// command (declared via Params.Cmds) to write to stdout. If this method is
// called outside of the execution of a user-supplied builtin command, the
// call panics.
func (ts *TestScript) Stdout() io.Writer {
	if !ts.runningBuiltin {
		panic("can only call TestScript.Stdout when running a builtin command")
	}
	ts.setBuiltinStd()
	return ts.builtinStdout
}

// Stderr returns an io.Writer that can be used by a user-supplied builtin
// command (declared via Params.Cmds) to write to stderr. If this method is
// called outside of the execution of a user-supplied builtin command, the
// call panics.
func (ts *TestScript) Stderr() io.Writer {
	if !ts.runningBuiltin {
		panic("can only call TestScript.Stderr when running a builtin command")
	}
	ts.setBuiltinStd()
	return ts.builtinStderr
}

// setBuiltinStd ensures that builtinStdout and builtinStderr are non nil.
func (ts *TestScript) setBuiltinStd() {
	// This method must maintain the invariant that both builtinStdout and
	// builtinStderr are set or neither are set

	// If both are set, nothing to do
	if ts.builtinStdout != nil && ts.builtinStderr != nil {
		return
	}
	ts.builtinStdout = new(strings.Builder)
	ts.builtinStderr = new(strings.Builder)
}

// clearBuiltinStd sets ts.stdout and ts.stderr from the builtin command
// buffers, logs both, and resets both builtinStdout and builtinStderr to nil.
func (ts *TestScript) clearBuiltinStd() {
	// This method must maintain the invariant that both builtinStdout and
	// builtinStderr are set or neither are set

	// If neither set, nothing to do
	if ts.builtinStdout == nil && ts.builtinStderr == nil {
		return
	}
	ts.stdout = ts.builtinStdout.String()
	ts.builtinStdout = nil
	ts.stderr = ts.builtinStderr.String()
	ts.builtinStderr = nil
	ts.logStd()
}

// Logf appends the given formatted message to the test log transcript.
func (ts *TestScript) Logf(format string, args ...any) {
	format = strings.TrimSuffix(format, "\n")
	fmt.Fprintf(&ts.log, format, args...)
	ts.log.WriteByte('\n')
}

// exec runs the given command line (an actual subprocess, not simulated)
// in ts.cd with environment ts.env and then returns collected standard output and standard error.
func (ts *TestScript) exec(command string, args ...string) (stdout, stderr string, err error) {
	cmd, err := ts.buildExecCmd(command, args...)
	if err != nil {
		return "", "", err
	}
	cmd.Dir = ts.cd
	cmd.Env = append(ts.env, "PWD="+ts.cd)
	cmd.Stdin = strings.NewReader(ts.stdin)
	var stdoutBuf, stderrBuf strings.Builder
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf
	if ts.ttyin != "" {
		ctrl, tty, err := pty.Open()
		if err != nil {
			return "", "", err
		}
		doneR, doneW := make(chan struct{}), make(chan struct{})
		var ptyBuf strings.Builder
		go func() {
			io.Copy(ctrl, strings.NewReader(ts.ttyin))
			ctrl.Write([]byte{4 /* EOT */})
			close(doneW)
		}()
		go func() {
			io.Copy(&ptyBuf, ctrl)
			close(doneR)
		}()
		defer func() {
			tty.Close()
			ctrl.Close()
			<-doneR
			<-doneW
			ts.ttyin = ""
			ts.ttyout = ptyBuf.String()
		}()
		pty.SetCtty(cmd, tty)
		if ts.stdinPty {
			cmd.Stdin = tty
		}
	}
	if err = cmd.Start(); err == nil {
		err = waitOrStop(ts.ctxt, cmd, ts.gracePeriod)
	}
	ts.stdin = ""
	ts.stdinPty = false
	return stdoutBuf.String(), stderrBuf.String(), err
}

// execBackground starts the given command line (an actual subprocess, not simulated)
// in ts.cd with environment ts.env.
func (ts *TestScript) execBackground(command string, args ...string) (*exec.Cmd, error) {
	if ts.ttyin != "" {
		return nil, errors.New("ttyin is not supported by background commands")
	}
	cmd, err := ts.buildExecCmd(command, args...)
	if err != nil {
		return nil, err
	}
	cmd.Dir = ts.cd
	cmd.Env = append(ts.env, "PWD="+ts.cd)
	var stdoutBuf, stderrBuf strings.Builder
	cmd.Stdin = strings.NewReader(ts.stdin)
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf
	ts.stdin = ""
	return cmd, cmd.Start()
}

func (ts *TestScript) buildExecCmd(command string, args ...string) (*exec.Cmd, error) {
	if filepath.Base(command) == command {
		if lp, err := execpath.Look(command, ts.Getenv); err != nil {
			return nil, err
		} else {
			command = lp
		}
	}
	return exec.Command(command, args...), nil
}

// BackgroundCmds returns a slice containing all the commands that have
// been started in the background since the most recent wait command, or
// the start of the script if wait has not been called.
func (ts *TestScript) BackgroundCmds() []*exec.Cmd {
	cmds := make([]*exec.Cmd, len(ts.background))
	for i, b := range ts.background {
		cmds[i] = b.cmd
	}
	return cmds
}

// Chdir changes the current directory of the script.
// The path may be relative to the current directory.
func (ts *TestScript) Chdir(dir string) error {
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(ts.cd, dir)
	}
	info, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory", dir)
	}

	ts.cd = dir
	return nil
}

// waitOrStop waits for the already-started command cmd by calling its Wait method.
//
// If cmd does not return before ctx is done, waitOrStop sends it an interrupt
// signal. If killDelay is positive, waitOrStop waits that additional period for
// Wait to return before sending os.Kill.
func waitOrStop(ctx context.Context, cmd *exec.Cmd, killDelay time.Duration) error {
	if cmd.Process == nil {
		panic("waitOrStop called with a nil cmd.Process — missing Start call?")
	}

	errc := make(chan error)
	go func() {
		select {
		case errc <- nil:
			return
		case <-ctx.Done():
		}

		var interrupt os.Signal = syscall.SIGQUIT
		if runtime.GOOS == "windows" {
			// Per https://golang.org/pkg/os/#Signal, “Interrupt is not implemented on
			// Windows; using it with os.Process.Signal will return an error.”
			// Fall back directly to Kill instead.
			interrupt = os.Kill
		}

		err := cmd.Process.Signal(interrupt)
		if err == nil {
			err = ctx.Err() // Report ctx.Err() as the reason we interrupted.
		} else if err == os.ErrProcessDone {
			errc <- nil
			return
		}

		if killDelay > 0 {
			timer := time.NewTimer(killDelay)
			select {
			// Report ctx.Err() as the reason we interrupted the process...
			case errc <- ctx.Err():
				timer.Stop()
				return
			// ...but after killDelay has elapsed, fall back to a stronger signal.
			case <-timer.C:
			}

			// Wait still hasn't returned.
			// Kill the process harder to make sure that it exits.
			//
			// Ignore any error: if cmd.Process has already terminated, we still
			// want to send ctx.Err() (or the error from the Interrupt call)
			// to properly attribute the signal that may have terminated it.
			_ = cmd.Process.Kill()
		}

		errc <- err
	}()

	waitErr := cmd.Wait()
	if interruptErr := <-errc; interruptErr != nil {
		return interruptErr
	}
	return waitErr
}

// interruptProcess sends os.Interrupt to p if supported, or os.Kill otherwise.
func interruptProcess(p *os.Process) {
	if err := p.Signal(os.Interrupt); err != nil {
		// Per https://golang.org/pkg/os/#Signal, “Interrupt is not implemented on
		// Windows; using it with os.Process.Signal will return an error.”
		// Fall back to Kill instead.
		p.Kill()
	}
}

// Exec runs the given command and saves its stdout and stderr so
// they can be inspected by subsequent script commands.
func (ts *TestScript) Exec(command string, args ...string) error {
	var err error
	ts.stdout, ts.stderr, err = ts.exec(command, args...)
	ts.logStd()
	return err
}

// logStd logs the current non-empty values of stdout and stderr.
func (ts *TestScript) logStd() {
	if ts.stdout != "" {
		ts.Logf("[stdout]\n%s", ts.stdout)
	}
	if ts.stderr != "" {
		ts.Logf("[stderr]\n%s", ts.stderr)
	}
}

// expand applies environment variable expansion to the string s.
func (ts *TestScript) expand(s string) string {
	return os.Expand(s, func(key string) string {
		if key1 := strings.TrimSuffix(key, "@R"); len(key1) != len(key) {
			return regexp.QuoteMeta(ts.Getenv(key1))
		}
		return ts.Getenv(key)
	})
}

// fatalf aborts the test with the given failure message.
func (ts *TestScript) Fatalf(format string, args ...any) {
	// In user-supplied builtins, the only way we have of aborting
	// is via Fatalf. Hence if we are aborting from a user-supplied
	// builtin, it's important we first log stdout and stderr. If
	// we are not, the following call is a no-op.
	ts.clearBuiltinStd()

	fmt.Fprintf(&ts.log, "FAIL: %s:%d: %s\n", ts.file, ts.lineno, fmt.Sprintf(format, args...))
	// This should be caught by the defer inside the TestScript.runLine method.
	// We do this rather than calling ts.t.FailNow directly because we want to
	// be able to continue on error when Params.ContinueOnError is set.
	panic(failNow)
}

// MkAbs interprets file relative to the test script's current directory
// and returns the corresponding absolute path.
func (ts *TestScript) MkAbs(file string) string {
	if filepath.IsAbs(file) {
		return file
	}
	return filepath.Join(ts.cd, file)
}

// ReadFile returns the contents of the file with the
// given name, interpreted relative to the test script's
// current directory. It interprets "stdout" and "stderr" to
// mean the standard output or standard error from
// the most recent exec or wait command respectively.
//
// If the file cannot be read, the script fails.
func (ts *TestScript) ReadFile(file string) string {
	switch file {
	case "stdout":
		return ts.stdout
	case "stderr":
		return ts.stderr
	case "ttyout":
		return ts.ttyout
	default:
		file = ts.MkAbs(file)
		data, err := os.ReadFile(file)
		ts.Check(err)
		return string(data)
	}
}

// Setenv sets the value of the environment variable named by the key.
func (ts *TestScript) Setenv(key, value string) {
	ts.env = append(ts.env, key+"="+value)
	ts.envMap[envvarname(key)] = value
}

// Getenv gets the value of the environment variable named by the key.
func (ts *TestScript) Getenv(key string) string {
	return ts.envMap[envvarname(key)]
}

// parse parses a single line as a list of space-separated arguments
// subject to environment variable expansion (but not resplitting).
// Single quotes around text disable splitting and expansion.
// To embed a single quote, double it: 'Don”t communicate by sharing memory.'
func (ts *TestScript) parse(line string) []string {
	ts.line = line

	var (
		args   []string
		arg    string  // text of current arg so far (need to add line[start:i])
		start  = -1    // if >= 0, position where current arg text chunk starts
		quoted = false // currently processing quoted text
	)
	for i := 0; ; i++ {
		if !quoted && (i >= len(line) || line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '#') {
			// Found arg-separating space.
			if start >= 0 {
				arg += ts.expand(line[start:i])
				args = append(args, arg)
				start = -1
				arg = ""
			}
			if i >= len(line) || line[i] == '#' {
				break
			}
			continue
		}
		if i >= len(line) {
			ts.Fatalf("unterminated quoted argument")
		}
		if line[i] == '\'' {
			if !quoted {
				// starting a quoted chunk
				if start >= 0 {
					arg += ts.expand(line[start:i])
				}
				start = i + 1
				quoted = true
				continue
			}
			// 'foo''bar' means foo'bar, like in rc shell and Pascal.
			if i+1 < len(line) && line[i+1] == '\'' {
				arg += line[start:i]
				start = i + 1
				i++ // skip over second ' before next iteration
				continue
			}
			// ending a quoted chunk
			arg += line[start:i]
			start = i + 1
			quoted = false
			continue
		}
		// found character worth saving; make sure we're saving
		if start < 0 {
			start = i
		}
	}
	return args
}

func removeAll(dir string) error {
	// module cache has 0o444 directories;
	// make them writable in order to remove content.
	filepath.WalkDir(dir, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return nil // ignore errors walking in file system
		}
		if entry.IsDir() {
			os.Chmod(path, 0o777)
		}
		return nil
	})
	return os.RemoveAll(dir)
}

func homeEnvName() string {
	switch runtime.GOOS {
	case "windows":
		return "USERPROFILE"
	case "plan9":
		return "home"
	default:
		return "HOME"
	}
}

func tempEnvName() string {
	switch runtime.GOOS {
	case "windows":
		return "TMP"
	case "plan9":
		return "TMPDIR" // actually plan 9 doesn't have one at all but this is fine
	default:
		return "TMPDIR"
	}
}
