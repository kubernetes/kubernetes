// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testscript

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"github.com/rogpeppe/go-internal/diff"
	"github.com/rogpeppe/go-internal/testscript/internal/pty"
	"github.com/rogpeppe/go-internal/txtar"
)

// scriptCmds are the script command implementations.
// Keep list and the implementations below sorted by name.
//
// NOTE: If you make changes here, update doc.go.
var scriptCmds = map[string]func(*TestScript, bool, []string){
	"cd":       (*TestScript).cmdCd,
	"chmod":    (*TestScript).cmdChmod,
	"cmp":      (*TestScript).cmdCmp,
	"cmpenv":   (*TestScript).cmdCmpenv,
	"cp":       (*TestScript).cmdCp,
	"env":      (*TestScript).cmdEnv,
	"exec":     (*TestScript).cmdExec,
	"exists":   (*TestScript).cmdExists,
	"grep":     (*TestScript).cmdGrep,
	"kill":     (*TestScript).cmdKill,
	"mkdir":    (*TestScript).cmdMkdir,
	"mv":       (*TestScript).cmdMv,
	"rm":       (*TestScript).cmdRm,
	"skip":     (*TestScript).cmdSkip,
	"stderr":   (*TestScript).cmdStderr,
	"stdin":    (*TestScript).cmdStdin,
	"stdout":   (*TestScript).cmdStdout,
	"ttyin":    (*TestScript).cmdTtyin,
	"ttyout":   (*TestScript).cmdTtyout,
	"stop":     (*TestScript).cmdStop,
	"symlink":  (*TestScript).cmdSymlink,
	"unix2dos": (*TestScript).cmdUNIX2DOS,
	"unquote":  (*TestScript).cmdUnquote,
	"wait":     (*TestScript).cmdWait,
}

// cd changes to a different directory.
func (ts *TestScript) cmdCd(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! cd")
	}
	if len(args) != 1 {
		ts.Fatalf("usage: cd dir")
	}

	dir := args[0]
	err := ts.Chdir(dir)
	if os.IsNotExist(err) {
		ts.Fatalf("directory %s does not exist", dir)
	}
	ts.Check(err)
	ts.Logf("%s\n", ts.cd)
}

func (ts *TestScript) cmdChmod(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! chmod")
	}
	if len(args) != 2 {
		ts.Fatalf("usage: chmod perm paths...")
	}
	perm, err := strconv.ParseUint(args[0], 8, 32)
	if err != nil || perm&uint64(os.ModePerm) != perm {
		ts.Fatalf("invalid mode: %s", args[0])
	}
	for _, arg := range args[1:] {
		path := arg
		if !filepath.IsAbs(path) {
			path = filepath.Join(ts.cd, arg)
		}
		err := os.Chmod(path, os.FileMode(perm))
		ts.Check(err)
	}
}

// cmp compares two files.
func (ts *TestScript) cmdCmp(neg bool, args []string) {
	if len(args) != 2 {
		ts.Fatalf("usage: cmp file1 file2")
	}

	ts.doCmdCmp(neg, args, false)
}

// cmpenv compares two files with environment variable substitution.
func (ts *TestScript) cmdCmpenv(neg bool, args []string) {
	if len(args) != 2 {
		ts.Fatalf("usage: cmpenv file1 file2")
	}
	ts.doCmdCmp(neg, args, true)
}

func (ts *TestScript) doCmdCmp(neg bool, args []string, env bool) {
	name1, name2 := args[0], args[1]
	text1 := ts.ReadFile(name1)

	absName2 := ts.MkAbs(name2)
	data, err := os.ReadFile(absName2)
	ts.Check(err)
	text2 := string(data)
	if env {
		text2 = ts.expand(text2)
	}
	eq := text1 == text2
	if neg {
		if eq {
			ts.Fatalf("%s and %s do not differ", name1, name2)
		}
		return // they differ, as expected
	}
	if eq {
		return // they are equal, as expected
	}
	if ts.params.UpdateScripts && !env {
		if scriptFile, ok := ts.scriptFiles[absName2]; ok {
			ts.scriptUpdates[scriptFile] = text1
			return
		}
		// The file being compared against isn't in the txtar archive, so don't
		// update the script.
	}

	unifiedDiff := diff.Diff(name1, []byte(text1), name2, []byte(text2))

	ts.Logf("%s", unifiedDiff)
	ts.Fatalf("%s and %s differ", name1, name2)
}

// cp copies files, maybe eventually directories.
func (ts *TestScript) cmdCp(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! cp")
	}
	if len(args) < 2 {
		ts.Fatalf("usage: cp src... dst")
	}

	dst := ts.MkAbs(args[len(args)-1])
	info, err := os.Stat(dst)
	dstDir := err == nil && info.IsDir()
	if len(args) > 2 && !dstDir {
		ts.Fatalf("cp: destination %s is not a directory", dst)
	}

	for _, arg := range args[:len(args)-1] {
		var (
			src  string
			data []byte
			mode os.FileMode
		)
		switch arg {
		case "stdout":
			src = arg
			data = []byte(ts.stdout)
			mode = 0o666
		case "stderr":
			src = arg
			data = []byte(ts.stderr)
			mode = 0o666
		case "ttyout":
			src = arg
			data = []byte(ts.ttyout)
			mode = 0o666
		default:
			src = ts.MkAbs(arg)
			info, err := os.Stat(src)
			ts.Check(err)
			mode = info.Mode() & 0o777
			data, err = os.ReadFile(src)
			ts.Check(err)
		}
		targ := dst
		if dstDir {
			targ = filepath.Join(dst, filepath.Base(src))
		}
		ts.Check(os.WriteFile(targ, data, mode))
	}
}

// env displays or adds to the environment.
func (ts *TestScript) cmdEnv(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! env")
	}
	if len(args) == 0 {
		printed := make(map[string]bool) // env list can have duplicates; only print effective value (from envMap) once
		for _, kv := range ts.env {
			k := envvarname(kv[:strings.Index(kv, "=")])
			if !printed[k] {
				printed[k] = true
				ts.Logf("%s=%s\n", k, ts.envMap[k])
			}
		}
		return
	}
	for _, env := range args {
		i := strings.Index(env, "=")
		if i < 0 {
			// Display value instead of setting it.
			ts.Logf("%s=%s\n", env, ts.Getenv(env))
			continue
		}
		ts.Setenv(env[:i], env[i+1:])
	}
}

var backgroundSpecifier = regexp.MustCompile(`^&([a-zA-Z_0-9]+&)?$`)

// exec runs the given command.
func (ts *TestScript) cmdExec(neg bool, args []string) {
	if len(args) < 1 || (len(args) == 1 && args[0] == "&") {
		ts.Fatalf("usage: exec program [args...] [&]")
	}

	var err error
	if len(args) > 0 && backgroundSpecifier.MatchString(args[len(args)-1]) {
		bgName := strings.TrimSuffix(strings.TrimPrefix(args[len(args)-1], "&"), "&")
		if ts.findBackground(bgName) != nil {
			ts.Fatalf("duplicate background process name %q", bgName)
		}
		var cmd *exec.Cmd
		cmd, err = ts.execBackground(args[0], args[1:len(args)-1]...)
		if err == nil {
			wait := make(chan struct{})
			go func() {
				waitOrStop(ts.ctxt, cmd, -1)
				close(wait)
			}()
			ts.background = append(ts.background, backgroundCmd{bgName, cmd, wait, neg})
		}
		ts.stdout, ts.stderr = "", ""
	} else {
		ts.stdout, ts.stderr, err = ts.exec(args[0], args[1:]...)
		if ts.stdout != "" {
			fmt.Fprintf(&ts.log, "[stdout]\n%s", ts.stdout)
		}
		if ts.stderr != "" {
			fmt.Fprintf(&ts.log, "[stderr]\n%s", ts.stderr)
		}
		if err == nil && neg {
			ts.Fatalf("unexpected command success")
		}
	}

	if err != nil {
		fmt.Fprintf(&ts.log, "[%v]\n", err)
		if ts.ctxt.Err() != nil {
			ts.Fatalf("test timed out while running command")
		} else if !neg {
			ts.Fatalf("unexpected command failure")
		}
	}
}

// exists checks that the list of files exists.
func (ts *TestScript) cmdExists(neg bool, args []string) {
	var readonly bool
	if len(args) > 0 && args[0] == "-readonly" {
		readonly = true
		args = args[1:]
	}
	if len(args) == 0 {
		ts.Fatalf("usage: exists [-readonly] file...")
	}

	for _, file := range args {
		file = ts.MkAbs(file)
		info, err := os.Stat(file)
		if err == nil && neg {
			what := "file"
			if info.IsDir() {
				what = "directory"
			}
			ts.Fatalf("%s %s unexpectedly exists", what, file)
		}
		if err != nil && !neg {
			ts.Fatalf("%s does not exist", file)
		}
		if err == nil && !neg && readonly && info.Mode()&0o222 != 0 {
			ts.Fatalf("%s exists but is writable", file)
		}
	}
}

// mkdir creates directories.
func (ts *TestScript) cmdMkdir(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! mkdir")
	}
	if len(args) < 1 {
		ts.Fatalf("usage: mkdir dir...")
	}
	for _, arg := range args {
		ts.Check(os.MkdirAll(ts.MkAbs(arg), 0o777))
	}
}

func (ts *TestScript) cmdMv(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! mv")
	}
	if len(args) != 2 {
		ts.Fatalf("usage: mv old new")
	}
	ts.Check(os.Rename(ts.MkAbs(args[0]), ts.MkAbs(args[1])))
}

// unquote unquotes files.
func (ts *TestScript) cmdUnquote(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! unquote")
	}
	for _, arg := range args {
		file := ts.MkAbs(arg)
		data, err := os.ReadFile(file)
		ts.Check(err)
		data, err = txtar.Unquote(data)
		ts.Check(err)
		err = os.WriteFile(file, data, 0o666)
		ts.Check(err)
	}
}

// rm removes files or directories.
func (ts *TestScript) cmdRm(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! rm")
	}
	if len(args) < 1 {
		ts.Fatalf("usage: rm file...")
	}
	for _, arg := range args {
		file := ts.MkAbs(arg)
		removeAll(file)              // does chmod and then attempts rm
		ts.Check(os.RemoveAll(file)) // report error
	}
}

// skip marks the test skipped.
func (ts *TestScript) cmdSkip(neg bool, args []string) {
	if len(args) > 1 {
		ts.Fatalf("usage: skip [msg]")
	}
	if neg {
		ts.Fatalf("unsupported: ! skip")
	}

	// Before we mark the test as skipped, shut down any background processes and
	// make sure they have returned the correct status.
	for _, bg := range ts.background {
		interruptProcess(bg.cmd.Process)
	}
	ts.cmdWait(false, nil)

	if len(args) == 1 {
		ts.t.Skip(args[0])
	}
	ts.t.Skip()
}

func (ts *TestScript) cmdStdin(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! stdin")
	}
	if len(args) != 1 {
		ts.Fatalf("usage: stdin filename")
	}
	if ts.stdinPty {
		ts.Fatalf("conflicting use of 'stdin' and 'ttyin -stdin'")
	}
	ts.stdin = ts.ReadFile(args[0])
}

// stdout checks that the last go command standard output matches a regexp.
func (ts *TestScript) cmdStdout(neg bool, args []string) {
	scriptMatch(ts, neg, args, ts.stdout, "stdout")
}

// stderr checks that the last go command standard output matches a regexp.
func (ts *TestScript) cmdStderr(neg bool, args []string) {
	scriptMatch(ts, neg, args, ts.stderr, "stderr")
}

// grep checks that file content matches a regexp.
// Like stdout/stderr and unlike Unix grep, it accepts Go regexp syntax.
func (ts *TestScript) cmdGrep(neg bool, args []string) {
	scriptMatch(ts, neg, args, "", "grep")
}

func (ts *TestScript) cmdTtyin(neg bool, args []string) {
	if !pty.Supported {
		ts.Fatalf("unsupported: ttyin on %s", runtime.GOOS)
	}
	if neg {
		ts.Fatalf("unsupported: ! ttyin")
	}
	switch len(args) {
	case 1:
		ts.ttyin = ts.ReadFile(args[0])
	case 2:
		if args[0] != "-stdin" {
			ts.Fatalf("usage: ttyin [-stdin] filename")
		}
		if ts.stdin != "" {
			ts.Fatalf("conflicting use of 'stdin' and 'ttyin -stdin'")
		}
		ts.stdinPty = true
		ts.ttyin = ts.ReadFile(args[1])
	default:
		ts.Fatalf("usage: ttyin [-stdin] filename")
	}
	if ts.ttyin == "" {
		ts.Fatalf("tty input file is empty")
	}
}

func (ts *TestScript) cmdTtyout(neg bool, args []string) {
	scriptMatch(ts, neg, args, ts.ttyout, "ttyout")
}

// stop stops execution of the test (marking it passed).
func (ts *TestScript) cmdStop(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! stop")
	}
	if len(args) > 1 {
		ts.Fatalf("usage: stop [msg]")
	}
	if len(args) == 1 {
		ts.Logf("stop: %s\n", args[0])
	} else {
		ts.Logf("stop\n")
	}
	ts.stopped = true
}

// symlink creates a symbolic link.
func (ts *TestScript) cmdSymlink(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! symlink")
	}
	if len(args) != 3 || args[1] != "->" {
		ts.Fatalf("usage: symlink file -> target")
	}
	// Note that the link target args[2] is not interpreted with MkAbs:
	// it will be interpreted relative to the directory file is in.
	ts.Check(os.Symlink(args[2], ts.MkAbs(args[0])))
}

// cmdUNIX2DOS converts files from UNIX line endings to DOS line endings.
func (ts *TestScript) cmdUNIX2DOS(neg bool, args []string) {
	if neg {
		ts.Fatalf("unsupported: ! unix2dos")
	}
	if len(args) < 1 {
		ts.Fatalf("usage: unix2dos paths...")
	}
	for _, arg := range args {
		filename := ts.MkAbs(arg)
		data, err := os.ReadFile(filename)
		ts.Check(err)
		dosData, err := unix2DOS(data)
		ts.Check(err)
		if err := os.WriteFile(filename, dosData, 0o666); err != nil {
			ts.Fatalf("%s: %v", filename, err)
		}
	}
}

// cmdKill kills background commands.
func (ts *TestScript) cmdKill(neg bool, args []string) {
	signals := map[string]os.Signal{
		"INT":  os.Interrupt,
		"KILL": os.Kill,
	}
	var (
		name   string
		signal os.Signal
	)
	switch len(args) {
	case 0:
	case 1, 2:
		sig, ok := strings.CutPrefix(args[0], "-")
		if ok {
			signal, ok = signals[sig]
			if !ok {
				ts.Fatalf("unknown signal: %s", sig)
			}
		} else {
			name = args[0]
			break
		}
		if len(args) == 2 {
			name = args[1]
		}
	default:
		ts.Fatalf("usage: kill [-SIGNAL] [name]")
	}
	if neg {
		ts.Fatalf("unsupported: ! kill")
	}
	if signal == nil {
		signal = os.Kill
	}
	if name != "" {
		ts.killBackgroundOne(name, signal)
	} else {
		ts.killBackground(signal)
	}
}

func (ts *TestScript) killBackgroundOne(bgName string, signal os.Signal) {
	bg := ts.findBackground(bgName)
	if bg == nil {
		ts.Fatalf("unknown background process %q", bgName)
	}
	err := bg.cmd.Process.Signal(signal)
	if err != nil {
		ts.Fatalf("unexpected error terminating background command %q: %v", bgName, err)
	}
}

func (ts *TestScript) killBackground(signal os.Signal) {
	for bgName, bg := range ts.background {
		err := bg.cmd.Process.Signal(signal)
		if err != nil {
			ts.Fatalf("unexpected error terminating background command %q: %v", bgName, err)
		}
	}
}

// cmdWait waits for background commands to exit, setting stderr and stdout to their result.
func (ts *TestScript) cmdWait(neg bool, args []string) {
	if len(args) > 1 {
		ts.Fatalf("usage: wait [name]")
	}
	if neg {
		ts.Fatalf("unsupported: ! wait")
	}
	if len(args) > 0 {
		ts.waitBackgroundOne(args[0])
	} else {
		ts.waitBackground(true)
	}
}

func (ts *TestScript) waitBackgroundOne(bgName string) {
	bg := ts.findBackground(bgName)
	if bg == nil {
		ts.Fatalf("unknown background process %q", bgName)
	}
	<-bg.wait
	ts.stdout = bg.cmd.Stdout.(*strings.Builder).String()
	ts.stderr = bg.cmd.Stderr.(*strings.Builder).String()
	if ts.stdout != "" {
		fmt.Fprintf(&ts.log, "[stdout]\n%s", ts.stdout)
	}
	if ts.stderr != "" {
		fmt.Fprintf(&ts.log, "[stderr]\n%s", ts.stderr)
	}
	// Note: ignore bg.neg, which only takes effect on the non-specific
	// wait command.
	if bg.cmd.ProcessState.Success() {
		if bg.neg {
			ts.Fatalf("unexpected command success")
		}
	} else {
		if ts.ctxt.Err() != nil {
			ts.Fatalf("test timed out while running command")
		} else if !bg.neg {
			ts.Fatalf("unexpected command failure")
		}
	}
	// Remove this process from the list of running background processes.
	for i := range ts.background {
		if bg == &ts.background[i] {
			ts.background = slices.Delete(ts.background, i, i+1)
			break
		}
	}
}

func (ts *TestScript) findBackground(bgName string) *backgroundCmd {
	if bgName == "" {
		return nil
	}
	for i := range ts.background {
		bg := &ts.background[i]
		if bg.name == bgName {
			return bg
		}
	}
	return nil
}

func (ts *TestScript) waitBackground(checkStatus bool) {
	var stdouts, stderrs []string
	for _, bg := range ts.background {
		<-bg.wait

		args := append([]string{filepath.Base(bg.cmd.Args[0])}, bg.cmd.Args[1:]...)
		fmt.Fprintf(&ts.log, "[background] %s: %v\n", strings.Join(args, " "), bg.cmd.ProcessState)

		cmdStdout := bg.cmd.Stdout.(*strings.Builder).String()
		if cmdStdout != "" {
			fmt.Fprintf(&ts.log, "[stdout]\n%s", cmdStdout)
			stdouts = append(stdouts, cmdStdout)
		}

		cmdStderr := bg.cmd.Stderr.(*strings.Builder).String()
		if cmdStderr != "" {
			fmt.Fprintf(&ts.log, "[stderr]\n%s", cmdStderr)
			stderrs = append(stderrs, cmdStderr)
		}

		if !checkStatus {
			continue
		}
		if bg.cmd.ProcessState.Success() {
			if bg.neg {
				ts.Fatalf("unexpected command success")
			}
		} else {
			if ts.ctxt.Err() != nil {
				ts.Fatalf("test timed out while running command")
			} else if !bg.neg {
				ts.Fatalf("unexpected command failure")
			}
		}
	}

	ts.stdout = strings.Join(stdouts, "")
	ts.stderr = strings.Join(stderrs, "")
	ts.background = nil
}

// scriptMatch implements both stdout and stderr.
func scriptMatch(ts *TestScript, neg bool, args []string, text, name string) {
	n := 0
	if len(args) >= 1 && strings.HasPrefix(args[0], "-count=") {
		if neg {
			ts.Fatalf("cannot use -count= with negated match")
		}
		var err error
		n, err = strconv.Atoi(args[0][len("-count="):])
		if err != nil {
			ts.Fatalf("bad -count=: %v", err)
		}
		if n < 1 {
			ts.Fatalf("bad -count=: must be at least 1")
		}
		args = args[1:]
	}

	extraUsage := ""
	want := 1
	if name == "grep" {
		extraUsage = " file"
		want = 2
	}
	if len(args) != want {
		ts.Fatalf("usage: %s [-count=N] 'pattern'%s", name, extraUsage)
	}

	pattern := args[0]
	re, err := regexp.Compile(`(?m)` + pattern)
	ts.Check(err)

	isGrep := name == "grep"
	if isGrep {
		name = args[1] // for error messages
		data, err := os.ReadFile(ts.MkAbs(args[1]))
		ts.Check(err)
		text = string(data)
	}

	if neg {
		if re.MatchString(text) {
			if isGrep {
				ts.Logf("[%s]\n%s\n", name, text)
			}
			ts.Fatalf("unexpected match for %#q found in %s: %s", pattern, name, re.FindString(text))
		}
	} else {
		if !re.MatchString(text) {
			if isGrep {
				ts.Logf("[%s]\n%s\n", name, text)
			}
			ts.Fatalf("no match for %#q found in %s", pattern, name)
		}
		if n > 0 {
			count := len(re.FindAllString(text, -1))
			if count != n {
				if isGrep {
					ts.Logf("[%s]\n%s\n", name, text)
				}
				ts.Fatalf("have %d matches for %#q, want %d", count, pattern, n)
			}
		}
	}
}

// unix2DOS returns data with UNIX line endings converted to DOS line endings.
func unix2DOS(data []byte) ([]byte, error) {
	sb := &strings.Builder{}
	s := bufio.NewScanner(bytes.NewReader(data))
	for s.Scan() {
		if _, err := sb.Write(s.Bytes()); err != nil {
			return nil, err
		}
		if _, err := sb.WriteString("\r\n"); err != nil {
			return nil, err
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return []byte(sb.String()), nil
}
