package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/tools/go/vcs"
)

// VCS represents a version control system.
type VCS struct {
	vcs *vcs.Cmd

	IdentifyCmd string
	DescribeCmd string
	DiffCmd     string
	ListCmd     string
	RootCmd     string

	// run in sandbox repos
	ExistsCmd string
}

var vcsBzr = &VCS{
	vcs: vcs.ByCmd("bzr"),

	IdentifyCmd: "version-info --custom --template {revision_id}",
	DescribeCmd: "revno", // TODO(kr): find tag names if possible
	DiffCmd:     "diff -r {rev}",
	ListCmd:     "ls --from-root -R",
	RootCmd:     "root",
}

var vcsGit = &VCS{
	vcs: vcs.ByCmd("git"),

	IdentifyCmd: "rev-parse HEAD",
	DescribeCmd: "describe --tags --abbrev=14",
	DiffCmd:     "diff {rev}",
	ListCmd:     "ls-files --full-name",
	RootCmd:     "rev-parse --show-cdup",

	ExistsCmd: "cat-file -e {rev}",
}

var vcsHg = &VCS{
	vcs: vcs.ByCmd("hg"),

	IdentifyCmd: "parents --template {node}",
	DescribeCmd: "log -r . --template {latesttag}-{latesttagdistance}",
	DiffCmd:     "diff -r {rev}",
	ListCmd:     "status --all --no-status",
	RootCmd:     "root",

	ExistsCmd: "cat -r {rev} .",
}

var cmd = map[*vcs.Cmd]*VCS{
	vcsBzr.vcs: vcsBzr,
	vcsGit.vcs: vcsGit,
	vcsHg.vcs:  vcsHg,
}

// VCSFromDir returns a VCS value from a directory.
func VCSFromDir(dir, srcRoot string) (*VCS, string, error) {
	vcscmd, reporoot, err := vcs.FromDir(dir, srcRoot)
	if err != nil {
		return nil, "", fmt.Errorf("error while inspecting %q: %v", dir, err)
	}
	vcsext := cmd[vcscmd]
	if vcsext == nil {
		return nil, "", fmt.Errorf("%s is unsupported: %s", vcscmd.Name, dir)
	}
	return vcsext, reporoot, nil
}

func (v *VCS) identify(dir string) (string, error) {
	out, err := v.runOutput(dir, v.IdentifyCmd)
	return string(bytes.TrimSpace(out)), err
}

func absRoot(dir, out string) string {
	if filepath.IsAbs(out) {
		return filepath.Clean(out)
	}
	return filepath.Join(dir, out)
}

func (v *VCS) root(dir string) (string, error) {
	out, err := v.runOutput(dir, v.RootCmd)
	return absRoot(dir, string(bytes.TrimSpace(out))), err
}

func (v *VCS) describe(dir, rev string) string {
	out, err := v.runOutputVerboseOnly(dir, v.DescribeCmd, "rev", rev)
	if err != nil {
		return ""
	}
	return string(bytes.TrimSpace(out))
}

func (v *VCS) isDirty(dir, rev string) bool {
	out, err := v.runOutput(dir, v.DiffCmd, "rev", rev)
	return err != nil || len(out) != 0
}

type vcsFiles map[string]bool

func (vf vcsFiles) Contains(path string) bool {
	// Fast path, we have the path
	if vf[path] {
		return true
	}

	// Slow path for case insensitive filesystems
	// See #310
	for f := range vf {
		if pathEqual(f, path) {
			return true
		}
		// git's root command (maybe other vcs as well) resolve symlinks, so try that too
		// FIXME: rev-parse --show-cdup + extra logic will fix this for git but also need to validate the other vcs commands. This is maybe temporary.
		p, err := filepath.EvalSymlinks(path)
		if err != nil {
			return false
		}
		if pathEqual(f, p) {
			return true
		}
	}

	// No matches by either method
	return false
}

// listFiles tracked by the VCS in the repo that contains dir, converted to absolute path.
func (v *VCS) listFiles(dir string) vcsFiles {
	root, err := v.root(dir)
	debugln("vcs dir", dir)
	debugln("vcs root", root)
	ppln(v)
	if err != nil {
		return nil
	}
	out, err := v.runOutput(dir, v.ListCmd)
	if err != nil {
		return nil
	}
	files := make(vcsFiles)
	for _, file := range bytes.Split(out, []byte{'\n'}) {
		if len(file) > 0 {
			path, err := filepath.Abs(filepath.Join(root, string(file)))
			if err != nil {
				panic(err) // this should not happen
			}

			if pathEqual(filepath.Dir(path), dir) {
				files[path] = true
			}
		}
	}
	return files
}

func (v *VCS) exists(dir, rev string) bool {
	err := v.runVerboseOnly(dir, v.ExistsCmd, "rev", rev)
	return err == nil
}

// RevSync checks out the revision given by rev in dir.
// The dir must exist and rev must be a valid revision.
func (v *VCS) RevSync(dir, rev string) error {
	return v.run(dir, v.vcs.TagSyncCmd, "tag", rev)
}

// run runs the command line cmd in the given directory.
// keyval is a list of key, value pairs.  run expands
// instances of {key} in cmd into value, but only after
// splitting cmd into individual arguments.
// If an error occurs, run prints the command line and the
// command's combined stdout+stderr to standard error.
// Otherwise run discards the command's output.
func (v *VCS) run(dir string, cmdline string, kv ...string) error {
	_, err := v.run1(dir, cmdline, kv, true)
	return err
}

// runVerboseOnly is like run but only generates error output to standard error in verbose mode.
func (v *VCS) runVerboseOnly(dir string, cmdline string, kv ...string) error {
	_, err := v.run1(dir, cmdline, kv, false)
	return err
}

// runOutput is like run but returns the output of the command.
func (v *VCS) runOutput(dir string, cmdline string, kv ...string) ([]byte, error) {
	return v.run1(dir, cmdline, kv, true)
}

// runOutputVerboseOnly is like runOutput but only generates error output to standard error in verbose mode.
func (v *VCS) runOutputVerboseOnly(dir string, cmdline string, kv ...string) ([]byte, error) {
	return v.run1(dir, cmdline, kv, false)
}

// run1 is the generalized implementation of run and runOutput.
func (v *VCS) run1(dir string, cmdline string, kv []string, verbose bool) ([]byte, error) {
	m := make(map[string]string)
	for i := 0; i < len(kv); i += 2 {
		m[kv[i]] = kv[i+1]
	}
	args := strings.Fields(cmdline)
	for i, arg := range args {
		args[i] = expand(m, arg)
	}

	_, err := exec.LookPath(v.vcs.Cmd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "godep: missing %s command.\n", v.vcs.Name)
		return nil, err
	}

	cmd := exec.Command(v.vcs.Cmd, args...)
	cmd.Dir = dir
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err = cmd.Run()
	out := buf.Bytes()
	if err != nil {
		if verbose {
			fmt.Fprintf(os.Stderr, "# cd %s; %s %s\n", dir, v.vcs.Cmd, strings.Join(args, " "))
			os.Stderr.Write(out)
		}
		return nil, err
	}
	return out, nil
}

func expand(m map[string]string, s string) string {
	for k, v := range m {
		s = strings.Replace(s, "{"+k+"}", v, -1)
	}
	return s
}

func gitDetached(r string) (bool, error) {
	o, err := vcsGit.runOutput(r, "status")
	if err != nil {
		return false, errors.New("unable to determine git status " + err.Error())
	}
	return bytes.Contains(o, []byte("HEAD detached at")), nil
}

func gitDefaultBranch(r string) (string, error) {
	o, err := vcsGit.runOutput(r, "remote show origin")
	if err != nil {
		return "", errors.New("Running git remote show origin errored with: " + err.Error())
	}
	return gitDetermineDefaultBranch(r, string(o))
}

func gitDetermineDefaultBranch(r, o string) (string, error) {
	e := "Unable to determine HEAD branch: "
	hb := "HEAD branch:"
	lbcfgp := "Local branch configured for 'git pull':"
	s := strings.Index(o, hb)
	if s < 0 {
		b := strings.Index(o, lbcfgp)
		if b < 0 {
			return "", errors.New(e + "Remote HEAD is ambiguous. Before godep can pull new commits you will need to:" + `
cd ` + r + `
git checkout <a HEAD branch>
Here is what was reported:
` + o)
		}
		s = b + len(lbcfgp)
	} else {
		s += len(hb)
	}
	f := strings.Fields(o[s:])
	if len(f) < 3 {
		return "", errors.New(e + "git output too short")
	}
	return f[0], nil
}

func gitCheckout(r, b string) error {
	return vcsGit.run(r, "checkout "+b)
}
