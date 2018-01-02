// +build linux

package apparmor

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"text/template"

	"github.com/pkg/errors"
)

const dir = "/etc/apparmor.d"

const defaultTemplate = `
{{range $value := .Imports}}
{{$value}}
{{end}}

profile {{.Name}} flags=(attach_disconnected,mediate_deleted) {
{{range $value := .InnerImports}}
  {{$value}}
{{end}}

  network,
  capability,
  file,
  umount,

  deny @{PROC}/* w,   # deny write for all files directly in /proc (not in a subdir)
  # deny write to files not in /proc/<number>/** or /proc/sys/**
  deny @{PROC}/{[^1-9],[^1-9][^0-9],[^1-9s][^0-9y][^0-9s],[^1-9][^0-9][^0-9][^0-9]*}/** w,
  deny @{PROC}/sys/[^k]** w,  # deny /proc/sys except /proc/sys/k* (effectively /proc/sys/kernel)
  deny @{PROC}/sys/kernel/{?,??,[^s][^h][^m]**} w,  # deny everything except shm* in /proc/sys/kernel/
  deny @{PROC}/sysrq-trigger rwklx,
  deny @{PROC}/mem rwklx,
  deny @{PROC}/kmem rwklx,
  deny @{PROC}/kcore rwklx,

  deny mount,

  deny /sys/[^f]*/** wklx,
  deny /sys/f[^s]*/** wklx,
  deny /sys/fs/[^c]*/** wklx,
  deny /sys/fs/c[^g]*/** wklx,
  deny /sys/fs/cg[^r]*/** wklx,
  deny /sys/firmware/** rwklx,
  deny /sys/kernel/security/** rwklx,

{{if ge .Version 208095}}
  ptrace (trace,read) peer={{.Name}},
{{end}}
}
`

type data struct {
	Name         string
	Imports      []string
	InnerImports []string
	Version      int
}

func loadData(name string) (*data, error) {
	p := data{
		Name: name,
	}

	if macroExists("tunables/global") {
		p.Imports = append(p.Imports, "#include <tunables/global>")
	} else {
		p.Imports = append(p.Imports, "@{PROC}=/proc/")
	}
	if macroExists("abstractions/base") {
		p.InnerImports = append(p.InnerImports, "#include <abstractions/base>")
	}
	ver, err := getVersion()
	if err != nil {
		return nil, errors.Wrap(err, "get apparmor_parser version")
	}
	p.Version = ver
	return &p, nil
}

func generate(p *data, o io.Writer) error {
	t, err := template.New("apparmor_profile").Parse(defaultTemplate)
	if err != nil {
		return err
	}
	return t.Execute(o, p)
}

func load(path string) error {
	out, err := aaParser("-Kr", path)
	if err != nil {
		return errors.Errorf("%s: %s", err, out)
	}
	return nil
}

// macrosExists checks if the passed macro exists.
func macroExists(m string) bool {
	_, err := os.Stat(path.Join(dir, m))
	return err == nil
}

func aaParser(args ...string) (string, error) {
	out, err := exec.Command("apparmor_parser", args...).CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func getVersion() (int, error) {
	out, err := aaParser("--version")
	if err != nil {
		return -1, err
	}
	return parseVersion(out)
}

// parseVersion takes the output from `apparmor_parser --version` and returns
// a representation of the {major, minor, patch} version as a single number of
// the form MMmmPPP {major, minor, patch}.
func parseVersion(output string) (int, error) {
	// output is in the form of the following:
	// AppArmor parser version 2.9.1
	// Copyright (C) 1999-2008 Novell Inc.
	// Copyright 2009-2012 Canonical Ltd.

	lines := strings.SplitN(output, "\n", 2)
	words := strings.Split(lines[0], " ")
	version := words[len(words)-1]

	// split by major minor version
	v := strings.Split(version, ".")
	if len(v) == 0 || len(v) > 3 {
		return -1, fmt.Errorf("parsing version failed for output: `%s`", output)
	}

	// Default the versions to 0.
	var majorVersion, minorVersion, patchLevel int

	majorVersion, err := strconv.Atoi(v[0])
	if err != nil {
		return -1, err
	}

	if len(v) > 1 {
		minorVersion, err = strconv.Atoi(v[1])
		if err != nil {
			return -1, err
		}
	}
	if len(v) > 2 {
		patchLevel, err = strconv.Atoi(v[2])
		if err != nil {
			return -1, err
		}
	}

	// major*10^5 + minor*10^3 + patch*10^0
	numericVersion := majorVersion*1e5 + minorVersion*1e3 + patchLevel
	return numericVersion, nil
}

func isLoaded(name string) (bool, error) {
	f, err := os.Open("/sys/kernel/security/apparmor/profiles")
	if err != nil {
		return false, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	for {
		p, err := r.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			return false, err
		}
		if strings.HasPrefix(p, name+" ") {
			return true, nil
		}
	}
	return false, nil
}
