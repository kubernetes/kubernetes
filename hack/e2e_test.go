/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Unit tests for hack/e2e.go shim
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

type FileInfo struct {
	when time.Time
}

func (f FileInfo) Name() string {
	return "fake-file"
}

func (f FileInfo) Size() int64 {
	return 0
}

func (f FileInfo) Mode() os.FileMode {
	return 0
}

func (f FileInfo) ModTime() time.Time {
	return f.when
}

func (f FileInfo) IsDir() bool {
	return false
}

func (f FileInfo) Sys() interface{} {
	return f
}

func TestParse(t *testing.T) {
	cases := []struct {
		args     []string
		expected flags
		err      error
	}{
		{
			[]string{"hello", "world"},
			flags{getDefault, oldDefault, []string{"world"}},
			nil,
		},
		{
			[]string{"hello", "--", "--venus", "--karaoke"},
			flags{getDefault, oldDefault, []string{"--venus", "--karaoke"}},
			nil,
		},
		{
			[]string{"hello", "--alpha", "--beta"},
			flags{getDefault, oldDefault, []string{"--alpha", "--beta"}},
			nil,
		},
		{
			[]string{"so", "--get", "--boo"},
			flags{true, oldDefault, []string{"--boo"}},
			nil,
		},
		{
			[]string{"omg", "--get=false", "--", "ugh"},
			flags{false, oldDefault, []string{"ugh"}},
			nil,
		},
		{
			[]string{"wee", "--old=5m", "--get"},
			flags{true, 5 * time.Minute, []string{}},
			nil,
		},
		{
			[]string{"fun", "--times", "--old=666s"},
			flags{getDefault, oldDefault, []string{"--times", "--old=666s"}},
			nil,
		},
		{
			[]string{"wut", "-h"},
			flags{},
			flag.ErrHelp,
		},
		{
			[]string{"wut", "--", "-h"},
			flags{getDefault, oldDefault, []string{"-h"}},
			nil,
		},
	}

	for i, c := range cases {
		a, err := parse(c.args)
		if err != c.err {
			t.Errorf("%d: a=%v != e%v", i, err, c.err)
		}
		e := c.expected
		if a.get != e.get {
			t.Errorf("%d: a=%v != e=%v", i, a.get, e.get)
		}
		if a.old != e.old {
			t.Errorf("%d: a=%v != e=%v", i, a.old, e.old)
		}
		if !reflect.DeepEqual(a.args, e.args) {
			t.Errorf("%d: a=%v != e=%v", i, a.args, e.args)
		}
	}
}

func TestLook(t *testing.T) {
	lpf := errors.New("LookPath failed")
	sf := errors.New("Stat failed")
	lpnc := errors.New("LookPath should not be called")
	snc := errors.New("Stat should not be called")
	cases := []struct {
		stat     error
		lookPath error
		goPath   string
		expected error
	}{
		{ // GOPATH set, stat succeeds returns gopath
			stat:     nil,
			lookPath: lpnc,
			goPath:   "fake-gopath/",
			expected: nil,
		},
		{ // GOPATH set, stat fails, terms on lookpath
			stat:     sf,
			lookPath: lpf,
			goPath:   "fake-gopath/",
			expected: lpf,
		},
		{ // GOPATH unset, stat not called, terms on lookpath
			stat:     snc,
			lookPath: lpf,
			goPath:   "",
			expected: lpf,
		},
		{ // GOPATH unset, stat not called, lookpath matches
			stat:     snc,
			lookPath: nil,
			goPath:   "",
			expected: nil,
		},
	}

	for _, c := range cases {
		l := tester{
			func(string) (os.FileInfo, error) {
				return FileInfo{}, c.stat
			},
			func(string) (string, error) {
				if c.lookPath != nil {
					return "FAILED", c.lookPath
				}
				return "$PATH-FOUND", nil
			},
			c.goPath,
			nil, // wait
		}
		if _, err := l.lookKubetest(); err != c.expected {
			t.Errorf("err: %s != %s", err, c.expected)
		}
	}
}

func TestGetKubetest(t *testing.T) {
	gp := "fake-gopath"
	gpk := filepath.Join(gp, "bin", "kubetest")
	p := "PATH"
	pk := filepath.Join(p, "kubetest")
	eu := errors.New("upgrade failed")
	euVerbose := fmt.Errorf("go get -u k8s.io/test-infra/kubetest: %v", eu)
	et := errors.New("touch failed")
	cases := []struct {
		name string
		get  bool
		old  time.Duration

		stat     string        // stat succeeds on this file
		path     bool          // file exists on path
		age      time.Duration // age of mod time on file
		upgraded bool          // go get -u succeeds
		touched  bool          // touch succeeds
		goPath   string        // GOPATH var

		returnPath  string
		returnError error
	}{
		{name: "0: Pass when on GOPATH/bin",
			get: false,
			old: 0,

			stat:     gpk,
			path:     false,
			age:      100,
			upgraded: false,
			touched:  false,
			goPath:   gp,

			returnPath:  gpk,
			returnError: nil,
		},
		{name: "1: Pass when on PATH",
			get: false,
			old: 0,

			stat:     pk,
			path:     true,
			age:      100,
			upgraded: false,
			touched:  false,
			goPath:   gp,

			returnPath:  pk,
			returnError: nil,
		},
		{name: "2: Don't upgrade if on PATH and GOPATH is ''",
			get: true,
			old: 0,

			stat:     pk,
			path:     true,
			age:      100,
			upgraded: false,
			touched:  false,
			goPath:   "",

			returnPath:  pk,
			returnError: nil,
		},
		{name: "3: Don't upgrade on PATH when young.",
			get: true,
			old: time.Hour,

			stat:     pk,
			path:     true,
			age:      time.Second,
			upgraded: false,
			touched:  false,
			goPath:   gp,

			returnPath:  pk,
			returnError: nil,
		},
		{name: "4: Upgrade if old but GOPATH is set.",
			get: true,
			old: 0,

			stat:     pk,
			path:     true,
			age:      time.Second,
			upgraded: true,
			touched:  true,
			goPath:   gp,

			returnPath:  pk,
			returnError: nil,
		},
		{name: "5: Fail if upgrade fails",
			get: true,
			old: 0,

			stat:     pk,
			path:     true,
			age:      time.Second,
			upgraded: false,
			touched:  false,
			goPath:   gpk,

			returnPath:  "",
			returnError: euVerbose,
		},
		{name: "6: Fail if touch fails",
			get: true,
			old: 0,

			stat:     pk,
			path:     true,
			age:      time.Second,
			upgraded: true,
			touched:  false,
			goPath:   gpk,

			returnPath:  "",
			returnError: et,
		},
	}

	for i, c := range cases {
		didUp := false
		didTouch := false
		l := tester{
			stat: func(p string) (os.FileInfo, error) {
				// stat
				if p != c.stat {
					return nil, fmt.Errorf("Failed to find %s", p)
				}
				return FileInfo{time.Now().Add(c.age * -1)}, nil
			},
			lookPath: func(name string) (string, error) {
				if c.path {
					return filepath.Join(p, name), nil
				}
				return "", fmt.Errorf("Not on path: %s", name)
			},
			goPath: c.goPath,
			wait: func(cmd string, args ...string) error {
				if cmd == "go" {
					if c.upgraded {
						didUp = true
						return nil
					}
					return eu
				}
				if c.touched {
					didTouch = true
					return nil
				}
				return et
			},
		}
		p, e := l.getKubetest(c.get, c.old)
		if p != c.returnPath {
			t.Errorf("%d: test=%q returnPath %q != %q", i, c.name, p, c.returnPath)
		}
		if e == nil || c.returnError == nil {
			if e != c.returnError {
				t.Errorf("%d: test=%q returnError %q != %q", i, c.name, e, c.returnError)
			}
		} else {
			if e.Error() != c.returnError.Error() {
				t.Errorf("%d: test=%q returnError %q != %q", i, c.name, e, c.returnError)
			}
		}
		if didUp != c.upgraded {
			t.Errorf("%d: test=%q bad upgrade state of %v", i, c.name, didUp)
		}
		if didTouch != c.touched {
			t.Errorf("%d: test=%q bad touch state of %v", i, c.name, didTouch)
		}
	}
}
