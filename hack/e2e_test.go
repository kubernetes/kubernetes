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
		args []string
		expected flags
	}{
		{
			[]string{"hello", "world"},
			flags{getDefault, oldDefault, []string{"world"}},
		},
		{
			[]string{"hello", "--", "--venus", "--karaoke"},
			flags{getDefault, oldDefault, []string{"--venus", "--karaoke"}},
		},
		{
			[]string{"hello", "--alpha", "--beta"},
			flags{getDefault, oldDefault, []string{"--alpha", "--beta"}},
		},
		{
			[]string{"so", "--get", "--boo"},
			flags{true, oldDefault, []string{"--boo"}},
		},
		{
			[]string{"omg", "--get=false", "--", "ugh"},
			flags{false, oldDefault, []string{"ugh"}},
		},
		{
			[]string{"wee", "--old=5m", "--get"},
			flags{true, 5*time.Minute, []string{}},
		},
		{
			[]string{"fun", "--times", "--old=666s"},
			flags{getDefault, oldDefault, []string{"--times", "--old=666s"}},
		},
	}

	for i, c := range cases {
		a := parse(c.args)
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
			nil,
			lpnc,
			"fake-gopath/",
			nil,
		},
		{ // GOPATH set, stat fails, terms on lookpath
			sf,
			lpf,
			"fake-gopath/",
			lpf,
		},
		{ // GOPATH unset, stat not called, terms on lookpath
			snc,
			lpf,
			"",
			lpf,
		},
		{ // GOPATH unset, stat not called, lookpath matches
			snc,
			nil,
			"",
			nil,
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
		if _, err := l.look(); err != c.expected {
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
	et := errors.New("touch failed")
	cases := []struct {
		get bool
		old time.Duration

		stat     string        // stat succeeds on this file
		path     bool          // file exists on path
		age      time.Duration // age of mod time on file
		upgraded bool          // go get -u succeeds
		touched  bool          // touch succeeds
		goPath   string        // GOPATH var

		returnPath  string
		returnError error
	}{
		{ // 0: Pass when on GOPATH/bin
			false,
			0,

			gpk,
			false,
			100,
			false,
			false,
			gp,

			gpk,
			nil,
		},
		{ // 1: Pass when on PATH
			false,
			0,

			pk,
			true,
			100,
			false,
			false,
			gp,

			pk,
			nil,
		},
		{ // 2: Don't upgrade if on PATH and GOPATH is ""
			true,
			0,

			pk,
			true,
			100,
			false,
			false,
			"",

			pk,
			nil,
		},
		{ // 3: Don't upgrade on PATH when young.
			true,
			time.Hour,

			pk,
			true,
			time.Second,
			false,
			false,
			gp,

			pk,
			nil,
		},
		{ // 4: Upgrade if old but GOPATH is set.
			true,
			0,

			pk,
			true,
			100,
			true,
			true,
			gp,

			pk,
			nil,
		},
		{ // 5: Fail if upgrade fails
			true,
			0,

			pk,
			true,
			100,
			false,
			false,
			gpk,

			"",
			eu,
		},
		{ // 6: Fail if touch fails
			true,
			0,

			pk,
			true,
			100,
			true,
			false,
			gpk,

			"",
			et,
		},
	}

	for i, c := range cases {
		didUp := false
		didTouch := false
		l := tester{
			func(p string) (os.FileInfo, error) {
				// stat
				if p != c.stat {
					return nil, fmt.Errorf("Failed to find %s", p)
				}
				return FileInfo{time.Now().Add(c.age)}, nil
			},
			func(name string) (string, error) {
				if c.path {
					return filepath.Join(p, name), nil
				}
				return "", fmt.Errorf("Not on path: %s", name)
			},
			c.goPath,
			func(cmd string, args ...string) error {
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
		if p, e := l.getKubetest(c.get, c.old); p != c.returnPath || e != c.returnError {
			t.Errorf("%d: c=%v p=%v e=%v", i, c, p, e)
		}
		if didUp != c.upgraded {
			t.Errorf("%d: bad upgrade state of %v", i, didUp)
		}
		if didTouch != c.touched {
			t.Errorf("%d: bad touch state of %v", i, didTouch)
		}
	}
}
