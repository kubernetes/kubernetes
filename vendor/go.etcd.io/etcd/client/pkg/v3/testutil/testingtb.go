// Copyright 2021 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testutil

import (
	"io/ioutil"
	"log"
	"os"
)

// TB is a subset of methods of testing.TB interface.
// We cannot implement testing.TB due to protection, so we expose this simplified interface.
type TB interface {
	Cleanup(func())
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Logf(format string, args ...interface{})
	Name() string
	TempDir() string
	Helper()
	Skip(args ...interface{})
}

// NewTestingTBProthesis creates a fake variant of testing.TB implementation.
// It's supposed to be used in contexts were real testing.T is not provided,
// e.g. in 'examples'.
//
// The `closef` goroutine should get executed when tb will not be needed any longer.
//
// The provided implementation is NOT thread safe (Cleanup() method).
func NewTestingTBProthesis(name string) (tb TB, closef func()) {
	testtb := &testingTBProthesis{name: name}
	return testtb, testtb.close
}

type testingTBProthesis struct {
	name     string
	failed   bool
	cleanups []func()
}

func (t *testingTBProthesis) Helper() {
	// Ignored
}

func (t *testingTBProthesis) Skip(args ...interface{}) {
	t.Log(append([]interface{}{"Skipping due to: "}, args...))
}

func (t *testingTBProthesis) Cleanup(f func()) {
	t.cleanups = append(t.cleanups, f)
}

func (t *testingTBProthesis) Error(args ...interface{}) {
	log.Println(args...)
	t.Fail()
}

func (t *testingTBProthesis) Errorf(format string, args ...interface{}) {
	log.Printf(format, args...)
	t.Fail()
}

func (t *testingTBProthesis) Fail() {
	t.failed = true
}

func (t *testingTBProthesis) FailNow() {
	t.failed = true
	panic("FailNow() called")
}

func (t *testingTBProthesis) Failed() bool {
	return t.failed
}

func (t *testingTBProthesis) Fatal(args ...interface{}) {
	log.Fatalln(args...)
}

func (t *testingTBProthesis) Fatalf(format string, args ...interface{}) {
	log.Fatalf(format, args...)
}

func (t *testingTBProthesis) Logf(format string, args ...interface{}) {
	log.Printf(format, args...)
}

func (t *testingTBProthesis) Log(args ...interface{}) {
	log.Println(args...)
}

func (t *testingTBProthesis) Name() string {
	return t.name
}

func (t *testingTBProthesis) TempDir() string {
	dir, err := ioutil.TempDir("", t.name)
	if err != nil {
		t.Fatal(err)
	}
	t.cleanups = append([]func(){func() {
		t.Logf("Cleaning UP: %v", dir)
		os.RemoveAll(dir)
	}}, t.cleanups...)
	return dir
}

func (t *testingTBProthesis) close() {
	for i := len(t.cleanups) - 1; i >= 0; i-- {
		t.cleanups[i]()
	}
}
