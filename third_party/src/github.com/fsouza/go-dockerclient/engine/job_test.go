// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package engine

import (
	"os"
	"testing"
)

func TestJobStatusOK(t *testing.T) {
	eng := newTestEngine(t)
	defer os.RemoveAll(eng.Root())
	eng.Register("return_ok", func(job *Job) Status { return StatusOK })
	err := eng.Job("return_ok").Run()
	if err != nil {
		t.Fatalf("Expected: err=%v\nReceived: err=%v", nil, err)
	}
}

func TestJobStatusErr(t *testing.T) {
	eng := newTestEngine(t)
	defer os.RemoveAll(eng.Root())
	eng.Register("return_err", func(job *Job) Status { return StatusErr })
	err := eng.Job("return_err").Run()
	if err == nil {
		t.Fatalf("When a job returns StatusErr, Run() should return an error")
	}
}

func TestJobStatusNotFound(t *testing.T) {
	eng := newTestEngine(t)
	defer os.RemoveAll(eng.Root())
	eng.Register("return_not_found", func(job *Job) Status { return StatusNotFound })
	err := eng.Job("return_not_found").Run()
	if err == nil {
		t.Fatalf("When a job returns StatusNotFound, Run() should return an error")
	}
}

func TestJobStdoutString(t *testing.T) {
	eng := newTestEngine(t)
	defer os.RemoveAll(eng.Root())
	// FIXME: test multiple combinations of output and status
	eng.Register("say_something_in_stdout", func(job *Job) Status {
		job.Printf("Hello world\n")
		return StatusOK
	})

	job := eng.Job("say_something_in_stdout")
	var output string
	if err := job.Stdout.AddString(&output); err != nil {
		t.Fatal(err)
	}
	if err := job.Run(); err != nil {
		t.Fatal(err)
	}
	if expectedOutput := "Hello world"; output != expectedOutput {
		t.Fatalf("Stdout last line:\nExpected: %v\nReceived: %v", expectedOutput, output)
	}
}

func TestJobStderrString(t *testing.T) {
	eng := newTestEngine(t)
	defer os.RemoveAll(eng.Root())
	// FIXME: test multiple combinations of output and status
	eng.Register("say_something_in_stderr", func(job *Job) Status {
		job.Errorf("Warning, something might happen\nHere it comes!\nOh no...\nSomething happened\n")
		return StatusOK
	})

	job := eng.Job("say_something_in_stderr")
	var output string
	if err := job.Stderr.AddString(&output); err != nil {
		t.Fatal(err)
	}
	if err := job.Run(); err != nil {
		t.Fatal(err)
	}
	if expectedOutput := "Something happened"; output != expectedOutput {
		t.Fatalf("Stderr last line:\nExpected: %v\nReceived: %v", expectedOutput, output)
	}
}
