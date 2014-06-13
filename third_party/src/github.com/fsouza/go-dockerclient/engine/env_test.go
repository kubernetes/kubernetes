// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package engine

import (
	"testing"
)

func TestNewJob(t *testing.T) {
	job := mkJob(t, "dummy", "--level=awesome")
	if job.Name != "dummy" {
		t.Fatalf("Wrong job name: %s", job.Name)
	}
	if len(job.Args) != 1 {
		t.Fatalf("Wrong number of job arguments: %d", len(job.Args))
	}
	if job.Args[0] != "--level=awesome" {
		t.Fatalf("Wrong job arguments: %s", job.Args[0])
	}
}

func TestSetenv(t *testing.T) {
	job := mkJob(t, "dummy")
	job.Setenv("foo", "bar")
	if val := job.Getenv("foo"); val != "bar" {
		t.Fatalf("Getenv returns incorrect value: %s", val)
	}

	job.Setenv("bar", "")
	if val := job.Getenv("bar"); val != "" {
		t.Fatalf("Getenv returns incorrect value: %s", val)
	}
	if val := job.Getenv("nonexistent"); val != "" {
		t.Fatalf("Getenv returns incorrect value: %s", val)
	}
}

func TestSetenvBool(t *testing.T) {
	job := mkJob(t, "dummy")
	job.SetenvBool("foo", true)
	if val := job.GetenvBool("foo"); !val {
		t.Fatalf("GetenvBool returns incorrect value: %t", val)
	}

	job.SetenvBool("bar", false)
	if val := job.GetenvBool("bar"); val {
		t.Fatalf("GetenvBool returns incorrect value: %t", val)
	}

	if val := job.GetenvBool("nonexistent"); val {
		t.Fatalf("GetenvBool returns incorrect value: %t", val)
	}
}

func TestSetenvInt(t *testing.T) {
	job := mkJob(t, "dummy")

	job.SetenvInt("foo", -42)
	if val := job.GetenvInt("foo"); val != -42 {
		t.Fatalf("GetenvInt returns incorrect value: %d", val)
	}

	job.SetenvInt("bar", 42)
	if val := job.GetenvInt("bar"); val != 42 {
		t.Fatalf("GetenvInt returns incorrect value: %d", val)
	}
	if val := job.GetenvInt("nonexistent"); val != -1 {
		t.Fatalf("GetenvInt returns incorrect value: %d", val)
	}
}

func TestSetenvList(t *testing.T) {
	job := mkJob(t, "dummy")

	job.SetenvList("foo", []string{"bar"})
	if val := job.GetenvList("foo"); len(val) != 1 || val[0] != "bar" {
		t.Fatalf("GetenvList returns incorrect value: %v", val)
	}

	job.SetenvList("bar", nil)
	if val := job.GetenvList("bar"); val != nil {
		t.Fatalf("GetenvList returns incorrect value: %v", val)
	}
	if val := job.GetenvList("nonexistent"); val != nil {
		t.Fatalf("GetenvList returns incorrect value: %v", val)
	}
}

func TestImportEnv(t *testing.T) {
	type dummy struct {
		DummyInt         int
		DummyStringArray []string
	}

	job := mkJob(t, "dummy")
	if err := job.ImportEnv(&dummy{42, []string{"foo", "bar"}}); err != nil {
		t.Fatal(err)
	}

	dmy := dummy{}
	if err := job.ExportEnv(&dmy); err != nil {
		t.Fatal(err)
	}

	if dmy.DummyInt != 42 {
		t.Fatalf("Expected 42, got %d", dmy.DummyInt)
	}

	if len(dmy.DummyStringArray) != 2 || dmy.DummyStringArray[0] != "foo" || dmy.DummyStringArray[1] != "bar" {
		t.Fatalf("Expected {foo, bar}, got %v", dmy.DummyStringArray)
	}

}

func TestEnviron(t *testing.T) {
	job := mkJob(t, "dummy")
	job.Setenv("foo", "bar")
	val, exists := job.Environ()["foo"]
	if !exists {
		t.Fatalf("foo not found in the environ")
	}
	if val != "bar" {
		t.Fatalf("bar not found in the environ")
	}
}
