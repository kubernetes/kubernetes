// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package svc_test

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

func getState(t *testing.T, s *mgr.Service) svc.State {
	status, err := s.Query()
	if err != nil {
		t.Fatalf("Query(%s) failed: %s", s.Name, err)
	}
	return status.State
}

func testState(t *testing.T, s *mgr.Service, want svc.State) {
	have := getState(t, s)
	if have != want {
		t.Fatalf("%s state is=%d want=%d", s.Name, have, want)
	}
}

func waitState(t *testing.T, s *mgr.Service, want svc.State) {
	for i := 0; ; i++ {
		have := getState(t, s)
		if have == want {
			return
		}
		if i > 10 {
			t.Fatalf("%s state is=%d, waiting timeout", s.Name, have)
		}
		time.Sleep(300 * time.Millisecond)
	}
}

// stopAndDeleteIfInstalled stops and deletes service name,
// if the service is running and / or installed.
func stopAndDeleteIfInstalled(t *testing.T, m *mgr.Mgr, name string) {
	s, err := m.OpenService(name)
	if err != nil {
		// Service is not installed.
		return

	}
	defer s.Close()

	// Make sure the service is not running, otherwise we won't be able to delete it.
	if getState(t, s) == svc.Running {
		_, err = s.Control(svc.Stop)
		if err != nil {
			t.Fatalf("Control(%s) failed: %s", s.Name, err)
		}
		waitState(t, s, svc.Stopped)
	}

	err = s.Delete()
	if err != nil {
		t.Fatalf("Delete failed: %s", err)
	}
}

func TestExample(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode - it modifies system services")
	}

	const name = "myservice"

	m, err := mgr.Connect()
	if err != nil {
		t.Fatalf("SCM connection failed: %s", err)
	}
	defer m.Disconnect()

	dir, err := ioutil.TempDir("", "svc")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	exepath := filepath.Join(dir, "a.exe")
	o, err := exec.Command("go", "build", "-o", exepath, "golang.org/x/sys/windows/svc/example").CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build service program: %v\n%v", err, string(o))
	}

	stopAndDeleteIfInstalled(t, m, name)

	s, err := m.CreateService(name, exepath, mgr.Config{DisplayName: "my service"}, "is", "auto-started")
	if err != nil {
		t.Fatalf("CreateService(%s) failed: %v", name, err)
	}
	defer s.Close()

	args := []string{"is", "manual-started", fmt.Sprintf("%d", rand.Int())}

	testState(t, s, svc.Stopped)
	err = s.Start(args...)
	if err != nil {
		t.Fatalf("Start(%s) failed: %s", s.Name, err)
	}
	waitState(t, s, svc.Running)
	time.Sleep(1 * time.Second)

	// testing deadlock from issues 4.
	_, err = s.Control(svc.Interrogate)
	if err != nil {
		t.Fatalf("Control(%s) failed: %s", s.Name, err)
	}
	_, err = s.Control(svc.Interrogate)
	if err != nil {
		t.Fatalf("Control(%s) failed: %s", s.Name, err)
	}
	time.Sleep(1 * time.Second)

	_, err = s.Control(svc.Stop)
	if err != nil {
		t.Fatalf("Control(%s) failed: %s", s.Name, err)
	}
	waitState(t, s, svc.Stopped)

	err = s.Delete()
	if err != nil {
		t.Fatalf("Delete failed: %s", err)
	}

	out, err := exec.Command("wevtutil.exe", "qe", "Application", "/q:*[System[Provider[@Name='myservice']]]", "/rd:true", "/c:10").CombinedOutput()
	if err != nil {
		t.Fatalf("wevtutil failed: %v\n%v", err, string(out))
	}
	want := strings.Join(append([]string{name}, args...), "-")
	// Test context passing (see servicemain in sys_386.s and sys_amd64.s).
	want += "-123456"
	if !strings.Contains(string(out), want) {
		t.Errorf("%q string does not contain %q", string(out), want)
	}
}

func TestIsAnInteractiveSession(t *testing.T) {
	isInteractive, err := svc.IsAnInteractiveSession()
	if err != nil {
		t.Fatal(err)
	}
	if !isInteractive {
		t.Error("IsAnInteractiveSession retuns false when running interactively.")
	}
}

func TestIsWindowsService(t *testing.T) {
	isSvc, err := svc.IsWindowsService()
	if err != nil {
		t.Fatal(err)
	}
	if isSvc {
		t.Error("IsWindowsService retuns true when not running in a service.")
	}
}
