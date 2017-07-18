/*
Copyright 2014 The Kubernetes Authors.

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

// User-interface for test-infra/kubetest/e2e.go
// Equivalent to go get -u k8s.io/test-infra/kubetest && kubetest "${@}"
package main

import (
	"flag"
	"fmt"
	"go/build"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type flags struct {
	get  bool
	old  time.Duration
	args []string
}

const (
	getDefault = true
	oldDefault = 24 * time.Hour
)

func parse(args []string) (flags, error) {
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	get := fs.Bool("get", getDefault, "go get -u kubetest if old or not installed")
	old := fs.Duration("old", oldDefault, "Consider kubetest old if it exceeds this")
	var a []string
	if err := fs.Parse(args[1:]); err == flag.ErrHelp {
		os.Stderr.WriteString("  -- kubetestArgs\n")
		os.Stderr.WriteString("        All flags after -- are passed to the kubetest program\n")
		return flags{}, err
	} else if err != nil {
		log.Print("NOTICE: go run hack/e2e.go is now a shim for test-infra/kubetest")
		log.Printf("  Usage: go run hack/e2e.go [--get=%v] [--old=%v] -- [KUBETEST_ARGS]", getDefault, oldDefault)
		log.Print("  The separator is required to use --get or --old flags")
		log.Print("  The -- flag separator also suppresses this message")
		a = args[len(args)-fs.NArg()-1:]
	} else {
		a = fs.Args()
	}
	return flags{*get, *old, a}, nil
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	f, err := parse(os.Args)
	if err != nil {
		os.Exit(2)
	}
	t := newTester()
	k, err := t.getKubetest(f.get, f.old)
	if err != nil {
		log.Fatalf("err: %v", err)
	}
	log.Printf("Calling kubetest %v...", strings.Join(f.args, " "))
	if err = t.wait(k, f.args...); err != nil {
		log.Fatalf("err: %v", err)
	}
	log.Print("Done")
}

func wait(cmd string, args ...string) error {
	c := exec.Command(cmd, args...)
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	if err := c.Start(); err != nil {
		return err
	}
	return c.Wait()
}

// Struct that allows unit tests to override functionality.
type tester struct {
	// os.Stat
	stat func(string) (os.FileInfo, error)
	// exec.LookPath
	lookPath func(string) (string, error)
	// build.Default.GOPATH
	goPath string
	wait   func(string, ...string) error
}

func newTester() tester {
	return tester{os.Stat, exec.LookPath, build.Default.GOPATH, wait}
}

// Try to find kubetest, either GOPATH/bin/kubetest or PATH
func (t tester) lookKubetest() (string, error) {
	// Check for kubetest in GOPATH/bin
	if t.goPath != "" {
		p := filepath.Join(t.goPath, "bin", "kubetest")
		_, err := t.stat(p)
		if err == nil {
			return p, nil
		}
	}

	// Check for kubetest in PATH
	p, err := t.lookPath("kubetest")
	return p, err
}

// Upgrade if kubetest does not exist or has not been updated today
func (t tester) getKubetest(get bool, old time.Duration) (string, error) {
	// Find kubetest installation
	p, err := t.lookKubetest()
	if err == nil && !get {
		return p, nil // Installed, Skip update
	}
	if err == nil {
		// Installed recently?
		if s, err := t.stat(p); err != nil {
			return p, err // Cannot stat
		} else if time.Since(s.ModTime()) <= old {
			return p, nil // Recently updated
		} else if t.goPath == "" {
			log.Print("Skipping kubetest upgrade because $GOPATH is empty")
			return p, nil
		}
		log.Printf("The kubetest binary is older than %s.", old)
	}
	if t.goPath == "" {
		return "", fmt.Errorf("Cannot install kubetest until $GOPATH is set")
	}
	log.Print("Updating kubetest binary...")
	cmd := []string{"go", "get", "-u", "k8s.io/test-infra/kubetest"}
	if err = t.wait(cmd[0], cmd[1:]...); err != nil {
		return "", fmt.Errorf("%s: %v", strings.Join(cmd, " "), err) // Could not upgrade
	}
	if p, err = t.lookKubetest(); err != nil {
		return "", err // Cannot find kubetest
	} else if err = t.wait("touch", p); err != nil {
		return "", err // Could not touch
	} else {
		return p, nil // Updated modtime
	}
}
