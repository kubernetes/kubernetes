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

package healthz

import (
	"bytes"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
)

// HealthzChecker is a named healthz checker.
type HealthzChecker interface {
	Name() string
	Check(req *http.Request) error
}

var defaultHealthz = sync.Once{}

// DefaultHealthz installs the default healthz check to the http.DefaultServeMux.
func DefaultHealthz(checks ...HealthzChecker) {
	defaultHealthz.Do(func() {
		InstallHandler(http.DefaultServeMux, checks...)
	})
}

// PingHealthz returns true automatically when checked
var PingHealthz HealthzChecker = ping{}

// ping implements the simplest possible healthz checker.
type ping struct{}

func (ping) Name() string {
	return "ping"
}

// PingHealthz is a health check that returns true.
func (ping) Check(_ *http.Request) error {
	return nil
}

// LogHealthz returns true if logging is not blocked
var LogHealthz HealthzChecker = &log{}

type log struct {
	startOnce    sync.Once
	lastVerified atomic.Value
}

func (l *log) Name() string {
	return "log"
}

func (l *log) Check(_ *http.Request) error {
	l.startOnce.Do(func() {
		l.lastVerified.Store(time.Now())
		go wait.Forever(func() {
			klog.Flush()
			l.lastVerified.Store(time.Now())
		}, time.Minute)
	})

	lastVerified := l.lastVerified.Load().(time.Time)
	if time.Since(lastVerified) < (2 * time.Minute) {
		return nil
	}
	return fmt.Errorf("logging blocked")
}

// NamedCheck returns a healthz checker for the given name and function.
func NamedCheck(name string, check func(r *http.Request) error) HealthzChecker {
	return &healthzCheck{name, check}
}

// InstallHandler registers handlers for health checking on the path
// "/healthz" to mux. *All handlers* for mux must be specified in
// exactly one call to InstallHandler. Calling InstallHandler more
// than once for the same mux will result in a panic.
func InstallHandler(mux mux, checks ...HealthzChecker) {
	InstallPathHandler(mux, "/healthz", checks...)
}

// InstallPathHandler registers handlers for health checking on
// a specific path to mux. *All handlers* for the path must be
// specified in exactly one call to InstallPathHandler. Calling
// InstallPathHandler more than once for the same path and mux will
// result in a panic.
func InstallPathHandler(mux mux, path string, checks ...HealthzChecker) {
	if len(checks) == 0 {
		klog.V(5).Info("No default health checks specified. Installing the ping handler.")
		checks = []HealthzChecker{PingHealthz}
	}

	klog.V(5).Info("Installing healthz checkers:", formatQuoted(checkerNames(checks...)...))

	mux.Handle(path, handleRootHealthz(checks...))
	for _, check := range checks {
		mux.Handle(fmt.Sprintf("%s/%v", path, check.Name()), adaptCheckToHandler(check.Check))
	}
}

// mux is an interface describing the methods InstallHandler requires.
type mux interface {
	Handle(pattern string, handler http.Handler)
}

// healthzCheck implements HealthzChecker on an arbitrary name and check function.
type healthzCheck struct {
	name  string
	check func(r *http.Request) error
}

var _ HealthzChecker = &healthzCheck{}

func (c *healthzCheck) Name() string {
	return c.name
}

func (c *healthzCheck) Check(r *http.Request) error {
	return c.check(r)
}

// getExcludedChecks extracts the health check names to be excluded from the query param
func getExcludedChecks(r *http.Request) sets.String {
	checks, found := r.URL.Query()["exclude"]
	if found {
		return sets.NewString(checks...)
	}
	return sets.NewString()
}

// handleRootHealthz returns an http.HandlerFunc that serves the provided checks.
func handleRootHealthz(checks ...HealthzChecker) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		failed := false
		excluded := getExcludedChecks(r)
		var verboseOut bytes.Buffer
		for _, check := range checks {
			// no-op the check if we've specified we want to exclude the check
			if excluded.Has(check.Name()) {
				excluded.Delete(check.Name())
				fmt.Fprintf(&verboseOut, "[+]%v excluded: ok\n", check.Name())
				continue
			}
			if err := check.Check(r); err != nil {
				// don't include the error since this endpoint is public.  If someone wants more detail
				// they should have explicit permission to the detailed checks.
				klog.V(4).Infof("healthz check %v failed: %v", check.Name(), err)
				fmt.Fprintf(&verboseOut, "[-]%v failed: reason withheld\n", check.Name())
				failed = true
			} else {
				fmt.Fprintf(&verboseOut, "[+]%v ok\n", check.Name())
			}
		}
		if excluded.Len() > 0 {
			fmt.Fprintf(&verboseOut, "warn: some health checks cannot be excluded: no matches for %v\n", formatQuoted(excluded.List()...))
			klog.Warningf("cannot exclude some health checks, no health checks are installed matching %v",
				formatQuoted(excluded.List()...))
		}
		// always be verbose on failure
		if failed {
			klog.V(2).Infof("%vhealthz check failed", verboseOut.String())
			http.Error(w, fmt.Sprintf("%vhealthz check failed", verboseOut.String()), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		if _, found := r.URL.Query()["verbose"]; !found {
			fmt.Fprint(w, "ok")
			return
		}

		verboseOut.WriteTo(w)
		fmt.Fprint(w, "healthz check passed\n")
	})
}

// adaptCheckToHandler returns an http.HandlerFunc that serves the provided checks.
func adaptCheckToHandler(c func(r *http.Request) error) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		err := c(r)
		if err != nil {
			http.Error(w, fmt.Sprintf("internal server error: %v", err), http.StatusInternalServerError)
		} else {
			fmt.Fprint(w, "ok")
		}
	})
}

// checkerNames returns the names of the checks in the same order as passed in.
func checkerNames(checks ...HealthzChecker) []string {
	// accumulate the names of checks for printing them out.
	checkerNames := make([]string, 0, len(checks))
	for _, check := range checks {
		checkerNames = append(checkerNames, check.Name())
	}
	return checkerNames
}

// formatQuoted returns a formatted string of the health check names,
// preserving the order passed in.
func formatQuoted(names ...string) string {
	quoted := make([]string, 0, len(names))
	for _, name := range names {
		quoted = append(quoted, fmt.Sprintf("%q", name))
	}
	return strings.Join(quoted, ",")
}
