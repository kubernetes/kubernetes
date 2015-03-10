/*
Copyright 2014 Google Inc. All rights reserved.

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
	"sync"
)

var (
	// guards names and checks
	lock = sync.RWMutex{}
	// used to ensure checks are performed in the order added
	names  = []string{}
	checks = map[string]*healthzCheck{}
)

func init() {
	http.HandleFunc("/healthz", handleRootHealthz)
	// add ping health check by default
	AddHealthzFunc("ping", func(_ *http.Request) error {
		return nil
	})
}

// AddHealthzFunc adds a health check under the url /healhz/{name}
func AddHealthzFunc(name string, check func(r *http.Request) error) {
	lock.Lock()
	defer lock.Unlock()
	if _, found := checks[name]; !found {
		names = append(names, name)
	}
	checks[name] = &healthzCheck{name, check}
}

// InstallHandler registers a handler for health checking on the path "/healthz" to mux.
func InstallHandler(mux mux) {
	lock.RLock()
	defer lock.RUnlock()
	mux.HandleFunc("/healthz", handleRootHealthz)
	for _, check := range checks {
		mux.HandleFunc(fmt.Sprintf("/healthz/%v", check.name), adaptCheckToHandler(check.check))
	}
}

// mux is an interface describing the methods InstallHandler requires.
type mux interface {
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

type healthzCheck struct {
	name  string
	check func(r *http.Request) error
}

func handleRootHealthz(w http.ResponseWriter, r *http.Request) {
	lock.RLock()
	defer lock.RUnlock()
	failed := false
	var verboseOut bytes.Buffer
	for _, name := range names {
		check, found := checks[name]
		if !found {
			// this should not happen
			http.Error(w, fmt.Sprintf("Internal server error: check \"%q\" not registered", name), http.StatusInternalServerError)
			return
		}
		err := check.check(r)
		if err != nil {
			fmt.Fprintf(&verboseOut, "[-]%v failed: %v\n", check.name, err)
			failed = true
		} else {
			fmt.Fprintf(&verboseOut, "[+]%v ok\n", check.name)
		}
	}
	// always be verbose on failure
	if failed {
		http.Error(w, fmt.Sprintf("%vhealthz check failed", verboseOut.String()), http.StatusInternalServerError)
		return
	}

	if _, found := r.URL.Query()["verbose"]; !found {
		fmt.Fprint(w, "ok")
		return
	} else {
		verboseOut.WriteTo(w)
		fmt.Fprint(w, "healthz check passed\n")
	}
}

func adaptCheckToHandler(c func(r *http.Request) error) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		err := c(r)
		if err != nil {
			http.Error(w, fmt.Sprintf("Internal server error: %v", err), http.StatusInternalServerError)
		} else {
			fmt.Fprint(w, "ok")
		}
	}
}
