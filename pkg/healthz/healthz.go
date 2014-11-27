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
	"net/http"
)

// mux is an interface describing the methods InstallHandler requires.
type mux interface {
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

func init() {
	http.HandleFunc("/healthz", handleHealthz)
}

func handleHealthz(w http.ResponseWriter, r *http.Request) {
	// TODO Support user supplied health functions too.
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

// InstallHandler registers a handler for health checking on the path "/healthz" to mux.
func InstallHandler(mux mux) {
	mux.HandleFunc("/healthz", handleHealthz)
}
