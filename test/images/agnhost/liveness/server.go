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

// A simple server that is alive for 10 seconds, then reports unhealthy for
// the rest of its (hopefully) short existence.

package liveness

import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/spf13/cobra"
)

// CmdLiveness is used by agnhost Cobra.
var CmdLiveness = &cobra.Command{
	Use:   "liveness",
	Short: "Starts a server that is alive for 10 seconds",
	Long: "A simple server that is alive for 10 seconds, then reports unhealthy for the rest of its (hopefully) short existence. " +
		"Failures can be emulated by passing parameters to specific URLs.",
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

func flush(w http.ResponseWriter) {
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func forciblyDisconnect(w http.ResponseWriter) {
	if c, _, err := w.(http.Hijacker).Hijack(); err == nil {
		_ = c.Close()
	} else {
		_, _ = w.Write([]byte(fmt.Sprintf("Forcible disconnection failed: %v", err)))
	}
}

func writeBody(w http.ResponseWriter, size, interval int) {
	for kb := 0; kb < size; kb++ {
		if interval > 0 {
			flush(w)
			time.Sleep(time.Duration(interval) * time.Second)
		}
		for b := 0; b < 1024; b += 16 {
			_, _ = w.Write([]byte(fmt.Sprintf("%16x", rand.Uint64())))
		}
	}
}

type params struct {
	statusCode    int
	sleepSec      int
	bodySize      int
	writeInterval int
}

func parseIntParam(r *http.Request, name string, defaultVal int) (int, error) {
	if param := r.URL.Query().Get(name); param == "" {
		return defaultVal, nil
	} else if intVal, err := strconv.Atoi(param); err != nil {
		return defaultVal, err
	} else {
		return intVal, nil
	}
}

func parseParams(r *http.Request) (*params, error) {
	params := &params{}
	if val, err := parseIntParam(r, "code", 500); err != nil {
		return nil, fmt.Errorf("parameter 'code' is invalid: %w", err)
	} else {
		params.statusCode = val
	}
	if val, err := parseIntParam(r, "time", 0); err != nil {
		return nil, fmt.Errorf("parameter 'time' is invalid: %w", err)
	} else {
		params.sleepSec = val
	}
	if val, err := parseIntParam(r, "size", 0); err != nil {
		return nil, fmt.Errorf("parameter 'size' is invalid: %w", err)
	} else {
		params.bodySize = val
	}
	if val, err := parseIntParam(r, "interval", 0); err != nil {
		return nil, fmt.Errorf("parameter 'interval' is invalid: %w", err)
	} else {
		params.writeInterval = val
	}

	return params, nil
}

func main(cmd *cobra.Command, args []string) {
	started := time.Now()
	http.HandleFunc("/started", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		data := (time.Since(started)).String()
		w.Write([]byte(data))
	})
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		duration := time.Since(started)
		if duration.Seconds() > 10 {
			w.WriteHeader(500)
			w.Write([]byte(fmt.Sprintf("error: %v", duration.Seconds())))
		} else {
			w.WriteHeader(200)
			w.Write([]byte("ok"))
		}
	})
	http.HandleFunc("/redirect", func(w http.ResponseWriter, r *http.Request) {
		loc, err := url.QueryUnescape(r.URL.Query().Get("loc"))
		if err != nil {
			http.Error(w, fmt.Sprintf("invalid redirect: %q", r.URL.Query().Get("loc")), http.StatusBadRequest)
			return
		}
		http.Redirect(w, r, loc, http.StatusFound)
	})
	http.HandleFunc("/sleep-headers", func(w http.ResponseWriter, r *http.Request) {
		params, err := parseParams(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}
		if params.sleepSec > 0 {
			time.Sleep(time.Duration(params.sleepSec) * time.Second)
		}

		w.WriteHeader(params.statusCode)
		_, _ = w.Write([]byte(fmt.Sprintf("Slept for %d secs\n", params.sleepSec)))
		writeBody(w, params.bodySize, 0)
	})
	http.HandleFunc("/disconnect-headers", func(w http.ResponseWriter, r *http.Request) {
		forciblyDisconnect(w)
	})
	http.HandleFunc("/sleep-body", func(w http.ResponseWriter, r *http.Request) {
		params, err := parseParams(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		w.WriteHeader(params.statusCode)
		writeBody(w, params.bodySize, params.writeInterval)

		_, _ = w.Write([]byte(fmt.Sprintf("Sleeping for %d secs\n", params.sleepSec)))
		if params.sleepSec > 0 {
			flush(w)
			time.Sleep(time.Duration(params.sleepSec) * time.Second)
		}
		_, _ = w.Write([]byte(fmt.Sprintf("Slept for %d secs\n", params.sleepSec)))
	})
	http.HandleFunc("/disconnect-body", func(w http.ResponseWriter, r *http.Request) {
		params, err := parseParams(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		w.WriteHeader(params.statusCode)
		writeBody(w, params.bodySize, 0)

		_, _ = w.Write([]byte("Disconnecting\n"))
		flush(w)
		forciblyDisconnect(w)
	})
	http.HandleFunc("/slow-response", func(w http.ResponseWriter, r *http.Request) {
		params, err := parseParams(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		w.WriteHeader(params.statusCode)
		writeBody(w, params.bodySize, params.writeInterval)
	})
	log.Fatal(http.ListenAndServe(":8080", nil))
}
