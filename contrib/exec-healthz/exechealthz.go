/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// A tiny web server that returns 200 on it's healthz endpoint if the command
// passed in via -cmd exits with 0. Returns 503 otherwise.
// Usage: exechealthz -port 8080 -period 2s -latency 30s -cmd 'nslookup localhost >/dev/null' -verbose true
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

// TODO:
// 1. Sigterm handler for docker stop
// 2. Meaningful default healthz
// 3. 404 for unknown endpoints

var (
	port       = flag.Int("port", 8080, "Port number to serve /healthz.")
	cmd        = flag.String("cmd", "echo healthz", "Command to run in response to a GET on /healthz. If the given command exits with 0, /healthz will respond with a 200.")
	period     = flag.Duration("period", 2*time.Second, "Period to run the given cmd in an async worker.")
	maxLatency = flag.Duration("latency", 30*time.Second, "If the async worker hasn't updated the probe command output in this long, return a 503.")
	verbose    = flag.Bool("verbose", true, "Print to console at each periodic exec")
	// prober is the async worker running the cmd, the output of which is used to service /healthz.
	prober *execWorker
)

// execResult holds the result of the latest exec from the execWorker.
type execResult struct {
	output []byte
	err    error
	ts     time.Time
}

func (r execResult) String() string {
	errMsg := "None"
	if r.err != nil {
		errMsg = fmt.Sprintf("%v", r.err)
	}
	return fmt.Sprintf("Result of last exec: %v, at %v, error %v", string(r.output), r.ts, errMsg)
}

// execWorker provides an async interface to exec.
type execWorker struct {
	result   execResult
	mutex    sync.Mutex
	period   time.Duration
	probeCmd string
	stopCh   chan struct{}
}

// getResults returns the results of the latest execWorker run.
// The caller should treat returned results as read-only.
func (h *execWorker) getResults() execResult {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	return h.result
}

// start attemtps to run the probeCmd every `period` seconds.
// Meant to be called as a goroutine.
func (h *execWorker) start() {
	ticker := time.NewTicker(h.period)
	defer ticker.Stop()

	for {
		select {
		// If the command takes > period, the command runs continuously.
		case <-ticker.C:
			if verbose {
				log.Printf("Worker running %v", *cmd)
			}
			output, err := exec.Command("sh", "-c", *cmd).CombinedOutput()
			ts := time.Now()
			func() {
				h.mutex.Lock()
				defer h.mutex.Unlock()
				h.result = execResult{output, err, ts}
			}()
		case <-h.stopCh:
			return
		}
	}
}

// newExecWorker is a constructor for execWorker.
func newExecWorker(probeCmd string, execPeriod time.Duration) *execWorker {
	return &execWorker{
		// Initializing the result with a timestamp here allows us to
		// wait maxLatency for the worker goroutine to start, and for each
		// iteration of the worker to complete.
		result:   execResult{[]byte{}, nil, time.Now()},
		period:   execPeriod,
		probeCmd: probeCmd,
		stopCh:   make(chan struct{}),
	}
}

func main() {
	flag.Parse()
	links := []struct {
		link, desc string
	}{
		{"/healthz", "healthz probe. Returns \"ok\" if the command given through -cmd exits with 0."},
		{"/quit", "Cause this container to exit."},
	}
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "<b> Kubernetes healthz sidecar container </b><br/><br/>")
		for _, v := range links {
			fmt.Fprintf(w, `<a href="%v">%v: %v</a><br/>`, v.link, v.link, v.desc)
		}
	})

	http.HandleFunc("/quit", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Shutdown requested via /quit by %v", r.RemoteAddr)
		os.Exit(0)
	})
	prober = newExecWorker(*cmd, *period)
	defer close(prober.stopCh)
	go prober.start()

	http.HandleFunc("/healthz", healthzHandler)
	log.Fatal(http.ListenAndServe(fmt.Sprintf("0.0.0.0:%d", *port), nil))
}

func healthzHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Client ip %v requesting /healthz probe servicing cmd %v", r.RemoteAddr, *cmd)
	result := prober.getResults()

	// return 503 if the last command exec returned a non-zero status, or the worker
	// hasn't run in maxLatency (including when the worker goroutine is cpu starved,
	// because the pod is probably unavailable too).
	if result.err != nil {
		msg := fmt.Sprintf("Healthz probe error: %v", result)
		log.Printf(msg)
		http.Error(w, msg, http.StatusServiceUnavailable)
	} else if time.Since(result.ts) > *maxLatency {
		msg := fmt.Sprintf("Latest result too old to be useful: %v.", result)
		log.Printf(msg)
		http.Error(w, msg, http.StatusServiceUnavailable)
	} else {
		fmt.Fprintf(w, "ok")
	}
}
