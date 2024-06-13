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
	"time"

	"github.com/spf13/cobra"
)

// CmdLiveness is used by agnhost Cobra.
var CmdLiveness = &cobra.Command{
	Use:   "liveness",
	Short: "Starts a server that is alive for 10 seconds",
	Long: "A simple server that is alive for 10 seconds, then reports unhealthy for the rest of its (hopefully) short existence. " +
		"Failures can be emulated with flags, which take effect after the server gets unhealthy.",
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

var healthspan int
var initDisconnect bool
var disconnect bool
var initSleepSec int
var sleepSec int
var bodySize int
var responseCode int
var writeInterval int

func init() {
	CmdLiveness.Flags().IntVar(&healthspan, "healthspan", 10, "Duration (in seconds) that the server is alive after startup. The default is 10.")
	CmdLiveness.Flags().BoolVar(&initDisconnect, "init-disconnect", false, "If true, forcibly disconnects before sending a header.")
	CmdLiveness.Flags().BoolVar(&disconnect, "disconnect", false, "If true, forcibly disconnects after sending a body.")
	CmdLiveness.Flags().IntVar(&initSleepSec, "init-sleep", 0, "Duration (in seconds) that the server sleeps before sending a header.")
	CmdLiveness.Flags().IntVar(&sleepSec, "sleep", 0, "Duration (in seconds) that the server sleeps after sending a body.")
	CmdLiveness.Flags().IntVar(&bodySize, "body-size", -1, "Size (in KB) of a response body that consists of random characters (0-9 and a-f) and whietespaces. If not set, a few words are sent.")
	CmdLiveness.Flags().IntVar(&responseCode, "response-code", 500, "Response code after healthspan passed. The default is 500.")
	CmdLiveness.Flags().IntVar(&writeInterval, "write-interval", 0, "Interval (in seconds) between sending 1KB chunk. The default is 0.")
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

func main(cmd *cobra.Command, args []string) {
	started := time.Now()
	http.HandleFunc("/started", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		data := (time.Since(started)).String()
		w.Write([]byte(data))
	})
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		duration := time.Since(started)
		if duration <= time.Duration(healthspan)*time.Second {
			w.WriteHeader(200)
			w.Write([]byte("ok"))
			return
		}

		if initSleepSec > 0 {
			time.Sleep(time.Duration(initSleepSec) * time.Second)
		}
		if initDisconnect {
			forciblyDisconnect(w)
			return
		}

		w.WriteHeader(responseCode)

		if bodySize < 0 {
			// For backward compatibility, write the same data as before `bodySize` was introduced.
			_, _ = w.Write([]byte(fmt.Sprintf("error: %v", duration.Seconds())))
		} else {
			for kb := 0; kb < bodySize; kb++ {
				if writeInterval > 0 {
					flush(w)
					time.Sleep(time.Duration(writeInterval) * time.Second)
				}
				for b := 0; b < 1024; b += 16 {
					_, _ = w.Write([]byte(fmt.Sprintf("%16x", rand.Uint64())))
				}
			}
		}

		if sleepSec > 0 {
			flush(w)
			time.Sleep(time.Duration(sleepSec) * time.Second)
		}
		if disconnect {
			flush(w)
			forciblyDisconnect(w)
			return
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
	log.Fatal(http.ListenAndServe(":8080", nil))
}
