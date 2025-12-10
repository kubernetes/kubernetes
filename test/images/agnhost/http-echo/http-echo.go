/*
Copyright 2025 The Kubernetes Authors.

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

// A small utility to echo back an HTTP Request as JSON
// This utility is an extraction of Gateway API Echo server: https://github.com/kubernetes-sigs/gateway-api/blob/main/conformance/echo-basic/echo-basic.go

package httpecho

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/spf13/cobra"
)

// CmdServeHostname is used by agnhost Cobra.
var CmdHTTPEcho = &cobra.Command{
	Use:   "http-echo",
	Short: "Echoes back the HTTP request as a JSON",
	Long:  `Echoes back the HTTP request as a JSON payload, including headers.`,
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

var (
	httpPort int
	context  Context
)

func init() {
	CmdHTTPEcho.Flags().IntVar(&httpPort, "port", 8080, "Port number.")
}

// Context contains information about the context where the echoserver is running
type Context struct {
	Namespace string `json:"namespace"`
	Pod       string `json:"pod"`
}

// RequestPayload contains information about the request
type RequestPayload struct {
	Path     string              `json:"path"`
	Host     string              `json:"host"`
	Method   string              `json:"method"`
	Proto    string              `json:"proto"`
	Headers  map[string][]string `json:"headers"`
	HTTPPort string              `json:"httpPort"`

	Context `json:",inline"`
}

func main(cmd *cobra.Command, args []string) {
	context = Context{
		Namespace: os.Getenv("NAMESPACE"),
		Pod:       os.Getenv("POD_NAME"),
	}

	http.HandleFunc("/", echoHandler)

	go func() {
		// Run in a closure so http.ListenAndServe doesn't block
		log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", httpPort), nil))
	}()

	log.Printf("Serving on port %d.\n", httpPort)
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM)
	sig := <-signals
	log.Printf("Shutting down after receiving signal: %s.\n", sig)
	log.Printf("Awaiting pod deletion.\n")
	time.Sleep(60 * time.Second)
}

func echoHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Echoing back request made to %s to client (%s)\n", r.RequestURI, r.RemoteAddr)

	requestPayload := RequestPayload{
		r.RequestURI,
		r.Host,
		r.Method,
		r.Proto,
		r.Header,
		strconv.Itoa(httpPort),

		context,
	}

	js, err := json.MarshalIndent(requestPayload, "", " ")
	if err != nil {
		http.Error(w, fmt.Sprintf("%v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	_, err = w.Write(js)
	if err != nil {
		log.Printf("error writing the response back: %s", err)
	}
}
