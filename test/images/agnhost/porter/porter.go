/*
Copyright 2015 The Kubernetes Authors.

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

// A tiny binary for testing ports.
//
// Reads env vars; for every var of the form SERVE_PORT_X, where X is a valid
// port number, porter starts an HTTP server which serves the env var's value
// in response to any query.

package porter

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

const prefix = "SERVE_PORT_"
const tlsPrefix = "SERVE_TLS_PORT_"

// CmdPorter is used by agnhost Cobra.
var CmdPorter = &cobra.Command{
	Use:   "porter",
	Short: "Serves requested data on ports specified in ENV variables",
	Long: `Serves requested data on ports specified in ENV variables. For example, if the environment variable "SERVE_PORT_9001" is set, then the subcommand will start serving on the port 9001.

Additionally, if the environment variable "SERVE_TLS_PORT_9002" is set, then the subcommand will start a TLS server on that port.

The included "localhost.crt" is a PEM-encoded TLS cert with SAN IPs "127.0.0.1" and "[::1]", expiring in January 2084, generated from "src/crypto/tls".

To use a different cert/key, mount them into the pod and set the "CERT_FILE" and "KEY_FILE" environment variables to the desired paths.`,
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

func main(cmd *cobra.Command, args []string) {
	for _, vk := range os.Environ() {
		// Put everything before the first = sign in parts[0], and
		// everything else in parts[1] (even if there are multiple =
		// characters).
		parts := strings.SplitN(vk, "=", 2)
		key := parts[0]
		value := parts[1]
		if strings.HasPrefix(key, prefix) {
			port := strings.TrimPrefix(key, prefix)
			go servePort(port, value)
		}
		if strings.HasPrefix(key, tlsPrefix) {
			port := strings.TrimPrefix(key, tlsPrefix)
			go serveTLSPort(port, value)
		}
	}

	select {}
}

func servePort(port, value string) {
	s := &http.Server{
		Addr: "0.0.0.0:" + port,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, value)
		}),
	}
	log.Printf("server on port %q failed: %v", port, s.ListenAndServe())
}

func serveTLSPort(port, value string) {
	s := &http.Server{
		Addr: "0.0.0.0:" + port,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, value)
		}),
	}
	certFile := os.Getenv("CERT_FILE")
	if len(certFile) == 0 {
		certFile = "localhost.crt"
	}
	keyFile := os.Getenv("KEY_FILE")
	if len(keyFile) == 0 {
		keyFile = "localhost.key"
	}
	log.Printf("tls server on port %q with certFile=%q, keyFile=%q failed: %v", port, certFile, keyFile, s.ListenAndServeTLS(certFile, keyFile))
}
