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
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/ishidawataru/sctp"
	"github.com/spf13/cobra"
)

const tcpPrefix = "SERVE_PORT_"
const sctpPrefix = "SERVE_SCTP_PORT_"
const tlsPrefix = "SERVE_TLS_PORT_"

// CmdPorter is used by agnhost Cobra.
var CmdPorter = &cobra.Command{
	Use:   "porter",
	Short: "Serves requested data on ports specified in ENV variables",
	Long: `Serves requested data on ports specified in environment variables of the form SERVE_{PORT,TLS_PORT,SCTP_PORT}_[NNNN]. 
	
eg:
* SERVE_PORT_9001 - serve TCP connections on port 9001
* SERVE_TLS_PORT_9002 - serve TLS-encrypted TCP connections on port 9002
* SERVE_SCTP_PORT_9003 - serve SCTP connections on port 9003

The included "localhost.crt" is a PEM-encoded TLS cert with SAN IPs "127.0.0.1" and "[::1]", expiring in January 2084, generated from "src/crypto/tls".

To use a different cert/key, mount them into the pod and set the "CERT_FILE" and "KEY_FILE" environment variables to the desired paths.`,
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

// JSONResponse enables --json-response flag
var JSONResponse bool

type jsonResponse struct {
	Method string
	Body   string
}

func init() {
	CmdPorter.Flags().BoolVar(&JSONResponse, "json-response", false, "Responds to requests with a json response that includes the default value with the http.Request Method")
}

func main(cmd *cobra.Command, args []string) {
	for _, vk := range os.Environ() {
		// Put everything before the first = sign in parts[0], and
		// everything else in parts[1] (even if there are multiple =
		// characters).
		parts := strings.SplitN(vk, "=", 2)
		key := parts[0]
		value := parts[1]

		switch {
		case strings.HasPrefix(key, tcpPrefix):
			port := strings.TrimPrefix(key, tcpPrefix)
			go servePort(port, value)
		case strings.HasPrefix(key, sctpPrefix):
			port := strings.TrimPrefix(key, sctpPrefix)
			go serveSCTPPort(port, value)
		case strings.HasPrefix(key, tlsPrefix):
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
			body := value
			if JSONResponse {
				j, err := json.Marshal(&jsonResponse{
					Method: r.Method,
					Body:   value})
				if err != nil {
					http.Error(w, fmt.Sprintf("Internal Server Error: %v", err), 500)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				body = string(j)
			}
			fmt.Fprint(w, body)
		}),
	}
	log.Printf("server on port %q failed: %v", port, s.ListenAndServe())
}

func serveTLSPort(port, value string) {
	s := &http.Server{
		Addr: "0.0.0.0:" + port,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body := value
			if JSONResponse {
				j, err := json.Marshal(&jsonResponse{
					Method: r.Method,
					Body:   value})
				if err != nil {
					http.Error(w, fmt.Sprintf("Internal Server Error: %v", err), 500)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				body = string(j)
			}
			fmt.Fprint(w, body)
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

func serveSCTPPort(port, value string) {
	serverAddress, err := sctp.ResolveSCTPAddr("sctp", "0.0.0.0:"+port)
	if err != nil {
		log.Fatal("Sctp: failed to resolve address. error:", err)
	}

	listener, err := sctp.ListenSCTP("sctp", serverAddress)
	if err != nil {
		log.Fatal("Failed to listen SCTP. error:", err)
	}
	log.Printf("Started SCTP server")

	defer listener.Close()
	defer func() {
		log.Printf("SCTP server exited")
	}()

	for {
		conn, err := listener.AcceptSCTP()
		if err != nil {
			log.Fatal("Failed to accept SCTP. error:", err)
		}
		go func(conn *sctp.SCTPConn) {
			defer conn.Close()
			log.Println("Sending response")
			_, err = conn.Write([]byte(value))
			if err != nil {
				log.Println("Failed to send response", err)
				return
			}
			log.Println("Response sent")
		}(conn)
	}
}
