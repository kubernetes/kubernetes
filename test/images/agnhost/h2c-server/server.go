/*
Copyright The Kubernetes Authors.

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

// Package h2cserver offers a tiny HTTP/2 cleartext (h2c) web server for probe testing.
package h2cserver

import (
	"fmt"
	"log"
	"net"
	"net/http"

	"github.com/spf13/cobra"
	"golang.org/x/net/http2"
)

// CmdH2CServer is used by agnhost Cobra.
var CmdH2CServer = &cobra.Command{
	Use:   "h2c-server",
	Short: "Starts a simple HTTP/2 cleartext (h2c) server",
	Long:  "Starts a simple HTTP/2 cleartext server with prior knowledge on the given --port. Responds with 200 OK to all requests.",
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

var (
	port int
)

func init() {
	CmdH2CServer.Flags().IntVar(&port, "port", 80, "Port number.")
}

func main(cmd *cobra.Command, args []string) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, "ok")
	})

	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("Error from net.Listen(): %s", err)
	}

	h2s := &http2.Server{}
	srv := &http.Server{Handler: handler}

	log.Printf("Serving h2c on port %d.\n", port)

	for {
		c, err := l.Accept()
		if err != nil {
			log.Fatalf("Error from Accept(): %s", err)
		}
		go h2s.ServeConn(c, &http2.ServeConnOpts{
			Handler:    handler,
			BaseConfig: srv,
		})
	}
}
