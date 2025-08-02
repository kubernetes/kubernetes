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

package fakeregistryserver

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

var (
	port        int
	host        string
	registryDir string
	private     bool
)

func init() {
	CmdFakeRegistryServer.Flags().IntVar(&port, "port", 5000, "Port number.")
	CmdFakeRegistryServer.Flags().StringVar(&host, "host", "0.0.0.0", "Host address.")
	CmdFakeRegistryServer.Flags().StringVar(&registryDir, "registry-dir", "/registry", "Directory containing the registry data.")
	CmdFakeRegistryServer.Flags().BoolVar(&private, "private", false, "Enable basic authentication with static credentials (test:test)")
}

// CmdFakeRegistryServer is the cobra command for the fake registry server
var CmdFakeRegistryServer = &cobra.Command{
	Use:   "fake-registry-server",
	Short: "Starts a fake registry server for testing",
	Long:  `Starts a fake registry server that serves static OCI image files`,
	Run:   main,
}

func main(cmd *cobra.Command, args []string) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v2/", handleV2)

	var handler http.Handler = mux
	if private {
		handler = auth(mux)
	}

	serverAdr := fmt.Sprintf("%s:%d", host, port)
	log.Printf("HTTP server starting to listen on %s", serverAdr)
	if err := http.ListenAndServe(serverAdr, handler); err != nil {
		log.Fatalf("Error while starting the HTTP server: %v", err)
	}
}

func auth(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, pass, ok := r.BasicAuth()
		if !ok || user != "test" || pass != "test" {
			w.Header().Set("WWW-Authenticate", `Basic realm="Restricted"`)
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte("Unauthorized\n"))
			return
		}
		h.ServeHTTP(w, r)
	})
}

func handleV2(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Docker-Distribution-Api-Version", "registry/2.0")

	if r.Method == "HEAD" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.URL.Path == "/v2/" {
		w.WriteHeader(http.StatusOK)
		return
	}

	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/v2/"), "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	imageName := parts[0]
	objectType := parts[1]
	identifier := parts[2]

	filePath := fmt.Sprintf("%s/%s/%s/%s", registryDir, imageName, objectType, identifier)

	switch objectType {
	case "blobs":
		w.Header().Set("Content-Type", "application/octet-stream")
	case "manifests":
		if _, err := os.Stat(filePath + "_index"); err == nil {
			filePath += "_index"
			w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.list.v2+json")
		} else {
			w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
		}
	default:
		http.NotFound(w, r)
		return
	}

	log.Printf("Serving file: %s", filePath)
	http.ServeFile(w, r, filePath)
}
