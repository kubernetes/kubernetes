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
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

var (
	port        int
	private     bool
	registryDir = "/var/registry"
)

const (
	privateRegistryUser = "user"
	privateRegistryPass = "password"
)

func init() {
	CmdFakeRegistryServer.Flags().IntVar(&port, "port", 5000, "Port number.")
	CmdFakeRegistryServer.Flags().BoolVar(&private, "private", false, "Enable authentication for the registry.")
}

// CmdFakeRegistryServer is the cobra command for the fake registry server
var CmdFakeRegistryServer = &cobra.Command{
	Use:   "fake-registry-server",
	Short: "Starts a fake registry server for testing",
	Long:  fmt.Sprintf("Starts a fake registry server that serves static OCI image files from %s folder", registryDir),
	Run:   main,
}

func main(cmd *cobra.Command, args []string) {
	registryMux := NewRegistryServerMux(private)

	addr := fmt.Sprintf(":%d", port)
	log.Printf("HTTP server starting to listen on %s", addr)
	if err := http.ListenAndServe(addr, registryMux); err != nil {
		log.Fatalf("Error while starting the HTTP server: %v", err)
	}
}

func NewRegistryServerMux(isPrivate bool) *http.ServeMux {
	mux := http.NewServeMux()

	var v2Handler http.Handler = http.HandlerFunc(handleV2)
	if isPrivate {
		v2Handler = auth(v2Handler)
	}
	mux.Handle("/v2/", v2Handler)

	return mux
}

func auth(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, pass, ok := r.BasicAuth()
		if !ok || user != privateRegistryUser || pass != privateRegistryPass {
			w.Header().Set("WWW-Authenticate", `Basic realm="Restricted"`)
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte("Unauthorized\n"))
			return
		}
		h.ServeHTTP(w, r)
	})
}

// handleBlobs serves blob requests
func handleBlobs(w http.ResponseWriter, r *http.Request, imageName, identifier string) {
	filePath := fmt.Sprintf("%s/%s/blobs/%s", registryDir, imageName, identifier)
	w.Header().Set("Content-Type", "application/octet-stream")
	log.Printf("Serving blob: %s", filePath)
	http.ServeFile(w, r, filePath)
}

// handleManifests serves manifest requests. It dynamically sets the Content-Type
// based on the manifest's mediaType field. If the identifier is a tag, it
// reads the digest from the tag file and issues a redirect.
func handleManifests(w http.ResponseWriter, r *http.Request, imageName, identifier string) {
	filePath := fmt.Sprintf("%s/%s/manifests/%s", registryDir, imageName, identifier)

	// if the identifier is not a digest, assume it's a tag and perform a redirect.
	if !strings.HasPrefix(identifier, "sha256:") {
		digest, err := os.ReadFile(filePath)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		redirectURL := strings.Replace(r.URL.String(), identifier, strings.TrimSpace(string(digest)), 1)
		w.Header().Set("Location", redirectURL)
		w.WriteHeader(http.StatusTemporaryRedirect)
		return
	}

	manifestContent, err := os.ReadFile(filePath)
	if err != nil {
		http.NotFound(w, r)
		return
	}

	var manifestData struct {
		MediaType string `json:"mediaType"`
	}

	if err := json.Unmarshal(manifestContent, &manifestData); err == nil && manifestData.MediaType != "" {
		w.Header().Set("Content-Type", manifestData.MediaType)
	}

	log.Printf("Serving manifest: %s", filePath)
	_, _ = w.Write(manifestContent)
}

func handleV2(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Docker-Distribution-Api-Version", "registry/2.0")

	if r.URL.Path == "/v2/" {
		w.WriteHeader(http.StatusOK)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/v2/")
	parts := strings.Split(path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	imageName := parts[0]
	objectType := parts[1]
	identifier := parts[2]

	switch objectType {
	case "blobs":
		handleBlobs(w, r, imageName, identifier)
	case "manifests":
		handleManifests(w, r, imageName, identifier)
	default:
		http.NotFound(w, r)
	}
}
