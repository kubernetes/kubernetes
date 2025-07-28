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

package registry

import (
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"sync"

	"github.com/google/go-containerregistry/pkg/crane"
	"github.com/google/go-containerregistry/pkg/registry"
	"github.com/google/go-containerregistry/pkg/v1/tarball"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/test/images/agnhost/registry/testimages"
)

const (
	certFileEnvName = "CERTFILE_PATH"
	keyFileEnvName  = "KEYFILE_PATH"
	testUsername    = "joe"
	testPassword    = "secretoftheuniverse"
)

var CmdRegistry = &cobra.Command{
	Use:   "registry",
	Short: "Runs a test container registry",
	Long: `Runs a test container registry providing the "pause" image with various tags.

Requires Basic HTTP authentication with hardcoded credentials joe/secretoftheuniverse.

The subcommand uses these environment variables:
	- CERTFILE_PATH - path to the serving certificate
	- KEYFILE_PATH - path to the private key that belongs to the serving certificate
`,
	Args: cobra.MaximumNArgs(0),
	RunE: runRegistry,
}

func runRegistry(cmd *cobra.Command, args []string) error {
	certFilePath := os.Getenv(certFileEnvName)
	keyFilePath := os.Getenv(keyFileEnvName)
	if len(certFilePath) == 0 || len(keyFilePath) == 0 {
		return fmt.Errorf("both %q and %q env vars must be set", certFileEnvName, keyFileEnvName)
	}

	insecureListener, err := net.Listen("tcp", "localhost:5001")
	if err != nil {
		return fmt.Errorf("failed to listen insecurely: %w", err)
	}
	defer insecureListener.Close()

	registryHandler := registry.New()

	wg := sync.WaitGroup{}
	wg.Add(1)

	go func() {
		defer wg.Done()
		http.Serve(insecureListener, registryHandler)
	}()

	cert, err := tls.LoadX509KeyPair(certFilePath, keyFilePath)
	if err != nil {
		return fmt.Errorf("failed to create a cert/key pair: %w", err)
	}

	listener, err := net.Listen("tcp", "localhost:5000")
	if err != nil {
		return fmt.Errorf("failed to start listener for secure connections: %w", err)
	}
	defer listener.Close()
	listener = tls.NewListener(listener, &tls.Config{
		Certificates: []tls.Certificate{cert},
	})

	wg.Add(1)
	go func() {
		defer wg.Done()
		http.Serve(listener, withAuthentication(registryHandler))
	}()

	err = initRegistry()
	if err != nil {
		return err
	}
	insecureListener.Close()

	wg.Wait()
	return nil
}

func withAuthentication(delegate http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		username, password, ok := req.BasicAuth()
		if !ok {
			w.Header().Add("WWW-Authenticate", `Basic realm="agnhost-testing"`)
			http.Error(w, "basic authentication required", http.StatusUnauthorized)
			return
		}

		if username != testUsername || password != testPassword {
			http.Error(w, "wrong username/password", http.StatusUnauthorized)
			return
		}

		delegate.ServeHTTP(w, req)
	}
}

// initRegistry initializes the registry via the insecure handler
//
// Unfortunately, go-containers don't allow simple initialization as it does not
// export any useful interfaces. Currently, it only allows supplying a custom
// BlobHandler, but one would still need to push manifests somehow.
func initRegistry() error {
	preloadedImages, err := testimages.Images.ReadDir("images")
	if err != nil {
		return fmt.Errorf("failed to read the directory with images: %w", err)
	}

	for _, image := range preloadedImages {
		imageFileOpener := func() (io.ReadCloser, error) { return testimages.Images.Open("images/" + image.Name()) }
		img, err := tarball.Image(imageFileOpener, nil)
		if err != nil {
			return err
		}

		if err := crane.Push(img, "localhost:5001/pause:test"); err != nil {
			return err
		}
	}

	return nil
}
