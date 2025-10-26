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

// Package mtlsserver is an agnhost subcommand implementing a server to build
// the mTLS with the client and echo the client identity.
package mtlsserver

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/component-base/logs"
)

var CmdMtlsServer = &cobra.Command{
	Use:   "mtlsserver",
	Short: "Server is configured to build mTLS with client and echo client's spiffe identity",
	Args:  cobra.MaximumNArgs(0),
	RunE:  main,
}

var (
	listen                string
	serverCredsFile       string
	spiffeTrustBundleFile string
)

func init() {
	CmdMtlsServer.Flags().StringVar(&listen, "listen", "", "<address>:<port> to listen on")
	CmdMtlsServer.Flags().StringVar(&serverCredsFile, "server-creds", "", "Credential bundle with the server key and certificate chain")
	CmdMtlsServer.Flags().StringVar(&spiffeTrustBundleFile, "spiffe-trust-bundle", "", "Trust bundle for verifying client certificates")
}

func main(cmd *cobra.Command, args []string) error {
	logs.InitLogs()
	defer logs.FlushLogs()

	if listen == "" || serverCredsFile == "" || spiffeTrustBundleFile == "" {
		return fmt.Errorf("missing required flags")
	}

	if err := serve(); err != nil {
		return fmt.Errorf("error while serving: %w", err)
	}
	return nil
}

func serve() error {
	serveMux := http.NewServeMux()
	serveMux.HandleFunc("GET /spiffe-echo", handleGetSPIFFEEcho)

	clientTrustBundlePEM, err := os.ReadFile(spiffeTrustBundleFile)
	if err != nil {
		return fmt.Errorf("while reading SPIFFE trust anchors: %w", err)
	}

	rootPool := x509.NewCertPool()
	if ok := rootPool.AppendCertsFromPEM(clientTrustBundlePEM); !ok {
		return fmt.Errorf("failed to append client certs from PEM")
	}

	// Pre-load server cert to check early if it's readable
	_, err = tls.LoadX509KeyPair(serverCredsFile, serverCredsFile)
	if err != nil {
		return fmt.Errorf("while loading server key pair: %w", err)
	}

	server := &http.Server{
		Addr: listen,

		Handler: serveMux,

		TLSConfig: &tls.Config{
			// Tell the client to send client certs if they have any.  Don't
			// pass in roots to verify the client certificate, since we expect
			// multiple types signed by different CAs.  Instead, each endpoint
			// will do the appropriate client certificate verification.
			ClientAuth: tls.RequestClientCert,
		},

		ReadTimeout:    30 * time.Second,
		WriteTimeout:   30 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}

	// TODO: Auto-reload the server creds
	if err := server.ListenAndServeTLS(serverCredsFile, serverCredsFile); err != nil {
		return fmt.Errorf("while listening: %w", err)
	}

	return nil
}

func handleGetSPIFFEEcho(w http.ResponseWriter, r *http.Request) {
	if len(r.TLS.PeerCertificates) == 0 {
		http.Error(w, "SPIFFE client certificate authentication required", http.StatusUnauthorized)
		return
	}

	leafCert := r.TLS.PeerCertificates[0]

	// TODO: Use the trust domain from the leaf certificate to select which
	// trust bundle to use, to permit federated use cases.

	intermediatePool := x509.NewCertPool()
	if len(r.TLS.PeerCertificates) > 1 {
		for _, intermediate := range r.TLS.PeerCertificates[1:] {
			intermediatePool.AddCert(intermediate)
		}
	}
	rootTrustBundlePEM, err := os.ReadFile(spiffeTrustBundleFile)
	if err != nil {
		log.Printf("Error while reading SPIFFE trust anchors: %v", err)
		http.Error(w, "Internal Error", http.StatusInternalServerError)
		return
	}
	rootPool := x509.NewCertPool()
	rootPool.AppendCertsFromPEM(rootTrustBundlePEM)

	chains, err := leafCert.Verify(x509.VerifyOptions{
		Intermediates: intermediatePool,
		Roots:         rootPool,
	})
	if err != nil {
		log.Printf("Error while verifying SPIFFE client certificate: %v", err)
		http.Error(w, "SPIFFE client certificate authentication required", http.StatusUnauthorized)
		return
	}
	if len(chains) == 0 {
		log.Print("Client certificate did not chain to any roots")
		http.Error(w, "SPIFFE client certificate authentication required", http.StatusUnauthorized)
		return
	}

	if len(leafCert.URIs) != 1 {
		log.Printf("SPIFFE client certificate did not have 1 URI SAN, count: %d", len(leafCert.URIs))
		http.Error(w, "Malformed SPIFFE certificate", http.StatusUnauthorized)
		return
	}

	if _, err := w.Write([]byte("Client Identity: " + leafCert.URIs[0].String())); err != nil {
		log.Printf("Error while writing response: %v", err)
	}
}
