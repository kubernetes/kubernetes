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

// Package mtlsclient is an agnhost subcommand implementing a client to build
// the mTLS with the server.
package mtlsclient

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"
)

var CmdMtlsClient = &cobra.Command{
	Use:   "mtlsclient",
	Short: "Client is configured to build mTLS with server",
	Args:  cobra.MaximumNArgs(0),
	RunE:  main,
}

var (
	fetchURL              string
	serverTrustBundleFile string
	clientCredBundleFile  string
)

func init() {
	CmdMtlsClient.Flags().StringVar(&fetchURL, "fetch-url", "", "server URL to poll")
	CmdMtlsClient.Flags().StringVar(&serverTrustBundleFile, "server-trust-bundle", "", "File with trust anchors to verify the server certificate")
	CmdMtlsClient.Flags().StringVar(&clientCredBundleFile, "client-cred-bundle", "", "File with client key and certificate chain")
}

func main(cmd *cobra.Command, args []string) error {
	logs.InitLogs()
	defer logs.FlushLogs()

	if fetchURL == "" || serverTrustBundleFile == "" || clientCredBundleFile == "" {
		return fmt.Errorf("missing required flags")
	}

	if _, err := url.ParseRequestURI(fetchURL); err != nil {
		return fmt.Errorf("invalid --fetch-url: %w", err)
	}

	for range time.Tick(10 * time.Second) {
		if err := pollOnce(); err != nil {
			klog.Errorf("while sending requests to the server: %v", err)
		}
	}
	return nil
}

func pollOnce() error {
	trustBundlePEM, err := os.ReadFile(serverTrustBundleFile)
	if err != nil {
		return fmt.Errorf("while reading service trust bundle: %w", err)
	}

	serverTrustAnchors := x509.NewCertPool()
	serverTrustAnchors.AppendCertsFromPEM(trustBundlePEM)

	tlsConfig := &tls.Config{
		RootCAs: serverTrustAnchors,
	}

	// Load and send client certificates if a bundle file was specified.
	if clientCredBundleFile != "" {
		bundlePEM, err := os.ReadFile(clientCredBundleFile)
		if err != nil {
			return fmt.Errorf("while reading client credential bundle: %w", err)
		}

		cert := tls.Certificate{}

		var block *pem.Block
		rest := bundlePEM
		for {
			block, rest = pem.Decode(rest)
			if block == nil {
				break
			}

			switch block.Type {
			case "PRIVATE KEY":
				cert.PrivateKey, err = x509.ParsePKCS8PrivateKey(block.Bytes)
				if err != nil {
					return fmt.Errorf("while parsing private key from credential bundle: %w", err)
				}
			case "CERTIFICATE":
				cert.Certificate = append(cert.Certificate, block.Bytes)
			}
		}

		if cert.PrivateKey == nil {
			return fmt.Errorf("client credential bundle had no private key")
		}

		if len(cert.Certificate) == 0 {
			return fmt.Errorf("client credential bundle had no certificates")
		}

		tlsConfig.Certificates = []tls.Certificate{cert}
	}

	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: tlsConfig,
		},
	}

	resp, err := client.Get(fetchURL)
	if err != nil {
		return fmt.Errorf("while getting URL: %w", err)

	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("non-2xx status %d %q", resp.StatusCode, resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("while reading body: %w", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Printf("while closing response body: %v", err)
		}
	}()

	log.Printf("Got response body: %s", string(body))
	return nil
}
