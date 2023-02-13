/*
Copyright 2023 The Kubernetes Authors.

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

package dynamiccertificates

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"sync/atomic"

	"k8s.io/klog/v2"
)

type ConnContext func(ctx context.Context, c net.Conn) context.Context
type GetConfigForClient func(info *tls.ClientHelloInfo) (*tls.Config, error)

// contextKey type is unexported to prevent collisions.
type contextKey int

const chainsKey contextKey = iota

func WithChainsConnContext(ctx context.Context, _ net.Conn) context.Context {
	// TODO may also track conn to allow the server to close it when the cert if expired but otherwise valid?
	c := &atomic.Pointer[chains]{} // safe for concurrent access
	return context.WithValue(ctx, chainsKey, c)
}

func setChainsOnce(ctx context.Context, c *chains) error {
	chainsPtr, err := getChains(ctx)
	if err != nil {
		return err
	}
	if !chainsPtr.CompareAndSwap(nil, c) {
		return fmt.Errorf("chains set more than once")
	}
	return nil
}

func getChains(ctx context.Context) (*atomic.Pointer[chains], error) {
	chainsPtr, _ := ctx.Value(chainsKey).(*atomic.Pointer[chains])
	if chainsPtr == nil {
		return nil, fmt.Errorf("chains not wired to server context")
	}
	return chainsPtr, nil
}

type chains map[string]chainData // TODO maybe use CAContentProvider as the map key?

type chainData struct {
	caBundleContent []byte
	chains          [][]*x509.Certificate
	err             error
}

func WithChainsGetConfigForClient(getConfig GetConfigForClient, clientCAUnion CAContentProvider) GetConfigForClient {
	return func(info *tls.ClientHelloInfo) (*tls.Config, error) {
		tlsConfig, err := getConfig(info)
		if err != nil {
			return nil, err
		}

		// TODO verify no overlapping names or something like that
		clientCAs := splitUnion(clientCAUnion)

		tlsConfig = tlsConfig.Clone()

		origVerifyConnection := tlsConfig.VerifyConnection
		tlsConfig.VerifyConnection = func(state tls.ConnectionState) error {
			if origVerifyConnection != nil {
				if err := origVerifyConnection(state); err != nil {
					return err
				}
			}

			if len(state.PeerCertificates) == 0 {
				return nil // either this is the call before peer certs are known or no certs provided
			}

			out := make(chains)
			for _, clientCA := range clientCAs {
				clientCA := clientCA

				// avoid any TOCTOU issues
				caBundle := clientCA.CurrentCABundleContent()
				opts, err := newCABundleAndVerifier(clientCA.Name(), caBundle)
				if err != nil {
					// if there are intentionally no verify options, then we cannot authenticate using this bundle
					klog.ErrorS(err, "invalid bundle prevented TLS verification", "name", clientCA.Name())
					continue // the CA bundle contents are always supposed to be valid so this should not happen in practice
				}

				optsCopy := opts.verifyOptions

				// Use intermediates, if provided
				if optsCopy.Intermediates == nil && len(state.PeerCertificates) > 1 {
					optsCopy.Intermediates = x509.NewCertPool()
					for _, intermediate := range state.PeerCertificates[1:] {
						optsCopy.Intermediates.AddCert(intermediate)
					}
				}

				verifyChains, err := state.PeerCertificates[0].Verify(optsCopy)

				out[clientCA.Name()] = chainData{
					chains: verifyChains,
					err:    err,
				}
			}

			if len(out) == 0 {
				return nil
			}

			return setChainsOnce(info.Context(), &out)
		}

		return tlsConfig, nil
	}
}

func WithChainsVerification(ctx context.Context, clientCA CAContentProvider) ([][]*x509.Certificate, error) {
	chainsPtr, err := getChains(ctx)
	if err != nil {
		return nil, err
	}

	chainsLoad := chainsPtr.Load()
	if chainsLoad == nil {
		return nil, fmt.Errorf("tls verification was not performed on this connection")
	}
	chainsLoaded := *chainsLoad

	data, ok := chainsLoaded[clientCA.Name()]
	if !ok {
		return nil, fmt.Errorf("tls verification was not performed on this connection for %q", clientCA.Name())
	}

	// TODO this is wrong
	//  the new caBundleContent may be able to verify the TLS details so we should not fail on that case
	if !bytes.Equal(data.caBundleContent, clientCA.CurrentCABundleContent()) {
		return nil, fmt.Errorf("tls verifcation options have changed and the connection must be closed")
	}

	return data.chains, data.err
}

// TODO this is a hack, we should be given all of the CAs as distinct input so we know how to bucket them
func splitUnion(clientCA CAContentProvider) unionCAContent {
	caContents, ok := clientCA.(unionCAContent)
	if !ok {
		return unionCAContent{caContents}
	}
	var out unionCAContent
	for _, ca := range caContents {
		ca := ca
		split := splitUnion(ca)
		out = append(out, split...)
	}
	return out
}
