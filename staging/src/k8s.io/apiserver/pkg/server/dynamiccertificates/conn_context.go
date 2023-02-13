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
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"sync/atomic"
)

type ConnContext func(ctx context.Context, c net.Conn) context.Context
type GetConfigForClient func(info *tls.ClientHelloInfo) (*tls.Config, error)

// contextKey type is unexported to prevent collisions.
type contextKey int

const chainsKey contextKey = iota

func WithChainsConnContext(ctx context.Context, _ net.Conn) context.Context {
	// TODO may also track conn to allow the server to close it?
	c := &atomic.Pointer[chains]{} // safe for concurrent access
	return context.WithValue(ctx, chainsKey, c)
}

func setChains(ctx context.Context, c *chains) error {
	cc, _ := ctx.Value(chainsKey).(*atomic.Pointer[chains])
	if cc == nil {
		return fmt.Errorf("chains not wired to server context")
	}
	if !cc.CompareAndSwap(nil, c) {
		return fmt.Errorf("chains set more than once")
	}
	return nil
}

type chains struct {
	opts   x509.VerifyOptions
	ok     bool
	chains [][]*x509.Certificate
	err    error
}

func WithChainsGetConfigForClient(getConfig GetConfigForClient, clientCA CAContentProvider) GetConfigForClient {
	return func(info *tls.ClientHelloInfo) (*tls.Config, error) {
		tlsConfig, err := getConfig(info)
		if err != nil {
			return nil, err
		}

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

			// Use intermediates, if provided
			optsCopy, ok := clientCA.VerifyOptions()
			// if there are intentionally no verify options, then we cannot authenticate this request
			if !ok {
				return nil
			}
			if optsCopy.Intermediates == nil && len(state.PeerCertificates) > 1 {
				optsCopy.Intermediates = x509.NewCertPool()
				for _, intermediate := range state.PeerCertificates[1:] {
					optsCopy.Intermediates.AddCert(intermediate)
				}
			}

			verifyChains, err := state.PeerCertificates[0].Verify(optsCopy)

			return setChains(info.Context(),
				&chains{
					opts:   optsCopy,
					ok:     true,
					chains: verifyChains,
					err:    err,
				})
		}

		return tlsConfig, nil
	}
}
