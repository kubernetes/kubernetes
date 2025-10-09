/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package service manages connections between the VM application and the ALTS
// handshaker service.
package service

import (
	"sync"
	"time"

	grpc "google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/internal/envconfig"
	"google.golang.org/grpc/keepalive"
)

var (
	// mu guards hsConnMap and hsDialer.
	mu sync.Mutex
	// hsConn represents a mapping from a hypervisor handshaker service address
	// to a corresponding connection to a hypervisor handshaker service
	// instance.
	hsConnMap = make(map[string]*grpc.ClientConn)
)

// Dial dials the handshake service in the hypervisor. If a connection has
// already been established, this function returns it. Otherwise, a new
// connection is created.
func Dial(hsAddress string) (*grpc.ClientConn, error) {
	mu.Lock()
	defer mu.Unlock()

	hsConn, ok := hsConnMap[hsAddress]
	if !ok {
		// Create a new connection to the handshaker service. Note that
		// this connection stays open until the application is closed.
		// Disable the service config to avoid unnecessary TXT record lookups that
		// cause timeouts with some versions of systemd-resolved.
		var err error
		opts := []grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDisableServiceConfig(),
		}
		if envconfig.ALTSHandshakerKeepaliveParams {
			opts = append(opts, grpc.WithKeepaliveParams(keepalive.ClientParameters{
				Timeout: 10 * time.Second,
				Time:    10 * time.Minute,
			}))
		}
		hsConn, err = grpc.NewClient(hsAddress, opts...)
		if err != nil {
			return nil, err
		}
		hsConnMap[hsAddress] = hsConn
	}
	return hsConn, nil
}

// CloseForTesting closes all open connections to the handshaker service.
//
// For testing purposes only.
func CloseForTesting() error {
	for _, hsConn := range hsConnMap {
		if hsConn == nil {
			continue
		}
		if err := hsConn.Close(); err != nil {
			return err
		}
	}

	// Reset the connection map.
	hsConnMap = make(map[string]*grpc.ClientConn)
	return nil
}
