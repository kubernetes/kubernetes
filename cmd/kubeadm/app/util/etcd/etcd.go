/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"context"
	"crypto/tls"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
)

// EtcdClusterInterrogator is an interface to get etcd cluster related information
type EtcdClusterInterrogator interface {
	GetEtcdClusterStatus() ([]*clientv3.StatusResponse, error)
}

// EtcdCluster provides connection parameters for an etcd cluster
type EtcdCluster struct {
	Endpoints []string
	TLS       *tls.Config
}

func New(endpoints []string, ca, cert, key string) (*EtcdCluster, error) {

	var tlsConfig *tls.Config
	var err error

	if ca != "" && cert != "" && key != "" {
		tlsInfo := transport.TLSInfo{
			CertFile:      cert,
			KeyFile:       key,
			TrustedCAFile: ca,
		}
		tlsConfig, err = tlsInfo.ClientConfig()
		if err != nil {
			return nil, err
		}
	}

	return &EtcdCluster{
		Endpoints: endpoints,
		TLS:       tlsConfig,
	}, nil
}

// GetEtcdClusterStatus returns nil for status Up or error for status Down
func (cluster EtcdCluster) GetEtcdClusterStatus() ([]*clientv3.StatusResponse, error) {

	var resp []*clientv3.StatusResponse
	for _, ep := range cluster.Endpoints {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   []string{ep},
			DialTimeout: 5 * time.Second,
			TLS:         cluster.TLS,
		})
		if err != nil {
			return nil, err
		}
		defer cli.Close()

		r, err := cli.Status(context.Background(), ep)
		if err != nil {
			return nil, err
		}
		resp = append(resp, r)
	}
	return resp, nil
}
