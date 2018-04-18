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
	"fmt"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
)

// EtcdCluster is an interface to get etcd cluster related information
type EtcdCluster interface {
	GetEtcdClusterStatus() (*clientv3.StatusResponse, error)
}

// LocalEtcdCluster represents an instance of a local etcd cluster
type LocalEtcdCluster struct {
	Endpoints     []string
	TLS           bool
	CertFile      string
	KeyFile       string
	TrustedCAFile string
}

func (c LocalEtcdCluster) getTLSConfig() (*tls.Config, error) {
	tlsInfo := transport.TLSInfo{
		CertFile:      c.CertFile,
		KeyFile:       c.KeyFile,
		TrustedCAFile: c.TrustedCAFile,
	}
	return tlsInfo.ClientConfig()
}

func (c *LocalEtcdCluster) setEndpoints() {
	if c.TLS {
		c.Endpoints = []string{"https://localhost:2379"}
	} else {
		c.Endpoints = []string{"http://localhost:2379"}
	}
}

func (c LocalEtcdCluster) getEtcdClientConfig() (*clientv3.Config, error) {
	var tlsConfig *tls.Config
	if len(c.Endpoints) == 0 {
		c.setEndpoints()
	}

	if c.TLS {
		var err error
		tlsConfig, err = c.getTLSConfig()
		if err != nil {
			return nil, err
		}
	}

	return &clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: 5 * time.Second,
		TLS:         tlsConfig,
	}, nil
}

// GetEtcdClusterStatus returns a StatusResponse for status Up or error for status Down
func (c LocalEtcdCluster) GetEtcdClusterStatus() (*clientv3.StatusResponse, error) {
	clientConfig, err := c.getEtcdClientConfig()
	if err != nil {
		return nil, err
	}
	return getEtcdClusterStatus(clientConfig)
}

// WaitForEtcdClusterStatus returns a StatusResponse after an inital delay and retry attempts
func (c LocalEtcdCluster) WaitForEtcdClusterStatus(delay time.Duration, retries int, retryInterval time.Duration) (*clientv3.StatusResponse, error) {
	time.Sleep(delay)
	for i := 0; i < retries+1; i++ {
		resp, err := c.GetEtcdClusterStatus()
		if err == nil {
			return resp, nil
		}
		time.Sleep(retryInterval)
	}
	return nil, fmt.Errorf("timeout waiting for etcd cluster status")
}

func getEtcdClusterStatus(clientConfig *clientv3.Config) (*clientv3.StatusResponse, error) {
	cli, err := clientv3.New(*clientConfig)
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	resp, err := cli.Status(context.Background(), clientConfig.Endpoints[0])
	if err != nil {
		return nil, err
	}

	return resp, nil
}
