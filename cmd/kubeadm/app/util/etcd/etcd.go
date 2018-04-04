/*
Copyright 2018 The Kubernetes Authors.

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

package etcd

import (
	"context"
	"crypto/tls"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

// ClusterInterrogator is an interface to get etcd cluster related information
type ClusterInterrogator interface {
	GetStatus() (*clientv3.StatusResponse, error)
	WaitForStatus(delay time.Duration, retries int, retryInterval time.Duration) (*clientv3.StatusResponse, error)
	HasTLS() bool
	GetClusterStatus() ([]*clientv3.StatusResponse, error)
}

// Client provides connection parameters for an etcd cluster
type Client struct {
	Endpoints []string
	TLS       *tls.Config
}

// HasTLS returns true if etcd is configured for TLS
func (c Client) HasTLS() bool {
	return c.TLS != nil
}

// PodManifestsHaveTLS reads the etcd staticpod manifest from disk and returns false if the TLS flags
// are missing from the command list. If all the flags are present it returns true.
func PodManifestsHaveTLS(ManifestDir string) (bool, error) {
	etcdPodPath := constants.GetStaticPodFilepath(constants.Etcd, ManifestDir)
	etcdPod, err := staticpod.ReadStaticPodFromDisk(etcdPodPath)
	if err != nil {
		return false, fmt.Errorf("failed to check if etcd pod implements TLS: %v", err)
	}

	tlsFlags := []string{
		"--cert-file=",
		"--key-file=",
		"--trusted-ca-file=",
		"--client-cert-auth=",
		"--peer-cert-file=",
		"--peer-key-file=",
		"--peer-trusted-ca-file=",
		"--peer-client-cert-auth=",
	}
FlagLoop:
	for _, flag := range tlsFlags {
		for _, container := range etcdPod.Spec.Containers {
			for _, arg := range container.Command {
				if strings.Contains(arg, flag) {
					continue FlagLoop
				}
			}
		}
		// flag not found in any container
		return false, nil
	}
	// all flags were found in container args; pod fully implements TLS
	return true, nil
}

// GetStatus gets server status
func (c Client) GetStatus() (*clientv3.StatusResponse, error) {
	const dialTimeout = 5 * time.Second
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: dialTimeout,
		TLS:         c.TLS,
	})
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	resp, err := cli.Status(ctx, c.Endpoints[0])
	cancel()
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// WaitForStatus returns a StatusResponse after an initial delay and retry attempts
func (c Client) WaitForStatus(delay time.Duration, retries int, retryInterval time.Duration) (*clientv3.StatusResponse, error) {
	fmt.Printf("[util/etcd] Waiting %v for initial delay\n", delay)
	time.Sleep(delay)
	for i := 0; i < retries; i++ {
		if i > 0 {
			fmt.Printf("[util/etcd] Waiting %v until next retry\n", retryInterval)
			time.Sleep(retryInterval)
		}
		fmt.Printf("[util/etcd] Attempting to get etcd status %d/%d\n", i+1, retries)
		resp, err := c.GetStatus()
		if err != nil {
			switch err {
			case context.DeadlineExceeded:
				fmt.Println("[util/etcd] Attempt timed out")
			default:
				fmt.Printf("[util/etcd] Attempt failed with error: %v\n", err)
			}
			continue
		}
		return resp, nil
	}
	return nil, fmt.Errorf("timeout waiting for etcd cluster status")
}

// New creates a new EtcdCluster client
func New(endpoints []string, ca, cert, key string) (*Client, error) {
	client := Client{Endpoints: endpoints}

	if ca != "" || cert != "" || key != "" {
		tlsInfo := transport.TLSInfo{
			CertFile:      cert,
			KeyFile:       key,
			TrustedCAFile: ca,
		}
		tlsConfig, err := tlsInfo.ClientConfig()
		if err != nil {
			return nil, err
		}
		client.TLS = tlsConfig
	}

	return &client, nil
}

// NewFromStaticPod creates a GenericClient from the given endpoints, manifestDir, and certificatesDir
func NewFromStaticPod(endpoints []string, manifestDir string, certificatesDir string) (*Client, error) {
	hasTLS, err := PodManifestsHaveTLS(manifestDir)
	if err != nil {
		return nil, fmt.Errorf("could not read manifests from: %s, error: %v", manifestDir, err)
	}
	if hasTLS {
		return New(
			endpoints,
			filepath.Join(certificatesDir, constants.EtcdCACertName),
			filepath.Join(certificatesDir, constants.EtcdHealthcheckClientCertName),
			filepath.Join(certificatesDir, constants.EtcdHealthcheckClientKeyName),
		)
	}
	return New(endpoints, "", "", "")
}

// GetClusterStatus returns nil for status Up or error for status Down
func (c Client) GetClusterStatus() ([]*clientv3.StatusResponse, error) {

	var resp []*clientv3.StatusResponse
	for _, ep := range c.Endpoints {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   []string{ep},
			DialTimeout: 5 * time.Second,
			TLS:         c.TLS,
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
