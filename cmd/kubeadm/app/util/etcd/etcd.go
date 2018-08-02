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
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

// ClusterInterrogator is an interface to get etcd cluster related information
type ClusterInterrogator interface {
	ClusterAvailable() (bool, error)
	GetClusterStatus() (map[string]*clientv3.StatusResponse, error)
	GetClusterVersions() (map[string]string, error)
	GetVersion() (string, error)
	HasTLS() bool
	WaitForClusterAvailable(delay time.Duration, retries int, retryInterval time.Duration) (bool, error)
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

// GetVersion returns the etcd version of the cluster.
// An error is returned if the version of all endpoints do not match
func (c Client) GetVersion() (string, error) {
	var clusterVersion string

	versions, err := c.GetClusterVersions()
	if err != nil {
		return "", err
	}
	for _, v := range versions {
		if clusterVersion != "" && clusterVersion != v {
			return "", fmt.Errorf("etcd cluster contains endpoints with mismatched versions: %v", versions)
		}
		clusterVersion = v
	}
	if clusterVersion == "" {
		return "", fmt.Errorf("could not determine cluster etcd version")
	}
	return clusterVersion, nil
}

// GetClusterVersions returns a map of the endpoints and their associated versions
func (c Client) GetClusterVersions() (map[string]string, error) {
	versions := make(map[string]string)
	statuses, err := c.GetClusterStatus()
	if err != nil {
		return versions, err
	}

	for ep, status := range statuses {
		versions[ep] = status.Version
	}
	return versions, nil
}

// ClusterAvailable returns true if the cluster status indicates the cluster is available.
func (c Client) ClusterAvailable() (bool, error) {
	_, err := c.GetClusterStatus()
	if err != nil {
		return false, err
	}
	return true, nil
}

// GetClusterStatus returns nil for status Up or error for status Down
func (c Client) GetClusterStatus() (map[string]*clientv3.StatusResponse, error) {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: 5 * time.Second,
		TLS:         c.TLS,
	})
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	clusterStatus := make(map[string]*clientv3.StatusResponse)
	for _, ep := range c.Endpoints {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		resp, err := cli.Status(ctx, ep)
		cancel()
		if err != nil {
			return nil, err
		}
		clusterStatus[ep] = resp
	}
	return clusterStatus, nil
}

// WaitForClusterAvailable returns true if all endpoints in the cluster are available after an initial delay and retry attempts, an error is returned otherwise
func (c Client) WaitForClusterAvailable(delay time.Duration, retries int, retryInterval time.Duration) (bool, error) {
	fmt.Printf("[util/etcd] Waiting %v for initial delay\n", delay)
	time.Sleep(delay)
	for i := 0; i < retries; i++ {
		if i > 0 {
			fmt.Printf("[util/etcd] Waiting %v until next retry\n", retryInterval)
			time.Sleep(retryInterval)
		}
		fmt.Printf("[util/etcd] Attempting to see if all cluster endpoints are available %d/%d\n", i+1, retries)
		resp, err := c.ClusterAvailable()
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
	return false, fmt.Errorf("timeout waiting for etcd cluster to be available")
}

// CheckConfigurationIsHA returns true if the given InitConfiguration etcd block appears to be an HA configuration.
func CheckConfigurationIsHA(cfg *kubeadmapi.Etcd) bool {
	return cfg.External != nil && len(cfg.External.Endpoints) > 1
}
