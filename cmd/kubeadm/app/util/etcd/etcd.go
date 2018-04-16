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

// Cluster is an interface to get etcd cluster related information
type Cluster interface {
	HasTLS() (bool, error)
	GetStatus() (*clientv3.StatusResponse, error)
}

// StaticPodCluster represents an instance of a static pod etcd cluster.
// CertificatesDir should contain the etcd CA and healthcheck client TLS identity.
// ManifestDir should contain the etcd static pod manifest.
type StaticPodCluster struct {
	Endpoints       []string
	CertificatesDir string
	ManifestDir     string
}

// HasTLS returns a boolean representing whether the static pod etcd cluster implements TLS.
// It may return an error for file I/O issues.
func (cluster StaticPodCluster) HasTLS() (bool, error) {
	return PodManifestsHaveTLS(cluster.ManifestDir)
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

// GetStatus invokes the proper protocol check based off of whether the cluster HasTLS() to get the cluster's status
func (cluster StaticPodCluster) GetStatus() (*clientv3.StatusResponse, error) {
	hasTLS, err := cluster.HasTLS()
	if err != nil {
		return nil, fmt.Errorf("failed to determine if current etcd static pod is using TLS: %v", err)
	}

	var tlsConfig *tls.Config
	if hasTLS {
		tlsConfig, err = NewTLSConfig(cluster.CertificatesDir)
		if err != nil {
			return nil, fmt.Errorf("failed to create a TLS Config using the cluster.CertificatesDir: %v", err)
		}
	}

	return GetClusterStatus(cluster.Endpoints, tlsConfig)
}

// NewTLSConfig generates a tlsConfig using credentials from the default sub-paths of the certificates directory
func NewTLSConfig(certificatesDir string) (*tls.Config, error) {
	tlsInfo := transport.TLSInfo{
		CertFile:      filepath.Join(certificatesDir, constants.EtcdHealthcheckClientCertName),
		KeyFile:       filepath.Join(certificatesDir, constants.EtcdHealthcheckClientKeyName),
		TrustedCAFile: filepath.Join(certificatesDir, constants.EtcdCACertName),
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return nil, err
	}

	return tlsConfig, nil
}

// GetClusterStatus returns nil for status Up or error for status Down
func GetClusterStatus(endpoints []string, tlsConfig *tls.Config) (*clientv3.StatusResponse, error) {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: 5 * time.Second,
		TLS:         tlsConfig,
	})
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	resp, err := cli.Status(context.Background(), endpoints[0])
	if err != nil {
		return nil, err
	}

	return resp, nil
}
