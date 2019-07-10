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
	"net"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

// Exponential backoff for MemberAdd/Remove (values exclude jitter):
// 0, 50, 150, 350, 750, 1550, 3150, 6350, 12750 ms
var addRemoveBackoff = wait.Backoff{
	Steps:    8,
	Duration: 50 * time.Millisecond,
	Factor:   2.0,
	Jitter:   0.1,
}

// ClusterInterrogator is an interface to get etcd cluster related information
type ClusterInterrogator interface {
	ClusterAvailable() (bool, error)
	GetClusterStatus() (map[string]*clientv3.StatusResponse, error)
	GetClusterVersions() (map[string]string, error)
	GetVersion() (string, error)
	WaitForClusterAvailable(retries int, retryInterval time.Duration) (bool, error)
	Sync() error
	AddMember(name string, peerAddrs string) ([]Member, error)
	GetMemberID(peerURL string) (uint64, error)
	RemoveMember(id uint64) ([]Member, error)
}

// Client provides connection parameters for an etcd cluster
type Client struct {
	Endpoints []string
	TLS       *tls.Config
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

// NewFromCluster creates an etcd client for the etcd endpoints defined in the ClusterStatus value stored in
// the kubeadm-config ConfigMap in kube-system namespace.
// Once created, the client synchronizes client's endpoints with the known endpoints from the etcd membership API (reality check).
func NewFromCluster(client clientset.Interface, certificatesDir string) (*Client, error) {
	// etcd is listening the API server advertise address on each control-plane node
	// so it is necessary to get the list of endpoints from kubeadm cluster status before connecting

	// Gets the cluster status
	clusterStatus, err := config.GetClusterStatus(client)
	if err != nil {
		return nil, err
	}

	// Get the list of etcd endpoints from cluster status
	endpoints := []string{}
	for _, e := range clusterStatus.APIEndpoints {
		endpoints = append(endpoints, GetClientURLByIP(e.AdvertiseAddress))
	}
	klog.V(1).Infof("etcd endpoints read from pods: %s", strings.Join(endpoints, ","))

	// Creates an etcd client
	etcdClient, err := New(
		endpoints,
		filepath.Join(certificatesDir, constants.EtcdCACertName),
		filepath.Join(certificatesDir, constants.EtcdHealthcheckClientCertName),
		filepath.Join(certificatesDir, constants.EtcdHealthcheckClientKeyName),
	)
	if err != nil {
		return nil, errors.Wrapf(err, "error creating etcd client for %v endpoints", endpoints)
	}

	// synchronizes client's endpoints with the known endpoints from the etcd membership.
	err = etcdClient.Sync()
	if err != nil {
		return nil, errors.Wrap(err, "error syncing endpoints with etc")
	}
	klog.V(1).Infof("update etcd endpoints: %s", strings.Join(etcdClient.Endpoints, ","))

	return etcdClient, nil
}

// Sync synchronizes client's endpoints with the known endpoints from the etcd membership.
func (c *Client) Sync() error {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: 20 * time.Second,
		TLS:         c.TLS,
	})
	if err != nil {
		return err
	}
	defer cli.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	err = cli.Sync(ctx)
	cancel()
	if err != nil {
		return err
	}
	klog.V(1).Infof("etcd endpoints read from etcd: %s", strings.Join(cli.Endpoints(), ","))

	c.Endpoints = cli.Endpoints()
	return nil
}

// Member struct defines an etcd member; it is used for avoiding to spread github.com/coreos/etcd dependency
// across kubeadm codebase
type Member struct {
	Name    string
	PeerURL string
}

// GetMemberID returns the member ID of the given peer URL
func (c *Client) GetMemberID(peerURL string) (uint64, error) {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: 30 * time.Second,
		TLS:         c.TLS,
	})
	if err != nil {
		return 0, err
	}
	defer cli.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	resp, err := cli.MemberList(ctx)
	cancel()
	if err != nil {
		return 0, err
	}

	for _, member := range resp.Members {
		if member.GetPeerURLs()[0] == peerURL {
			return member.GetID(), nil
		}
	}
	return 0, nil
}

// RemoveMember notifies an etcd cluster to remove an existing member
func (c *Client) RemoveMember(id uint64) ([]Member, error) {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: 30 * time.Second,
		TLS:         c.TLS,
	})
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	// Remove an existing member from the cluster
	var lastError error
	var resp *clientv3.MemberRemoveResponse
	err = wait.ExponentialBackoff(addRemoveBackoff, func() (bool, error) {
		resp, err = cli.MemberRemove(context.Background(), id)
		if err == nil {
			return true, nil
		}
		lastError = err
		return false, nil
	})
	if err != nil {
		return nil, lastError
	}

	// Returns the updated list of etcd members
	ret := []Member{}
	for _, m := range resp.Members {
		ret = append(ret, Member{Name: m.Name, PeerURL: m.PeerURLs[0]})
	}

	return ret, nil
}

// AddMember notifies an existing etcd cluster that a new member is joining
func (c *Client) AddMember(name string, peerAddrs string) ([]Member, error) {
	// Parse the peer address, required to add the client URL later to the list
	// of endpoints for this client. Parsing as a first operation to make sure that
	// if this fails no member addition is performed on the etcd cluster.
	parsedPeerAddrs, err := url.Parse(peerAddrs)
	if err != nil {
		return nil, errors.Wrapf(err, "error parsing peer address %s", peerAddrs)
	}

	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: 20 * time.Second,
		TLS:         c.TLS,
	})
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	// Adds a new member to the cluster
	var lastError error
	var resp *clientv3.MemberAddResponse
	err = wait.ExponentialBackoff(addRemoveBackoff, func() (bool, error) {
		resp, err = cli.MemberAdd(context.Background(), []string{peerAddrs})
		if err == nil {
			return true, nil
		}
		lastError = err
		return false, nil
	})
	if err != nil {
		return nil, lastError
	}

	// Returns the updated list of etcd members
	ret := []Member{}
	for _, m := range resp.Members {
		// fixes the entry for the joining member (that doesn't have a name set in the initialCluster returned by etcd)
		if m.Name == "" {
			ret = append(ret, Member{Name: name, PeerURL: m.PeerURLs[0]})
		} else {
			ret = append(ret, Member{Name: m.Name, PeerURL: m.PeerURLs[0]})
		}
	}

	// Add the new member client address to the list of endpoints
	c.Endpoints = append(c.Endpoints, GetClientURLByIP(parsedPeerAddrs.Hostname()))

	return ret, nil
}

// GetVersion returns the etcd version of the cluster.
// An error is returned if the version of all endpoints do not match
func (c *Client) GetVersion() (string, error) {
	var clusterVersion string

	versions, err := c.GetClusterVersions()
	if err != nil {
		return "", err
	}
	for _, v := range versions {
		if clusterVersion != "" && clusterVersion != v {
			return "", errors.Errorf("etcd cluster contains endpoints with mismatched versions: %v", versions)
		}
		clusterVersion = v
	}
	if clusterVersion == "" {
		return "", errors.New("could not determine cluster etcd version")
	}
	return clusterVersion, nil
}

// GetClusterVersions returns a map of the endpoints and their associated versions
func (c *Client) GetClusterVersions() (map[string]string, error) {
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
func (c *Client) ClusterAvailable() (bool, error) {
	_, err := c.GetClusterStatus()
	if err != nil {
		return false, err
	}
	return true, nil
}

// GetClusterStatus returns nil for status Up or error for status Down
func (c *Client) GetClusterStatus() (map[string]*clientv3.StatusResponse, error) {
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

// WaitForClusterAvailable returns true if all endpoints in the cluster are available after retry attempts, an error is returned otherwise
func (c *Client) WaitForClusterAvailable(retries int, retryInterval time.Duration) (bool, error) {
	for i := 0; i < retries; i++ {
		if i > 0 {
			klog.V(1).Infof("[etcd] Waiting %v until next retry\n", retryInterval)
			time.Sleep(retryInterval)
		}
		klog.V(2).Infof("[etcd] attempting to see if all cluster endpoints (%s) are available %d/%d", c.Endpoints, i+1, retries)
		resp, err := c.ClusterAvailable()
		if err != nil {
			switch err {
			case context.DeadlineExceeded:
				klog.V(1).Infof("[etcd] Attempt timed out")
			default:
				klog.V(1).Infof("[etcd] Attempt failed with error: %v\n", err)
			}
			continue
		}
		return resp, nil
	}
	return false, errors.New("timeout waiting for etcd cluster to be available")
}

// CheckConfigurationIsHA returns true if the given InitConfiguration etcd block appears to be an HA configuration.
func CheckConfigurationIsHA(cfg *kubeadmapi.Etcd) bool {
	return cfg.External != nil && len(cfg.External.Endpoints) > 1
}

// GetClientURL creates an HTTPS URL that uses the configured advertise
// address and client port for the API controller
func GetClientURL(localEndpoint *kubeadmapi.APIEndpoint) string {
	return "https://" + net.JoinHostPort(localEndpoint.AdvertiseAddress, strconv.Itoa(constants.EtcdListenClientPort))
}

// GetPeerURL creates an HTTPS URL that uses the configured advertise
// address and peer port for the API controller
func GetPeerURL(localEndpoint *kubeadmapi.APIEndpoint) string {
	return "https://" + net.JoinHostPort(localEndpoint.AdvertiseAddress, strconv.Itoa(constants.EtcdListenPeerPort))
}

// GetClientURLByIP creates an HTTPS URL based on an IP address
// and the client listening port.
func GetClientURLByIP(ip string) string {
	return "https://" + net.JoinHostPort(ip, strconv.Itoa(constants.EtcdListenClientPort))
}
