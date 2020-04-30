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

	"github.com/pkg/errors"
	"go.etcd.io/etcd/clientv3"
	"go.etcd.io/etcd/pkg/transport"
	"google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

const etcdTimeout = 2 * time.Second

// Exponential backoff for etcd operations
var etcdBackoff = wait.Backoff{
	Steps:    11,
	Duration: 50 * time.Millisecond,
	Factor:   2.0,
	Jitter:   0.1,
}

// ClusterInterrogator is an interface to get etcd cluster related information
type ClusterInterrogator interface {
	CheckClusterHealth() error
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

// dialTimeout is the timeout for failing to establish a connection.
// It is set to >20 seconds as times shorter than that will cause TLS connections to fail
// on heavily loaded arm64 CPUs (issue #64649)
const dialTimeout = 40 * time.Second

// Sync synchronizes client's endpoints with the known endpoints from the etcd membership.
func (c *Client) Sync() error {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: dialTimeout,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: c.TLS,
	})
	if err != nil {
		return err
	}
	defer cli.Close()

	// Syncs the list of endpoints
	var lastError error
	err = wait.ExponentialBackoff(etcdBackoff, func() (bool, error) {
		ctx, cancel := context.WithTimeout(context.Background(), etcdTimeout)
		err = cli.Sync(ctx)
		cancel()
		if err == nil {
			return true, nil
		}
		klog.V(5).Infof("Failed to sync etcd endpoints: %v", err)
		lastError = err
		return false, nil
	})
	if err != nil {
		return lastError
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
		DialTimeout: dialTimeout,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: c.TLS,
	})
	if err != nil {
		return 0, err
	}
	defer cli.Close()

	// Gets the member list
	var lastError error
	var resp *clientv3.MemberListResponse
	err = wait.ExponentialBackoff(etcdBackoff, func() (bool, error) {
		ctx, cancel := context.WithTimeout(context.Background(), etcdTimeout)
		resp, err = cli.MemberList(ctx)
		cancel()
		if err == nil {
			return true, nil
		}
		klog.V(5).Infof("Failed to get etcd member list: %v", err)
		lastError = err
		return false, nil
	})
	if err != nil {
		return 0, lastError
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
		DialTimeout: dialTimeout,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: c.TLS,
	})
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	// Remove an existing member from the cluster
	var lastError error
	var resp *clientv3.MemberRemoveResponse
	err = wait.ExponentialBackoff(etcdBackoff, func() (bool, error) {
		ctx, cancel := context.WithTimeout(context.Background(), etcdTimeout)
		resp, err = cli.MemberRemove(ctx, id)
		cancel()
		if err == nil {
			return true, nil
		}
		klog.V(5).Infof("Failed to remove etcd member: %v", err)
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

	// Exponential backoff for the MemberAdd operation (up to ~200 seconds)
	etcdBackoffAdd := wait.Backoff{
		Steps:    18,
		Duration: 100 * time.Millisecond,
		Factor:   1.5,
		Jitter:   0.1,
	}

	// Adds a new member to the cluster
	var lastError error
	var resp *clientv3.MemberAddResponse
	err = wait.ExponentialBackoff(etcdBackoffAdd, func() (bool, error) {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   c.Endpoints,
			DialTimeout: etcdTimeout,
			DialOptions: []grpc.DialOption{
				grpc.WithBlock(), // block until the underlying connection is up
			},
			TLS: c.TLS,
		})
		if err != nil {
			lastError = err
			return false, nil
		}
		defer cli.Close()

		ctx, cancel := context.WithTimeout(context.Background(), etcdTimeout)
		resp, err = cli.MemberAdd(ctx, []string{peerAddrs})
		cancel()
		if err == nil {
			return true, nil
		}
		klog.V(5).Infof("Failed to add etcd member: %v", err)
		lastError = err
		return false, nil
	})
	if err != nil {
		return nil, lastError
	}

	// Returns the updated list of etcd members
	ret := []Member{}
	for _, m := range resp.Members {
		// If the peer address matches, this is the member we are adding.
		// Use the name we passed to the function.
		if peerAddrs == m.PeerURLs[0] {
			ret = append(ret, Member{Name: name, PeerURL: peerAddrs})
			continue
		}
		// Otherwise, we are processing other existing etcd members returned by AddMembers.
		memberName := m.Name
		// In some cases during concurrent join, some members can end up without a name.
		// Use the member ID as name for those.
		if len(memberName) == 0 {
			memberName = strconv.FormatUint(m.ID, 16)
		}
		ret = append(ret, Member{Name: memberName, PeerURL: m.PeerURLs[0]})
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
	statuses, err := c.getClusterStatus()
	if err != nil {
		return versions, err
	}

	for ep, status := range statuses {
		versions[ep] = status.Version
	}
	return versions, nil
}

// CheckClusterHealth returns nil for status Up or error for status Down
func (c *Client) CheckClusterHealth() error {
	_, err := c.getClusterStatus()
	return err
}

// getClusterStatus returns nil for status Up (along with endpoint status response map) or error for status Down
func (c *Client) getClusterStatus() (map[string]*clientv3.StatusResponse, error) {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.Endpoints,
		DialTimeout: dialTimeout,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: c.TLS,
	})
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	clusterStatus := make(map[string]*clientv3.StatusResponse)
	for _, ep := range c.Endpoints {
		// Gets the member status
		var lastError error
		var resp *clientv3.StatusResponse
		err = wait.ExponentialBackoff(etcdBackoff, func() (bool, error) {
			ctx, cancel := context.WithTimeout(context.Background(), etcdTimeout)
			resp, err = cli.Status(ctx, ep)
			cancel()
			if err == nil {
				return true, nil
			}
			klog.V(5).Infof("Failed to get etcd status for %s: %v", ep, err)
			lastError = err
			return false, nil
		})
		if err != nil {
			return nil, lastError
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
		_, err := c.getClusterStatus()
		if err != nil {
			switch err {
			case context.DeadlineExceeded:
				klog.V(1).Infof("[etcd] Attempt timed out")
			default:
				klog.V(1).Infof("[etcd] Attempt failed with error: %v\n", err)
			}
			continue
		}
		return true, nil
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
