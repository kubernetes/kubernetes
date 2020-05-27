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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
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
	WaitForClusterAvailable(retries int, retryInterval time.Duration) (bool, error)
	Sync() error
	ListMembers() ([]Member, error)
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

// NewFromCluster creates an etcd client for the etcd endpoints present in etcd member list. In order to compose this information,
// it will first discover at least one etcd endpoint to connect to. Once created, the client synchronizes client's endpoints with
// the known endpoints from the etcd membership API, since it is the authoritative source of truth for the list of available members.
func NewFromCluster(client clientset.Interface, certificatesDir string) (*Client, error) {
	// Discover at least one etcd endpoint to connect to by inspecting the existing etcd pods

	// Get the list of etcd endpoints
	endpoints, err := getEtcdEndpoints(client)
	if err != nil {
		return nil, err
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
		return nil, errors.Wrap(err, "error syncing endpoints with etcd")
	}
	klog.V(1).Infof("update etcd endpoints: %s", strings.Join(etcdClient.Endpoints, ","))

	return etcdClient, nil
}

// getEtcdEndpoints returns the list of etcd endpoints.
func getEtcdEndpoints(client clientset.Interface) ([]string, error) {
	return getEtcdEndpointsWithBackoff(client, constants.StaticPodMirroringDefaultRetry)
}

func getEtcdEndpointsWithBackoff(client clientset.Interface, backoff wait.Backoff) ([]string, error) {
	etcdEndpoints, err := getRawEtcdEndpointsFromPodAnnotation(client, backoff)
	if err != nil {
		// NB: this is a fallback when there is no annotation found in the etcd pods that contains
		//     the client URL, and so we fallback to reading the ClusterStatus struct present in the
		//     kubeadm-config ConfigMap. This can happen for example, when performing the first
		//     `kubeadm upgrade apply`. This logic will be removed when the cluster status struct
		//     is removed from the kubeadm-config ConfigMap.
		return getRawEtcdEndpointsFromClusterStatus(client)
	}
	return etcdEndpoints, nil
}

// getRawEtcdEndpointsFromPodAnnotation returns the list of endpoints as reported on etcd's pod annotations using the given backoff
func getRawEtcdEndpointsFromPodAnnotation(client clientset.Interface, backoff wait.Backoff) ([]string, error) {
	etcdEndpoints := []string{}
	var lastErr error
	// Let's tolerate some unexpected transient failures from the API server or load balancers. Also, if
	// static pods were not yet mirrored into the API server we want to wait for this propagation.
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		var overallEtcdPodCount int
		if etcdEndpoints, overallEtcdPodCount, lastErr = getRawEtcdEndpointsFromPodAnnotationWithoutRetry(client); lastErr != nil {
			return false, nil
		}
		// TODO (ereslibre): this logic will need tweaking once that we get rid of the ClusterStatus, since we won't have
		// the ClusterStatus safety net we will have to retry in both cases.
		if len(etcdEndpoints) == 0 {
			if overallEtcdPodCount == 0 {
				return false, nil
			}
			// Fail fast scenario, to be removed once we get rid of the ClusterStatus
			return true, errors.New("etcd Pods exist, but no etcd endpoint annotations were found")
		}
		return true, nil
	})
	if err != nil {
		if lastErr != nil {
			return []string{}, errors.Wrap(lastErr, "could not retrieve the list of etcd endpoints")
		}
		return []string{}, errors.Wrap(err, "could not retrieve the list of etcd endpoints")
	}
	return etcdEndpoints, nil
}

// getRawEtcdEndpointsFromPodAnnotationWithoutRetry returns the list of etcd endpoints as reported by etcd Pod annotations,
// along with the number of global etcd pods. This allows for callers to tell the difference between "no endpoints found",
// and "no endpoints found and pods were listed", so they can skip retrying.
func getRawEtcdEndpointsFromPodAnnotationWithoutRetry(client clientset.Interface) ([]string, int, error) {
	klog.V(3).Infof("retrieving etcd endpoints from %q annotation in etcd Pods", constants.EtcdAdvertiseClientUrlsAnnotationKey)
	podList, err := client.CoreV1().Pods(metav1.NamespaceSystem).List(
		context.TODO(),
		metav1.ListOptions{
			LabelSelector: fmt.Sprintf("component=%s,tier=%s", constants.Etcd, constants.ControlPlaneTier),
		},
	)
	if err != nil {
		return []string{}, 0, err
	}
	etcdEndpoints := []string{}
	for _, pod := range podList.Items {
		etcdEndpoint, ok := pod.ObjectMeta.Annotations[constants.EtcdAdvertiseClientUrlsAnnotationKey]
		if !ok {
			klog.V(3).Infof("etcd Pod %q is missing the %q annotation; cannot infer etcd advertise client URL using the Pod annotation", pod.ObjectMeta.Name, constants.EtcdAdvertiseClientUrlsAnnotationKey)
			continue
		}
		etcdEndpoints = append(etcdEndpoints, etcdEndpoint)
	}
	return etcdEndpoints, len(podList.Items), nil
}

// TODO: remove after 1.20, when the ClusterStatus struct is removed from the kubeadm-config ConfigMap.
func getRawEtcdEndpointsFromClusterStatus(client clientset.Interface) ([]string, error) {
	klog.V(3).Info("retrieving etcd endpoints from the cluster status")
	clusterStatus, err := config.GetClusterStatus(client)
	if err != nil {
		return []string{}, err
	}
	etcdEndpoints := []string{}
	for _, e := range clusterStatus.APIEndpoints {
		etcdEndpoints = append(etcdEndpoints, GetClientURLByIP(e.AdvertiseAddress))
	}
	return etcdEndpoints, nil
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

func (c *Client) listMembers() (*clientv3.MemberListResponse, error) {
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
		return nil, lastError
	}
	return resp, nil
}

// GetMemberID returns the member ID of the given peer URL
func (c *Client) GetMemberID(peerURL string) (uint64, error) {
	resp, err := c.listMembers()
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

// ListMembers returns the member list.
func (c *Client) ListMembers() ([]Member, error) {
	resp, err := c.listMembers()
	if err != nil {
		return nil, err
	}

	ret := make([]Member, 0, len(resp.Members))
	for _, m := range resp.Members {
		ret = append(ret, Member{Name: m.Name, PeerURL: m.PeerURLs[0]})
	}
	return ret, nil
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
