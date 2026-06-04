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

package reconcilers

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"path"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	storagefactory "k8s.io/apiserver/pkg/storage/storagebackend/factory"
)

const (
	APIServerIdentityLabel = "apiserverIdentity"
)

type PeerAdvertiseAddress struct {
	PeerAdvertiseIP   string
	PeerAdvertisePort string
}

type peerEndpointLeases struct {
	storage   storage.Interface
	destroyFn func()
	baseKey   string
	leaseTime time.Duration
}

type PeerEndpointLeaseReconciler interface {
	// GetEndpoint retrieves the endpoint for a given apiserverId
	GetEndpoint(serverId string) (string, error)
	// UpdateLease updates the ip and port of peer servers
	UpdateLease(serverId string, ip string, endpointPorts []corev1.EndpointPort) error
	// RemoveEndpoints removes this apiserver's peer endpoint lease.
	RemoveLease(serverId string) error
	// Destroy cleans up everything on shutdown.
	Destroy()
	// StopReconciling turns any later ReconcileEndpoints call into a noop.
	StopReconciling()
}

type peerEndpointLeaseReconciler struct {
	serverLeases          *peerEndpointLeases
	stopReconcilingCalled atomic.Bool
}

// NewPeerEndpointLeaseReconciler creates a new peer endpoint lease reconciler
func NewPeerEndpointLeaseReconciler(config *storagebackend.ConfigForResource, baseKey string, leaseTime time.Duration) (PeerEndpointLeaseReconciler, error) {
	// note that newFunc, newListFunc
	// can be left blank unless the storage.Watch method is used
	leaseStorage, destroyFn, err := storagefactory.Create(*config, nil, nil, baseKey)
	if err != nil {
		return nil, fmt.Errorf("error creating storage factory: %v", err)
	}
	var once sync.Once
	return &peerEndpointLeaseReconciler{
		serverLeases: &peerEndpointLeases{
			storage:   leaseStorage,
			destroyFn: func() { once.Do(destroyFn) },
			baseKey:   baseKey,
			leaseTime: leaseTime,
		},
	}, nil
}

// PeerEndpointController is the controller manager for updating the peer endpoint leases.
// This provides a separate independent reconciliation loop for peer endpoint leases
// which ensures that the peer kube-apiservers are fetching the updated endpoint info for a given apiserver
// in the case when the peer wants to proxy the request to the given apiserver because it can not serve the
// request itself due to version mismatch.
type PeerEndpointLeaseController struct {
	reconciler       PeerEndpointLeaseReconciler
	endpointInterval time.Duration
	serverId         string
	// peeraddress stores the IP and port of this kube-apiserver. Used by peer kube-apiservers to
	// route request to this apiserver in case of a version skew.
	peeraddress string

	client kubernetes.Interface

	lock   sync.Mutex
	stopCh chan struct{} // closed by Stop()
}

func New(serverId string, peeraddress string,
	reconciler PeerEndpointLeaseReconciler, endpointInterval time.Duration, client kubernetes.Interface) *PeerEndpointLeaseController {
	return &PeerEndpointLeaseController{
		reconciler: reconciler,
		serverId:   serverId,
		// peeraddress stores the IP and port of this kube-apiserver. Used by peer kube-apiservers to
		// route request to this apiserver in case of a version skew.
		peeraddress:      peeraddress,
		endpointInterval: endpointInterval,
		client:           client,
		stopCh:           make(chan struct{}),
	}
}

// Start begins the peer endpoint lease reconciler loop that must exist for bootstrapping
// a cluster.
func (c *PeerEndpointLeaseController) Start(stopCh <-chan struct{}) {
	localStopCh := make(chan struct{})
	go func() {
		defer close(localStopCh)
		select {
		case <-stopCh: // from Start
		case <-c.stopCh: // from Stop
		}
	}()
	go c.Run(localStopCh)
}

// RunPeerEndpointReconciler periodically updates the peer endpoint leases
func (c *PeerEndpointLeaseController) Run(stopCh <-chan struct{}) {
	// wait until process is ready
	wait.PollImmediateUntil(100*time.Millisecond, func() (bool, error) {
		var code int
		c.client.CoreV1().RESTClient().Get().AbsPath("/readyz").Do(context.TODO()).StatusCode(&code)
		return code == http.StatusOK, nil
	}, stopCh)

	wait.NonSlidingUntil(func() {
		if err := c.UpdatePeerEndpointLeases(); err != nil {
			runtime.HandleError(fmt.Errorf("unable to update peer endpoint leases: %v", err))
		}
	}, c.endpointInterval, stopCh)
}

// Stop cleans up this apiserver's peer endpoint leases.
func (c *PeerEndpointLeaseController) Stop() {
	c.lock.Lock()
	defer c.lock.Unlock()

	select {
	case <-c.stopCh:
		return // only close once
	default:
		close(c.stopCh)
	}
	finishedReconciling := make(chan struct{})
	go func() {
		defer close(finishedReconciling)
		klog.Infof("Shutting down peer endpoint lease reconciler")
		// stop reconciliation
		c.reconciler.StopReconciling()

		// Ensure that there will be no race condition with the ReconcileEndpointLeases.
		if err := c.reconciler.RemoveLease(c.serverId); err != nil {
			klog.Errorf("Unable to remove peer endpoint leases: %v", err)
		}
		c.reconciler.Destroy()
	}()

	select {
	case <-finishedReconciling:
		// done
	case <-time.After(2 * c.endpointInterval):
		// don't block server shutdown forever if we can't reach etcd to remove ourselves
		klog.Warning("peer_endpoint_controller's RemoveEndpoints() timed out")
	}
}

// UpdatePeerEndpointLeases attempts to update the peer endpoint leases.
func (c *PeerEndpointLeaseController) UpdatePeerEndpointLeases() error {
	host, port, err := net.SplitHostPort(c.peeraddress)
	if err != nil {
		return err
	}

	p, err := strconv.Atoi(port)
	if err != nil {
		return err
	}
	endpointPorts := createEndpointPortSpec(p, "https")

	// Ensure that there will be no race condition with the RemoveEndpointLeases.
	c.lock.Lock()
	defer c.lock.Unlock()

	// Refresh the TTL on our key, independently of whether any error or
	// update conflict happens below. This makes sure that at least some of
	// the servers will add our endpoint lease.
	if err := c.reconciler.UpdateLease(c.serverId, host, endpointPorts); err != nil {
		return err
	}
	return nil
}

// UpdateLease resets the TTL on a server IP in storage
// UpdateLease will create a new key if it doesn't exist.
// We use the first element in endpointPorts as a part of the lease's base key
// This is done to support out tests that simulate 2 apiservers running on the same ip but
// different ports

// It will also do the following if UnknownVersionInteroperabilityProxy feature is enabled
// 1. store the apiserverId as a label
// 2. store the values passed to --peer-advertise-ip and --peer-advertise-port flags to kube-apiserver as an annotation
// with value of format <ip:port>
func (r *peerEndpointLeaseReconciler) UpdateLease(serverId string, ip string, endpointPorts []corev1.EndpointPort) error {
	// reconcile endpoints only if apiserver was not shutdown
	if r.stopReconcilingCalled.Load() {
		return nil
	}

	// we use the serverID as the key to avoid using the server IP, port as the key.
	// note: this means that this lease doesn't enforce mutual exclusion of ip/port usage between apiserver.
	key := path.Join(r.serverLeases.baseKey, serverId)
	return r.serverLeases.storage.GuaranteedUpdate(apirequest.NewDefaultContext(), key, &corev1.Endpoints{}, true, nil, func(input kruntime.Object, respMeta storage.ResponseMeta) (kruntime.Object, *uint64, error) {
		existing := input.(*corev1.Endpoints)
		existing.Subsets = []corev1.EndpointSubset{
			{
				Addresses: []corev1.EndpointAddress{{IP: ip}},
				Ports:     endpointPorts,
			},
		}

		// store this server's identity (serverId) as a label. This will be used by
		// peers to find the IP of this server when the peer can not serve a request
		// due to version skew.
		if existing.Labels == nil {
			existing.Labels = map[string]string{}
		}
		existing.Labels[APIServerIdentityLabel] = serverId

		// leaseTime needs to be in seconds
		leaseTime := uint64(r.serverLeases.leaseTime / time.Second)

		// NB: GuaranteedUpdate does not perform the store operation unless
		// something changed between load and store (not including resource
		// version), meaning we can't refresh the TTL without actually
		// changing a field.
		existing.Generation++

		klog.V(6).Infof("Resetting TTL on server IP %q listed in storage to %v", ip, leaseTime)
		return existing, &leaseTime, nil
	}, nil)
}

// ListLeases retrieves a list of the current server IPs from storage
func (r *peerEndpointLeaseReconciler) ListLeases() ([]string, error) {
	storageOpts := storage.ListOptions{
		ResourceVersion:      "0",
		ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
		Predicate:            storage.Everything,
		Recursive:            true,
	}
	ipInfoList, err := r.getIpInfoList(storageOpts)
	if err != nil {
		return nil, err
	}
	ipList := make([]string, 0, len(ipInfoList.Items))
	for _, ip := range ipInfoList.Items {
		if len(ip.Subsets) > 0 && len(ip.Subsets[0].Addresses) > 0 && len(ip.Subsets[0].Addresses[0].IP) > 0 {
			ipList = append(ipList, ip.Subsets[0].Addresses[0].IP)
		}
	}
	klog.V(6).Infof("Current server IPs listed in storage are %v", ipList)
	return ipList, nil
}

// GetLease retrieves the server IP and port for a specific server id
func (r *peerEndpointLeaseReconciler) GetLease(serverId string) (string, error) {
	var fullAddr string
	if serverId == "" {
		return "", fmt.Errorf("error getting endpoint for serverId: empty serverId")
	}
	storageOpts := storage.ListOptions{
		ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
		Predicate:            storage.Everything,
		Recursive:            true,
	}
	ipInfoList, err := r.getIpInfoList(storageOpts)
	if err != nil {
		return "", err
	}

	for _, ip := range ipInfoList.Items {
		if ip.Labels[APIServerIdentityLabel] == serverId {
			if len(ip.Subsets) > 0 {
				var ipStr, portStr string
				if len(ip.Subsets[0].Addresses) > 0 {
					if len(ip.Subsets[0].Addresses[0].IP) > 0 {
						ipStr = ip.Subsets[0].Addresses[0].IP
					}
				}
				if len(ip.Subsets[0].Ports) > 0 {
					portStr = fmt.Sprint(ip.Subsets[0].Ports[0].Port)
				}
				fullAddr = net.JoinHostPort(ipStr, portStr)
				break
			}
		}
	}
	klog.V(6).Infof("Fetched this server IP for the specified apiserverId %v, %v", serverId, fullAddr)
	return fullAddr, nil
}

func (r *peerEndpointLeaseReconciler) StopReconciling() {
	r.stopReconcilingCalled.Store(true)
}

// RemoveLease removes the lease on a server IP in storage
// We use the first element in endpointPorts as a part of the lease's base key
// This is done to support out tests that simulate 2 apiservers running on the same ip but
// different ports
func (r *peerEndpointLeaseReconciler) RemoveLease(serverId string) error {
	key := path.Join(r.serverLeases.baseKey, serverId)
	return r.serverLeases.storage.Delete(apirequest.NewDefaultContext(), key, &corev1.Endpoints{}, nil, rest.ValidateAllObjectFunc, nil, storage.DeleteOptions{})
}

func (r *peerEndpointLeaseReconciler) Destroy() {
	r.serverLeases.destroyFn()
}

func (r *peerEndpointLeaseReconciler) GetEndpoint(serverId string) (string, error) {
	return r.GetLease(serverId)
}

func (r *peerEndpointLeaseReconciler) getIpInfoList(storageOpts storage.ListOptions) (*corev1.EndpointsList, error) {
	ipInfoList := &corev1.EndpointsList{}
	if err := r.serverLeases.storage.GetList(apirequest.NewDefaultContext(), r.serverLeases.baseKey, storageOpts, ipInfoList); err != nil {
		return nil, err
	}
	return ipInfoList, nil
}

// createEndpointPortSpec creates the endpoint ports
func createEndpointPortSpec(endpointPort int, endpointPortName string) []corev1.EndpointPort {
	return []corev1.EndpointPort{{
		Protocol: corev1.ProtocolTCP,
		Port:     int32(endpointPort),
		Name:     endpointPortName,
	}}
}
