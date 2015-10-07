package kubelet

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"encoding/json"
	flannelIp "github.com/coreos/flannel/pkg/ip"
	flannelSubnet "github.com/coreos/flannel/subnet"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// Lease is a single flannel subnet lease.
type Lease struct {
	flannelSubnet.Lease
	ResourceVersion string
}

// LeaseList is a list of flannel subnet leases.
type LeaseList struct {
	Items           []Lease
	ResourceVersion string
}

// LeaseEvent is a single watch event.
type LeaseEvent struct {
	Lease
	Type watch.EventType
}

// nodeGetterSetter is a client for nodes alone.
type nodeGetterSetter interface {
	Get(name string) (*api.Node, error)
	Set(*api.Node) (*api.Node, error)
	List() (*api.NodeList, error)
}

type rawClientGetterSetter struct {
	client *client.Client
}

func (r *rawClientGetterSetter) Get(name string) (*api.Node, error) {
	return r.client.Nodes().Get(name)
}

func (r *rawClientGetterSetter) Set(node *api.Node) (*api.Node, error) {
	return r.client.Nodes().Update(node)
}

func (r *rawClientGetterSetter) List() (*api.NodeList, error) {
	return r.client.Nodes().List(labels.Everything(), fields.Everything())
}

// nodeAnnotationHelper sets and gets node annotations.
type nodeAnnotationHelper struct {
	nodeGetterSetter
}

func (nah *nodeAnnotationHelper) getAnnotation(key string, node *api.Node) (string, error) {
	if node.Annotations == nil {
		return "", fmt.Errorf("Node %v has no annotations", node.Name)
	}
	an, ok := node.Annotations[key]
	if !ok {
		return "", fmt.Errorf("Key %v not found in node annotations for %v",
			key, node.Name)
	}
	return an, nil
}

func (nah *nodeAnnotationHelper) setAnnotation(key, value string, node *api.Node) error {
	if node.Annotations == nil {
		node.Annotations = map[string]string{}
	}
	if an, ok := node.Annotations[key]; ok && an == value {
		return nil
	}
	node.Annotations[key] = value
	_, err := nah.Set(node)
	if err != nil {
		return err
	}
	return nil
}

// nodeIpHelper parses node ips into a format flannel understands.
type nodeIpHelper struct {
	nodeGetterSetter
}

func (nih *nodeIpHelper) getIp(node *api.Node) (*flannelIp.IP4, error) {
	// This is confusing, by default flannel uses the ip of the
	// interface associated with the gateway route. On GCE this
	// is the "internal" ip of the node. We want the flannel ip
	// to match the node ip, so getIp returns the same.

	for _, a := range node.Status.Addresses {
		if a.Type == api.NodeInternalIP {
			if ip, err := flannelIp.ParseIP4(a.Address); err != nil {
				return nil, err
			} else {
				return &ip, nil
			}
		}
	}
	return nil, fmt.Errorf("Ip of node %v not found", node.Name)
}

func (nih *nodeIpHelper) getSubnet(node *api.Node) (*flannelIp.IP4Net, error) {
	cidr := strings.Split(node.Spec.PodCIDR, "/")
	if len(cidr) != 2 {
		return nil, fmt.Errorf("No pod cidr for node %v", node.Name)
	}

	ip, err := flannelIp.ParseIP4(cidr[0])
	if err != nil {
		return nil, err
	}
	iCidr, err := strconv.Atoi(cidr[1])
	if err != nil {
		return nil, fmt.Errorf("Bad cidr for node %v: %v", iCidr, node.Name)
	}

	return &flannelIp.IP4Net{IP: ip, PrefixLen: uint(iCidr)}, nil
}

func (nih *nodeIpHelper) getNodeByIp(ip string) (*api.Node, error) {
	nodes, err := nih.List()
	if err != nil {
		return nil, err
	}
	for _, n := range nodes.Items {
		flannelIp, err := nih.getIp(&n)
		if err != nil {
			return nil, err
		}
		if flannelIp.String() == ip {
			return &n, nil
		}
	}
	return nil, fmt.Errorf("Node with ip %v not found", ip)
}

// nodeToLeaseTranslator converts nodes into their respective subnet leases.
type nodeToLeaseTranslator struct {
	nodeIpHelper
	nodeAnnotationHelper
}

func (nt *nodeToLeaseTranslator) getLease(node *api.Node) (lease Lease, err error) {
	ip, err := nt.getIp(node)
	if err != nil {
		return lease, err
	}

	s, err := nt.getSubnet(node)
	if err != nil {
		return lease, err
	}

	backendData, err := nt.getAnnotation(flannelBackendDataAnnotation, node)
	if err != nil {
		// TODO: Can we survive without this?
		return lease, err
	}

	attrs := &flannelSubnet.LeaseAttrs{
		PublicIP: *ip,
		// TODO: populate this from network spec
		BackendType: networkType,
		BackendData: json.RawMessage([]byte(backendData)),
	}

	// TODO: Do we really want to expire leases?
	return Lease{flannelSubnet.Lease{*s, attrs, time.Now().Add(day)}, node.ResourceVersion}, nil
}

// NewNodeToLeaseTranslator returns a translator that can convert nodes to leases.
func NewNodeToLeaseTranslator(c *client.Client) nodeToLeaseTranslator {
	nodeClient := &rawClientGetterSetter{c}
	return nodeToLeaseTranslator{
		nodeIpHelper{nodeClient},
		nodeAnnotationHelper{nodeClient},
	}
}

// LeaseClient is a kubernetes client for a virtual lease resource backed by nodes.
type LeaseClient struct {
	nodeToLeaseTranslator
	kubeClient *client.Client
}

// List lists all flannel subnet leases.
func (l *LeaseClient) List() (LeaseList, error) {

	nodes, err := l.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return LeaseList{}, err
	}

	leases := []Lease{}
	for _, n := range nodes.Items {
		lease, err := l.getLease(&n)
		if err != nil {
			glog.Infof("Error converting node %v to lease: %v", n.Name, err)
			continue
		}
		leases = append(leases, lease)
	}
	return LeaseList{leases, nodes.ResourceVersion}, nil
}

// Watch watches for a *single* update to the virtual lease resource, after the given resource version.
func (l *LeaseClient) Watch(rv string) (LeaseEvent, error) {
	var err error
	leaseEvent := LeaseEvent{}

	w, err := l.kubeClient.Nodes().Watch(labels.Everything(), fields.Everything(), api.ListOptions{ResourceVersion: rv})
	if err != nil {
		return leaseEvent, err
	}
	defer w.Stop()

	// This is a performance bottleneck waiting to happen, but it isn't in the
	// the critical path, and hanging is just so simple for now. The level of
	// optimization required here depends on how the flannel client behaves.
	event, ok := <-w.ResultChan()
	if !ok {
		return leaseEvent, fmt.Errorf("Watch error")
	}

	node := event.Object.(*api.Node)
	lease, err := l.getLease(node)
	if err != nil {
		return leaseEvent, err
	}
	return LeaseEvent{lease, event.Type}, nil
}

// KubeFlannelClient is a client capable of watching kubernetes and responding to flannel.
type KubeFlannelClient struct {
	client LeaseClient
}

func (k *KubeFlannelClient) ListLeases() (wr flannelSubnet.LeaseWatchResult, err error) {
	leases, err := k.client.List()
	if err != nil {
		return wr, err
	}
	snapshot := []flannelSubnet.Lease{}
	for _, l := range leases.Items {
		snapshot = append(snapshot, l.Lease)
	}
	return flannelSubnet.LeaseWatchResult{
		Snapshot: snapshot,
		Cursor:   leases.ResourceVersion,
	}, nil
}

func (k *KubeFlannelClient) WatchLeases(rv string) (flannelSubnet.LeaseWatchResult, error) {

	// Force clients to pass in a legit resource version by givng them a snapshot
	if rv == "0" || rv == "" {
		return k.ListLeases()
	}

	e, err := k.client.Watch(rv)
	if err != nil {
		return k.ListLeases()
	}

	var flannelEventType flannelSubnet.EventType
	switch t := e.Type; t {
	case watch.Added, watch.Modified:
		flannelEventType = flannelSubnet.EventAdded
	case watch.Deleted:
		flannelEventType = flannelSubnet.EventRemoved
	default:
		// Kubernetes doesn't understand the etcd expired. So that results
		// in a relist handled above via the err clause.
		return flannelSubnet.LeaseWatchResult{}, fmt.Errorf("Unknown event %v", t)
	}

	ev := flannelSubnet.Event{
		Type:    flannelEventType,
		Lease:   e.Lease.Lease,
		Network: "",
	}
	return flannelSubnet.LeaseWatchResult{
		Events: []flannelSubnet.Event{ev},
		Cursor: e.ResourceVersion,
	}, nil
}

// NewKubeFlannelClient creates a kubernetes x flannel client.
func NewKubeFlannelClient(c *client.Client) *KubeFlannelClient {
	translator := NewNodeToLeaseTranslator(c)
	return &KubeFlannelClient{
		LeaseClient{
			nodeToLeaseTranslator: translator,
			kubeClient:            c,
		}}
}
