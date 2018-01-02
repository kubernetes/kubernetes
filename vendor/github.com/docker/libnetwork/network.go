package libnetwork

import (
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/libnetwork/common"
	"github.com/docker/libnetwork/config"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/etchosts"
	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/netutils"
	"github.com/docker/libnetwork/networkdb"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

// A Network represents a logical connectivity zone that containers may
// join using the Link method. A Network is managed by a specific driver.
type Network interface {
	// A user chosen name for this network.
	Name() string

	// A system generated id for this network.
	ID() string

	// The type of network, which corresponds to its managing driver.
	Type() string

	// Create a new endpoint to this network symbolically identified by the
	// specified unique name. The options parameter carries driver specific options.
	CreateEndpoint(name string, options ...EndpointOption) (Endpoint, error)

	// Delete the network.
	Delete() error

	// Endpoints returns the list of Endpoint(s) in this network.
	Endpoints() []Endpoint

	// WalkEndpoints uses the provided function to walk the Endpoints
	WalkEndpoints(walker EndpointWalker)

	// EndpointByName returns the Endpoint which has the passed name. If not found, the error ErrNoSuchEndpoint is returned.
	EndpointByName(name string) (Endpoint, error)

	// EndpointByID returns the Endpoint which has the passed id. If not found, the error ErrNoSuchEndpoint is returned.
	EndpointByID(id string) (Endpoint, error)

	// Return certain operational data belonging to this network
	Info() NetworkInfo
}

// NetworkInfo returns some configuration and operational information about the network
type NetworkInfo interface {
	IpamConfig() (string, map[string]string, []*IpamConf, []*IpamConf)
	IpamInfo() ([]*IpamInfo, []*IpamInfo)
	DriverOptions() map[string]string
	Scope() string
	IPv6Enabled() bool
	Internal() bool
	Attachable() bool
	Ingress() bool
	ConfigFrom() string
	ConfigOnly() bool
	Labels() map[string]string
	Dynamic() bool
	Created() time.Time
	// Peers returns a slice of PeerInfo structures which has the information about the peer
	// nodes participating in the same overlay network. This is currently the per-network
	// gossip cluster. For non-dynamic overlay networks and bridge networks it returns an
	// empty slice
	Peers() []networkdb.PeerInfo
	//Services returns a map of services keyed by the service name with the details
	//of all the tasks that belong to the service. Applicable only in swarm mode.
	Services() map[string]ServiceInfo
}

// EndpointWalker is a client provided function which will be used to walk the Endpoints.
// When the function returns true, the walk will stop.
type EndpointWalker func(ep Endpoint) bool

// ipInfo is the reverse mapping from IP to service name to serve the PTR query.
// extResolver is set if an externl server resolves a service name to this IP.
// Its an indication to defer PTR queries also to that external server.
type ipInfo struct {
	name        string
	serviceID   string
	extResolver bool
}

// svcMapEntry is the body of the element into the svcMap
// The ip is a string because the SetMatrix does not accept non hashable values
type svcMapEntry struct {
	ip        string
	serviceID string
}

type svcInfo struct {
	svcMap     common.SetMatrix
	svcIPv6Map common.SetMatrix
	ipMap      common.SetMatrix
	service    map[string][]servicePorts
}

// backing container or host's info
type serviceTarget struct {
	name string
	ip   net.IP
	port uint16
}

type servicePorts struct {
	portName string
	proto    string
	target   []serviceTarget
}

type networkDBTable struct {
	name    string
	objType driverapi.ObjectType
}

// IpamConf contains all the ipam related configurations for a network
type IpamConf struct {
	// The master address pool for containers and network interfaces
	PreferredPool string
	// A subset of the master pool. If specified,
	// this becomes the container pool
	SubPool string
	// Preferred Network Gateway address (optional)
	Gateway string
	// Auxiliary addresses for network driver. Must be within the master pool.
	// libnetwork will reserve them if they fall into the container pool
	AuxAddresses map[string]string
}

// Validate checks whether the configuration is valid
func (c *IpamConf) Validate() error {
	if c.Gateway != "" && nil == net.ParseIP(c.Gateway) {
		return types.BadRequestErrorf("invalid gateway address %s in Ipam configuration", c.Gateway)
	}
	return nil
}

// IpamInfo contains all the ipam related operational info for a network
type IpamInfo struct {
	PoolID string
	Meta   map[string]string
	driverapi.IPAMData
}

// MarshalJSON encodes IpamInfo into json message
func (i *IpamInfo) MarshalJSON() ([]byte, error) {
	m := map[string]interface{}{
		"PoolID": i.PoolID,
	}
	v, err := json.Marshal(&i.IPAMData)
	if err != nil {
		return nil, err
	}
	m["IPAMData"] = string(v)

	if i.Meta != nil {
		m["Meta"] = i.Meta
	}
	return json.Marshal(m)
}

// UnmarshalJSON decodes json message into PoolData
func (i *IpamInfo) UnmarshalJSON(data []byte) error {
	var (
		m   map[string]interface{}
		err error
	)
	if err = json.Unmarshal(data, &m); err != nil {
		return err
	}
	i.PoolID = m["PoolID"].(string)
	if v, ok := m["Meta"]; ok {
		b, _ := json.Marshal(v)
		if err = json.Unmarshal(b, &i.Meta); err != nil {
			return err
		}
	}
	if v, ok := m["IPAMData"]; ok {
		if err = json.Unmarshal([]byte(v.(string)), &i.IPAMData); err != nil {
			return err
		}
	}
	return nil
}

type network struct {
	ctrlr        *controller
	name         string
	networkType  string
	id           string
	created      time.Time
	scope        string // network data scope
	labels       map[string]string
	ipamType     string
	ipamOptions  map[string]string
	addrSpace    string
	ipamV4Config []*IpamConf
	ipamV6Config []*IpamConf
	ipamV4Info   []*IpamInfo
	ipamV6Info   []*IpamInfo
	enableIPv6   bool
	postIPv6     bool
	epCnt        *endpointCnt
	generic      options.Generic
	dbIndex      uint64
	dbExists     bool
	persist      bool
	stopWatchCh  chan struct{}
	drvOnce      *sync.Once
	resolverOnce sync.Once
	resolver     []Resolver
	internal     bool
	attachable   bool
	inDelete     bool
	ingress      bool
	driverTables []networkDBTable
	dynamic      bool
	configOnly   bool
	configFrom   string
	sync.Mutex
}

func (n *network) Name() string {
	n.Lock()
	defer n.Unlock()

	return n.name
}

func (n *network) ID() string {
	n.Lock()
	defer n.Unlock()

	return n.id
}

func (n *network) Created() time.Time {
	n.Lock()
	defer n.Unlock()

	return n.created
}

func (n *network) Type() string {
	n.Lock()
	defer n.Unlock()

	return n.networkType
}

func (n *network) Key() []string {
	n.Lock()
	defer n.Unlock()
	return []string{datastore.NetworkKeyPrefix, n.id}
}

func (n *network) KeyPrefix() []string {
	return []string{datastore.NetworkKeyPrefix}
}

func (n *network) Value() []byte {
	n.Lock()
	defer n.Unlock()
	b, err := json.Marshal(n)
	if err != nil {
		return nil
	}
	return b
}

func (n *network) SetValue(value []byte) error {
	return json.Unmarshal(value, n)
}

func (n *network) Index() uint64 {
	n.Lock()
	defer n.Unlock()
	return n.dbIndex
}

func (n *network) SetIndex(index uint64) {
	n.Lock()
	n.dbIndex = index
	n.dbExists = true
	n.Unlock()
}

func (n *network) Exists() bool {
	n.Lock()
	defer n.Unlock()
	return n.dbExists
}

func (n *network) Skip() bool {
	n.Lock()
	defer n.Unlock()
	return !n.persist
}

func (n *network) New() datastore.KVObject {
	n.Lock()
	defer n.Unlock()

	return &network{
		ctrlr:   n.ctrlr,
		drvOnce: &sync.Once{},
		scope:   n.scope,
	}
}

// CopyTo deep copies to the destination IpamConfig
func (c *IpamConf) CopyTo(dstC *IpamConf) error {
	dstC.PreferredPool = c.PreferredPool
	dstC.SubPool = c.SubPool
	dstC.Gateway = c.Gateway
	if c.AuxAddresses != nil {
		dstC.AuxAddresses = make(map[string]string, len(c.AuxAddresses))
		for k, v := range c.AuxAddresses {
			dstC.AuxAddresses[k] = v
		}
	}
	return nil
}

// CopyTo deep copies to the destination IpamInfo
func (i *IpamInfo) CopyTo(dstI *IpamInfo) error {
	dstI.PoolID = i.PoolID
	if i.Meta != nil {
		dstI.Meta = make(map[string]string)
		for k, v := range i.Meta {
			dstI.Meta[k] = v
		}
	}

	dstI.AddressSpace = i.AddressSpace
	dstI.Pool = types.GetIPNetCopy(i.Pool)
	dstI.Gateway = types.GetIPNetCopy(i.Gateway)

	if i.AuxAddresses != nil {
		dstI.AuxAddresses = make(map[string]*net.IPNet)
		for k, v := range i.AuxAddresses {
			dstI.AuxAddresses[k] = types.GetIPNetCopy(v)
		}
	}

	return nil
}

func (n *network) validateConfiguration() error {
	if n.configOnly {
		// Only supports network specific configurations.
		// Network operator configurations are not supported.
		if n.ingress || n.internal || n.attachable || n.scope != "" {
			return types.ForbiddenErrorf("configuration network can only contain network " +
				"specific fields. Network operator fields like " +
				"[ ingress | internal | attachable | scope ] are not supported.")
		}
	}
	if n.configFrom != "" {
		if n.configOnly {
			return types.ForbiddenErrorf("a configuration network cannot depend on another configuration network")
		}
		if n.ipamType != "" &&
			n.ipamType != defaultIpamForNetworkType(n.networkType) ||
			n.enableIPv6 ||
			len(n.labels) > 0 || len(n.ipamOptions) > 0 ||
			len(n.ipamV4Config) > 0 || len(n.ipamV6Config) > 0 {
			return types.ForbiddenErrorf("user specified configurations are not supported if the network depends on a configuration network")
		}
		if len(n.generic) > 0 {
			if data, ok := n.generic[netlabel.GenericData]; ok {
				var (
					driverOptions map[string]string
					opts          interface{}
				)
				switch data.(type) {
				case map[string]interface{}:
					opts = data.(map[string]interface{})
				case map[string]string:
					opts = data.(map[string]string)
				}
				ba, err := json.Marshal(opts)
				if err != nil {
					return fmt.Errorf("failed to validate network configuration: %v", err)
				}
				if err := json.Unmarshal(ba, &driverOptions); err != nil {
					return fmt.Errorf("failed to validate network configuration: %v", err)
				}
				if len(driverOptions) > 0 {
					return types.ForbiddenErrorf("network driver options are not supported if the network depends on a configuration network")
				}
			}
		}
	}
	return nil
}

// Applies network specific configurations
func (n *network) applyConfigurationTo(to *network) error {
	to.enableIPv6 = n.enableIPv6
	if len(n.labels) > 0 {
		to.labels = make(map[string]string, len(n.labels))
		for k, v := range n.labels {
			if _, ok := to.labels[k]; !ok {
				to.labels[k] = v
			}
		}
	}
	if len(n.ipamType) != 0 {
		to.ipamType = n.ipamType
	}
	if len(n.ipamOptions) > 0 {
		to.ipamOptions = make(map[string]string, len(n.ipamOptions))
		for k, v := range n.ipamOptions {
			if _, ok := to.ipamOptions[k]; !ok {
				to.ipamOptions[k] = v
			}
		}
	}
	if len(n.ipamV4Config) > 0 {
		to.ipamV4Config = make([]*IpamConf, 0, len(n.ipamV4Config))
		to.ipamV4Config = append(to.ipamV4Config, n.ipamV4Config...)
	}
	if len(n.ipamV6Config) > 0 {
		to.ipamV6Config = make([]*IpamConf, 0, len(n.ipamV6Config))
		to.ipamV6Config = append(to.ipamV6Config, n.ipamV6Config...)
	}
	if len(n.generic) > 0 {
		to.generic = options.Generic{}
		for k, v := range n.generic {
			to.generic[k] = v
		}
	}
	return nil
}

func (n *network) CopyTo(o datastore.KVObject) error {
	n.Lock()
	defer n.Unlock()

	dstN := o.(*network)
	dstN.name = n.name
	dstN.id = n.id
	dstN.created = n.created
	dstN.networkType = n.networkType
	dstN.scope = n.scope
	dstN.dynamic = n.dynamic
	dstN.ipamType = n.ipamType
	dstN.enableIPv6 = n.enableIPv6
	dstN.persist = n.persist
	dstN.postIPv6 = n.postIPv6
	dstN.dbIndex = n.dbIndex
	dstN.dbExists = n.dbExists
	dstN.drvOnce = n.drvOnce
	dstN.internal = n.internal
	dstN.attachable = n.attachable
	dstN.inDelete = n.inDelete
	dstN.ingress = n.ingress
	dstN.configOnly = n.configOnly
	dstN.configFrom = n.configFrom

	// copy labels
	if dstN.labels == nil {
		dstN.labels = make(map[string]string, len(n.labels))
	}
	for k, v := range n.labels {
		dstN.labels[k] = v
	}

	if n.ipamOptions != nil {
		dstN.ipamOptions = make(map[string]string, len(n.ipamOptions))
		for k, v := range n.ipamOptions {
			dstN.ipamOptions[k] = v
		}
	}

	for _, v4conf := range n.ipamV4Config {
		dstV4Conf := &IpamConf{}
		v4conf.CopyTo(dstV4Conf)
		dstN.ipamV4Config = append(dstN.ipamV4Config, dstV4Conf)
	}

	for _, v4info := range n.ipamV4Info {
		dstV4Info := &IpamInfo{}
		v4info.CopyTo(dstV4Info)
		dstN.ipamV4Info = append(dstN.ipamV4Info, dstV4Info)
	}

	for _, v6conf := range n.ipamV6Config {
		dstV6Conf := &IpamConf{}
		v6conf.CopyTo(dstV6Conf)
		dstN.ipamV6Config = append(dstN.ipamV6Config, dstV6Conf)
	}

	for _, v6info := range n.ipamV6Info {
		dstV6Info := &IpamInfo{}
		v6info.CopyTo(dstV6Info)
		dstN.ipamV6Info = append(dstN.ipamV6Info, dstV6Info)
	}

	dstN.generic = options.Generic{}
	for k, v := range n.generic {
		dstN.generic[k] = v
	}

	return nil
}

func (n *network) DataScope() string {
	s := n.Scope()
	// All swarm scope networks have local datascope
	if s == datastore.SwarmScope {
		s = datastore.LocalScope
	}
	return s
}

func (n *network) getEpCnt() *endpointCnt {
	n.Lock()
	defer n.Unlock()

	return n.epCnt
}

// TODO : Can be made much more generic with the help of reflection (but has some golang limitations)
func (n *network) MarshalJSON() ([]byte, error) {
	netMap := make(map[string]interface{})
	netMap["name"] = n.name
	netMap["id"] = n.id
	netMap["created"] = n.created
	netMap["networkType"] = n.networkType
	netMap["scope"] = n.scope
	netMap["labels"] = n.labels
	netMap["ipamType"] = n.ipamType
	netMap["ipamOptions"] = n.ipamOptions
	netMap["addrSpace"] = n.addrSpace
	netMap["enableIPv6"] = n.enableIPv6
	if n.generic != nil {
		netMap["generic"] = n.generic
	}
	netMap["persist"] = n.persist
	netMap["postIPv6"] = n.postIPv6
	if len(n.ipamV4Config) > 0 {
		ics, err := json.Marshal(n.ipamV4Config)
		if err != nil {
			return nil, err
		}
		netMap["ipamV4Config"] = string(ics)
	}
	if len(n.ipamV4Info) > 0 {
		iis, err := json.Marshal(n.ipamV4Info)
		if err != nil {
			return nil, err
		}
		netMap["ipamV4Info"] = string(iis)
	}
	if len(n.ipamV6Config) > 0 {
		ics, err := json.Marshal(n.ipamV6Config)
		if err != nil {
			return nil, err
		}
		netMap["ipamV6Config"] = string(ics)
	}
	if len(n.ipamV6Info) > 0 {
		iis, err := json.Marshal(n.ipamV6Info)
		if err != nil {
			return nil, err
		}
		netMap["ipamV6Info"] = string(iis)
	}
	netMap["internal"] = n.internal
	netMap["attachable"] = n.attachable
	netMap["inDelete"] = n.inDelete
	netMap["ingress"] = n.ingress
	netMap["configOnly"] = n.configOnly
	netMap["configFrom"] = n.configFrom
	return json.Marshal(netMap)
}

// TODO : Can be made much more generic with the help of reflection (but has some golang limitations)
func (n *network) UnmarshalJSON(b []byte) (err error) {
	var netMap map[string]interface{}
	if err := json.Unmarshal(b, &netMap); err != nil {
		return err
	}
	n.name = netMap["name"].(string)
	n.id = netMap["id"].(string)
	// "created" is not available in older versions
	if v, ok := netMap["created"]; ok {
		// n.created is time.Time but marshalled as string
		if err = n.created.UnmarshalText([]byte(v.(string))); err != nil {
			logrus.Warnf("failed to unmarshal creation time %v: %v", v, err)
			n.created = time.Time{}
		}
	}
	n.networkType = netMap["networkType"].(string)
	n.enableIPv6 = netMap["enableIPv6"].(bool)

	// if we weren't unmarshaling to netMap we could simply set n.labels
	// unfortunately, we can't because map[string]interface{} != map[string]string
	if labels, ok := netMap["labels"].(map[string]interface{}); ok {
		n.labels = make(map[string]string, len(labels))
		for label, value := range labels {
			n.labels[label] = value.(string)
		}
	}

	if v, ok := netMap["ipamOptions"]; ok {
		if iOpts, ok := v.(map[string]interface{}); ok {
			n.ipamOptions = make(map[string]string, len(iOpts))
			for k, v := range iOpts {
				n.ipamOptions[k] = v.(string)
			}
		}
	}

	if v, ok := netMap["generic"]; ok {
		n.generic = v.(map[string]interface{})
		// Restore opts in their map[string]string form
		if v, ok := n.generic[netlabel.GenericData]; ok {
			var lmap map[string]string
			ba, err := json.Marshal(v)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(ba, &lmap); err != nil {
				return err
			}
			n.generic[netlabel.GenericData] = lmap
		}
	}
	if v, ok := netMap["persist"]; ok {
		n.persist = v.(bool)
	}
	if v, ok := netMap["postIPv6"]; ok {
		n.postIPv6 = v.(bool)
	}
	if v, ok := netMap["ipamType"]; ok {
		n.ipamType = v.(string)
	} else {
		n.ipamType = ipamapi.DefaultIPAM
	}
	if v, ok := netMap["addrSpace"]; ok {
		n.addrSpace = v.(string)
	}
	if v, ok := netMap["ipamV4Config"]; ok {
		if err := json.Unmarshal([]byte(v.(string)), &n.ipamV4Config); err != nil {
			return err
		}
	}
	if v, ok := netMap["ipamV4Info"]; ok {
		if err := json.Unmarshal([]byte(v.(string)), &n.ipamV4Info); err != nil {
			return err
		}
	}
	if v, ok := netMap["ipamV6Config"]; ok {
		if err := json.Unmarshal([]byte(v.(string)), &n.ipamV6Config); err != nil {
			return err
		}
	}
	if v, ok := netMap["ipamV6Info"]; ok {
		if err := json.Unmarshal([]byte(v.(string)), &n.ipamV6Info); err != nil {
			return err
		}
	}
	if v, ok := netMap["internal"]; ok {
		n.internal = v.(bool)
	}
	if v, ok := netMap["attachable"]; ok {
		n.attachable = v.(bool)
	}
	if s, ok := netMap["scope"]; ok {
		n.scope = s.(string)
	}
	if v, ok := netMap["inDelete"]; ok {
		n.inDelete = v.(bool)
	}
	if v, ok := netMap["ingress"]; ok {
		n.ingress = v.(bool)
	}
	if v, ok := netMap["configOnly"]; ok {
		n.configOnly = v.(bool)
	}
	if v, ok := netMap["configFrom"]; ok {
		n.configFrom = v.(string)
	}
	// Reconcile old networks with the recently added `--ipv6` flag
	if !n.enableIPv6 {
		n.enableIPv6 = len(n.ipamV6Info) > 0
	}
	return nil
}

// NetworkOption is an option setter function type used to pass various options to
// NewNetwork method. The various setter functions of type NetworkOption are
// provided by libnetwork, they look like NetworkOptionXXXX(...)
type NetworkOption func(n *network)

// NetworkOptionGeneric function returns an option setter for a Generic option defined
// in a Dictionary of Key-Value pair
func NetworkOptionGeneric(generic map[string]interface{}) NetworkOption {
	return func(n *network) {
		if n.generic == nil {
			n.generic = make(map[string]interface{})
		}
		if val, ok := generic[netlabel.EnableIPv6]; ok {
			n.enableIPv6 = val.(bool)
		}
		if val, ok := generic[netlabel.Internal]; ok {
			n.internal = val.(bool)
		}
		for k, v := range generic {
			n.generic[k] = v
		}
	}
}

// NetworkOptionIngress returns an option setter to indicate if a network is
// an ingress network.
func NetworkOptionIngress(ingress bool) NetworkOption {
	return func(n *network) {
		n.ingress = ingress
	}
}

// NetworkOptionPersist returns an option setter to set persistence policy for a network
func NetworkOptionPersist(persist bool) NetworkOption {
	return func(n *network) {
		n.persist = persist
	}
}

// NetworkOptionEnableIPv6 returns an option setter to explicitly configure IPv6
func NetworkOptionEnableIPv6(enableIPv6 bool) NetworkOption {
	return func(n *network) {
		if n.generic == nil {
			n.generic = make(map[string]interface{})
		}
		n.enableIPv6 = enableIPv6
		n.generic[netlabel.EnableIPv6] = enableIPv6
	}
}

// NetworkOptionInternalNetwork returns an option setter to config the network
// to be internal which disables default gateway service
func NetworkOptionInternalNetwork() NetworkOption {
	return func(n *network) {
		if n.generic == nil {
			n.generic = make(map[string]interface{})
		}
		n.internal = true
		n.generic[netlabel.Internal] = true
	}
}

// NetworkOptionAttachable returns an option setter to set attachable for a network
func NetworkOptionAttachable(attachable bool) NetworkOption {
	return func(n *network) {
		n.attachable = attachable
	}
}

// NetworkOptionScope returns an option setter to overwrite the network's scope.
// By default the network's scope is set to the network driver's datascope.
func NetworkOptionScope(scope string) NetworkOption {
	return func(n *network) {
		n.scope = scope
	}
}

// NetworkOptionIpam function returns an option setter for the ipam configuration for this network
func NetworkOptionIpam(ipamDriver string, addrSpace string, ipV4 []*IpamConf, ipV6 []*IpamConf, opts map[string]string) NetworkOption {
	return func(n *network) {
		if ipamDriver != "" {
			n.ipamType = ipamDriver
			if ipamDriver == ipamapi.DefaultIPAM {
				n.ipamType = defaultIpamForNetworkType(n.Type())
			}
		}
		n.ipamOptions = opts
		n.addrSpace = addrSpace
		n.ipamV4Config = ipV4
		n.ipamV6Config = ipV6
	}
}

// NetworkOptionDriverOpts function returns an option setter for any driver parameter described by a map
func NetworkOptionDriverOpts(opts map[string]string) NetworkOption {
	return func(n *network) {
		if n.generic == nil {
			n.generic = make(map[string]interface{})
		}
		if opts == nil {
			opts = make(map[string]string)
		}
		// Store the options
		n.generic[netlabel.GenericData] = opts
	}
}

// NetworkOptionLabels function returns an option setter for labels specific to a network
func NetworkOptionLabels(labels map[string]string) NetworkOption {
	return func(n *network) {
		n.labels = labels
	}
}

// NetworkOptionDynamic function returns an option setter for dynamic option for a network
func NetworkOptionDynamic() NetworkOption {
	return func(n *network) {
		n.dynamic = true
	}
}

// NetworkOptionDeferIPv6Alloc instructs the network to defer the IPV6 address allocation until after the endpoint has been created
// It is being provided to support the specific docker daemon flags where user can deterministically assign an IPv6 address
// to a container as combination of fixed-cidr-v6 + mac-address
// TODO: Remove this option setter once we support endpoint ipam options
func NetworkOptionDeferIPv6Alloc(enable bool) NetworkOption {
	return func(n *network) {
		n.postIPv6 = enable
	}
}

// NetworkOptionConfigOnly tells controller this network is
// a configuration only network. It serves as a configuration
// for other networks.
func NetworkOptionConfigOnly() NetworkOption {
	return func(n *network) {
		n.configOnly = true
	}
}

// NetworkOptionConfigFrom tells controller to pick the
// network configuration from a configuration only network
func NetworkOptionConfigFrom(name string) NetworkOption {
	return func(n *network) {
		n.configFrom = name
	}
}

func (n *network) processOptions(options ...NetworkOption) {
	for _, opt := range options {
		if opt != nil {
			opt(n)
		}
	}
}

func (n *network) resolveDriver(name string, load bool) (driverapi.Driver, *driverapi.Capability, error) {
	c := n.getController()

	// Check if a driver for the specified network type is available
	d, cap := c.drvRegistry.Driver(name)
	if d == nil {
		if load {
			err := c.loadDriver(name)
			if err != nil {
				return nil, nil, err
			}

			d, cap = c.drvRegistry.Driver(name)
			if d == nil {
				return nil, nil, fmt.Errorf("could not resolve driver %s in registry", name)
			}
		} else {
			// don't fail if driver loading is not required
			return nil, nil, nil
		}
	}

	return d, cap, nil
}

func (n *network) driverScope() string {
	_, cap, err := n.resolveDriver(n.networkType, true)
	if err != nil {
		// If driver could not be resolved simply return an empty string
		return ""
	}

	return cap.DataScope
}

func (n *network) driverIsMultihost() bool {
	_, cap, err := n.resolveDriver(n.networkType, true)
	if err != nil {
		return false
	}
	return cap.ConnectivityScope == datastore.GlobalScope
}

func (n *network) driver(load bool) (driverapi.Driver, error) {
	d, cap, err := n.resolveDriver(n.networkType, load)
	if err != nil {
		return nil, err
	}

	n.Lock()
	// If load is not required, driver, cap and err may all be nil
	if n.scope == "" && cap != nil {
		n.scope = cap.DataScope
	}
	if n.dynamic {
		// If the network is dynamic, then it is swarm
		// scoped regardless of the backing driver.
		n.scope = datastore.SwarmScope
	}
	n.Unlock()
	return d, nil
}

func (n *network) Delete() error {
	return n.delete(false)
}

func (n *network) delete(force bool) error {
	n.Lock()
	c := n.ctrlr
	name := n.name
	id := n.id
	n.Unlock()

	c.networkLocker.Lock(id)
	defer c.networkLocker.Unlock(id)

	n, err := c.getNetworkFromStore(id)
	if err != nil {
		return &UnknownNetworkError{name: name, id: id}
	}

	if !force && n.getEpCnt().EndpointCnt() != 0 {
		if n.configOnly {
			return types.ForbiddenErrorf("configuration network %q is in use", n.Name())
		}
		return &ActiveEndpointsError{name: n.name, id: n.id}
	}

	// Mark the network for deletion
	n.inDelete = true
	if err = c.updateToStore(n); err != nil {
		return fmt.Errorf("error marking network %s (%s) for deletion: %v", n.Name(), n.ID(), err)
	}

	if n.ConfigFrom() != "" {
		if t, err := c.getConfigNetwork(n.ConfigFrom()); err == nil {
			if err := t.getEpCnt().DecEndpointCnt(); err != nil {
				logrus.Warnf("Failed to update reference count for configuration network %q on removal of network %q: %v",
					t.Name(), n.Name(), err)
			}
		} else {
			logrus.Warnf("Could not find configuration network %q during removal of network %q", n.configOnly, n.Name())
		}
	}

	if n.configOnly {
		goto removeFromStore
	}

	if err = n.deleteNetwork(); err != nil {
		if !force {
			return err
		}
		logrus.Debugf("driver failed to delete stale network %s (%s): %v", n.Name(), n.ID(), err)
	}

	n.ipamRelease()
	if err = c.updateToStore(n); err != nil {
		logrus.Warnf("Failed to update store after ipam release for network %s (%s): %v", n.Name(), n.ID(), err)
	}

	// We are about to delete the network. Leave the gossip
	// cluster for the network to stop all incoming network
	// specific gossip updates before cleaning up all the service
	// bindings for the network. But cleanup service binding
	// before deleting the network from the store since service
	// bindings cleanup requires the network in the store.
	n.cancelDriverWatches()
	if err = n.leaveCluster(); err != nil {
		logrus.Errorf("Failed leaving network %s from the agent cluster: %v", n.Name(), err)
	}

	c.cleanupServiceBindings(n.ID())

removeFromStore:
	// deleteFromStore performs an atomic delete operation and the
	// network.epCnt will help prevent any possible
	// race between endpoint join and network delete
	if err = c.deleteFromStore(n.getEpCnt()); err != nil {
		if !force {
			return fmt.Errorf("error deleting network endpoint count from store: %v", err)
		}
		logrus.Debugf("Error deleting endpoint count from store for stale network %s (%s) for deletion: %v", n.Name(), n.ID(), err)
	}

	if err = c.deleteFromStore(n); err != nil {
		return fmt.Errorf("error deleting network from store: %v", err)
	}

	return nil
}

func (n *network) deleteNetwork() error {
	d, err := n.driver(true)
	if err != nil {
		return fmt.Errorf("failed deleting network: %v", err)
	}

	if err := d.DeleteNetwork(n.ID()); err != nil {
		// Forbidden Errors should be honored
		if _, ok := err.(types.ForbiddenError); ok {
			return err
		}

		if _, ok := err.(types.MaskableError); !ok {
			logrus.Warnf("driver error deleting network %s : %v", n.name, err)
		}
	}

	for _, resolver := range n.resolver {
		resolver.Stop()
	}
	return nil
}

func (n *network) addEndpoint(ep *endpoint) error {
	d, err := n.driver(true)
	if err != nil {
		return fmt.Errorf("failed to add endpoint: %v", err)
	}

	err = d.CreateEndpoint(n.id, ep.id, ep.Interface(), ep.generic)
	if err != nil {
		return types.InternalErrorf("failed to create endpoint %s on network %s: %v",
			ep.Name(), n.Name(), err)
	}

	return nil
}

func (n *network) CreateEndpoint(name string, options ...EndpointOption) (Endpoint, error) {
	var err error
	if !config.IsValidName(name) {
		return nil, ErrInvalidName(name)
	}

	if n.ConfigOnly() {
		return nil, types.ForbiddenErrorf("cannot create endpoint on configuration-only network")
	}

	if _, err = n.EndpointByName(name); err == nil {
		return nil, types.ForbiddenErrorf("endpoint with name %s already exists in network %s", name, n.Name())
	}

	ep := &endpoint{name: name, generic: make(map[string]interface{}), iface: &endpointInterface{}}
	ep.id = stringid.GenerateRandomID()

	n.ctrlr.networkLocker.Lock(n.id)
	defer n.ctrlr.networkLocker.Unlock(n.id)

	// Initialize ep.network with a possibly stale copy of n. We need this to get network from
	// store. But once we get it from store we will have the most uptodate copy possibly.
	ep.network = n
	ep.locator = n.getController().clusterHostID()
	ep.network, err = ep.getNetworkFromStore()
	if err != nil {
		return nil, fmt.Errorf("failed to get network during CreateEndpoint: %v", err)
	}
	n = ep.network

	ep.processOptions(options...)

	for _, llIPNet := range ep.Iface().LinkLocalAddresses() {
		if !llIPNet.IP.IsLinkLocalUnicast() {
			return nil, types.BadRequestErrorf("invalid link local IP address: %v", llIPNet.IP)
		}
	}

	if opt, ok := ep.generic[netlabel.MacAddress]; ok {
		if mac, ok := opt.(net.HardwareAddr); ok {
			ep.iface.mac = mac
		}
	}

	ipam, cap, err := n.getController().getIPAMDriver(n.ipamType)
	if err != nil {
		return nil, err
	}

	if cap.RequiresMACAddress {
		if ep.iface.mac == nil {
			ep.iface.mac = netutils.GenerateRandomMAC()
		}
		if ep.ipamOptions == nil {
			ep.ipamOptions = make(map[string]string)
		}
		ep.ipamOptions[netlabel.MacAddress] = ep.iface.mac.String()
	}

	if err = ep.assignAddress(ipam, true, n.enableIPv6 && !n.postIPv6); err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			ep.releaseAddress()
		}
	}()

	if err = n.addEndpoint(ep); err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			if e := ep.deleteEndpoint(false); e != nil {
				logrus.Warnf("cleaning up endpoint failed %s : %v", name, e)
			}
		}
	}()

	if err = ep.assignAddress(ipam, false, n.enableIPv6 && n.postIPv6); err != nil {
		return nil, err
	}

	if err = n.getController().updateToStore(ep); err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			if e := n.getController().deleteFromStore(ep); e != nil {
				logrus.Warnf("error rolling back endpoint %s from store: %v", name, e)
			}
		}
	}()

	// Watch for service records
	n.getController().watchSvcRecord(ep)
	defer func() {
		if err != nil {
			n.getController().unWatchSvcRecord(ep)
		}
	}()

	// Increment endpoint count to indicate completion of endpoint addition
	if err = n.getEpCnt().IncEndpointCnt(); err != nil {
		return nil, err
	}

	return ep, nil
}

func (n *network) Endpoints() []Endpoint {
	var list []Endpoint

	endpoints, err := n.getEndpointsFromStore()
	if err != nil {
		logrus.Error(err)
	}

	for _, ep := range endpoints {
		list = append(list, ep)
	}

	return list
}

func (n *network) WalkEndpoints(walker EndpointWalker) {
	for _, e := range n.Endpoints() {
		if walker(e) {
			return
		}
	}
}

func (n *network) EndpointByName(name string) (Endpoint, error) {
	if name == "" {
		return nil, ErrInvalidName(name)
	}
	var e Endpoint

	s := func(current Endpoint) bool {
		if current.Name() == name {
			e = current
			return true
		}
		return false
	}

	n.WalkEndpoints(s)

	if e == nil {
		return nil, ErrNoSuchEndpoint(name)
	}

	return e, nil
}

func (n *network) EndpointByID(id string) (Endpoint, error) {
	if id == "" {
		return nil, ErrInvalidID(id)
	}

	ep, err := n.getEndpointFromStore(id)
	if err != nil {
		return nil, ErrNoSuchEndpoint(id)
	}

	return ep, nil
}

func (n *network) updateSvcRecord(ep *endpoint, localEps []*endpoint, isAdd bool) {
	var ipv6 net.IP
	epName := ep.Name()
	if iface := ep.Iface(); iface.Address() != nil {
		myAliases := ep.MyAliases()
		if iface.AddressIPv6() != nil {
			ipv6 = iface.AddressIPv6().IP
		}

		serviceID := ep.svcID
		if serviceID == "" {
			serviceID = ep.ID()
		}
		if isAdd {
			// If anonymous endpoint has an alias use the first alias
			// for ip->name mapping. Not having the reverse mapping
			// breaks some apps
			if ep.isAnonymous() {
				if len(myAliases) > 0 {
					n.addSvcRecords(ep.ID(), myAliases[0], serviceID, iface.Address().IP, ipv6, true, "updateSvcRecord")
				}
			} else {
				n.addSvcRecords(ep.ID(), epName, serviceID, iface.Address().IP, ipv6, true, "updateSvcRecord")
			}
			for _, alias := range myAliases {
				n.addSvcRecords(ep.ID(), alias, serviceID, iface.Address().IP, ipv6, false, "updateSvcRecord")
			}
		} else {
			if ep.isAnonymous() {
				if len(myAliases) > 0 {
					n.deleteSvcRecords(ep.ID(), myAliases[0], serviceID, iface.Address().IP, ipv6, true, "updateSvcRecord")
				}
			} else {
				n.deleteSvcRecords(ep.ID(), epName, serviceID, iface.Address().IP, ipv6, true, "updateSvcRecord")
			}
			for _, alias := range myAliases {
				n.deleteSvcRecords(ep.ID(), alias, serviceID, iface.Address().IP, ipv6, false, "updateSvcRecord")
			}
		}
	}
}

func addIPToName(ipMap common.SetMatrix, name, serviceID string, ip net.IP) {
	reverseIP := netutils.ReverseIP(ip.String())
	ipMap.Insert(reverseIP, ipInfo{
		name:      name,
		serviceID: serviceID,
	})
}

func delIPToName(ipMap common.SetMatrix, name, serviceID string, ip net.IP) {
	reverseIP := netutils.ReverseIP(ip.String())
	ipMap.Remove(reverseIP, ipInfo{
		name:      name,
		serviceID: serviceID,
	})
}

func addNameToIP(svcMap common.SetMatrix, name, serviceID string, epIP net.IP) {
	svcMap.Insert(name, svcMapEntry{
		ip:        epIP.String(),
		serviceID: serviceID,
	})
}

func delNameToIP(svcMap common.SetMatrix, name, serviceID string, epIP net.IP) {
	svcMap.Remove(name, svcMapEntry{
		ip:        epIP.String(),
		serviceID: serviceID,
	})
}

func (n *network) addSvcRecords(eID, name, serviceID string, epIP, epIPv6 net.IP, ipMapUpdate bool, method string) {
	// Do not add service names for ingress network as this is a
	// routing only network
	if n.ingress {
		return
	}

	logrus.Debugf("%s (%s).addSvcRecords(%s, %s, %s, %t) %s sid:%s", eID, n.ID()[0:7], name, epIP, epIPv6, ipMapUpdate, method, serviceID)

	c := n.getController()
	c.Lock()
	defer c.Unlock()

	sr, ok := c.svcRecords[n.ID()]
	if !ok {
		sr = svcInfo{
			svcMap:     common.NewSetMatrix(),
			svcIPv6Map: common.NewSetMatrix(),
			ipMap:      common.NewSetMatrix(),
		}
		c.svcRecords[n.ID()] = sr
	}

	if ipMapUpdate {
		addIPToName(sr.ipMap, name, serviceID, epIP)
		if epIPv6 != nil {
			addIPToName(sr.ipMap, name, serviceID, epIPv6)
		}
	}

	addNameToIP(sr.svcMap, name, serviceID, epIP)
	if epIPv6 != nil {
		addNameToIP(sr.svcIPv6Map, name, serviceID, epIPv6)
	}
}

func (n *network) deleteSvcRecords(eID, name, serviceID string, epIP net.IP, epIPv6 net.IP, ipMapUpdate bool, method string) {
	// Do not delete service names from ingress network as this is a
	// routing only network
	if n.ingress {
		return
	}

	logrus.Debugf("%s (%s).deleteSvcRecords(%s, %s, %s, %t) %s sid:%s ", eID, n.ID()[0:7], name, epIP, epIPv6, ipMapUpdate, method, serviceID)

	c := n.getController()
	c.Lock()
	defer c.Unlock()

	sr, ok := c.svcRecords[n.ID()]
	if !ok {
		return
	}

	if ipMapUpdate {
		delIPToName(sr.ipMap, name, serviceID, epIP)

		if epIPv6 != nil {
			delIPToName(sr.ipMap, name, serviceID, epIPv6)
		}
	}

	delNameToIP(sr.svcMap, name, serviceID, epIP)

	if epIPv6 != nil {
		delNameToIP(sr.svcIPv6Map, name, serviceID, epIPv6)
	}
}

func (n *network) getSvcRecords(ep *endpoint) []etchosts.Record {
	n.Lock()
	defer n.Unlock()

	if ep == nil {
		return nil
	}

	var recs []etchosts.Record

	epName := ep.Name()

	n.ctrlr.Lock()
	defer n.ctrlr.Unlock()
	sr, ok := n.ctrlr.svcRecords[n.id]
	if !ok || sr.svcMap == nil {
		return nil
	}

	svcMapKeys := sr.svcMap.Keys()
	// Loop on service names on this network
	for _, k := range svcMapKeys {
		if strings.Split(k, ".")[0] == epName {
			continue
		}
		// Get all the IPs associated to this service
		mapEntryList, ok := sr.svcMap.Get(k)
		if !ok {
			// The key got deleted
			continue
		}
		if len(mapEntryList) == 0 {
			logrus.Warnf("Found empty list of IP addresses for service %s on network %s (%s)", k, n.name, n.id)
			continue
		}

		recs = append(recs, etchosts.Record{
			Hosts: k,
			IP:    mapEntryList[0].(svcMapEntry).ip,
		})
	}

	return recs
}

func (n *network) getController() *controller {
	n.Lock()
	defer n.Unlock()
	return n.ctrlr
}

func (n *network) ipamAllocate() error {
	if n.hasSpecialDriver() {
		return nil
	}

	ipam, _, err := n.getController().getIPAMDriver(n.ipamType)
	if err != nil {
		return err
	}

	if n.addrSpace == "" {
		if n.addrSpace, err = n.deriveAddressSpace(); err != nil {
			return err
		}
	}

	err = n.ipamAllocateVersion(4, ipam)
	if err != nil {
		return err
	}

	defer func() {
		if err != nil {
			n.ipamReleaseVersion(4, ipam)
		}
	}()

	if !n.enableIPv6 {
		return nil
	}

	err = n.ipamAllocateVersion(6, ipam)
	return err
}

func (n *network) requestPoolHelper(ipam ipamapi.Ipam, addressSpace, preferredPool, subPool string, options map[string]string, v6 bool) (string, *net.IPNet, map[string]string, error) {
	for {
		poolID, pool, meta, err := ipam.RequestPool(addressSpace, preferredPool, subPool, options, v6)
		if err != nil {
			return "", nil, nil, err
		}

		// If the network belongs to global scope or the pool was
		// explicitly chosen or it is invalid, do not perform the overlap check.
		if n.Scope() == datastore.GlobalScope || preferredPool != "" || !types.IsIPNetValid(pool) {
			return poolID, pool, meta, nil
		}

		// Check for overlap and if none found, we have found the right pool.
		if _, err := netutils.FindAvailableNetwork([]*net.IPNet{pool}); err == nil {
			return poolID, pool, meta, nil
		}

		// Pool obtained in this iteration is
		// overlapping. Hold onto the pool and don't release
		// it yet, because we don't want ipam to give us back
		// the same pool over again. But make sure we still do
		// a deferred release when we have either obtained a
		// non-overlapping pool or ran out of pre-defined
		// pools.
		defer func() {
			if err := ipam.ReleasePool(poolID); err != nil {
				logrus.Warnf("Failed to release overlapping pool %s while returning from pool request helper for network %s", pool, n.Name())
			}
		}()

		// If this is a preferred pool request and the network
		// is local scope and there is an overlap, we fail the
		// network creation right here. The pool will be
		// released in the defer.
		if preferredPool != "" {
			return "", nil, nil, fmt.Errorf("requested subnet %s overlaps in the host", preferredPool)
		}
	}
}

func (n *network) ipamAllocateVersion(ipVer int, ipam ipamapi.Ipam) error {
	var (
		cfgList  *[]*IpamConf
		infoList *[]*IpamInfo
		err      error
	)

	switch ipVer {
	case 4:
		cfgList = &n.ipamV4Config
		infoList = &n.ipamV4Info
	case 6:
		cfgList = &n.ipamV6Config
		infoList = &n.ipamV6Info
	default:
		return types.InternalErrorf("incorrect ip version passed to ipam allocate: %d", ipVer)
	}

	if len(*cfgList) == 0 {
		*cfgList = []*IpamConf{{}}
	}

	*infoList = make([]*IpamInfo, len(*cfgList))

	logrus.Debugf("Allocating IPv%d pools for network %s (%s)", ipVer, n.Name(), n.ID())

	for i, cfg := range *cfgList {
		if err = cfg.Validate(); err != nil {
			return err
		}
		d := &IpamInfo{}
		(*infoList)[i] = d

		d.AddressSpace = n.addrSpace
		d.PoolID, d.Pool, d.Meta, err = n.requestPoolHelper(ipam, n.addrSpace, cfg.PreferredPool, cfg.SubPool, n.ipamOptions, ipVer == 6)
		if err != nil {
			return err
		}

		defer func() {
			if err != nil {
				if err := ipam.ReleasePool(d.PoolID); err != nil {
					logrus.Warnf("Failed to release address pool %s after failure to create network %s (%s)", d.PoolID, n.Name(), n.ID())
				}
			}
		}()

		if gws, ok := d.Meta[netlabel.Gateway]; ok {
			if d.Gateway, err = types.ParseCIDR(gws); err != nil {
				return types.BadRequestErrorf("failed to parse gateway address (%v) returned by ipam driver: %v", gws, err)
			}
		}

		// If user requested a specific gateway, libnetwork will allocate it
		// irrespective of whether ipam driver returned a gateway already.
		// If none of the above is true, libnetwork will allocate one.
		if cfg.Gateway != "" || d.Gateway == nil {
			var gatewayOpts = map[string]string{
				ipamapi.RequestAddressType: netlabel.Gateway,
			}
			if d.Gateway, _, err = ipam.RequestAddress(d.PoolID, net.ParseIP(cfg.Gateway), gatewayOpts); err != nil {
				return types.InternalErrorf("failed to allocate gateway (%v): %v", cfg.Gateway, err)
			}
		}

		// Auxiliary addresses must be part of the master address pool
		// If they fall into the container addressable pool, libnetwork will reserve them
		if cfg.AuxAddresses != nil {
			var ip net.IP
			d.IPAMData.AuxAddresses = make(map[string]*net.IPNet, len(cfg.AuxAddresses))
			for k, v := range cfg.AuxAddresses {
				if ip = net.ParseIP(v); ip == nil {
					return types.BadRequestErrorf("non parsable secondary ip address (%s:%s) passed for network %s", k, v, n.Name())
				}
				if !d.Pool.Contains(ip) {
					return types.ForbiddenErrorf("auxilairy address: (%s:%s) must belong to the master pool: %s", k, v, d.Pool)
				}
				// Attempt reservation in the container addressable pool, silent the error if address does not belong to that pool
				if d.IPAMData.AuxAddresses[k], _, err = ipam.RequestAddress(d.PoolID, ip, nil); err != nil && err != ipamapi.ErrIPOutOfRange {
					return types.InternalErrorf("failed to allocate secondary ip address (%s:%s): %v", k, v, err)
				}
			}
		}
	}

	return nil
}

func (n *network) ipamRelease() {
	if n.hasSpecialDriver() {
		return
	}
	ipam, _, err := n.getController().getIPAMDriver(n.ipamType)
	if err != nil {
		logrus.Warnf("Failed to retrieve ipam driver to release address pool(s) on delete of network %s (%s): %v", n.Name(), n.ID(), err)
		return
	}
	n.ipamReleaseVersion(4, ipam)
	n.ipamReleaseVersion(6, ipam)
}

func (n *network) ipamReleaseVersion(ipVer int, ipam ipamapi.Ipam) {
	var infoList *[]*IpamInfo

	switch ipVer {
	case 4:
		infoList = &n.ipamV4Info
	case 6:
		infoList = &n.ipamV6Info
	default:
		logrus.Warnf("incorrect ip version passed to ipam release: %d", ipVer)
		return
	}

	if len(*infoList) == 0 {
		return
	}

	logrus.Debugf("releasing IPv%d pools from network %s (%s)", ipVer, n.Name(), n.ID())

	for _, d := range *infoList {
		if d.Gateway != nil {
			if err := ipam.ReleaseAddress(d.PoolID, d.Gateway.IP); err != nil {
				logrus.Warnf("Failed to release gateway ip address %s on delete of network %s (%s): %v", d.Gateway.IP, n.Name(), n.ID(), err)
			}
		}
		if d.IPAMData.AuxAddresses != nil {
			for k, nw := range d.IPAMData.AuxAddresses {
				if d.Pool.Contains(nw.IP) {
					if err := ipam.ReleaseAddress(d.PoolID, nw.IP); err != nil && err != ipamapi.ErrIPOutOfRange {
						logrus.Warnf("Failed to release secondary ip address %s (%v) on delete of network %s (%s): %v", k, nw.IP, n.Name(), n.ID(), err)
					}
				}
			}
		}
		if err := ipam.ReleasePool(d.PoolID); err != nil {
			logrus.Warnf("Failed to release address pool %s on delete of network %s (%s): %v", d.PoolID, n.Name(), n.ID(), err)
		}
	}

	*infoList = nil
}

func (n *network) getIPInfo(ipVer int) []*IpamInfo {
	var info []*IpamInfo
	switch ipVer {
	case 4:
		info = n.ipamV4Info
	case 6:
		info = n.ipamV6Info
	default:
		return nil
	}
	l := make([]*IpamInfo, 0, len(info))
	n.Lock()
	l = append(l, info...)
	n.Unlock()
	return l
}

func (n *network) getIPData(ipVer int) []driverapi.IPAMData {
	var info []*IpamInfo
	switch ipVer {
	case 4:
		info = n.ipamV4Info
	case 6:
		info = n.ipamV6Info
	default:
		return nil
	}
	l := make([]driverapi.IPAMData, 0, len(info))
	n.Lock()
	for _, d := range info {
		l = append(l, d.IPAMData)
	}
	n.Unlock()
	return l
}

func (n *network) deriveAddressSpace() (string, error) {
	local, global, err := n.getController().drvRegistry.IPAMDefaultAddressSpaces(n.ipamType)
	if err != nil {
		return "", types.NotFoundErrorf("failed to get default address space: %v", err)
	}
	if n.DataScope() == datastore.GlobalScope {
		return global, nil
	}
	return local, nil
}

func (n *network) Info() NetworkInfo {
	return n
}

func (n *network) Peers() []networkdb.PeerInfo {
	if !n.Dynamic() {
		return []networkdb.PeerInfo{}
	}

	agent := n.getController().getAgent()
	if agent == nil {
		return []networkdb.PeerInfo{}
	}

	return agent.networkDB.Peers(n.ID())
}

func (n *network) DriverOptions() map[string]string {
	n.Lock()
	defer n.Unlock()
	if n.generic != nil {
		if m, ok := n.generic[netlabel.GenericData]; ok {
			return m.(map[string]string)
		}
	}
	return map[string]string{}
}

func (n *network) Scope() string {
	n.Lock()
	defer n.Unlock()
	return n.scope
}

func (n *network) IpamConfig() (string, map[string]string, []*IpamConf, []*IpamConf) {
	n.Lock()
	defer n.Unlock()

	v4L := make([]*IpamConf, len(n.ipamV4Config))
	v6L := make([]*IpamConf, len(n.ipamV6Config))

	for i, c := range n.ipamV4Config {
		cc := &IpamConf{}
		c.CopyTo(cc)
		v4L[i] = cc
	}

	for i, c := range n.ipamV6Config {
		cc := &IpamConf{}
		c.CopyTo(cc)
		v6L[i] = cc
	}

	return n.ipamType, n.ipamOptions, v4L, v6L
}

func (n *network) IpamInfo() ([]*IpamInfo, []*IpamInfo) {
	n.Lock()
	defer n.Unlock()

	v4Info := make([]*IpamInfo, len(n.ipamV4Info))
	v6Info := make([]*IpamInfo, len(n.ipamV6Info))

	for i, info := range n.ipamV4Info {
		ic := &IpamInfo{}
		info.CopyTo(ic)
		v4Info[i] = ic
	}

	for i, info := range n.ipamV6Info {
		ic := &IpamInfo{}
		info.CopyTo(ic)
		v6Info[i] = ic
	}

	return v4Info, v6Info
}

func (n *network) Internal() bool {
	n.Lock()
	defer n.Unlock()

	return n.internal
}

func (n *network) Attachable() bool {
	n.Lock()
	defer n.Unlock()

	return n.attachable
}

func (n *network) Ingress() bool {
	n.Lock()
	defer n.Unlock()

	return n.ingress
}

func (n *network) Dynamic() bool {
	n.Lock()
	defer n.Unlock()

	return n.dynamic
}

func (n *network) IPv6Enabled() bool {
	n.Lock()
	defer n.Unlock()

	return n.enableIPv6
}

func (n *network) ConfigFrom() string {
	n.Lock()
	defer n.Unlock()

	return n.configFrom
}

func (n *network) ConfigOnly() bool {
	n.Lock()
	defer n.Unlock()

	return n.configOnly
}

func (n *network) Labels() map[string]string {
	n.Lock()
	defer n.Unlock()

	var lbls = make(map[string]string, len(n.labels))
	for k, v := range n.labels {
		lbls[k] = v
	}

	return lbls
}

func (n *network) TableEventRegister(tableName string, objType driverapi.ObjectType) error {
	if !driverapi.IsValidType(objType) {
		return fmt.Errorf("invalid object type %v in registering table, %s", objType, tableName)
	}

	t := networkDBTable{
		name:    tableName,
		objType: objType,
	}
	n.Lock()
	defer n.Unlock()
	n.driverTables = append(n.driverTables, t)
	return nil
}

// Special drivers are ones which do not need to perform any network plumbing
func (n *network) hasSpecialDriver() bool {
	return n.Type() == "host" || n.Type() == "null"
}

func (n *network) ResolveName(req string, ipType int) ([]net.IP, bool) {
	var ipv6Miss bool

	c := n.getController()
	c.Lock()
	defer c.Unlock()
	sr, ok := c.svcRecords[n.ID()]

	if !ok {
		return nil, false
	}

	req = strings.TrimSuffix(req, ".")
	ipSet, ok := sr.svcMap.Get(req)

	if ipType == types.IPv6 {
		// If the name resolved to v4 address then its a valid name in
		// the docker network domain. If the network is not v6 enabled
		// set ipv6Miss to filter the DNS query from going to external
		// resolvers.
		if ok && !n.enableIPv6 {
			ipv6Miss = true
		}
		ipSet, ok = sr.svcIPv6Map.Get(req)
	}

	if ok && len(ipSet) > 0 {
		// this map is to avoid IP duplicates, this can happen during a transition period where 2 services are using the same IP
		noDup := make(map[string]bool)
		var ipLocal []net.IP
		for _, ip := range ipSet {
			if _, dup := noDup[ip.(svcMapEntry).ip]; !dup {
				noDup[ip.(svcMapEntry).ip] = true
				ipLocal = append(ipLocal, net.ParseIP(ip.(svcMapEntry).ip))
			}
		}
		return ipLocal, ok
	}

	return nil, ipv6Miss
}

func (n *network) HandleQueryResp(name string, ip net.IP) {
	c := n.getController()
	c.Lock()
	defer c.Unlock()
	sr, ok := c.svcRecords[n.ID()]

	if !ok {
		return
	}

	ipStr := netutils.ReverseIP(ip.String())
	// If an object with extResolver == true is already in the set this call will fail
	// but anyway it means that has already been inserted before
	if ok, _ := sr.ipMap.Contains(ipStr, ipInfo{name: name}); ok {
		sr.ipMap.Remove(ipStr, ipInfo{name: name})
		sr.ipMap.Insert(ipStr, ipInfo{name: name, extResolver: true})
	}
}

func (n *network) ResolveIP(ip string) string {
	c := n.getController()
	c.Lock()
	defer c.Unlock()
	sr, ok := c.svcRecords[n.ID()]

	if !ok {
		return ""
	}

	nwName := n.Name()

	elemSet, ok := sr.ipMap.Get(ip)
	if !ok || len(elemSet) == 0 {
		return ""
	}
	// NOTE it is possible to have more than one element in the Set, this will happen
	// because of interleave of different events from different sources (local container create vs
	// network db notifications)
	// In such cases the resolution will be based on the first element of the set, and can vary
	// during the system stabilitation
	elem, ok := elemSet[0].(ipInfo)
	if !ok {
		setStr, b := sr.ipMap.String(ip)
		logrus.Errorf("expected set of ipInfo type for key %s set:%t %s", ip, b, setStr)
		return ""
	}

	if elem.extResolver {
		return ""
	}

	return elem.name + "." + nwName
}

func (n *network) ResolveService(name string) ([]*net.SRV, []net.IP) {
	c := n.getController()

	srv := []*net.SRV{}
	ip := []net.IP{}

	logrus.Debugf("Service name To resolve: %v", name)

	// There are DNS implementaions that allow SRV queries for names not in
	// the format defined by RFC 2782. Hence specific validations checks are
	// not done
	parts := strings.Split(name, ".")
	if len(parts) < 3 {
		return nil, nil
	}

	portName := parts[0]
	proto := parts[1]
	svcName := strings.Join(parts[2:], ".")

	c.Lock()
	defer c.Unlock()
	sr, ok := c.svcRecords[n.ID()]

	if !ok {
		return nil, nil
	}

	svcs, ok := sr.service[svcName]
	if !ok {
		return nil, nil
	}

	for _, svc := range svcs {
		if svc.portName != portName {
			continue
		}
		if svc.proto != proto {
			continue
		}
		for _, t := range svc.target {
			srv = append(srv,
				&net.SRV{
					Target: t.name,
					Port:   t.port,
				})

			ip = append(ip, t.ip)
		}
	}

	return srv, ip
}

func (n *network) ExecFunc(f func()) error {
	return types.NotImplementedErrorf("ExecFunc not supported by network")
}

func (n *network) NdotsSet() bool {
	return false
}

// config-only network is looked up by name
func (c *controller) getConfigNetwork(name string) (*network, error) {
	var n Network

	s := func(current Network) bool {
		if current.Info().ConfigOnly() && current.Name() == name {
			n = current
			return true
		}
		return false
	}

	c.WalkNetworks(s)

	if n == nil {
		return nil, types.NotFoundErrorf("configuration network %q not found", name)
	}

	return n.(*network), nil
}
