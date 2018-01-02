package bridge

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/iptables"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/netutils"
	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/osl"
	"github.com/docker/libnetwork/portmapper"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
)

const (
	networkType                = "bridge"
	vethPrefix                 = "veth"
	vethLen                    = 7
	defaultContainerVethPrefix = "eth"
	maxAllocatePortAttempts    = 10
)

const (
	// DefaultGatewayV4AuxKey represents the default-gateway configured by the user
	DefaultGatewayV4AuxKey = "DefaultGatewayIPv4"
	// DefaultGatewayV6AuxKey represents the ipv6 default-gateway configured by the user
	DefaultGatewayV6AuxKey = "DefaultGatewayIPv6"
)

type iptableCleanFunc func() error
type iptablesCleanFuncs []iptableCleanFunc

// configuration info for the "bridge" driver.
type configuration struct {
	EnableIPForwarding  bool
	EnableIPTables      bool
	EnableUserlandProxy bool
	UserlandProxyPath   string
}

// networkConfiguration for network specific configuration
type networkConfiguration struct {
	ID                   string
	BridgeName           string
	EnableIPv6           bool
	EnableIPMasquerade   bool
	EnableICC            bool
	Mtu                  int
	DefaultBindingIP     net.IP
	DefaultBridge        bool
	ContainerIfacePrefix string
	// Internal fields set after ipam data parsing
	AddressIPv4        *net.IPNet
	AddressIPv6        *net.IPNet
	DefaultGatewayIPv4 net.IP
	DefaultGatewayIPv6 net.IP
	dbIndex            uint64
	dbExists           bool
	Internal           bool

	BridgeIfaceCreator ifaceCreator
}

// ifaceCreator represents how the bridge interface was created
type ifaceCreator int8

const (
	ifaceCreatorUnknown ifaceCreator = iota
	ifaceCreatedByLibnetwork
	ifaceCreatedByUser
)

// endpointConfiguration represents the user specified configuration for the sandbox endpoint
type endpointConfiguration struct {
	MacAddress net.HardwareAddr
}

// containerConfiguration represents the user specified configuration for a container
type containerConfiguration struct {
	ParentEndpoints []string
	ChildEndpoints  []string
}

// cnnectivityConfiguration represents the user specified configuration regarding the external connectivity
type connectivityConfiguration struct {
	PortBindings []types.PortBinding
	ExposedPorts []types.TransportPort
}

type bridgeEndpoint struct {
	id              string
	nid             string
	srcName         string
	addr            *net.IPNet
	addrv6          *net.IPNet
	macAddress      net.HardwareAddr
	config          *endpointConfiguration // User specified parameters
	containerConfig *containerConfiguration
	extConnConfig   *connectivityConfiguration
	portMapping     []types.PortBinding // Operation port bindings
	dbIndex         uint64
	dbExists        bool
}

type bridgeNetwork struct {
	id            string
	bridge        *bridgeInterface // The bridge's L3 interface
	config        *networkConfiguration
	endpoints     map[string]*bridgeEndpoint // key: endpoint id
	portMapper    *portmapper.PortMapper
	driver        *driver // The network's driver
	iptCleanFuncs iptablesCleanFuncs
	sync.Mutex
}

type driver struct {
	config         *configuration
	network        *bridgeNetwork
	natChain       *iptables.ChainInfo
	filterChain    *iptables.ChainInfo
	isolationChain *iptables.ChainInfo
	networks       map[string]*bridgeNetwork
	store          datastore.DataStore
	nlh            *netlink.Handle
	sync.Mutex
}

// New constructs a new bridge driver
func newDriver() *driver {
	return &driver{networks: map[string]*bridgeNetwork{}, config: &configuration{}}
}

// Init registers a new instance of bridge driver
func Init(dc driverapi.DriverCallback, config map[string]interface{}) error {
	d := newDriver()
	if err := d.configure(config); err != nil {
		return err
	}

	c := driverapi.Capability{
		DataScope:         datastore.LocalScope,
		ConnectivityScope: datastore.LocalScope,
	}
	return dc.RegisterDriver(networkType, d, c)
}

// Validate performs a static validation on the network configuration parameters.
// Whatever can be assessed a priori before attempting any programming.
func (c *networkConfiguration) Validate() error {
	if c.Mtu < 0 {
		return ErrInvalidMtu(c.Mtu)
	}

	// If bridge v4 subnet is specified
	if c.AddressIPv4 != nil {
		// If default gw is specified, it must be part of bridge subnet
		if c.DefaultGatewayIPv4 != nil {
			if !c.AddressIPv4.Contains(c.DefaultGatewayIPv4) {
				return &ErrInvalidGateway{}
			}
		}
	}

	// If default v6 gw is specified, AddressIPv6 must be specified and gw must belong to AddressIPv6 subnet
	if c.EnableIPv6 && c.DefaultGatewayIPv6 != nil {
		if c.AddressIPv6 == nil || !c.AddressIPv6.Contains(c.DefaultGatewayIPv6) {
			return &ErrInvalidGateway{}
		}
	}
	return nil
}

// Conflicts check if two NetworkConfiguration objects overlap
func (c *networkConfiguration) Conflicts(o *networkConfiguration) error {
	if o == nil {
		return errors.New("same configuration")
	}

	// Also empty, because only one network with empty name is allowed
	if c.BridgeName == o.BridgeName {
		return errors.New("networks have same bridge name")
	}

	// They must be in different subnets
	if (c.AddressIPv4 != nil && o.AddressIPv4 != nil) &&
		(c.AddressIPv4.Contains(o.AddressIPv4.IP) || o.AddressIPv4.Contains(c.AddressIPv4.IP)) {
		return errors.New("networks have overlapping IPv4")
	}

	// They must be in different v6 subnets
	if (c.AddressIPv6 != nil && o.AddressIPv6 != nil) &&
		(c.AddressIPv6.Contains(o.AddressIPv6.IP) || o.AddressIPv6.Contains(c.AddressIPv6.IP)) {
		return errors.New("networks have overlapping IPv6")
	}

	return nil
}

func (c *networkConfiguration) fromLabels(labels map[string]string) error {
	var err error
	for label, value := range labels {
		switch label {
		case BridgeName:
			c.BridgeName = value
		case netlabel.DriverMTU:
			if c.Mtu, err = strconv.Atoi(value); err != nil {
				return parseErr(label, value, err.Error())
			}
		case netlabel.EnableIPv6:
			if c.EnableIPv6, err = strconv.ParseBool(value); err != nil {
				return parseErr(label, value, err.Error())
			}
		case EnableIPMasquerade:
			if c.EnableIPMasquerade, err = strconv.ParseBool(value); err != nil {
				return parseErr(label, value, err.Error())
			}
		case EnableICC:
			if c.EnableICC, err = strconv.ParseBool(value); err != nil {
				return parseErr(label, value, err.Error())
			}
		case DefaultBridge:
			if c.DefaultBridge, err = strconv.ParseBool(value); err != nil {
				return parseErr(label, value, err.Error())
			}
		case DefaultBindingIP:
			if c.DefaultBindingIP = net.ParseIP(value); c.DefaultBindingIP == nil {
				return parseErr(label, value, "nil ip")
			}
		case netlabel.ContainerIfacePrefix:
			c.ContainerIfacePrefix = value
		}
	}

	return nil
}

func parseErr(label, value, errString string) error {
	return types.BadRequestErrorf("failed to parse %s value: %v (%s)", label, value, errString)
}

func (n *bridgeNetwork) registerIptCleanFunc(clean iptableCleanFunc) {
	n.iptCleanFuncs = append(n.iptCleanFuncs, clean)
}

func (n *bridgeNetwork) getDriverChains() (*iptables.ChainInfo, *iptables.ChainInfo, *iptables.ChainInfo, error) {
	n.Lock()
	defer n.Unlock()

	if n.driver == nil {
		return nil, nil, nil, types.BadRequestErrorf("no driver found")
	}

	return n.driver.natChain, n.driver.filterChain, n.driver.isolationChain, nil
}

func (n *bridgeNetwork) getNetworkBridgeName() string {
	n.Lock()
	config := n.config
	n.Unlock()

	return config.BridgeName
}

func (n *bridgeNetwork) getEndpoint(eid string) (*bridgeEndpoint, error) {
	n.Lock()
	defer n.Unlock()

	if eid == "" {
		return nil, InvalidEndpointIDError(eid)
	}

	if ep, ok := n.endpoints[eid]; ok {
		return ep, nil
	}

	return nil, nil
}

// Install/Removes the iptables rules needed to isolate this network
// from each of the other networks
func (n *bridgeNetwork) isolateNetwork(others []*bridgeNetwork, enable bool) error {
	n.Lock()
	thisConfig := n.config
	n.Unlock()

	if thisConfig.Internal {
		return nil
	}

	// Install the rules to isolate this networks against each of the other networks
	for _, o := range others {
		o.Lock()
		otherConfig := o.config
		o.Unlock()

		if otherConfig.Internal {
			continue
		}

		if thisConfig.BridgeName != otherConfig.BridgeName {
			if err := setINC(thisConfig.BridgeName, otherConfig.BridgeName, enable); err != nil {
				return err
			}
		}
	}

	return nil
}

// Checks whether this network's configuration for the network with this id conflicts with any of the passed networks
func (c *networkConfiguration) conflictsWithNetworks(id string, others []*bridgeNetwork) error {
	for _, nw := range others {

		nw.Lock()
		nwID := nw.id
		nwConfig := nw.config
		nwBridge := nw.bridge
		nw.Unlock()

		if nwID == id {
			continue
		}
		// Verify the name (which may have been set by newInterface()) does not conflict with
		// existing bridge interfaces. Ironically the system chosen name gets stored in the config...
		// Basically we are checking if the two original configs were both empty.
		if nwConfig.BridgeName == c.BridgeName {
			return types.ForbiddenErrorf("conflicts with network %s (%s) by bridge name", nwID, nwConfig.BridgeName)
		}
		// If this network config specifies the AddressIPv4, we need
		// to make sure it does not conflict with any previously allocated
		// bridges. This could not be completely caught by the config conflict
		// check, because networks which config does not specify the AddressIPv4
		// get their address and subnet selected by the driver (see electBridgeIPv4())
		if c.AddressIPv4 != nil && nwBridge.bridgeIPv4 != nil {
			if nwBridge.bridgeIPv4.Contains(c.AddressIPv4.IP) ||
				c.AddressIPv4.Contains(nwBridge.bridgeIPv4.IP) {
				return types.ForbiddenErrorf("conflicts with network %s (%s) by ip network", nwID, nwConfig.BridgeName)
			}
		}
	}

	return nil
}

func (d *driver) configure(option map[string]interface{}) error {
	var (
		config         *configuration
		err            error
		natChain       *iptables.ChainInfo
		filterChain    *iptables.ChainInfo
		isolationChain *iptables.ChainInfo
	)

	genericData, ok := option[netlabel.GenericData]
	if !ok || genericData == nil {
		return nil
	}

	switch opt := genericData.(type) {
	case options.Generic:
		opaqueConfig, err := options.GenerateFromModel(opt, &configuration{})
		if err != nil {
			return err
		}
		config = opaqueConfig.(*configuration)
	case *configuration:
		config = opt
	default:
		return &ErrInvalidDriverConfig{}
	}

	if config.EnableIPTables {
		if _, err := os.Stat("/proc/sys/net/bridge"); err != nil {
			if out, err := exec.Command("modprobe", "-va", "bridge", "br_netfilter").CombinedOutput(); err != nil {
				logrus.Warnf("Running modprobe bridge br_netfilter failed with message: %s, error: %v", out, err)
			}
		}
		removeIPChains()
		natChain, filterChain, isolationChain, err = setupIPChains(config)
		if err != nil {
			return err
		}
		// Make sure on firewall reload, first thing being re-played is chains creation
		iptables.OnReloaded(func() { logrus.Debugf("Recreating iptables chains on firewall reload"); setupIPChains(config) })
	}

	if config.EnableIPForwarding {
		err = setupIPForwarding(config.EnableIPTables)
		if err != nil {
			logrus.Warn(err)
			return err
		}
	}

	d.Lock()
	d.natChain = natChain
	d.filterChain = filterChain
	d.isolationChain = isolationChain
	d.config = config
	d.Unlock()

	err = d.initStore(option)
	if err != nil {
		return err
	}

	return nil
}

func (d *driver) getNetwork(id string) (*bridgeNetwork, error) {
	d.Lock()
	defer d.Unlock()

	if id == "" {
		return nil, types.BadRequestErrorf("invalid network id: %s", id)
	}

	if nw, ok := d.networks[id]; ok {
		return nw, nil
	}

	return nil, types.NotFoundErrorf("network not found: %s", id)
}

func parseNetworkGenericOptions(data interface{}) (*networkConfiguration, error) {
	var (
		err    error
		config *networkConfiguration
	)

	switch opt := data.(type) {
	case *networkConfiguration:
		config = opt
	case map[string]string:
		config = &networkConfiguration{
			EnableICC:          true,
			EnableIPMasquerade: true,
		}
		err = config.fromLabels(opt)
	case options.Generic:
		var opaqueConfig interface{}
		if opaqueConfig, err = options.GenerateFromModel(opt, config); err == nil {
			config = opaqueConfig.(*networkConfiguration)
		}
	default:
		err = types.BadRequestErrorf("do not recognize network configuration format: %T", opt)
	}

	return config, err
}

func (c *networkConfiguration) processIPAM(id string, ipamV4Data, ipamV6Data []driverapi.IPAMData) error {
	if len(ipamV4Data) > 1 || len(ipamV6Data) > 1 {
		return types.ForbiddenErrorf("bridge driver doesn't support multiple subnets")
	}

	if len(ipamV4Data) == 0 {
		return types.BadRequestErrorf("bridge network %s requires ipv4 configuration", id)
	}

	if ipamV4Data[0].Gateway != nil {
		c.AddressIPv4 = types.GetIPNetCopy(ipamV4Data[0].Gateway)
	}

	if gw, ok := ipamV4Data[0].AuxAddresses[DefaultGatewayV4AuxKey]; ok {
		c.DefaultGatewayIPv4 = gw.IP
	}

	if len(ipamV6Data) > 0 {
		c.AddressIPv6 = ipamV6Data[0].Pool

		if ipamV6Data[0].Gateway != nil {
			c.AddressIPv6 = types.GetIPNetCopy(ipamV6Data[0].Gateway)
		}

		if gw, ok := ipamV6Data[0].AuxAddresses[DefaultGatewayV6AuxKey]; ok {
			c.DefaultGatewayIPv6 = gw.IP
		}
	}

	return nil
}

func parseNetworkOptions(id string, option options.Generic) (*networkConfiguration, error) {
	var (
		err    error
		config = &networkConfiguration{}
	)

	// Parse generic label first, config will be re-assigned
	if genData, ok := option[netlabel.GenericData]; ok && genData != nil {
		if config, err = parseNetworkGenericOptions(genData); err != nil {
			return nil, err
		}
	}

	// Process well-known labels next
	if val, ok := option[netlabel.EnableIPv6]; ok {
		config.EnableIPv6 = val.(bool)
	}

	if val, ok := option[netlabel.Internal]; ok {
		if internal, ok := val.(bool); ok && internal {
			config.Internal = true
		}
	}

	// Finally validate the configuration
	if err = config.Validate(); err != nil {
		return nil, err
	}

	if config.BridgeName == "" && config.DefaultBridge == false {
		config.BridgeName = "br-" + id[:12]
	}

	exists, err := bridgeInterfaceExists(config.BridgeName)
	if err != nil {
		return nil, err
	}

	if !exists {
		config.BridgeIfaceCreator = ifaceCreatedByLibnetwork
	} else {
		config.BridgeIfaceCreator = ifaceCreatedByUser
	}

	config.ID = id
	return config, nil
}

// Returns the non link-local IPv6 subnet for the containers attached to this bridge if found, nil otherwise
func getV6Network(config *networkConfiguration, i *bridgeInterface) *net.IPNet {
	if config.AddressIPv6 != nil {
		return config.AddressIPv6
	}
	if i.bridgeIPv6 != nil && i.bridgeIPv6.IP != nil && !i.bridgeIPv6.IP.IsLinkLocalUnicast() {
		return i.bridgeIPv6
	}

	return nil
}

// Return a slice of networks over which caller can iterate safely
func (d *driver) getNetworks() []*bridgeNetwork {
	d.Lock()
	defer d.Unlock()

	ls := make([]*bridgeNetwork, 0, len(d.networks))
	for _, nw := range d.networks {
		ls = append(ls, nw)
	}
	return ls
}

func (d *driver) NetworkAllocate(id string, option map[string]string, ipV4Data, ipV6Data []driverapi.IPAMData) (map[string]string, error) {
	return nil, types.NotImplementedErrorf("not implemented")
}

func (d *driver) NetworkFree(id string) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) EventNotify(etype driverapi.EventType, nid, tableName, key string, value []byte) {
}

func (d *driver) DecodeTableEntry(tablename string, key string, value []byte) (string, map[string]string) {
	return "", nil
}

// Create a new network using bridge plugin
func (d *driver) CreateNetwork(id string, option map[string]interface{}, nInfo driverapi.NetworkInfo, ipV4Data, ipV6Data []driverapi.IPAMData) error {
	if len(ipV4Data) == 0 || ipV4Data[0].Pool.String() == "0.0.0.0/0" {
		return types.BadRequestErrorf("ipv4 pool is empty")
	}
	// Sanity checks
	d.Lock()
	if _, ok := d.networks[id]; ok {
		d.Unlock()
		return types.ForbiddenErrorf("network %s exists", id)
	}
	d.Unlock()

	// Parse and validate the config. It should not be conflict with existing networks' config
	config, err := parseNetworkOptions(id, option)
	if err != nil {
		return err
	}

	err = config.processIPAM(id, ipV4Data, ipV6Data)
	if err != nil {
		return err
	}

	if err = d.createNetwork(config); err != nil {
		return err
	}

	return d.storeUpdate(config)
}

func (d *driver) createNetwork(config *networkConfiguration) error {
	var err error

	defer osl.InitOSContext()()

	networkList := d.getNetworks()
	for i, nw := range networkList {
		nw.Lock()
		nwConfig := nw.config
		nw.Unlock()
		if err := nwConfig.Conflicts(config); err != nil {
			if config.DefaultBridge {
				// We encountered and identified a stale default network
				// We must delete it as libnetwork is the source of thruth
				// The default network being created must be the only one
				// This can happen only from docker 1.12 on ward
				logrus.Infof("Removing stale default bridge network %s (%s)", nwConfig.ID, nwConfig.BridgeName)
				if err := d.DeleteNetwork(nwConfig.ID); err != nil {
					logrus.Warnf("Failed to remove stale default network: %s (%s): %v. Will remove from store.", nwConfig.ID, nwConfig.BridgeName, err)
					d.storeDelete(nwConfig)
				}
				networkList = append(networkList[:i], networkList[i+1:]...)
			} else {
				return types.ForbiddenErrorf("cannot create network %s (%s): conflicts with network %s (%s): %s",
					config.ID, config.BridgeName, nwConfig.ID, nwConfig.BridgeName, err.Error())
			}
		}
	}

	// Create and set network handler in driver
	network := &bridgeNetwork{
		id:         config.ID,
		endpoints:  make(map[string]*bridgeEndpoint),
		config:     config,
		portMapper: portmapper.New(d.config.UserlandProxyPath),
		driver:     d,
	}

	d.Lock()
	d.networks[config.ID] = network
	d.Unlock()

	// On failure make sure to reset driver network handler to nil
	defer func() {
		if err != nil {
			d.Lock()
			delete(d.networks, config.ID)
			d.Unlock()
		}
	}()

	// Initialize handle when needed
	d.Lock()
	if d.nlh == nil {
		d.nlh = ns.NlHandle()
	}
	d.Unlock()

	// Create or retrieve the bridge L3 interface
	bridgeIface, err := newInterface(d.nlh, config)
	if err != nil {
		return err
	}
	network.bridge = bridgeIface

	// Verify the network configuration does not conflict with previously installed
	// networks. This step is needed now because driver might have now set the bridge
	// name on this config struct. And because we need to check for possible address
	// conflicts, so we need to check against operationa lnetworks.
	if err = config.conflictsWithNetworks(config.ID, networkList); err != nil {
		return err
	}

	setupNetworkIsolationRules := func(config *networkConfiguration, i *bridgeInterface) error {
		if err := network.isolateNetwork(networkList, true); err != nil {
			if err := network.isolateNetwork(networkList, false); err != nil {
				logrus.Warnf("Failed on removing the inter-network iptables rules on cleanup: %v", err)
			}
			return err
		}
		network.registerIptCleanFunc(func() error {
			nwList := d.getNetworks()
			return network.isolateNetwork(nwList, false)
		})
		return nil
	}

	// Prepare the bridge setup configuration
	bridgeSetup := newBridgeSetup(config, bridgeIface)

	// If the bridge interface doesn't exist, we need to start the setup steps
	// by creating a new device and assigning it an IPv4 address.
	bridgeAlreadyExists := bridgeIface.exists()
	if !bridgeAlreadyExists {
		bridgeSetup.queueStep(setupDevice)
	}

	// Even if a bridge exists try to setup IPv4.
	bridgeSetup.queueStep(setupBridgeIPv4)

	enableIPv6Forwarding := d.config.EnableIPForwarding && config.AddressIPv6 != nil

	// Conditionally queue setup steps depending on configuration values.
	for _, step := range []struct {
		Condition bool
		Fn        setupStep
	}{
		// Enable IPv6 on the bridge if required. We do this even for a
		// previously  existing bridge, as it may be here from a previous
		// installation where IPv6 wasn't supported yet and needs to be
		// assigned an IPv6 link-local address.
		{config.EnableIPv6, setupBridgeIPv6},

		// We ensure that the bridge has the expectedIPv4 and IPv6 addresses in
		// the case of a previously existing device.
		{bridgeAlreadyExists, setupVerifyAndReconcile},

		// Enable IPv6 Forwarding
		{enableIPv6Forwarding, setupIPv6Forwarding},

		// Setup Loopback Adresses Routing
		{!d.config.EnableUserlandProxy, setupLoopbackAdressesRouting},

		// Setup IPTables.
		{d.config.EnableIPTables, network.setupIPTables},

		//We want to track firewalld configuration so that
		//if it is started/reloaded, the rules can be applied correctly
		{d.config.EnableIPTables, network.setupFirewalld},

		// Setup DefaultGatewayIPv4
		{config.DefaultGatewayIPv4 != nil, setupGatewayIPv4},

		// Setup DefaultGatewayIPv6
		{config.DefaultGatewayIPv6 != nil, setupGatewayIPv6},

		// Add inter-network communication rules.
		{d.config.EnableIPTables, setupNetworkIsolationRules},

		//Configure bridge networking filtering if ICC is off and IP tables are enabled
		{!config.EnableICC && d.config.EnableIPTables, setupBridgeNetFiltering},
	} {
		if step.Condition {
			bridgeSetup.queueStep(step.Fn)
		}
	}

	// Apply the prepared list of steps, and abort at the first error.
	bridgeSetup.queueStep(setupDeviceUp)
	if err = bridgeSetup.apply(); err != nil {
		return err
	}

	return nil
}

func (d *driver) DeleteNetwork(nid string) error {
	var err error

	defer osl.InitOSContext()()

	// Get network handler and remove it from driver
	d.Lock()
	n, ok := d.networks[nid]
	d.Unlock()

	if !ok {
		return types.InternalMaskableErrorf("network %s does not exist", nid)
	}

	n.Lock()
	config := n.config
	n.Unlock()

	// delele endpoints belong to this network
	for _, ep := range n.endpoints {
		if err := n.releasePorts(ep); err != nil {
			logrus.Warn(err)
		}
		if link, err := d.nlh.LinkByName(ep.srcName); err == nil {
			d.nlh.LinkDel(link)
		}

		if err := d.storeDelete(ep); err != nil {
			logrus.Warnf("Failed to remove bridge endpoint %s from store: %v", ep.id[0:7], err)
		}
	}

	d.Lock()
	delete(d.networks, nid)
	d.Unlock()

	// On failure set network handler back in driver, but
	// only if is not already taken over by some other thread
	defer func() {
		if err != nil {
			d.Lock()
			if _, ok := d.networks[nid]; !ok {
				d.networks[nid] = n
			}
			d.Unlock()
		}
	}()

	// Sanity check
	if n == nil {
		err = driverapi.ErrNoNetwork(nid)
		return err
	}

	switch config.BridgeIfaceCreator {
	case ifaceCreatedByLibnetwork, ifaceCreatorUnknown:
		// We only delete the bridge if it was created by the bridge driver and
		// it is not the default one (to keep the backward compatible behavior.)
		if !config.DefaultBridge {
			if err := d.nlh.LinkDel(n.bridge.Link); err != nil {
				logrus.Warnf("Failed to remove bridge interface %s on network %s delete: %v", config.BridgeName, nid, err)
			}
		}
	case ifaceCreatedByUser:
		// Don't delete the bridge interface if it was not created by libnetwork.
	}

	// clean all relevant iptables rules
	for _, cleanFunc := range n.iptCleanFuncs {
		if errClean := cleanFunc(); errClean != nil {
			logrus.Warnf("Failed to clean iptables rules for bridge network: %v", errClean)
		}
	}
	return d.storeDelete(config)
}

func addToBridge(nlh *netlink.Handle, ifaceName, bridgeName string) error {
	link, err := nlh.LinkByName(ifaceName)
	if err != nil {
		return fmt.Errorf("could not find interface %s: %v", ifaceName, err)
	}
	if err = nlh.LinkSetMaster(link,
		&netlink.Bridge{LinkAttrs: netlink.LinkAttrs{Name: bridgeName}}); err != nil {
		logrus.Debugf("Failed to add %s to bridge via netlink.Trying ioctl: %v", ifaceName, err)
		iface, err := net.InterfaceByName(ifaceName)
		if err != nil {
			return fmt.Errorf("could not find network interface %s: %v", ifaceName, err)
		}

		master, err := net.InterfaceByName(bridgeName)
		if err != nil {
			return fmt.Errorf("could not find bridge %s: %v", bridgeName, err)
		}

		return ioctlAddToBridge(iface, master)
	}
	return nil
}

func setHairpinMode(nlh *netlink.Handle, link netlink.Link, enable bool) error {
	err := nlh.LinkSetHairpin(link, enable)
	if err != nil && err != syscall.EINVAL {
		// If error is not EINVAL something else went wrong, bail out right away
		return fmt.Errorf("unable to set hairpin mode on %s via netlink: %v",
			link.Attrs().Name, err)
	}

	// Hairpin mode successfully set up
	if err == nil {
		return nil
	}

	// The netlink method failed with EINVAL which is probably because of an older
	// kernel. Try one more time via the sysfs method.
	path := filepath.Join("/sys/class/net", link.Attrs().Name, "brport/hairpin_mode")

	var val []byte
	if enable {
		val = []byte{'1', '\n'}
	} else {
		val = []byte{'0', '\n'}
	}

	if err := ioutil.WriteFile(path, val, 0644); err != nil {
		return fmt.Errorf("unable to set hairpin mode on %s via sysfs: %v", link.Attrs().Name, err)
	}

	return nil
}

func (d *driver) CreateEndpoint(nid, eid string, ifInfo driverapi.InterfaceInfo, epOptions map[string]interface{}) error {
	defer osl.InitOSContext()()

	if ifInfo == nil {
		return errors.New("invalid interface info passed")
	}

	// Get the network handler and make sure it exists
	d.Lock()
	n, ok := d.networks[nid]
	dconfig := d.config
	d.Unlock()

	if !ok {
		return types.NotFoundErrorf("network %s does not exist", nid)
	}
	if n == nil {
		return driverapi.ErrNoNetwork(nid)
	}

	// Sanity check
	n.Lock()
	if n.id != nid {
		n.Unlock()
		return InvalidNetworkIDError(nid)
	}
	n.Unlock()

	// Check if endpoint id is good and retrieve correspondent endpoint
	ep, err := n.getEndpoint(eid)
	if err != nil {
		return err
	}

	// Endpoint with that id exists either on desired or other sandbox
	if ep != nil {
		return driverapi.ErrEndpointExists(eid)
	}

	// Try to convert the options to endpoint configuration
	epConfig, err := parseEndpointOptions(epOptions)
	if err != nil {
		return err
	}

	// Create and add the endpoint
	n.Lock()
	endpoint := &bridgeEndpoint{id: eid, nid: nid, config: epConfig}
	n.endpoints[eid] = endpoint
	n.Unlock()

	// On failure make sure to remove the endpoint
	defer func() {
		if err != nil {
			n.Lock()
			delete(n.endpoints, eid)
			n.Unlock()
		}
	}()

	// Generate a name for what will be the host side pipe interface
	hostIfName, err := netutils.GenerateIfaceName(d.nlh, vethPrefix, vethLen)
	if err != nil {
		return err
	}

	// Generate a name for what will be the sandbox side pipe interface
	containerIfName, err := netutils.GenerateIfaceName(d.nlh, vethPrefix, vethLen)
	if err != nil {
		return err
	}

	// Generate and add the interface pipe host <-> sandbox
	veth := &netlink.Veth{
		LinkAttrs: netlink.LinkAttrs{Name: hostIfName, TxQLen: 0},
		PeerName:  containerIfName}
	if err = d.nlh.LinkAdd(veth); err != nil {
		return types.InternalErrorf("failed to add the host (%s) <=> sandbox (%s) pair interfaces: %v", hostIfName, containerIfName, err)
	}

	// Get the host side pipe interface handler
	host, err := d.nlh.LinkByName(hostIfName)
	if err != nil {
		return types.InternalErrorf("failed to find host side interface %s: %v", hostIfName, err)
	}
	defer func() {
		if err != nil {
			d.nlh.LinkDel(host)
		}
	}()

	// Get the sandbox side pipe interface handler
	sbox, err := d.nlh.LinkByName(containerIfName)
	if err != nil {
		return types.InternalErrorf("failed to find sandbox side interface %s: %v", containerIfName, err)
	}
	defer func() {
		if err != nil {
			d.nlh.LinkDel(sbox)
		}
	}()

	n.Lock()
	config := n.config
	n.Unlock()

	// Add bridge inherited attributes to pipe interfaces
	if config.Mtu != 0 {
		err = d.nlh.LinkSetMTU(host, config.Mtu)
		if err != nil {
			return types.InternalErrorf("failed to set MTU on host interface %s: %v", hostIfName, err)
		}
		err = d.nlh.LinkSetMTU(sbox, config.Mtu)
		if err != nil {
			return types.InternalErrorf("failed to set MTU on sandbox interface %s: %v", containerIfName, err)
		}
	}

	// Attach host side pipe interface into the bridge
	if err = addToBridge(d.nlh, hostIfName, config.BridgeName); err != nil {
		return fmt.Errorf("adding interface %s to bridge %s failed: %v", hostIfName, config.BridgeName, err)
	}

	if !dconfig.EnableUserlandProxy {
		err = setHairpinMode(d.nlh, host, true)
		if err != nil {
			return err
		}
	}

	// Store the sandbox side pipe interface parameters
	endpoint.srcName = containerIfName
	endpoint.macAddress = ifInfo.MacAddress()
	endpoint.addr = ifInfo.Address()
	endpoint.addrv6 = ifInfo.AddressIPv6()

	// Set the sbox's MAC if not provided. If specified, use the one configured by user, otherwise generate one based on IP.
	if endpoint.macAddress == nil {
		endpoint.macAddress = electMacAddress(epConfig, endpoint.addr.IP)
		if err = ifInfo.SetMacAddress(endpoint.macAddress); err != nil {
			return err
		}
	}

	// Up the host interface after finishing all netlink configuration
	if err = d.nlh.LinkSetUp(host); err != nil {
		return fmt.Errorf("could not set link up for host interface %s: %v", hostIfName, err)
	}

	if endpoint.addrv6 == nil && config.EnableIPv6 {
		var ip6 net.IP
		network := n.bridge.bridgeIPv6
		if config.AddressIPv6 != nil {
			network = config.AddressIPv6
		}

		ones, _ := network.Mask.Size()
		if ones > 80 {
			err = types.ForbiddenErrorf("Cannot self generate an IPv6 address on network %v: At least 48 host bits are needed.", network)
			return err
		}

		ip6 = make(net.IP, len(network.IP))
		copy(ip6, network.IP)
		for i, h := range endpoint.macAddress {
			ip6[i+10] = h
		}

		endpoint.addrv6 = &net.IPNet{IP: ip6, Mask: network.Mask}
		if err = ifInfo.SetIPAddress(endpoint.addrv6); err != nil {
			return err
		}
	}

	if err = d.storeUpdate(endpoint); err != nil {
		return fmt.Errorf("failed to save bridge endpoint %s to store: %v", endpoint.id[0:7], err)
	}

	return nil
}

func (d *driver) DeleteEndpoint(nid, eid string) error {
	var err error

	defer osl.InitOSContext()()

	// Get the network handler and make sure it exists
	d.Lock()
	n, ok := d.networks[nid]
	d.Unlock()

	if !ok {
		return types.InternalMaskableErrorf("network %s does not exist", nid)
	}
	if n == nil {
		return driverapi.ErrNoNetwork(nid)
	}

	// Sanity Check
	n.Lock()
	if n.id != nid {
		n.Unlock()
		return InvalidNetworkIDError(nid)
	}
	n.Unlock()

	// Check endpoint id and if an endpoint is actually there
	ep, err := n.getEndpoint(eid)
	if err != nil {
		return err
	}
	if ep == nil {
		return EndpointNotFoundError(eid)
	}

	// Remove it
	n.Lock()
	delete(n.endpoints, eid)
	n.Unlock()

	// On failure make sure to set back ep in n.endpoints, but only
	// if it hasn't been taken over already by some other thread.
	defer func() {
		if err != nil {
			n.Lock()
			if _, ok := n.endpoints[eid]; !ok {
				n.endpoints[eid] = ep
			}
			n.Unlock()
		}
	}()

	// Try removal of link. Discard error: it is a best effort.
	// Also make sure defer does not see this error either.
	if link, err := d.nlh.LinkByName(ep.srcName); err == nil {
		d.nlh.LinkDel(link)
	}

	if err := d.storeDelete(ep); err != nil {
		logrus.Warnf("Failed to remove bridge endpoint %s from store: %v", ep.id[0:7], err)
	}

	return nil
}

func (d *driver) EndpointOperInfo(nid, eid string) (map[string]interface{}, error) {
	// Get the network handler and make sure it exists
	d.Lock()
	n, ok := d.networks[nid]
	d.Unlock()
	if !ok {
		return nil, types.NotFoundErrorf("network %s does not exist", nid)
	}
	if n == nil {
		return nil, driverapi.ErrNoNetwork(nid)
	}

	// Sanity check
	n.Lock()
	if n.id != nid {
		n.Unlock()
		return nil, InvalidNetworkIDError(nid)
	}
	n.Unlock()

	// Check if endpoint id is good and retrieve correspondent endpoint
	ep, err := n.getEndpoint(eid)
	if err != nil {
		return nil, err
	}
	if ep == nil {
		return nil, driverapi.ErrNoEndpoint(eid)
	}

	m := make(map[string]interface{})

	if ep.extConnConfig != nil && ep.extConnConfig.ExposedPorts != nil {
		// Return a copy of the config data
		epc := make([]types.TransportPort, 0, len(ep.extConnConfig.ExposedPorts))
		for _, tp := range ep.extConnConfig.ExposedPorts {
			epc = append(epc, tp.GetCopy())
		}
		m[netlabel.ExposedPorts] = epc
	}

	if ep.portMapping != nil {
		// Return a copy of the operational data
		pmc := make([]types.PortBinding, 0, len(ep.portMapping))
		for _, pm := range ep.portMapping {
			pmc = append(pmc, pm.GetCopy())
		}
		m[netlabel.PortMap] = pmc
	}

	if len(ep.macAddress) != 0 {
		m[netlabel.MacAddress] = ep.macAddress
	}

	return m, nil
}

// Join method is invoked when a Sandbox is attached to an endpoint.
func (d *driver) Join(nid, eid string, sboxKey string, jinfo driverapi.JoinInfo, options map[string]interface{}) error {
	defer osl.InitOSContext()()

	network, err := d.getNetwork(nid)
	if err != nil {
		return err
	}

	endpoint, err := network.getEndpoint(eid)
	if err != nil {
		return err
	}

	if endpoint == nil {
		return EndpointNotFoundError(eid)
	}

	endpoint.containerConfig, err = parseContainerOptions(options)
	if err != nil {
		return err
	}

	iNames := jinfo.InterfaceName()
	containerVethPrefix := defaultContainerVethPrefix
	if network.config.ContainerIfacePrefix != "" {
		containerVethPrefix = network.config.ContainerIfacePrefix
	}
	err = iNames.SetNames(endpoint.srcName, containerVethPrefix)
	if err != nil {
		return err
	}

	err = jinfo.SetGateway(network.bridge.gatewayIPv4)
	if err != nil {
		return err
	}

	err = jinfo.SetGatewayIPv6(network.bridge.gatewayIPv6)
	if err != nil {
		return err
	}

	return nil
}

// Leave method is invoked when a Sandbox detaches from an endpoint.
func (d *driver) Leave(nid, eid string) error {
	defer osl.InitOSContext()()

	network, err := d.getNetwork(nid)
	if err != nil {
		return types.InternalMaskableErrorf("%s", err)
	}

	endpoint, err := network.getEndpoint(eid)
	if err != nil {
		return err
	}

	if endpoint == nil {
		return EndpointNotFoundError(eid)
	}

	if !network.config.EnableICC {
		if err = d.link(network, endpoint, false); err != nil {
			return err
		}
	}

	return nil
}

func (d *driver) ProgramExternalConnectivity(nid, eid string, options map[string]interface{}) error {
	defer osl.InitOSContext()()

	network, err := d.getNetwork(nid)
	if err != nil {
		return err
	}

	endpoint, err := network.getEndpoint(eid)
	if err != nil {
		return err
	}

	if endpoint == nil {
		return EndpointNotFoundError(eid)
	}

	endpoint.extConnConfig, err = parseConnectivityOptions(options)
	if err != nil {
		return err
	}

	// Program any required port mapping and store them in the endpoint
	endpoint.portMapping, err = network.allocatePorts(endpoint, network.config.DefaultBindingIP, d.config.EnableUserlandProxy)
	if err != nil {
		return err
	}

	defer func() {
		if err != nil {
			if e := network.releasePorts(endpoint); e != nil {
				logrus.Errorf("Failed to release ports allocated for the bridge endpoint %s on failure %v because of %v",
					eid, err, e)
			}
			endpoint.portMapping = nil
		}
	}()

	if err = d.storeUpdate(endpoint); err != nil {
		return fmt.Errorf("failed to update bridge endpoint %s to store: %v", endpoint.id[0:7], err)
	}

	if !network.config.EnableICC {
		return d.link(network, endpoint, true)
	}

	return nil
}

func (d *driver) RevokeExternalConnectivity(nid, eid string) error {
	defer osl.InitOSContext()()

	network, err := d.getNetwork(nid)
	if err != nil {
		return err
	}

	endpoint, err := network.getEndpoint(eid)
	if err != nil {
		return err
	}

	if endpoint == nil {
		return EndpointNotFoundError(eid)
	}

	err = network.releasePorts(endpoint)
	if err != nil {
		logrus.Warn(err)
	}

	endpoint.portMapping = nil

	// Clean the connection tracker state of the host for the specific endpoint
	// The host kernel keeps track of the connections (TCP and UDP), so if a new endpoint gets the same IP of
	// this one (that is going down), is possible that some of the packets would not be routed correctly inside
	// the new endpoint
	// Deeper details: https://github.com/docker/docker/issues/8795
	clearEndpointConnections(d.nlh, endpoint)

	if err = d.storeUpdate(endpoint); err != nil {
		return fmt.Errorf("failed to update bridge endpoint %s to store: %v", endpoint.id[0:7], err)
	}

	return nil
}

func (d *driver) link(network *bridgeNetwork, endpoint *bridgeEndpoint, enable bool) error {
	var err error

	cc := endpoint.containerConfig
	if cc == nil {
		return nil
	}
	ec := endpoint.extConnConfig
	if ec == nil {
		return nil
	}

	if ec.ExposedPorts != nil {
		for _, p := range cc.ParentEndpoints {
			var parentEndpoint *bridgeEndpoint
			parentEndpoint, err = network.getEndpoint(p)
			if err != nil {
				return err
			}
			if parentEndpoint == nil {
				err = InvalidEndpointIDError(p)
				return err
			}

			l := newLink(parentEndpoint.addr.IP.String(),
				endpoint.addr.IP.String(),
				ec.ExposedPorts, network.config.BridgeName)
			if enable {
				err = l.Enable()
				if err != nil {
					return err
				}
				defer func() {
					if err != nil {
						l.Disable()
					}
				}()
			} else {
				l.Disable()
			}
		}
	}

	for _, c := range cc.ChildEndpoints {
		var childEndpoint *bridgeEndpoint
		childEndpoint, err = network.getEndpoint(c)
		if err != nil {
			return err
		}
		if childEndpoint == nil {
			err = InvalidEndpointIDError(c)
			return err
		}
		if childEndpoint.extConnConfig == nil || childEndpoint.extConnConfig.ExposedPorts == nil {
			continue
		}

		l := newLink(endpoint.addr.IP.String(),
			childEndpoint.addr.IP.String(),
			childEndpoint.extConnConfig.ExposedPorts, network.config.BridgeName)
		if enable {
			err = l.Enable()
			if err != nil {
				return err
			}
			defer func() {
				if err != nil {
					l.Disable()
				}
			}()
		} else {
			l.Disable()
		}
	}

	return nil
}

func (d *driver) Type() string {
	return networkType
}

func (d *driver) IsBuiltIn() bool {
	return true
}

// DiscoverNew is a notification for a new discovery event, such as a new node joining a cluster
func (d *driver) DiscoverNew(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

// DiscoverDelete is a notification for a discovery delete event, such as a node leaving a cluster
func (d *driver) DiscoverDelete(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

func parseEndpointOptions(epOptions map[string]interface{}) (*endpointConfiguration, error) {
	if epOptions == nil {
		return nil, nil
	}

	ec := &endpointConfiguration{}

	if opt, ok := epOptions[netlabel.MacAddress]; ok {
		if mac, ok := opt.(net.HardwareAddr); ok {
			ec.MacAddress = mac
		} else {
			return nil, &ErrInvalidEndpointConfig{}
		}
	}

	return ec, nil
}

func parseContainerOptions(cOptions map[string]interface{}) (*containerConfiguration, error) {
	if cOptions == nil {
		return nil, nil
	}
	genericData := cOptions[netlabel.GenericData]
	if genericData == nil {
		return nil, nil
	}
	switch opt := genericData.(type) {
	case options.Generic:
		opaqueConfig, err := options.GenerateFromModel(opt, &containerConfiguration{})
		if err != nil {
			return nil, err
		}
		return opaqueConfig.(*containerConfiguration), nil
	case *containerConfiguration:
		return opt, nil
	default:
		return nil, nil
	}
}

func parseConnectivityOptions(cOptions map[string]interface{}) (*connectivityConfiguration, error) {
	if cOptions == nil {
		return nil, nil
	}

	cc := &connectivityConfiguration{}

	if opt, ok := cOptions[netlabel.PortMap]; ok {
		if pb, ok := opt.([]types.PortBinding); ok {
			cc.PortBindings = pb
		} else {
			return nil, types.BadRequestErrorf("Invalid port mapping data in connectivity configuration: %v", opt)
		}
	}

	if opt, ok := cOptions[netlabel.ExposedPorts]; ok {
		if ports, ok := opt.([]types.TransportPort); ok {
			cc.ExposedPorts = ports
		} else {
			return nil, types.BadRequestErrorf("Invalid exposed ports data in connectivity configuration: %v", opt)
		}
	}

	return cc, nil
}

func electMacAddress(epConfig *endpointConfiguration, ip net.IP) net.HardwareAddr {
	if epConfig != nil && epConfig.MacAddress != nil {
		return epConfig.MacAddress
	}
	return netutils.GenerateMACFromIP(ip)
}
