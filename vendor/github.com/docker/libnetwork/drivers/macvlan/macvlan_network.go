package macvlan

import (
	"fmt"

	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/osl"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

// CreateNetwork the network for the specified driver type
func (d *driver) CreateNetwork(nid string, option map[string]interface{}, nInfo driverapi.NetworkInfo, ipV4Data, ipV6Data []driverapi.IPAMData) error {
	defer osl.InitOSContext()()
	kv, err := kernel.GetKernelVersion()
	if err != nil {
		return fmt.Errorf("failed to check kernel version for %s driver support: %v", macvlanType, err)
	}
	// ensure Kernel version is >= v3.9 for macvlan support
	if kv.Kernel < macvlanKernelVer || (kv.Kernel == macvlanKernelVer && kv.Major < macvlanMajorVer) {
		return fmt.Errorf("kernel version failed to meet the minimum macvlan kernel requirement of %d.%d, found %d.%d.%d",
			macvlanKernelVer, macvlanMajorVer, kv.Kernel, kv.Major, kv.Minor)
	}
	// reject a null v4 network
	if len(ipV4Data) == 0 || ipV4Data[0].Pool.String() == "0.0.0.0/0" {
		return fmt.Errorf("ipv4 pool is empty")
	}
	// parse and validate the config and bind to networkConfiguration
	config, err := parseNetworkOptions(nid, option)
	if err != nil {
		return err
	}
	config.ID = nid
	err = config.processIPAM(nid, ipV4Data, ipV6Data)
	if err != nil {
		return err
	}
	// verify the macvlan mode from -o macvlan_mode option
	switch config.MacvlanMode {
	case "", modeBridge:
		// default to macvlan bridge mode if -o macvlan_mode is empty
		config.MacvlanMode = modeBridge
	case modePrivate:
		config.MacvlanMode = modePrivate
	case modePassthru:
		config.MacvlanMode = modePassthru
	case modeVepa:
		config.MacvlanMode = modeVepa
	default:
		return fmt.Errorf("requested macvlan mode '%s' is not valid, 'bridge' mode is the macvlan driver default", config.MacvlanMode)
	}
	// loopback is not a valid parent link
	if config.Parent == "lo" {
		return fmt.Errorf("loopback interface is not a valid %s parent link", macvlanType)
	}
	// if parent interface not specified, create a dummy type link to use named dummy+net_id
	if config.Parent == "" {
		config.Parent = getDummyName(stringid.TruncateID(config.ID))
		// empty parent and --internal are handled the same. Set here to update k/v
		config.Internal = true
	}
	err = d.createNetwork(config)
	if err != nil {
		return err
	}
	// update persistent db, rollback on fail
	err = d.storeUpdate(config)
	if err != nil {
		d.deleteNetwork(config.ID)
		logrus.Debugf("encoutered an error rolling back a network create for %s : %v", config.ID, err)
		return err
	}

	return nil
}

// createNetwork is used by new network callbacks and persistent network cache
func (d *driver) createNetwork(config *configuration) error {
	networkList := d.getNetworks()
	for _, nw := range networkList {
		if config.Parent == nw.config.Parent {
			return fmt.Errorf("network %s is already using parent interface %s",
				getDummyName(stringid.TruncateID(nw.config.ID)), config.Parent)
		}
	}
	if !parentExists(config.Parent) {
		// if the --internal flag is set, create a dummy link
		if config.Internal {
			err := createDummyLink(config.Parent, getDummyName(stringid.TruncateID(config.ID)))
			if err != nil {
				return err
			}
			config.CreatedSlaveLink = true
			// notify the user in logs they have limited comunicatins
			if config.Parent == getDummyName(stringid.TruncateID(config.ID)) {
				logrus.Debugf("Empty -o parent= and --internal flags limit communications to other containers inside of network: %s",
					config.Parent)
			}
		} else {
			// if the subinterface parent_iface.vlan_id checks do not pass, return err.
			//  a valid example is 'eth0.10' for a parent iface 'eth0' with a vlan id '10'
			err := createVlanLink(config.Parent)
			if err != nil {
				return err
			}
			// if driver created the networks slave link, record it for future deletion
			config.CreatedSlaveLink = true
		}
	}
	n := &network{
		id:        config.ID,
		driver:    d,
		endpoints: endpointTable{},
		config:    config,
	}
	// add the *network
	d.addNetwork(n)

	return nil
}

// DeleteNetwork deletes the network for the specified driver type
func (d *driver) DeleteNetwork(nid string) error {
	defer osl.InitOSContext()()
	n := d.network(nid)
	if n == nil {
		return fmt.Errorf("network id %s not found", nid)
	}
	// if the driver created the slave interface, delete it, otherwise leave it
	if ok := n.config.CreatedSlaveLink; ok {
		// if the interface exists, only delete if it matches iface.vlan or dummy.net_id naming
		if ok := parentExists(n.config.Parent); ok {
			// only delete the link if it is named the net_id
			if n.config.Parent == getDummyName(stringid.TruncateID(nid)) {
				err := delDummyLink(n.config.Parent)
				if err != nil {
					logrus.Debugf("link %s was not deleted, continuing the delete network operation: %v",
						n.config.Parent, err)
				}
			} else {
				// only delete the link if it matches iface.vlan naming
				err := delVlanLink(n.config.Parent)
				if err != nil {
					logrus.Debugf("link %s was not deleted, continuing the delete network operation: %v",
						n.config.Parent, err)
				}
			}
		}
	}
	for _, ep := range n.endpoints {
		if link, err := ns.NlHandle().LinkByName(ep.srcName); err == nil {
			ns.NlHandle().LinkDel(link)
		}

		if err := d.storeDelete(ep); err != nil {
			logrus.Warnf("Failed to remove macvlan endpoint %s from store: %v", ep.id[0:7], err)
		}
	}
	// delete the *network
	d.deleteNetwork(nid)
	// delete the network record from persistent cache
	err := d.storeDelete(n.config)
	if err != nil {
		return fmt.Errorf("error deleting deleting id %s from datastore: %v", nid, err)
	}
	return nil
}

// parseNetworkOptions parses docker network options
func parseNetworkOptions(id string, option options.Generic) (*configuration, error) {
	var (
		err    error
		config = &configuration{}
	)
	// parse generic labels first
	if genData, ok := option[netlabel.GenericData]; ok && genData != nil {
		if config, err = parseNetworkGenericOptions(genData); err != nil {
			return nil, err
		}
	}
	// setting the parent to "" will trigger an isolated network dummy parent link
	if _, ok := option[netlabel.Internal]; ok {
		config.Internal = true
		// empty --parent= and --internal are handled the same.
		config.Parent = ""
	}

	return config, nil
}

// parseNetworkGenericOptions parses generic driver docker network options
func parseNetworkGenericOptions(data interface{}) (*configuration, error) {
	var (
		err    error
		config *configuration
	)
	switch opt := data.(type) {
	case *configuration:
		config = opt
	case map[string]string:
		config = &configuration{}
		err = config.fromOptions(opt)
	case options.Generic:
		var opaqueConfig interface{}
		if opaqueConfig, err = options.GenerateFromModel(opt, config); err == nil {
			config = opaqueConfig.(*configuration)
		}
	default:
		err = types.BadRequestErrorf("unrecognized network configuration format: %v", opt)
	}

	return config, err
}

// fromOptions binds the generic options to networkConfiguration to cache
func (config *configuration) fromOptions(labels map[string]string) error {
	for label, value := range labels {
		switch label {
		case parentOpt:
			// parse driver option '-o parent'
			config.Parent = value
		case driverModeOpt:
			// parse driver option '-o macvlan_mode'
			config.MacvlanMode = value
		}
	}

	return nil
}

// processIPAM parses v4 and v6 IP information and binds it to the network configuration
func (config *configuration) processIPAM(id string, ipamV4Data, ipamV6Data []driverapi.IPAMData) error {
	if len(ipamV4Data) > 0 {
		for _, ipd := range ipamV4Data {
			s := &ipv4Subnet{
				SubnetIP: ipd.Pool.String(),
				GwIP:     ipd.Gateway.String(),
			}
			config.Ipv4Subnets = append(config.Ipv4Subnets, s)
		}
	}
	if len(ipamV6Data) > 0 {
		for _, ipd := range ipamV6Data {
			s := &ipv6Subnet{
				SubnetIP: ipd.Pool.String(),
				GwIP:     ipd.Gateway.String(),
			}
			config.Ipv6Subnets = append(config.Ipv6Subnets, s)
		}
	}

	return nil
}
