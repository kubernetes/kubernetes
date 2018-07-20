package daemon

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path"
	"runtime"
	"strings"
	"time"

	containertypes "github.com/docker/docker/api/types/container"
	networktypes "github.com/docker/docker/api/types/network"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/network"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/runconfig"
	"github.com/docker/go-connections/nat"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

var (
	// ErrRootFSReadOnly is returned when a container
	// rootfs is marked readonly.
	ErrRootFSReadOnly = errors.New("container rootfs is marked read-only")
	getPortMapInfo    = container.GetSandboxPortMapInfo
)

func (daemon *Daemon) getDNSSearchSettings(container *container.Container) []string {
	if len(container.HostConfig.DNSSearch) > 0 {
		return container.HostConfig.DNSSearch
	}

	if len(daemon.configStore.DNSSearch) > 0 {
		return daemon.configStore.DNSSearch
	}

	return nil
}

func (daemon *Daemon) buildSandboxOptions(container *container.Container) ([]libnetwork.SandboxOption, error) {
	var (
		sboxOptions []libnetwork.SandboxOption
		err         error
		dns         []string
		dnsOptions  []string
		bindings    = make(nat.PortMap)
		pbList      []types.PortBinding
		exposeList  []types.TransportPort
	)

	defaultNetName := runconfig.DefaultDaemonNetworkMode().NetworkName()
	sboxOptions = append(sboxOptions, libnetwork.OptionHostname(container.Config.Hostname),
		libnetwork.OptionDomainname(container.Config.Domainname))

	if container.HostConfig.NetworkMode.IsHost() {
		sboxOptions = append(sboxOptions, libnetwork.OptionUseDefaultSandbox())
		if len(container.HostConfig.ExtraHosts) == 0 {
			sboxOptions = append(sboxOptions, libnetwork.OptionOriginHostsPath("/etc/hosts"))
		}
		if len(container.HostConfig.DNS) == 0 && len(daemon.configStore.DNS) == 0 &&
			len(container.HostConfig.DNSSearch) == 0 && len(daemon.configStore.DNSSearch) == 0 &&
			len(container.HostConfig.DNSOptions) == 0 && len(daemon.configStore.DNSOptions) == 0 {
			sboxOptions = append(sboxOptions, libnetwork.OptionOriginResolvConfPath("/etc/resolv.conf"))
		}
	} else {
		// OptionUseExternalKey is mandatory for userns support.
		// But optional for non-userns support
		sboxOptions = append(sboxOptions, libnetwork.OptionUseExternalKey())
	}

	if err = setupPathsAndSandboxOptions(container, &sboxOptions); err != nil {
		return nil, err
	}

	if len(container.HostConfig.DNS) > 0 {
		dns = container.HostConfig.DNS
	} else if len(daemon.configStore.DNS) > 0 {
		dns = daemon.configStore.DNS
	}

	for _, d := range dns {
		sboxOptions = append(sboxOptions, libnetwork.OptionDNS(d))
	}

	dnsSearch := daemon.getDNSSearchSettings(container)

	for _, ds := range dnsSearch {
		sboxOptions = append(sboxOptions, libnetwork.OptionDNSSearch(ds))
	}

	if len(container.HostConfig.DNSOptions) > 0 {
		dnsOptions = container.HostConfig.DNSOptions
	} else if len(daemon.configStore.DNSOptions) > 0 {
		dnsOptions = daemon.configStore.DNSOptions
	}

	for _, ds := range dnsOptions {
		sboxOptions = append(sboxOptions, libnetwork.OptionDNSOptions(ds))
	}

	if container.NetworkSettings.SecondaryIPAddresses != nil {
		name := container.Config.Hostname
		if container.Config.Domainname != "" {
			name = name + "." + container.Config.Domainname
		}

		for _, a := range container.NetworkSettings.SecondaryIPAddresses {
			sboxOptions = append(sboxOptions, libnetwork.OptionExtraHost(name, a.Addr))
		}
	}

	for _, extraHost := range container.HostConfig.ExtraHosts {
		// allow IPv6 addresses in extra hosts; only split on first ":"
		if _, err := opts.ValidateExtraHost(extraHost); err != nil {
			return nil, err
		}
		parts := strings.SplitN(extraHost, ":", 2)
		sboxOptions = append(sboxOptions, libnetwork.OptionExtraHost(parts[0], parts[1]))
	}

	if container.HostConfig.PortBindings != nil {
		for p, b := range container.HostConfig.PortBindings {
			bindings[p] = []nat.PortBinding{}
			for _, bb := range b {
				bindings[p] = append(bindings[p], nat.PortBinding{
					HostIP:   bb.HostIP,
					HostPort: bb.HostPort,
				})
			}
		}
	}

	portSpecs := container.Config.ExposedPorts
	ports := make([]nat.Port, len(portSpecs))
	var i int
	for p := range portSpecs {
		ports[i] = p
		i++
	}
	nat.SortPortMap(ports, bindings)
	for _, port := range ports {
		expose := types.TransportPort{}
		expose.Proto = types.ParseProtocol(port.Proto())
		expose.Port = uint16(port.Int())
		exposeList = append(exposeList, expose)

		pb := types.PortBinding{Port: expose.Port, Proto: expose.Proto}
		binding := bindings[port]
		for i := 0; i < len(binding); i++ {
			pbCopy := pb.GetCopy()
			newP, err := nat.NewPort(nat.SplitProtoPort(binding[i].HostPort))
			var portStart, portEnd int
			if err == nil {
				portStart, portEnd, err = newP.Range()
			}
			if err != nil {
				return nil, fmt.Errorf("Error parsing HostPort value(%s):%v", binding[i].HostPort, err)
			}
			pbCopy.HostPort = uint16(portStart)
			pbCopy.HostPortEnd = uint16(portEnd)
			pbCopy.HostIP = net.ParseIP(binding[i].HostIP)
			pbList = append(pbList, pbCopy)
		}

		if container.HostConfig.PublishAllPorts && len(binding) == 0 {
			pbList = append(pbList, pb)
		}
	}

	sboxOptions = append(sboxOptions,
		libnetwork.OptionPortMapping(pbList),
		libnetwork.OptionExposedPorts(exposeList))

	// Legacy Link feature is supported only for the default bridge network.
	// return if this call to build join options is not for default bridge network
	// Legacy Link is only supported by docker run --link
	bridgeSettings, ok := container.NetworkSettings.Networks[defaultNetName]
	if !ok || bridgeSettings.EndpointSettings == nil {
		return sboxOptions, nil
	}

	if bridgeSettings.EndpointID == "" {
		return sboxOptions, nil
	}

	var (
		childEndpoints, parentEndpoints []string
		cEndpointID                     string
	)

	children := daemon.children(container)
	for linkAlias, child := range children {
		if !isLinkable(child) {
			return nil, fmt.Errorf("Cannot link to %s, as it does not belong to the default network", child.Name)
		}
		_, alias := path.Split(linkAlias)
		// allow access to the linked container via the alias, real name, and container hostname
		aliasList := alias + " " + child.Config.Hostname
		// only add the name if alias isn't equal to the name
		if alias != child.Name[1:] {
			aliasList = aliasList + " " + child.Name[1:]
		}
		sboxOptions = append(sboxOptions, libnetwork.OptionExtraHost(aliasList, child.NetworkSettings.Networks[defaultNetName].IPAddress))
		cEndpointID = child.NetworkSettings.Networks[defaultNetName].EndpointID
		if cEndpointID != "" {
			childEndpoints = append(childEndpoints, cEndpointID)
		}
	}

	for alias, parent := range daemon.parents(container) {
		if daemon.configStore.DisableBridge || !container.HostConfig.NetworkMode.IsPrivate() {
			continue
		}

		_, alias = path.Split(alias)
		logrus.Debugf("Update /etc/hosts of %s for alias %s with ip %s", parent.ID, alias, bridgeSettings.IPAddress)
		sboxOptions = append(sboxOptions, libnetwork.OptionParentUpdate(
			parent.ID,
			alias,
			bridgeSettings.IPAddress,
		))
		if cEndpointID != "" {
			parentEndpoints = append(parentEndpoints, cEndpointID)
		}
	}

	linkOptions := options.Generic{
		netlabel.GenericData: options.Generic{
			"ParentEndpoints": parentEndpoints,
			"ChildEndpoints":  childEndpoints,
		},
	}

	sboxOptions = append(sboxOptions, libnetwork.OptionGeneric(linkOptions))
	return sboxOptions, nil
}

func (daemon *Daemon) updateNetworkSettings(container *container.Container, n libnetwork.Network, endpointConfig *networktypes.EndpointSettings) error {
	if container.NetworkSettings == nil {
		container.NetworkSettings = &network.Settings{Networks: make(map[string]*network.EndpointSettings)}
	}

	if !container.HostConfig.NetworkMode.IsHost() && containertypes.NetworkMode(n.Type()).IsHost() {
		return runconfig.ErrConflictHostNetwork
	}

	for s := range container.NetworkSettings.Networks {
		sn, err := daemon.FindNetwork(s)
		if err != nil {
			continue
		}

		if sn.Name() == n.Name() {
			// Avoid duplicate config
			return nil
		}
		if !containertypes.NetworkMode(sn.Type()).IsPrivate() ||
			!containertypes.NetworkMode(n.Type()).IsPrivate() {
			return runconfig.ErrConflictSharedNetwork
		}
		if containertypes.NetworkMode(sn.Name()).IsNone() ||
			containertypes.NetworkMode(n.Name()).IsNone() {
			return runconfig.ErrConflictNoNetwork
		}
	}

	if _, ok := container.NetworkSettings.Networks[n.Name()]; !ok {
		container.NetworkSettings.Networks[n.Name()] = &network.EndpointSettings{
			EndpointSettings: endpointConfig,
		}
	}

	return nil
}

func (daemon *Daemon) updateEndpointNetworkSettings(container *container.Container, n libnetwork.Network, ep libnetwork.Endpoint) error {
	if err := container.BuildEndpointInfo(n, ep); err != nil {
		return err
	}

	if container.HostConfig.NetworkMode == runconfig.DefaultDaemonNetworkMode() {
		container.NetworkSettings.Bridge = daemon.configStore.BridgeConfig.Iface
	}

	return nil
}

// UpdateNetwork is used to update the container's network (e.g. when linked containers
// get removed/unlinked).
func (daemon *Daemon) updateNetwork(container *container.Container) error {
	var (
		start = time.Now()
		ctrl  = daemon.netController
		sid   = container.NetworkSettings.SandboxID
	)

	sb, err := ctrl.SandboxByID(sid)
	if err != nil {
		return fmt.Errorf("error locating sandbox id %s: %v", sid, err)
	}

	// Find if container is connected to the default bridge network
	var n libnetwork.Network
	for name := range container.NetworkSettings.Networks {
		sn, err := daemon.FindNetwork(name)
		if err != nil {
			continue
		}
		if sn.Name() == runconfig.DefaultDaemonNetworkMode().NetworkName() {
			n = sn
			break
		}
	}

	if n == nil {
		// Not connected to the default bridge network; Nothing to do
		return nil
	}

	options, err := daemon.buildSandboxOptions(container)
	if err != nil {
		return fmt.Errorf("Update network failed: %v", err)
	}

	if err := sb.Refresh(options...); err != nil {
		return fmt.Errorf("Update network failed: Failure in refresh sandbox %s: %v", sid, err)
	}

	networkActions.WithValues("update").UpdateSince(start)

	return nil
}

func (daemon *Daemon) findAndAttachNetwork(container *container.Container, idOrName string, epConfig *networktypes.EndpointSettings) (libnetwork.Network, *networktypes.NetworkingConfig, error) {
	n, err := daemon.FindNetwork(idOrName)
	if err != nil {
		// We should always be able to find the network for a
		// managed container.
		if container.Managed {
			return nil, nil, err
		}
	}

	// If we found a network and if it is not dynamically created
	// we should never attempt to attach to that network here.
	if n != nil {
		if container.Managed || !n.Info().Dynamic() {
			return n, nil, nil
		}
	}

	var addresses []string
	if epConfig != nil && epConfig.IPAMConfig != nil {
		if epConfig.IPAMConfig.IPv4Address != "" {
			addresses = append(addresses, epConfig.IPAMConfig.IPv4Address)
		}

		if epConfig.IPAMConfig.IPv6Address != "" {
			addresses = append(addresses, epConfig.IPAMConfig.IPv6Address)
		}
	}

	var (
		config     *networktypes.NetworkingConfig
		retryCount int
	)

	for {
		// In all other cases, attempt to attach to the network to
		// trigger attachment in the swarm cluster manager.
		if daemon.clusterProvider != nil {
			var err error
			config, err = daemon.clusterProvider.AttachNetwork(idOrName, container.ID, addresses)
			if err != nil {
				return nil, nil, err
			}
		}

		n, err = daemon.FindNetwork(idOrName)
		if err != nil {
			if daemon.clusterProvider != nil {
				if err := daemon.clusterProvider.DetachNetwork(idOrName, container.ID); err != nil {
					logrus.Warnf("Could not rollback attachment for container %s to network %s: %v", container.ID, idOrName, err)
				}
			}

			// Retry network attach again if we failed to
			// find the network after successful
			// attachment because the only reason that
			// would happen is if some other container
			// attached to the swarm scope network went down
			// and removed the network while we were in
			// the process of attaching.
			if config != nil {
				if _, ok := err.(libnetwork.ErrNoSuchNetwork); ok {
					if retryCount >= 5 {
						return nil, nil, fmt.Errorf("could not find network %s after successful attachment", idOrName)
					}
					retryCount++
					continue
				}
			}

			return nil, nil, err
		}

		break
	}

	// This container has attachment to a swarm scope
	// network. Update the container network settings accordingly.
	container.NetworkSettings.HasSwarmEndpoint = true
	return n, config, nil
}

// updateContainerNetworkSettings updates the network settings
func (daemon *Daemon) updateContainerNetworkSettings(container *container.Container, endpointsConfig map[string]*networktypes.EndpointSettings) {
	var n libnetwork.Network

	mode := container.HostConfig.NetworkMode
	if container.Config.NetworkDisabled || mode.IsContainer() {
		return
	}

	networkName := mode.NetworkName()
	if mode.IsDefault() {
		networkName = daemon.netController.Config().Daemon.DefaultNetwork
	}

	if mode.IsUserDefined() {
		var err error

		n, err = daemon.FindNetwork(networkName)
		if err == nil {
			networkName = n.Name()
		}
	}

	if container.NetworkSettings == nil {
		container.NetworkSettings = &network.Settings{}
	}

	if len(endpointsConfig) > 0 {
		if container.NetworkSettings.Networks == nil {
			container.NetworkSettings.Networks = make(map[string]*network.EndpointSettings)
		}

		for name, epConfig := range endpointsConfig {
			container.NetworkSettings.Networks[name] = &network.EndpointSettings{
				EndpointSettings: epConfig,
			}
		}
	}

	if container.NetworkSettings.Networks == nil {
		container.NetworkSettings.Networks = make(map[string]*network.EndpointSettings)
		container.NetworkSettings.Networks[networkName] = &network.EndpointSettings{
			EndpointSettings: &networktypes.EndpointSettings{},
		}
	}

	// Convert any settings added by client in default name to
	// engine's default network name key
	if mode.IsDefault() {
		if nConf, ok := container.NetworkSettings.Networks[mode.NetworkName()]; ok {
			container.NetworkSettings.Networks[networkName] = nConf
			delete(container.NetworkSettings.Networks, mode.NetworkName())
		}
	}

	if !mode.IsUserDefined() {
		return
	}
	// Make sure to internally store the per network endpoint config by network name
	if _, ok := container.NetworkSettings.Networks[networkName]; ok {
		return
	}

	if n != nil {
		if nwConfig, ok := container.NetworkSettings.Networks[n.ID()]; ok {
			container.NetworkSettings.Networks[networkName] = nwConfig
			delete(container.NetworkSettings.Networks, n.ID())
			return
		}
	}
}

func (daemon *Daemon) allocateNetwork(container *container.Container) error {
	start := time.Now()
	controller := daemon.netController

	if daemon.netController == nil {
		return nil
	}

	// Cleanup any stale sandbox left over due to ungraceful daemon shutdown
	if err := controller.SandboxDestroy(container.ID); err != nil {
		logrus.Errorf("failed to cleanup up stale network sandbox for container %s", container.ID)
	}

	if container.Config.NetworkDisabled || container.HostConfig.NetworkMode.IsContainer() {
		return nil
	}

	updateSettings := false

	if len(container.NetworkSettings.Networks) == 0 {
		daemon.updateContainerNetworkSettings(container, nil)
		updateSettings = true
	}

	// always connect default network first since only default
	// network mode support link and we need do some setting
	// on sandbox initialize for link, but the sandbox only be initialized
	// on first network connecting.
	defaultNetName := runconfig.DefaultDaemonNetworkMode().NetworkName()
	if nConf, ok := container.NetworkSettings.Networks[defaultNetName]; ok {
		cleanOperationalData(nConf)
		if err := daemon.connectToNetwork(container, defaultNetName, nConf.EndpointSettings, updateSettings); err != nil {
			return err
		}

	}

	// the intermediate map is necessary because "connectToNetwork" modifies "container.NetworkSettings.Networks"
	networks := make(map[string]*network.EndpointSettings)
	for n, epConf := range container.NetworkSettings.Networks {
		if n == defaultNetName {
			continue
		}

		networks[n] = epConf
	}

	for netName, epConf := range networks {
		cleanOperationalData(epConf)
		if err := daemon.connectToNetwork(container, netName, epConf.EndpointSettings, updateSettings); err != nil {
			return err
		}
	}

	// If the container is not to be connected to any network,
	// create its network sandbox now if not present
	if len(networks) == 0 {
		if nil == daemon.getNetworkSandbox(container) {
			options, err := daemon.buildSandboxOptions(container)
			if err != nil {
				return err
			}
			sb, err := daemon.netController.NewSandbox(container.ID, options...)
			if err != nil {
				return err
			}
			container.UpdateSandboxNetworkSettings(sb)
			defer func() {
				if err != nil {
					sb.Delete()
				}
			}()
		}

	}

	if _, err := container.WriteHostConfig(); err != nil {
		return err
	}
	networkActions.WithValues("allocate").UpdateSince(start)
	return nil
}

func (daemon *Daemon) getNetworkSandbox(container *container.Container) libnetwork.Sandbox {
	var sb libnetwork.Sandbox
	daemon.netController.WalkSandboxes(func(s libnetwork.Sandbox) bool {
		if s.ContainerID() == container.ID {
			sb = s
			return true
		}
		return false
	})
	return sb
}

// hasUserDefinedIPAddress returns whether the passed endpoint configuration contains IP address configuration
func hasUserDefinedIPAddress(epConfig *networktypes.EndpointSettings) bool {
	return epConfig != nil && epConfig.IPAMConfig != nil && (len(epConfig.IPAMConfig.IPv4Address) > 0 || len(epConfig.IPAMConfig.IPv6Address) > 0)
}

// User specified ip address is acceptable only for networks with user specified subnets.
func validateNetworkingConfig(n libnetwork.Network, epConfig *networktypes.EndpointSettings) error {
	if n == nil || epConfig == nil {
		return nil
	}
	if !hasUserDefinedIPAddress(epConfig) {
		return nil
	}
	_, _, nwIPv4Configs, nwIPv6Configs := n.Info().IpamConfig()
	for _, s := range []struct {
		ipConfigured  bool
		subnetConfigs []*libnetwork.IpamConf
	}{
		{
			ipConfigured:  len(epConfig.IPAMConfig.IPv4Address) > 0,
			subnetConfigs: nwIPv4Configs,
		},
		{
			ipConfigured:  len(epConfig.IPAMConfig.IPv6Address) > 0,
			subnetConfigs: nwIPv6Configs,
		},
	} {
		if s.ipConfigured {
			foundSubnet := false
			for _, cfg := range s.subnetConfigs {
				if len(cfg.PreferredPool) > 0 {
					foundSubnet = true
					break
				}
			}
			if !foundSubnet {
				return runconfig.ErrUnsupportedNetworkNoSubnetAndIP
			}
		}
	}

	return nil
}

// cleanOperationalData resets the operational data from the passed endpoint settings
func cleanOperationalData(es *network.EndpointSettings) {
	es.EndpointID = ""
	es.Gateway = ""
	es.IPAddress = ""
	es.IPPrefixLen = 0
	es.IPv6Gateway = ""
	es.GlobalIPv6Address = ""
	es.GlobalIPv6PrefixLen = 0
	es.MacAddress = ""
	if es.IPAMOperational {
		es.IPAMConfig = nil
	}
}

func (daemon *Daemon) updateNetworkConfig(container *container.Container, n libnetwork.Network, endpointConfig *networktypes.EndpointSettings, updateSettings bool) error {

	if !containertypes.NetworkMode(n.Name()).IsUserDefined() {
		if hasUserDefinedIPAddress(endpointConfig) && !enableIPOnPredefinedNetwork() {
			return runconfig.ErrUnsupportedNetworkAndIP
		}
		if endpointConfig != nil && len(endpointConfig.Aliases) > 0 && !container.EnableServiceDiscoveryOnDefaultNetwork() {
			return runconfig.ErrUnsupportedNetworkAndAlias
		}
	} else {
		addShortID := true
		shortID := stringid.TruncateID(container.ID)
		for _, alias := range endpointConfig.Aliases {
			if alias == shortID {
				addShortID = false
				break
			}
		}
		if addShortID {
			endpointConfig.Aliases = append(endpointConfig.Aliases, shortID)
		}
	}

	if err := validateNetworkingConfig(n, endpointConfig); err != nil {
		return err
	}

	if updateSettings {
		if err := daemon.updateNetworkSettings(container, n, endpointConfig); err != nil {
			return err
		}
	}
	return nil
}

func (daemon *Daemon) connectToNetwork(container *container.Container, idOrName string, endpointConfig *networktypes.EndpointSettings, updateSettings bool) (err error) {
	start := time.Now()
	if container.HostConfig.NetworkMode.IsContainer() {
		return runconfig.ErrConflictSharedNetwork
	}
	if containertypes.NetworkMode(idOrName).IsBridge() &&
		daemon.configStore.DisableBridge {
		container.Config.NetworkDisabled = true
		return nil
	}
	if endpointConfig == nil {
		endpointConfig = &networktypes.EndpointSettings{}
	}

	n, config, err := daemon.findAndAttachNetwork(container, idOrName, endpointConfig)
	if err != nil {
		return err
	}
	if n == nil {
		return nil
	}

	var operIPAM bool
	if config != nil {
		if epConfig, ok := config.EndpointsConfig[n.Name()]; ok {
			if endpointConfig.IPAMConfig == nil ||
				(endpointConfig.IPAMConfig.IPv4Address == "" &&
					endpointConfig.IPAMConfig.IPv6Address == "" &&
					len(endpointConfig.IPAMConfig.LinkLocalIPs) == 0) {
				operIPAM = true
			}

			// copy IPAMConfig and NetworkID from epConfig via AttachNetwork
			endpointConfig.IPAMConfig = epConfig.IPAMConfig
			endpointConfig.NetworkID = epConfig.NetworkID
		}
	}

	err = daemon.updateNetworkConfig(container, n, endpointConfig, updateSettings)
	if err != nil {
		return err
	}

	controller := daemon.netController
	sb := daemon.getNetworkSandbox(container)
	createOptions, err := container.BuildCreateEndpointOptions(n, endpointConfig, sb, daemon.configStore.DNS)
	if err != nil {
		return err
	}

	endpointName := strings.TrimPrefix(container.Name, "/")
	ep, err := n.CreateEndpoint(endpointName, createOptions...)
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			if e := ep.Delete(false); e != nil {
				logrus.Warnf("Could not rollback container connection to network %s", idOrName)
			}
		}
	}()
	container.NetworkSettings.Networks[n.Name()] = &network.EndpointSettings{
		EndpointSettings: endpointConfig,
		IPAMOperational:  operIPAM,
	}
	if _, ok := container.NetworkSettings.Networks[n.ID()]; ok {
		delete(container.NetworkSettings.Networks, n.ID())
	}

	if err := daemon.updateEndpointNetworkSettings(container, n, ep); err != nil {
		return err
	}

	if sb == nil {
		options, err := daemon.buildSandboxOptions(container)
		if err != nil {
			return err
		}
		sb, err = controller.NewSandbox(container.ID, options...)
		if err != nil {
			return err
		}

		container.UpdateSandboxNetworkSettings(sb)
	}

	joinOptions, err := container.BuildJoinOptions(n)
	if err != nil {
		return err
	}

	if err := ep.Join(sb, joinOptions...); err != nil {
		return err
	}

	if !container.Managed {
		// add container name/alias to DNS
		if err := daemon.ActivateContainerServiceBinding(container.Name); err != nil {
			return fmt.Errorf("Activate container service binding for %s failed: %v", container.Name, err)
		}
	}

	if err := container.UpdateJoinInfo(n, ep); err != nil {
		return fmt.Errorf("Updating join info failed: %v", err)
	}

	container.NetworkSettings.Ports = getPortMapInfo(sb)

	daemon.LogNetworkEventWithAttributes(n, "connect", map[string]string{"container": container.ID})
	networkActions.WithValues("connect").UpdateSince(start)
	return nil
}

// ForceEndpointDelete deletes an endpoint from a network forcefully
func (daemon *Daemon) ForceEndpointDelete(name string, networkName string) error {
	n, err := daemon.FindNetwork(networkName)
	if err != nil {
		return err
	}

	ep, err := n.EndpointByName(name)
	if err != nil {
		return err
	}
	return ep.Delete(true)
}

func (daemon *Daemon) disconnectFromNetwork(container *container.Container, n libnetwork.Network, force bool) error {
	var (
		ep   libnetwork.Endpoint
		sbox libnetwork.Sandbox
	)

	s := func(current libnetwork.Endpoint) bool {
		epInfo := current.Info()
		if epInfo == nil {
			return false
		}
		if sb := epInfo.Sandbox(); sb != nil {
			if sb.ContainerID() == container.ID {
				ep = current
				sbox = sb
				return true
			}
		}
		return false
	}
	n.WalkEndpoints(s)

	if ep == nil && force {
		epName := strings.TrimPrefix(container.Name, "/")
		ep, err := n.EndpointByName(epName)
		if err != nil {
			return err
		}
		return ep.Delete(force)
	}

	if ep == nil {
		return fmt.Errorf("container %s is not connected to network %s", container.ID, n.Name())
	}

	if err := ep.Leave(sbox); err != nil {
		return fmt.Errorf("container %s failed to leave network %s: %v", container.ID, n.Name(), err)
	}

	container.NetworkSettings.Ports = getPortMapInfo(sbox)

	if err := ep.Delete(false); err != nil {
		return fmt.Errorf("endpoint delete failed for container %s on network %s: %v", container.ID, n.Name(), err)
	}

	delete(container.NetworkSettings.Networks, n.Name())

	daemon.tryDetachContainerFromClusterNetwork(n, container)

	return nil
}

func (daemon *Daemon) tryDetachContainerFromClusterNetwork(network libnetwork.Network, container *container.Container) {
	if daemon.clusterProvider != nil && network.Info().Dynamic() && !container.Managed {
		if err := daemon.clusterProvider.DetachNetwork(network.Name(), container.ID); err != nil {
			logrus.Warnf("error detaching from network %s: %v", network.Name(), err)
			if err := daemon.clusterProvider.DetachNetwork(network.ID(), container.ID); err != nil {
				logrus.Warnf("error detaching from network %s: %v", network.ID(), err)
			}
		}
	}
	attributes := map[string]string{
		"container": container.ID,
	}
	daemon.LogNetworkEventWithAttributes(network, "disconnect", attributes)
}

func (daemon *Daemon) initializeNetworking(container *container.Container) error {
	var err error

	if container.HostConfig.NetworkMode.IsContainer() {
		// we need to get the hosts files from the container to join
		nc, err := daemon.getNetworkedContainer(container.ID, container.HostConfig.NetworkMode.ConnectedContainer())
		if err != nil {
			return err
		}

		err = daemon.initializeNetworkingPaths(container, nc)
		if err != nil {
			return err
		}

		container.Config.Hostname = nc.Config.Hostname
		container.Config.Domainname = nc.Config.Domainname
		return nil
	}

	if container.HostConfig.NetworkMode.IsHost() {
		if container.Config.Hostname == "" {
			container.Config.Hostname, err = os.Hostname()
			if err != nil {
				return err
			}
		}
	}

	if err := daemon.allocateNetwork(container); err != nil {
		return err
	}

	return container.BuildHostnameFile()
}

func (daemon *Daemon) getNetworkedContainer(containerID, connectedContainerID string) (*container.Container, error) {
	nc, err := daemon.GetContainer(connectedContainerID)
	if err != nil {
		return nil, err
	}
	if containerID == nc.ID {
		return nil, fmt.Errorf("cannot join own network")
	}
	if !nc.IsRunning() {
		err := fmt.Errorf("cannot join network of a non running container: %s", connectedContainerID)
		return nil, stateConflictError{err}
	}
	if nc.IsRestarting() {
		return nil, errContainerIsRestarting(connectedContainerID)
	}
	return nc, nil
}

func (daemon *Daemon) releaseNetwork(container *container.Container) {
	start := time.Now()
	if daemon.netController == nil {
		return
	}
	if container.HostConfig.NetworkMode.IsContainer() || container.Config.NetworkDisabled {
		return
	}

	sid := container.NetworkSettings.SandboxID
	settings := container.NetworkSettings.Networks
	container.NetworkSettings.Ports = nil

	if sid == "" {
		return
	}

	var networks []libnetwork.Network
	for n, epSettings := range settings {
		if nw, err := daemon.FindNetwork(n); err == nil {
			networks = append(networks, nw)
		}

		if epSettings.EndpointSettings == nil {
			continue
		}

		cleanOperationalData(epSettings)
	}

	sb, err := daemon.netController.SandboxByID(sid)
	if err != nil {
		logrus.Warnf("error locating sandbox id %s: %v", sid, err)
		return
	}

	if err := sb.Delete(); err != nil {
		logrus.Errorf("Error deleting sandbox id %s for container %s: %v", sid, container.ID, err)
	}

	for _, nw := range networks {
		daemon.tryDetachContainerFromClusterNetwork(nw, container)
	}
	networkActions.WithValues("release").UpdateSince(start)
}

func errRemovalContainer(containerID string) error {
	return fmt.Errorf("Container %s is marked for removal and cannot be connected or disconnected to the network", containerID)
}

// ConnectToNetwork connects a container to a network
func (daemon *Daemon) ConnectToNetwork(container *container.Container, idOrName string, endpointConfig *networktypes.EndpointSettings) error {
	if endpointConfig == nil {
		endpointConfig = &networktypes.EndpointSettings{}
	}
	container.Lock()
	defer container.Unlock()

	if !container.Running {
		if container.RemovalInProgress || container.Dead {
			return errRemovalContainer(container.ID)
		}

		n, err := daemon.FindNetwork(idOrName)
		if err == nil && n != nil {
			if err := daemon.updateNetworkConfig(container, n, endpointConfig, true); err != nil {
				return err
			}
		} else {
			container.NetworkSettings.Networks[idOrName] = &network.EndpointSettings{
				EndpointSettings: endpointConfig,
			}
		}
	} else if !daemon.isNetworkHotPluggable() {
		return fmt.Errorf(runtime.GOOS + " does not support connecting a running container to a network")
	} else {
		if err := daemon.connectToNetwork(container, idOrName, endpointConfig, true); err != nil {
			return err
		}
	}

	return container.CheckpointTo(daemon.containersReplica)
}

// DisconnectFromNetwork disconnects container from network n.
func (daemon *Daemon) DisconnectFromNetwork(container *container.Container, networkName string, force bool) error {
	n, err := daemon.FindNetwork(networkName)
	container.Lock()
	defer container.Unlock()

	if !container.Running || (err != nil && force) {
		if container.RemovalInProgress || container.Dead {
			return errRemovalContainer(container.ID)
		}
		// In case networkName is resolved we will use n.Name()
		// this will cover the case where network id is passed.
		if n != nil {
			networkName = n.Name()
		}
		if _, ok := container.NetworkSettings.Networks[networkName]; !ok {
			return fmt.Errorf("container %s is not connected to the network %s", container.ID, networkName)
		}
		delete(container.NetworkSettings.Networks, networkName)
	} else if err == nil && !daemon.isNetworkHotPluggable() {
		return fmt.Errorf(runtime.GOOS + " does not support connecting a running container to a network")
	} else if err == nil {
		if container.HostConfig.NetworkMode.IsHost() && containertypes.NetworkMode(n.Type()).IsHost() {
			return runconfig.ErrConflictHostNetwork
		}

		if err := daemon.disconnectFromNetwork(container, n, false); err != nil {
			return err
		}
	} else {
		return err
	}

	if err := container.CheckpointTo(daemon.containersReplica); err != nil {
		return err
	}

	if n != nil {
		daemon.LogNetworkEventWithAttributes(n, "disconnect", map[string]string{
			"container": container.ID,
		})
	}

	return nil
}

// ActivateContainerServiceBinding puts this container into load balancer active rotation and DNS response
func (daemon *Daemon) ActivateContainerServiceBinding(containerName string) error {
	container, err := daemon.GetContainer(containerName)
	if err != nil {
		return err
	}
	sb := daemon.getNetworkSandbox(container)
	if sb == nil {
		return fmt.Errorf("network sandbox does not exist for container %s", containerName)
	}
	return sb.EnableService()
}

// DeactivateContainerServiceBinding removes this container from load balancer active rotation, and DNS response
func (daemon *Daemon) DeactivateContainerServiceBinding(containerName string) error {
	container, err := daemon.GetContainer(containerName)
	if err != nil {
		return err
	}
	sb := daemon.getNetworkSandbox(container)
	if sb == nil {
		// If the network sandbox is not found, then there is nothing to deactivate
		logrus.Debugf("Could not find network sandbox for container %s on service binding deactivation request", containerName)
		return nil
	}
	return sb.DisableService()
}
