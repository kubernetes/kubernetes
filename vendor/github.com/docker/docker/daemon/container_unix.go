// +build !windows

package daemon

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/daemon/network"
	"github.com/docker/docker/links"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/directory"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/nat"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/ulimit"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/utils"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/types"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/devices"
)

const DefaultPathEnv = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

type Container struct {
	CommonContainer

	// Fields below here are platform specific.

	AppArmorProfile string
	activeLinks     map[string]*links.Link
}

func killProcessDirectly(container *Container) error {
	if _, err := container.WaitStop(10 * time.Second); err != nil {
		// Ensure that we don't kill ourselves
		if pid := container.GetPid(); pid != 0 {
			logrus.Infof("Container %s failed to exit within 10 seconds of kill - trying direct SIGKILL", stringid.TruncateID(container.ID))
			if err := syscall.Kill(pid, 9); err != nil {
				if err != syscall.ESRCH {
					return err
				}
				logrus.Debugf("Cannot kill process (pid=%d) with signal 9: no such process.", pid)
			}
		}
	}
	return nil
}

func (container *Container) setupLinkedContainers() ([]string, error) {
	var (
		env    []string
		daemon = container.daemon
	)
	children, err := daemon.Children(container.Name)
	if err != nil {
		return nil, err
	}

	if len(children) > 0 {
		container.activeLinks = make(map[string]*links.Link, len(children))

		// If we encounter an error make sure that we rollback any network
		// config and iptables changes
		rollback := func() {
			for _, link := range container.activeLinks {
				link.Disable()
			}
			container.activeLinks = nil
		}

		for linkAlias, child := range children {
			if !child.IsRunning() {
				return nil, fmt.Errorf("Cannot link to a non running container: %s AS %s", child.Name, linkAlias)
			}

			link, err := links.NewLink(
				container.NetworkSettings.IPAddress,
				child.NetworkSettings.IPAddress,
				linkAlias,
				child.Config.Env,
				child.Config.ExposedPorts,
			)

			if err != nil {
				rollback()
				return nil, err
			}

			container.activeLinks[link.Alias()] = link
			if err := link.Enable(); err != nil {
				rollback()
				return nil, err
			}

			for _, envVar := range link.ToEnv() {
				env = append(env, envVar)
			}
		}
	}
	return env, nil
}

func (container *Container) createDaemonEnvironment(linkedEnv []string) []string {
	// if a domain name was specified, append it to the hostname (see #7851)
	fullHostname := container.Config.Hostname
	if container.Config.Domainname != "" {
		fullHostname = fmt.Sprintf("%s.%s", fullHostname, container.Config.Domainname)
	}
	// Setup environment
	env := []string{
		"PATH=" + DefaultPathEnv,
		"HOSTNAME=" + fullHostname,
		// Note: we don't set HOME here because it'll get autoset intelligently
		// based on the value of USER inside dockerinit, but only if it isn't
		// set already (ie, that can be overridden by setting HOME via -e or ENV
		// in a Dockerfile).
	}
	if container.Config.Tty {
		env = append(env, "TERM=xterm")
	}
	env = append(env, linkedEnv...)
	// because the env on the container can override certain default values
	// we need to replace the 'env' keys where they match and append anything
	// else.
	env = utils.ReplaceOrAppendEnvValues(env, container.Config.Env)

	return env
}

func getDevicesFromPath(deviceMapping runconfig.DeviceMapping) (devs []*configs.Device, err error) {
	device, err := devices.DeviceFromPath(deviceMapping.PathOnHost, deviceMapping.CgroupPermissions)
	// if there was no error, return the device
	if err == nil {
		device.Path = deviceMapping.PathInContainer
		return append(devs, device), nil
	}

	// if the device is not a device node
	// try to see if it's a directory holding many devices
	if err == devices.ErrNotADevice {

		// check if it is a directory
		if src, e := os.Stat(deviceMapping.PathOnHost); e == nil && src.IsDir() {

			// mount the internal devices recursively
			filepath.Walk(deviceMapping.PathOnHost, func(dpath string, f os.FileInfo, e error) error {
				childDevice, e := devices.DeviceFromPath(dpath, deviceMapping.CgroupPermissions)
				if e != nil {
					// ignore the device
					return nil
				}

				// add the device to userSpecified devices
				childDevice.Path = strings.Replace(dpath, deviceMapping.PathOnHost, deviceMapping.PathInContainer, 1)
				devs = append(devs, childDevice)

				return nil
			})
		}
	}

	if len(devs) > 0 {
		return devs, nil
	}

	return devs, fmt.Errorf("error gathering device information while adding custom device %q: %s", deviceMapping.PathOnHost, err)
}

func populateCommand(c *Container, env []string) error {
	var en *execdriver.Network
	if !c.Config.NetworkDisabled {
		en = &execdriver.Network{
			NamespacePath: c.NetworkSettings.SandboxKey,
		}

		parts := strings.SplitN(string(c.hostConfig.NetworkMode), ":", 2)
		if parts[0] == "container" {
			nc, err := c.getNetworkedContainer()
			if err != nil {
				return err
			}
			en.ContainerID = nc.ID
		}
	}

	ipc := &execdriver.Ipc{}

	if c.hostConfig.IpcMode.IsContainer() {
		ic, err := c.getIpcContainer()
		if err != nil {
			return err
		}
		ipc.ContainerID = ic.ID
	} else {
		ipc.HostIpc = c.hostConfig.IpcMode.IsHost()
	}

	pid := &execdriver.Pid{}
	pid.HostPid = c.hostConfig.PidMode.IsHost()

	uts := &execdriver.UTS{
		HostUTS: c.hostConfig.UTSMode.IsHost(),
	}

	// Build lists of devices allowed and created within the container.
	var userSpecifiedDevices []*configs.Device
	for _, deviceMapping := range c.hostConfig.Devices {
		devs, err := getDevicesFromPath(deviceMapping)
		if err != nil {
			return err
		}

		userSpecifiedDevices = append(userSpecifiedDevices, devs...)
	}

	allowedDevices := mergeDevices(configs.DefaultAllowedDevices, userSpecifiedDevices)

	autoCreatedDevices := mergeDevices(configs.DefaultAutoCreatedDevices, userSpecifiedDevices)

	// TODO: this can be removed after lxc-conf is fully deprecated
	lxcConfig, err := mergeLxcConfIntoOptions(c.hostConfig)
	if err != nil {
		return err
	}

	var rlimits []*ulimit.Rlimit
	ulimits := c.hostConfig.Ulimits

	// Merge ulimits with daemon defaults
	ulIdx := make(map[string]*ulimit.Ulimit)
	for _, ul := range ulimits {
		ulIdx[ul.Name] = ul
	}
	for name, ul := range c.daemon.config.Ulimits {
		if _, exists := ulIdx[name]; !exists {
			ulimits = append(ulimits, ul)
		}
	}

	for _, limit := range ulimits {
		rl, err := limit.GetRlimit()
		if err != nil {
			return err
		}
		rlimits = append(rlimits, rl)
	}

	resources := &execdriver.Resources{
		Memory:           c.hostConfig.Memory,
		MemorySwap:       c.hostConfig.MemorySwap,
		CpuShares:        c.hostConfig.CpuShares,
		CpusetCpus:       c.hostConfig.CpusetCpus,
		CpusetMems:       c.hostConfig.CpusetMems,
		CpuPeriod:        c.hostConfig.CpuPeriod,
		CpuQuota:         c.hostConfig.CpuQuota,
		BlkioWeight:      c.hostConfig.BlkioWeight,
		Rlimits:          rlimits,
		OomKillDisable:   c.hostConfig.OomKillDisable,
		MemorySwappiness: c.hostConfig.MemorySwappiness,
	}

	processConfig := execdriver.ProcessConfig{
		Privileged: c.hostConfig.Privileged,
		Entrypoint: c.Path,
		Arguments:  c.Args,
		Tty:        c.Config.Tty,
		User:       c.Config.User,
	}

	processConfig.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	processConfig.Env = env

	c.command = &execdriver.Command{
		ID:                 c.ID,
		Rootfs:             c.RootfsPath(),
		ReadonlyRootfs:     c.hostConfig.ReadonlyRootfs,
		InitPath:           "/.dockerinit",
		WorkingDir:         c.Config.WorkingDir,
		Network:            en,
		Ipc:                ipc,
		Pid:                pid,
		UTS:                uts,
		Resources:          resources,
		AllowedDevices:     allowedDevices,
		AutoCreatedDevices: autoCreatedDevices,
		CapAdd:             c.hostConfig.CapAdd.Slice(),
		CapDrop:            c.hostConfig.CapDrop.Slice(),
		GroupAdd:           c.hostConfig.GroupAdd,
		ProcessConfig:      processConfig,
		ProcessLabel:       c.GetProcessLabel(),
		MountLabel:         c.GetMountLabel(),
		LxcConfig:          lxcConfig,
		AppArmorProfile:    c.AppArmorProfile,
		CgroupParent:       c.hostConfig.CgroupParent,
	}

	return nil
}

func mergeDevices(defaultDevices, userDevices []*configs.Device) []*configs.Device {
	if len(userDevices) == 0 {
		return defaultDevices
	}

	paths := map[string]*configs.Device{}
	for _, d := range userDevices {
		paths[d.Path] = d
	}

	var devs []*configs.Device
	for _, d := range defaultDevices {
		if _, defined := paths[d.Path]; !defined {
			devs = append(devs, d)
		}
	}
	return append(devs, userDevices...)
}

// GetSize, return real size, virtual size
func (container *Container) GetSize() (int64, int64) {
	var (
		sizeRw, sizeRootfs int64
		err                error
		driver             = container.daemon.driver
	)

	if err := container.Mount(); err != nil {
		logrus.Errorf("Failed to compute size of container rootfs %s: %s", container.ID, err)
		return sizeRw, sizeRootfs
	}
	defer container.Unmount()

	initID := fmt.Sprintf("%s-init", container.ID)
	sizeRw, err = driver.DiffSize(container.ID, initID)
	if err != nil {
		logrus.Errorf("Driver %s couldn't return diff size of container %s: %s", driver, container.ID, err)
		// FIXME: GetSize should return an error. Not changing it now in case
		// there is a side-effect.
		sizeRw = -1
	}

	if _, err = os.Stat(container.basefs); err == nil {
		if sizeRootfs, err = directory.Size(container.basefs); err != nil {
			sizeRootfs = -1
		}
	}
	return sizeRw, sizeRootfs
}

func (container *Container) buildHostnameFile() error {
	hostnamePath, err := container.GetRootResourcePath("hostname")
	if err != nil {
		return err
	}
	container.HostnamePath = hostnamePath

	if container.Config.Domainname != "" {
		return ioutil.WriteFile(container.HostnamePath, []byte(fmt.Sprintf("%s.%s\n", container.Config.Hostname, container.Config.Domainname)), 0644)
	}
	return ioutil.WriteFile(container.HostnamePath, []byte(container.Config.Hostname+"\n"), 0644)
}

func (container *Container) buildJoinOptions() ([]libnetwork.EndpointOption, error) {
	var (
		joinOptions []libnetwork.EndpointOption
		err         error
		dns         []string
		dnsSearch   []string
	)

	joinOptions = append(joinOptions, libnetwork.JoinOptionHostname(container.Config.Hostname),
		libnetwork.JoinOptionDomainname(container.Config.Domainname))

	if container.hostConfig.NetworkMode.IsHost() {
		joinOptions = append(joinOptions, libnetwork.JoinOptionUseDefaultSandbox())
	}

	container.HostsPath, err = container.GetRootResourcePath("hosts")
	if err != nil {
		return nil, err
	}
	joinOptions = append(joinOptions, libnetwork.JoinOptionHostsPath(container.HostsPath))

	container.ResolvConfPath, err = container.GetRootResourcePath("resolv.conf")
	if err != nil {
		return nil, err
	}
	joinOptions = append(joinOptions, libnetwork.JoinOptionResolvConfPath(container.ResolvConfPath))

	if len(container.hostConfig.Dns) > 0 {
		dns = container.hostConfig.Dns
	} else if len(container.daemon.config.Dns) > 0 {
		dns = container.daemon.config.Dns
	}

	for _, d := range dns {
		joinOptions = append(joinOptions, libnetwork.JoinOptionDNS(d))
	}

	if len(container.hostConfig.DnsSearch) > 0 {
		dnsSearch = container.hostConfig.DnsSearch
	} else if len(container.daemon.config.DnsSearch) > 0 {
		dnsSearch = container.daemon.config.DnsSearch
	}

	for _, ds := range dnsSearch {
		joinOptions = append(joinOptions, libnetwork.JoinOptionDNSSearch(ds))
	}

	if container.NetworkSettings.SecondaryIPAddresses != nil {
		name := container.Config.Hostname
		if container.Config.Domainname != "" {
			name = name + "." + container.Config.Domainname
		}

		for _, a := range container.NetworkSettings.SecondaryIPAddresses {
			joinOptions = append(joinOptions, libnetwork.JoinOptionExtraHost(name, a.Addr))
		}
	}

	var childEndpoints, parentEndpoints []string

	children, err := container.daemon.Children(container.Name)
	if err != nil {
		return nil, err
	}

	for linkAlias, child := range children {
		_, alias := path.Split(linkAlias)
		// allow access to the linked container via the alias, real name, and container hostname
		aliasList := alias + " " + child.Config.Hostname
		// only add the name if alias isn't equal to the name
		if alias != child.Name[1:] {
			aliasList = aliasList + " " + child.Name[1:]
		}
		joinOptions = append(joinOptions, libnetwork.JoinOptionExtraHost(aliasList, child.NetworkSettings.IPAddress))
		if child.NetworkSettings.EndpointID != "" {
			childEndpoints = append(childEndpoints, child.NetworkSettings.EndpointID)
		}
	}

	for _, extraHost := range container.hostConfig.ExtraHosts {
		// allow IPv6 addresses in extra hosts; only split on first ":"
		parts := strings.SplitN(extraHost, ":", 2)
		joinOptions = append(joinOptions, libnetwork.JoinOptionExtraHost(parts[0], parts[1]))
	}

	refs := container.daemon.ContainerGraph().RefPaths(container.ID)
	for _, ref := range refs {
		if ref.ParentID == "0" {
			continue
		}

		c, err := container.daemon.Get(ref.ParentID)
		if err != nil {
			logrus.Error(err)
		}

		if c != nil && !container.daemon.config.DisableBridge && container.hostConfig.NetworkMode.IsPrivate() {
			logrus.Debugf("Update /etc/hosts of %s for alias %s with ip %s", c.ID, ref.Name, container.NetworkSettings.IPAddress)
			joinOptions = append(joinOptions, libnetwork.JoinOptionParentUpdate(c.NetworkSettings.EndpointID, ref.Name, container.NetworkSettings.IPAddress))
			if c.NetworkSettings.EndpointID != "" {
				parentEndpoints = append(parentEndpoints, c.NetworkSettings.EndpointID)
			}
		}
	}

	linkOptions := options.Generic{
		netlabel.GenericData: options.Generic{
			"ParentEndpoints": parentEndpoints,
			"ChildEndpoints":  childEndpoints,
		},
	}

	joinOptions = append(joinOptions, libnetwork.JoinOptionGeneric(linkOptions))

	return joinOptions, nil
}

func (container *Container) buildPortMapInfo(n libnetwork.Network, ep libnetwork.Endpoint, networkSettings *network.Settings) (*network.Settings, error) {
	if ep == nil {
		return nil, fmt.Errorf("invalid endpoint while building port map info")
	}

	if networkSettings == nil {
		return nil, fmt.Errorf("invalid networksettings while building port map info")
	}

	driverInfo, err := ep.DriverInfo()
	if err != nil {
		return nil, err
	}

	if driverInfo == nil {
		// It is not an error for epInfo to be nil
		return networkSettings, nil
	}

	if mac, ok := driverInfo[netlabel.MacAddress]; ok {
		networkSettings.MacAddress = mac.(net.HardwareAddr).String()
	}

	networkSettings.Ports = nat.PortMap{}

	if expData, ok := driverInfo[netlabel.ExposedPorts]; ok {
		if exposedPorts, ok := expData.([]types.TransportPort); ok {
			for _, tp := range exposedPorts {
				natPort, err := nat.NewPort(tp.Proto.String(), strconv.Itoa(int(tp.Port)))
				if err != nil {
					return nil, fmt.Errorf("Error parsing Port value(%s):%v", tp.Port, err)
				}
				networkSettings.Ports[natPort] = nil
			}
		}
	}

	mapData, ok := driverInfo[netlabel.PortMap]
	if !ok {
		return networkSettings, nil
	}

	if portMapping, ok := mapData.([]types.PortBinding); ok {
		for _, pp := range portMapping {
			natPort, err := nat.NewPort(pp.Proto.String(), strconv.Itoa(int(pp.Port)))
			if err != nil {
				return nil, err
			}
			natBndg := nat.PortBinding{HostIP: pp.HostIP.String(), HostPort: strconv.Itoa(int(pp.HostPort))}
			networkSettings.Ports[natPort] = append(networkSettings.Ports[natPort], natBndg)
		}
	}

	return networkSettings, nil
}

func (container *Container) buildEndpointInfo(n libnetwork.Network, ep libnetwork.Endpoint, networkSettings *network.Settings) (*network.Settings, error) {
	if ep == nil {
		return nil, fmt.Errorf("invalid endpoint while building port map info")
	}

	if networkSettings == nil {
		return nil, fmt.Errorf("invalid networksettings while building port map info")
	}

	epInfo := ep.Info()
	if epInfo == nil {
		// It is not an error to get an empty endpoint info
		return networkSettings, nil
	}

	ifaceList := epInfo.InterfaceList()
	if len(ifaceList) == 0 {
		return networkSettings, nil
	}

	iface := ifaceList[0]

	ones, _ := iface.Address().Mask.Size()
	networkSettings.IPAddress = iface.Address().IP.String()
	networkSettings.IPPrefixLen = ones

	if iface.AddressIPv6().IP.To16() != nil {
		onesv6, _ := iface.AddressIPv6().Mask.Size()
		networkSettings.GlobalIPv6Address = iface.AddressIPv6().IP.String()
		networkSettings.GlobalIPv6PrefixLen = onesv6
	}

	if len(ifaceList) == 1 {
		return networkSettings, nil
	}

	networkSettings.SecondaryIPAddresses = make([]network.Address, 0, len(ifaceList)-1)
	networkSettings.SecondaryIPv6Addresses = make([]network.Address, 0, len(ifaceList)-1)
	for _, iface := range ifaceList[1:] {
		ones, _ := iface.Address().Mask.Size()
		addr := network.Address{Addr: iface.Address().IP.String(), PrefixLen: ones}
		networkSettings.SecondaryIPAddresses = append(networkSettings.SecondaryIPAddresses, addr)

		if iface.AddressIPv6().IP.To16() != nil {
			onesv6, _ := iface.AddressIPv6().Mask.Size()
			addrv6 := network.Address{Addr: iface.AddressIPv6().IP.String(), PrefixLen: onesv6}
			networkSettings.SecondaryIPv6Addresses = append(networkSettings.SecondaryIPv6Addresses, addrv6)
		}
	}

	return networkSettings, nil
}

func (container *Container) updateJoinInfo(ep libnetwork.Endpoint) error {
	epInfo := ep.Info()
	if epInfo == nil {
		// It is not an error to get an empty endpoint info
		return nil
	}

	container.NetworkSettings.Gateway = epInfo.Gateway().String()
	if epInfo.GatewayIPv6().To16() != nil {
		container.NetworkSettings.IPv6Gateway = epInfo.GatewayIPv6().String()
	}

	container.NetworkSettings.SandboxKey = epInfo.SandboxKey()

	return nil
}

func (container *Container) updateNetworkSettings(n libnetwork.Network, ep libnetwork.Endpoint) error {
	networkSettings := &network.Settings{NetworkID: n.ID(), EndpointID: ep.ID()}

	networkSettings, err := container.buildPortMapInfo(n, ep, networkSettings)
	if err != nil {
		return err
	}

	networkSettings, err = container.buildEndpointInfo(n, ep, networkSettings)
	if err != nil {
		return err
	}

	if container.hostConfig.NetworkMode == runconfig.NetworkMode("bridge") {
		networkSettings.Bridge = container.daemon.config.Bridge.Iface
	}

	container.NetworkSettings = networkSettings
	return nil
}

func (container *Container) UpdateNetwork() error {
	n, err := container.daemon.netController.NetworkByID(container.NetworkSettings.NetworkID)
	if err != nil {
		return fmt.Errorf("error locating network id %s: %v", container.NetworkSettings.NetworkID, err)
	}

	ep, err := n.EndpointByID(container.NetworkSettings.EndpointID)
	if err != nil {
		return fmt.Errorf("error locating endpoint id %s: %v", container.NetworkSettings.EndpointID, err)
	}

	if err := ep.Leave(container.ID); err != nil {
		return fmt.Errorf("endpoint leave failed: %v", err)

	}

	joinOptions, err := container.buildJoinOptions()
	if err != nil {
		return fmt.Errorf("Update network failed: %v", err)
	}

	if err := ep.Join(container.ID, joinOptions...); err != nil {
		return fmt.Errorf("endpoint join failed: %v", err)
	}

	if err := container.updateJoinInfo(ep); err != nil {
		return fmt.Errorf("Updating join info failed: %v", err)
	}

	return nil
}

func (container *Container) buildCreateEndpointOptions() ([]libnetwork.EndpointOption, error) {
	var (
		portSpecs     = make(nat.PortSet)
		bindings      = make(nat.PortMap)
		pbList        []types.PortBinding
		exposeList    []types.TransportPort
		createOptions []libnetwork.EndpointOption
	)

	if container.Config.ExposedPorts != nil {
		portSpecs = container.Config.ExposedPorts
	}

	if container.hostConfig.PortBindings != nil {
		for p, b := range container.hostConfig.PortBindings {
			bindings[p] = []nat.PortBinding{}
			for _, bb := range b {
				bindings[p] = append(bindings[p], nat.PortBinding{
					HostIP:   bb.HostIP,
					HostPort: bb.HostPort,
				})
			}
		}
	}

	container.NetworkSettings.PortMapping = nil

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
			if err != nil {
				return nil, fmt.Errorf("Error parsing HostPort value(%s):%v", binding[i].HostPort, err)
			}
			pbCopy.HostPort = uint16(newP.Int())
			pbCopy.HostIP = net.ParseIP(binding[i].HostIP)
			pbList = append(pbList, pbCopy)
		}

		if container.hostConfig.PublishAllPorts && len(binding) == 0 {
			pbList = append(pbList, pb)
		}
	}

	createOptions = append(createOptions,
		libnetwork.CreateOptionPortMapping(pbList),
		libnetwork.CreateOptionExposedPorts(exposeList))

	if container.Config.MacAddress != "" {
		mac, err := net.ParseMAC(container.Config.MacAddress)
		if err != nil {
			return nil, err
		}

		genericOption := options.Generic{
			netlabel.MacAddress: mac,
		}

		createOptions = append(createOptions, libnetwork.EndpointOptionGeneric(genericOption))
	}

	return createOptions, nil
}

func parseService(controller libnetwork.NetworkController, service string) (string, string, string) {
	dn := controller.Config().Daemon.DefaultNetwork
	dd := controller.Config().Daemon.DefaultDriver

	snd := strings.Split(service, ".")
	if len(snd) > 2 {
		return strings.Join(snd[:len(snd)-2], "."), snd[len(snd)-2], snd[len(snd)-1]
	}
	if len(snd) > 1 {
		return snd[0], snd[1], dd
	}
	return snd[0], dn, dd
}

func createNetwork(controller libnetwork.NetworkController, dnet string, driver string) (libnetwork.Network, error) {
	createOptions := []libnetwork.NetworkOption{}
	genericOption := options.Generic{}

	// Bridge driver is special due to legacy reasons
	if runconfig.NetworkMode(driver).IsBridge() {
		genericOption[netlabel.GenericData] = map[string]interface{}{
			"BridgeName":            dnet,
			"AllowNonDefaultBridge": "true",
		}
		networkOption := libnetwork.NetworkOptionGeneric(genericOption)
		createOptions = append(createOptions, networkOption)
	}

	return controller.NewNetwork(driver, dnet, createOptions...)
}

func (container *Container) secondaryNetworkRequired(primaryNetworkType string) bool {
	switch primaryNetworkType {
	case "bridge", "none", "host", "container":
		return false
	}

	if container.daemon.config.DisableBridge {
		return false
	}

	if container.Config.ExposedPorts != nil && len(container.Config.ExposedPorts) > 0 {
		return true
	}
	if container.hostConfig.PortBindings != nil && len(container.hostConfig.PortBindings) > 0 {
		return true
	}
	return false
}

func (container *Container) AllocateNetwork() error {
	mode := container.hostConfig.NetworkMode
	controller := container.daemon.netController
	if container.Config.NetworkDisabled || mode.IsContainer() {
		return nil
	}

	networkDriver := string(mode)
	service := container.Config.PublishService
	networkName := mode.NetworkName()
	if mode.IsDefault() {
		if service != "" {
			service, networkName, networkDriver = parseService(controller, service)
		} else {
			networkName = controller.Config().Daemon.DefaultNetwork
			networkDriver = controller.Config().Daemon.DefaultDriver
		}
	} else if service != "" {
		return fmt.Errorf("conflicting options: publishing a service and network mode")
	}

	if runconfig.NetworkMode(networkDriver).IsBridge() && container.daemon.config.DisableBridge {
		container.Config.NetworkDisabled = true
		return nil
	}

	if service == "" {
		// dot character "." has a special meaning to support SERVICE[.NETWORK] format.
		// For backward compatiblity, replacing "." with "-", instead of failing
		service = strings.Replace(container.Name, ".", "-", -1)
		// Service names dont like "/" in them. removing it instead of failing for backward compatibility
		service = strings.Replace(service, "/", "", -1)
	}

	if container.secondaryNetworkRequired(networkDriver) {
		// Configure Bridge as secondary network for port binding purposes
		if err := container.configureNetwork("bridge", service, "bridge", false); err != nil {
			return err
		}
	}

	if err := container.configureNetwork(networkName, service, networkDriver, mode.IsDefault()); err != nil {
		return err
	}

	return container.WriteHostConfig()
}

func (container *Container) configureNetwork(networkName, service, networkDriver string, canCreateNetwork bool) error {
	controller := container.daemon.netController
	n, err := controller.NetworkByName(networkName)
	if err != nil {
		if _, ok := err.(libnetwork.ErrNoSuchNetwork); !ok || !canCreateNetwork {
			return err
		}

		if n, err = createNetwork(controller, networkName, networkDriver); err != nil {
			return err
		}
	}

	ep, err := n.EndpointByName(service)
	if err != nil {
		if _, ok := err.(libnetwork.ErrNoSuchEndpoint); !ok {
			return err
		}

		createOptions, err := container.buildCreateEndpointOptions()
		if err != nil {
			return err
		}

		ep, err = n.CreateEndpoint(service, createOptions...)
		if err != nil {
			return err
		}
	}

	if err := container.updateNetworkSettings(n, ep); err != nil {
		return err
	}

	joinOptions, err := container.buildJoinOptions()
	if err != nil {
		return err
	}

	if err := ep.Join(container.ID, joinOptions...); err != nil {
		return err
	}

	if err := container.updateJoinInfo(ep); err != nil {
		return fmt.Errorf("Updating join info failed: %v", err)
	}

	return nil
}

func (container *Container) initializeNetworking() error {
	var err error

	// Make sure NetworkMode has an acceptable value before
	// initializing networking.
	if container.hostConfig.NetworkMode == runconfig.NetworkMode("") {
		container.hostConfig.NetworkMode = runconfig.NetworkMode("default")
	}
	if container.hostConfig.NetworkMode.IsContainer() {
		// we need to get the hosts files from the container to join
		nc, err := container.getNetworkedContainer()
		if err != nil {
			return err
		}
		container.HostnamePath = nc.HostnamePath
		container.HostsPath = nc.HostsPath
		container.ResolvConfPath = nc.ResolvConfPath
		container.Config.Hostname = nc.Config.Hostname
		container.Config.Domainname = nc.Config.Domainname
		return nil
	}

	if container.hostConfig.NetworkMode.IsHost() {
		container.Config.Hostname, err = os.Hostname()
		if err != nil {
			return err
		}

		parts := strings.SplitN(container.Config.Hostname, ".", 2)
		if len(parts) > 1 {
			container.Config.Hostname = parts[0]
			container.Config.Domainname = parts[1]
		}

	}

	if err := container.AllocateNetwork(); err != nil {
		return err
	}

	return container.buildHostnameFile()
}

func (container *Container) ExportRw() (archive.Archive, error) {
	if container.daemon == nil {
		return nil, fmt.Errorf("Can't load storage driver for unregistered container %s", container.ID)
	}
	archive, err := container.daemon.Diff(container)
	if err != nil {
		return nil, err
	}
	return ioutils.NewReadCloserWrapper(archive, func() error {
			err := archive.Close()
			return err
		}),
		nil
}

func (container *Container) getIpcContainer() (*Container, error) {
	containerID := container.hostConfig.IpcMode.Container()
	c, err := container.daemon.Get(containerID)
	if err != nil {
		return nil, err
	}
	if !c.IsRunning() {
		return nil, fmt.Errorf("cannot join IPC of a non running container: %s", containerID)
	}
	return c, nil
}

func (container *Container) setupWorkingDirectory() error {
	if container.Config.WorkingDir != "" {
		container.Config.WorkingDir = filepath.Clean(container.Config.WorkingDir)

		pth, err := container.GetResourcePath(container.Config.WorkingDir)
		if err != nil {
			return err
		}

		pthInfo, err := os.Stat(pth)
		if err != nil {
			if !os.IsNotExist(err) {
				return err
			}

			if err := system.MkdirAll(pth, 0755); err != nil {
				return err
			}
		}
		if pthInfo != nil && !pthInfo.IsDir() {
			return fmt.Errorf("Cannot mkdir: %s is not a directory", container.Config.WorkingDir)
		}
	}
	return nil
}

func (container *Container) getNetworkedContainer() (*Container, error) {
	parts := strings.SplitN(string(container.hostConfig.NetworkMode), ":", 2)
	switch parts[0] {
	case "container":
		if len(parts) != 2 {
			return nil, fmt.Errorf("no container specified to join network")
		}
		nc, err := container.daemon.Get(parts[1])
		if err != nil {
			return nil, err
		}
		if container == nc {
			return nil, fmt.Errorf("cannot join own network")
		}
		if !nc.IsRunning() {
			return nil, fmt.Errorf("cannot join network of a non running container: %s", parts[1])
		}
		return nc, nil
	default:
		return nil, fmt.Errorf("network mode not set to container")
	}
}

func (container *Container) ReleaseNetwork() {
	if container.hostConfig.NetworkMode.IsContainer() || container.Config.NetworkDisabled {
		return
	}

	eid := container.NetworkSettings.EndpointID
	nid := container.NetworkSettings.NetworkID

	container.NetworkSettings = &network.Settings{}

	if nid == "" || eid == "" {
		return
	}

	n, err := container.daemon.netController.NetworkByID(nid)
	if err != nil {
		logrus.Errorf("error locating network id %s: %v", nid, err)
		return
	}

	ep, err := n.EndpointByID(eid)
	if err != nil {
		logrus.Errorf("error locating endpoint id %s: %v", eid, err)
		return
	}

	switch {
	case container.hostConfig.NetworkMode.IsHost():
		if err := ep.Leave(container.ID); err != nil {
			logrus.Errorf("Error leaving endpoint id %s for container %s: %v", eid, container.ID, err)
			return
		}
	default:
		if err := container.daemon.netController.LeaveAll(container.ID); err != nil {
			logrus.Errorf("Leave all failed for  %s: %v", container.ID, err)
			return
		}
	}

	// In addition to leaving all endpoints, delete implicitly created endpoint
	if container.Config.PublishService == "" {
		if err := ep.Delete(); err != nil {
			logrus.Errorf("deleting endpoint failed: %v", err)
		}
	}

}

func disableAllActiveLinks(container *Container) {
	if container.activeLinks != nil {
		for _, link := range container.activeLinks {
			link.Disable()
		}
	}
}

func (container *Container) DisableLink(name string) {
	if container.activeLinks != nil {
		if link, exists := container.activeLinks[name]; exists {
			link.Disable()
			delete(container.activeLinks, name)
			if err := container.UpdateNetwork(); err != nil {
				logrus.Debugf("Could not update network to remove link: %v", err)
			}
		} else {
			logrus.Debugf("Could not find active link for %s", name)
		}
	}
}

func (container *Container) UnmountVolumes(forceSyscall bool) error {
	var volumeMounts []mountPoint

	for _, mntPoint := range container.MountPoints {
		dest, err := container.GetResourcePath(mntPoint.Destination)
		if err != nil {
			return err
		}

		volumeMounts = append(volumeMounts, mountPoint{Destination: dest, Volume: mntPoint.Volume})
	}

	for _, mnt := range container.networkMounts() {
		dest, err := container.GetResourcePath(mnt.Destination)
		if err != nil {
			return err
		}

		volumeMounts = append(volumeMounts, mountPoint{Destination: dest})
	}

	for _, volumeMount := range volumeMounts {
		if forceSyscall {
			syscall.Unmount(volumeMount.Destination, 0)
		}

		if volumeMount.Volume != nil {
			if err := volumeMount.Volume.Unmount(); err != nil {
				return err
			}
		}
	}

	return nil
}

func (container *Container) PrepareStorage() error {
	return nil
}

func (container *Container) CleanupStorage() error {
	return nil
}
