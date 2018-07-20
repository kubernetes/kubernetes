package daemon

import (
	"fmt"
	"net"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/network"
	clustertypes "github.com/docker/docker/daemon/cluster/provider"
	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/runconfig"
	"github.com/docker/libnetwork"
	lncluster "github.com/docker/libnetwork/cluster"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/ipamapi"
	networktypes "github.com/docker/libnetwork/types"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

// NetworkControllerEnabled checks if the networking stack is enabled.
// This feature depends on OS primitives and it's disabled in systems like Windows.
func (daemon *Daemon) NetworkControllerEnabled() bool {
	return daemon.netController != nil
}

// FindNetwork function finds a network for a given string that can represent network name or id
func (daemon *Daemon) FindNetwork(idName string) (libnetwork.Network, error) {
	// 1. match by full ID.
	n, err := daemon.GetNetworkByID(idName)
	if err == nil || !isNoSuchNetworkError(err) {
		return n, err
	}

	// 2. match by full name
	n, err = daemon.GetNetworkByName(idName)
	if err == nil || !isNoSuchNetworkError(err) {
		return n, err
	}

	// 3. match by ID prefix
	list := daemon.GetNetworksByIDPrefix(idName)
	if len(list) == 0 {
		// Be very careful to change the error type here, the libnetwork.ErrNoSuchNetwork error is used by the controller
		// to retry the creation of the network as managed through the swarm manager
		return nil, errors.WithStack(notFound(libnetwork.ErrNoSuchNetwork(idName)))
	}
	if len(list) > 1 {
		return nil, errors.WithStack(invalidIdentifier(idName))
	}
	return list[0], nil
}

func isNoSuchNetworkError(err error) bool {
	_, ok := err.(libnetwork.ErrNoSuchNetwork)
	return ok
}

// GetNetworkByID function returns a network whose ID matches the given ID.
// It fails with an error if no matching network is found.
func (daemon *Daemon) GetNetworkByID(id string) (libnetwork.Network, error) {
	c := daemon.netController
	if c == nil {
		return nil, libnetwork.ErrNoSuchNetwork(id)
	}
	return c.NetworkByID(id)
}

// GetNetworkByName function returns a network for a given network name.
// If no network name is given, the default network is returned.
func (daemon *Daemon) GetNetworkByName(name string) (libnetwork.Network, error) {
	c := daemon.netController
	if c == nil {
		return nil, libnetwork.ErrNoSuchNetwork(name)
	}
	if name == "" {
		name = c.Config().Daemon.DefaultNetwork
	}
	return c.NetworkByName(name)
}

// GetNetworksByIDPrefix returns a list of networks whose ID partially matches zero or more networks
func (daemon *Daemon) GetNetworksByIDPrefix(partialID string) []libnetwork.Network {
	c := daemon.netController
	if c == nil {
		return nil
	}
	list := []libnetwork.Network{}
	l := func(nw libnetwork.Network) bool {
		if strings.HasPrefix(nw.ID(), partialID) {
			list = append(list, nw)
		}
		return false
	}
	c.WalkNetworks(l)

	return list
}

// getAllNetworks returns a list containing all networks
func (daemon *Daemon) getAllNetworks() []libnetwork.Network {
	return daemon.netController.Networks()
}

type ingressJob struct {
	create  *clustertypes.NetworkCreateRequest
	ip      net.IP
	jobDone chan struct{}
}

var (
	ingressWorkerOnce  sync.Once
	ingressJobsChannel chan *ingressJob
	ingressID          string
)

func (daemon *Daemon) startIngressWorker() {
	ingressJobsChannel = make(chan *ingressJob, 100)
	go func() {
		// nolint: gosimple
		for {
			select {
			case r := <-ingressJobsChannel:
				if r.create != nil {
					daemon.setupIngress(r.create, r.ip, ingressID)
					ingressID = r.create.ID
				} else {
					daemon.releaseIngress(ingressID)
					ingressID = ""
				}
				close(r.jobDone)
			}
		}
	}()
}

// enqueueIngressJob adds a ingress add/rm request to the worker queue.
// It guarantees the worker is started.
func (daemon *Daemon) enqueueIngressJob(job *ingressJob) {
	ingressWorkerOnce.Do(daemon.startIngressWorker)
	ingressJobsChannel <- job
}

// SetupIngress setups ingress networking.
// The function returns a channel which will signal the caller when the programming is completed.
func (daemon *Daemon) SetupIngress(create clustertypes.NetworkCreateRequest, nodeIP string) (<-chan struct{}, error) {
	ip, _, err := net.ParseCIDR(nodeIP)
	if err != nil {
		return nil, err
	}
	done := make(chan struct{})
	daemon.enqueueIngressJob(&ingressJob{&create, ip, done})
	return done, nil
}

// ReleaseIngress releases the ingress networking.
// The function returns a channel which will signal the caller when the programming is completed.
func (daemon *Daemon) ReleaseIngress() (<-chan struct{}, error) {
	done := make(chan struct{})
	daemon.enqueueIngressJob(&ingressJob{nil, nil, done})
	return done, nil
}

func (daemon *Daemon) setupIngress(create *clustertypes.NetworkCreateRequest, ip net.IP, staleID string) {
	controller := daemon.netController
	controller.AgentInitWait()

	if staleID != "" && staleID != create.ID {
		daemon.releaseIngress(staleID)
	}

	if _, err := daemon.createNetwork(create.NetworkCreateRequest, create.ID, true); err != nil {
		// If it is any other error other than already
		// exists error log error and return.
		if _, ok := err.(libnetwork.NetworkNameError); !ok {
			logrus.Errorf("Failed creating ingress network: %v", err)
			return
		}
		// Otherwise continue down the call to create or recreate sandbox.
	}

	_, err := daemon.GetNetworkByID(create.ID)
	if err != nil {
		logrus.Errorf("Failed getting ingress network by id after creating: %v", err)
	}
}

func (daemon *Daemon) releaseIngress(id string) {
	controller := daemon.netController

	if id == "" {
		return
	}

	n, err := controller.NetworkByID(id)
	if err != nil {
		logrus.Errorf("failed to retrieve ingress network %s: %v", id, err)
		return
	}

	if err := n.Delete(); err != nil {
		logrus.Errorf("Failed to delete ingress network %s: %v", n.ID(), err)
		return
	}
}

// SetNetworkBootstrapKeys sets the bootstrap keys.
func (daemon *Daemon) SetNetworkBootstrapKeys(keys []*networktypes.EncryptionKey) error {
	err := daemon.netController.SetKeys(keys)
	if err == nil {
		// Upon successful key setting dispatch the keys available event
		daemon.cluster.SendClusterEvent(lncluster.EventNetworkKeysAvailable)
	}
	return err
}

// UpdateAttachment notifies the attacher about the attachment config.
func (daemon *Daemon) UpdateAttachment(networkName, networkID, containerID string, config *network.NetworkingConfig) error {
	if daemon.clusterProvider == nil {
		return fmt.Errorf("cluster provider is not initialized")
	}

	if err := daemon.clusterProvider.UpdateAttachment(networkName, containerID, config); err != nil {
		return daemon.clusterProvider.UpdateAttachment(networkID, containerID, config)
	}

	return nil
}

// WaitForDetachment makes the cluster manager wait for detachment of
// the container from the network.
func (daemon *Daemon) WaitForDetachment(ctx context.Context, networkName, networkID, taskID, containerID string) error {
	if daemon.clusterProvider == nil {
		return fmt.Errorf("cluster provider is not initialized")
	}

	return daemon.clusterProvider.WaitForDetachment(ctx, networkName, networkID, taskID, containerID)
}

// CreateManagedNetwork creates an agent network.
func (daemon *Daemon) CreateManagedNetwork(create clustertypes.NetworkCreateRequest) error {
	_, err := daemon.createNetwork(create.NetworkCreateRequest, create.ID, true)
	return err
}

// CreateNetwork creates a network with the given name, driver and other optional parameters
func (daemon *Daemon) CreateNetwork(create types.NetworkCreateRequest) (*types.NetworkCreateResponse, error) {
	resp, err := daemon.createNetwork(create, "", false)
	if err != nil {
		return nil, err
	}
	return resp, err
}

func (daemon *Daemon) createNetwork(create types.NetworkCreateRequest, id string, agent bool) (*types.NetworkCreateResponse, error) {
	if runconfig.IsPreDefinedNetwork(create.Name) && !agent {
		err := fmt.Errorf("%s is a pre-defined network and cannot be created", create.Name)
		return nil, notAllowedError{err}
	}

	var warning string
	nw, err := daemon.GetNetworkByName(create.Name)
	if err != nil {
		if _, ok := err.(libnetwork.ErrNoSuchNetwork); !ok {
			return nil, err
		}
	}
	if nw != nil {
		// check if user defined CheckDuplicate, if set true, return err
		// otherwise prepare a warning message
		if create.CheckDuplicate {
			return nil, libnetwork.NetworkNameError(create.Name)
		}
		warning = fmt.Sprintf("Network with name %s (id : %s) already exists", nw.Name(), nw.ID())
	}

	c := daemon.netController
	driver := create.Driver
	if driver == "" {
		driver = c.Config().Daemon.DefaultDriver
	}

	nwOptions := []libnetwork.NetworkOption{
		libnetwork.NetworkOptionEnableIPv6(create.EnableIPv6),
		libnetwork.NetworkOptionDriverOpts(create.Options),
		libnetwork.NetworkOptionLabels(create.Labels),
		libnetwork.NetworkOptionAttachable(create.Attachable),
		libnetwork.NetworkOptionIngress(create.Ingress),
		libnetwork.NetworkOptionScope(create.Scope),
	}

	if create.ConfigOnly {
		nwOptions = append(nwOptions, libnetwork.NetworkOptionConfigOnly())
	}

	if create.IPAM != nil {
		ipam := create.IPAM
		v4Conf, v6Conf, err := getIpamConfig(ipam.Config)
		if err != nil {
			return nil, err
		}
		nwOptions = append(nwOptions, libnetwork.NetworkOptionIpam(ipam.Driver, "", v4Conf, v6Conf, ipam.Options))
	}

	if create.Internal {
		nwOptions = append(nwOptions, libnetwork.NetworkOptionInternalNetwork())
	}
	if agent {
		nwOptions = append(nwOptions, libnetwork.NetworkOptionDynamic())
		nwOptions = append(nwOptions, libnetwork.NetworkOptionPersist(false))
	}

	if create.ConfigFrom != nil {
		nwOptions = append(nwOptions, libnetwork.NetworkOptionConfigFrom(create.ConfigFrom.Network))
	}

	if agent && driver == "overlay" && (create.Ingress || runtime.GOOS == "windows") {
		nodeIP, exists := daemon.GetAttachmentStore().GetIPForNetwork(id)
		if !exists {
			return nil, fmt.Errorf("Failed to find a load balancer IP to use for network: %v", id)
		}

		nwOptions = append(nwOptions, libnetwork.NetworkOptionLBEndpoint(nodeIP))
	}

	n, err := c.NewNetwork(driver, create.Name, id, nwOptions...)
	if err != nil {
		if _, ok := err.(libnetwork.ErrDataStoreNotInitialized); ok {
			// nolint: golint
			return nil, errors.New("This node is not a swarm manager. Use \"docker swarm init\" or \"docker swarm join\" to connect this node to swarm and try again.")
		}
		return nil, err
	}

	daemon.pluginRefCount(driver, driverapi.NetworkPluginEndpointType, plugingetter.Acquire)
	if create.IPAM != nil {
		daemon.pluginRefCount(create.IPAM.Driver, ipamapi.PluginEndpointType, plugingetter.Acquire)
	}
	daemon.LogNetworkEvent(n, "create")

	return &types.NetworkCreateResponse{
		ID:      n.ID(),
		Warning: warning,
	}, nil
}

func (daemon *Daemon) pluginRefCount(driver, capability string, mode int) {
	var builtinDrivers []string

	if capability == driverapi.NetworkPluginEndpointType {
		builtinDrivers = daemon.netController.BuiltinDrivers()
	} else if capability == ipamapi.PluginEndpointType {
		builtinDrivers = daemon.netController.BuiltinIPAMDrivers()
	}

	for _, d := range builtinDrivers {
		if d == driver {
			return
		}
	}

	if daemon.PluginStore != nil {
		_, err := daemon.PluginStore.Get(driver, capability, mode)
		if err != nil {
			logrus.WithError(err).WithFields(logrus.Fields{"mode": mode, "driver": driver}).Error("Error handling plugin refcount operation")
		}
	}
}

func getIpamConfig(data []network.IPAMConfig) ([]*libnetwork.IpamConf, []*libnetwork.IpamConf, error) {
	ipamV4Cfg := []*libnetwork.IpamConf{}
	ipamV6Cfg := []*libnetwork.IpamConf{}
	for _, d := range data {
		iCfg := libnetwork.IpamConf{}
		iCfg.PreferredPool = d.Subnet
		iCfg.SubPool = d.IPRange
		iCfg.Gateway = d.Gateway
		iCfg.AuxAddresses = d.AuxAddress
		ip, _, err := net.ParseCIDR(d.Subnet)
		if err != nil {
			return nil, nil, fmt.Errorf("Invalid subnet %s : %v", d.Subnet, err)
		}
		if ip.To4() != nil {
			ipamV4Cfg = append(ipamV4Cfg, &iCfg)
		} else {
			ipamV6Cfg = append(ipamV6Cfg, &iCfg)
		}
	}
	return ipamV4Cfg, ipamV6Cfg, nil
}

// UpdateContainerServiceConfig updates a service configuration.
func (daemon *Daemon) UpdateContainerServiceConfig(containerName string, serviceConfig *clustertypes.ServiceConfig) error {
	container, err := daemon.GetContainer(containerName)
	if err != nil {
		return err
	}

	container.NetworkSettings.Service = serviceConfig
	return nil
}

// ConnectContainerToNetwork connects the given container to the given
// network. If either cannot be found, an err is returned. If the
// network cannot be set up, an err is returned.
func (daemon *Daemon) ConnectContainerToNetwork(containerName, networkName string, endpointConfig *network.EndpointSettings) error {
	container, err := daemon.GetContainer(containerName)
	if err != nil {
		return err
	}
	return daemon.ConnectToNetwork(container, networkName, endpointConfig)
}

// DisconnectContainerFromNetwork disconnects the given container from
// the given network. If either cannot be found, an err is returned.
func (daemon *Daemon) DisconnectContainerFromNetwork(containerName string, networkName string, force bool) error {
	container, err := daemon.GetContainer(containerName)
	if err != nil {
		if force {
			return daemon.ForceEndpointDelete(containerName, networkName)
		}
		return err
	}
	return daemon.DisconnectFromNetwork(container, networkName, force)
}

// GetNetworkDriverList returns the list of plugins drivers
// registered for network.
func (daemon *Daemon) GetNetworkDriverList() []string {
	if !daemon.NetworkControllerEnabled() {
		return nil
	}

	pluginList := daemon.netController.BuiltinDrivers()

	managedPlugins := daemon.PluginStore.GetAllManagedPluginsByCap(driverapi.NetworkPluginEndpointType)

	for _, plugin := range managedPlugins {
		pluginList = append(pluginList, plugin.Name())
	}

	pluginMap := make(map[string]bool)
	for _, plugin := range pluginList {
		pluginMap[plugin] = true
	}

	networks := daemon.netController.Networks()

	for _, network := range networks {
		if !pluginMap[network.Type()] {
			pluginList = append(pluginList, network.Type())
			pluginMap[network.Type()] = true
		}
	}

	sort.Strings(pluginList)

	return pluginList
}

// DeleteManagedNetwork deletes an agent network.
func (daemon *Daemon) DeleteManagedNetwork(networkID string) error {
	return daemon.deleteNetwork(networkID, true)
}

// DeleteNetwork destroys a network unless it's one of docker's predefined networks.
func (daemon *Daemon) DeleteNetwork(networkID string) error {
	return daemon.deleteNetwork(networkID, false)
}

func (daemon *Daemon) deleteNetwork(networkID string, dynamic bool) error {
	nw, err := daemon.FindNetwork(networkID)
	if err != nil {
		return err
	}

	if nw.Info().Ingress() {
		return nil
	}

	if runconfig.IsPreDefinedNetwork(nw.Name()) && !dynamic {
		err := fmt.Errorf("%s is a pre-defined network and cannot be removed", nw.Name())
		return notAllowedError{err}
	}

	if dynamic && !nw.Info().Dynamic() {
		if runconfig.IsPreDefinedNetwork(nw.Name()) {
			// Predefined networks now support swarm services. Make this
			// a no-op when cluster requests to remove the predefined network.
			return nil
		}
		err := fmt.Errorf("%s is not a dynamic network", nw.Name())
		return notAllowedError{err}
	}

	if err := nw.Delete(); err != nil {
		return err
	}

	// If this is not a configuration only network, we need to
	// update the corresponding remote drivers' reference counts
	if !nw.Info().ConfigOnly() {
		daemon.pluginRefCount(nw.Type(), driverapi.NetworkPluginEndpointType, plugingetter.Release)
		ipamType, _, _, _ := nw.Info().IpamConfig()
		daemon.pluginRefCount(ipamType, ipamapi.PluginEndpointType, plugingetter.Release)
		daemon.LogNetworkEvent(nw, "destroy")
	}

	return nil
}

// GetNetworks returns a list of all networks
func (daemon *Daemon) GetNetworks() []libnetwork.Network {
	return daemon.getAllNetworks()
}

// clearAttachableNetworks removes the attachable networks
// after disconnecting any connected container
func (daemon *Daemon) clearAttachableNetworks() {
	for _, n := range daemon.GetNetworks() {
		if !n.Info().Attachable() {
			continue
		}
		for _, ep := range n.Endpoints() {
			epInfo := ep.Info()
			if epInfo == nil {
				continue
			}
			sb := epInfo.Sandbox()
			if sb == nil {
				continue
			}
			containerID := sb.ContainerID()
			if err := daemon.DisconnectContainerFromNetwork(containerID, n.ID(), true); err != nil {
				logrus.Warnf("Failed to disconnect container %s from swarm network %s on cluster leave: %v",
					containerID, n.Name(), err)
			}
		}
		if err := daemon.DeleteManagedNetwork(n.ID()); err != nil {
			logrus.Warnf("Failed to remove swarm network %s on cluster leave: %v", n.Name(), err)
		}
	}
}
