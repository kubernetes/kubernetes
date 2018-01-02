package container

import (
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	enginecontainer "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/events"
	"github.com/docker/docker/api/types/filters"
	enginemount "github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/network"
	volumetypes "github.com/docker/docker/api/types/volume"
	"github.com/docker/docker/daemon/cluster/convert"
	executorpkg "github.com/docker/docker/daemon/cluster/executor"
	clustertypes "github.com/docker/docker/daemon/cluster/provider"
	"github.com/docker/go-connections/nat"
	netconst "github.com/docker/libnetwork/datastore"
	"github.com/docker/swarmkit/agent/exec"
	"github.com/docker/swarmkit/api"
	"github.com/docker/swarmkit/api/genericresource"
	"github.com/docker/swarmkit/template"
	gogotypes "github.com/gogo/protobuf/types"
)

const (
	// Explicitly use the kernel's default setting for CPU quota of 100ms.
	// https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.txt
	cpuQuotaPeriod = 100 * time.Millisecond

	// systemLabelPrefix represents the reserved namespace for system labels.
	systemLabelPrefix = "com.docker.swarm"
)

// containerConfig converts task properties into docker container compatible
// components.
type containerConfig struct {
	task                *api.Task
	networksAttachments map[string]*api.NetworkAttachment
}

// newContainerConfig returns a validated container config. No methods should
// return an error if this function returns without error.
func newContainerConfig(t *api.Task) (*containerConfig, error) {
	var c containerConfig
	return &c, c.setTask(t)
}

func (c *containerConfig) setTask(t *api.Task) error {
	if t.Spec.GetContainer() == nil && t.Spec.GetAttachment() == nil {
		return exec.ErrRuntimeUnsupported
	}

	container := t.Spec.GetContainer()
	if container != nil {
		if container.Image == "" {
			return ErrImageRequired
		}

		if err := validateMounts(container.Mounts); err != nil {
			return err
		}
	}

	// index the networks by name
	c.networksAttachments = make(map[string]*api.NetworkAttachment, len(t.Networks))
	for _, attachment := range t.Networks {
		c.networksAttachments[attachment.Network.Spec.Annotations.Name] = attachment
	}

	c.task = t

	if t.Spec.GetContainer() != nil {
		preparedSpec, err := template.ExpandContainerSpec(nil, t)
		if err != nil {
			return err
		}
		c.task.Spec.Runtime = &api.TaskSpec_Container{
			Container: preparedSpec,
		}
	}

	return nil
}

func (c *containerConfig) networkAttachmentContainerID() string {
	attachment := c.task.Spec.GetAttachment()
	if attachment == nil {
		return ""
	}

	return attachment.ContainerID
}

func (c *containerConfig) taskID() string {
	return c.task.ID
}

func (c *containerConfig) endpoint() *api.Endpoint {
	return c.task.Endpoint
}

func (c *containerConfig) spec() *api.ContainerSpec {
	return c.task.Spec.GetContainer()
}

func (c *containerConfig) nameOrID() string {
	if c.task.Spec.GetContainer() != nil {
		return c.name()
	}

	return c.networkAttachmentContainerID()
}

func (c *containerConfig) name() string {
	if c.task.Annotations.Name != "" {
		// if set, use the container Annotations.Name field, set in the orchestrator.
		return c.task.Annotations.Name
	}

	slot := fmt.Sprint(c.task.Slot)
	if slot == "" || c.task.Slot == 0 {
		slot = c.task.NodeID
	}

	// fallback to service.slot.id.
	return fmt.Sprintf("%s.%s.%s", c.task.ServiceAnnotations.Name, slot, c.task.ID)
}

func (c *containerConfig) image() string {
	raw := c.spec().Image
	ref, err := reference.ParseNormalizedNamed(raw)
	if err != nil {
		return raw
	}
	return reference.FamiliarString(reference.TagNameOnly(ref))
}

func (c *containerConfig) portBindings() nat.PortMap {
	portBindings := nat.PortMap{}
	if c.task.Endpoint == nil {
		return portBindings
	}

	for _, portConfig := range c.task.Endpoint.Ports {
		if portConfig.PublishMode != api.PublishModeHost {
			continue
		}

		port := nat.Port(fmt.Sprintf("%d/%s", portConfig.TargetPort, strings.ToLower(portConfig.Protocol.String())))
		binding := []nat.PortBinding{
			{},
		}

		if portConfig.PublishedPort != 0 {
			binding[0].HostPort = strconv.Itoa(int(portConfig.PublishedPort))
		}
		portBindings[port] = binding
	}

	return portBindings
}

func (c *containerConfig) exposedPorts() map[nat.Port]struct{} {
	exposedPorts := make(map[nat.Port]struct{})
	if c.task.Endpoint == nil {
		return exposedPorts
	}

	for _, portConfig := range c.task.Endpoint.Ports {
		if portConfig.PublishMode != api.PublishModeHost {
			continue
		}

		port := nat.Port(fmt.Sprintf("%d/%s", portConfig.TargetPort, strings.ToLower(portConfig.Protocol.String())))
		exposedPorts[port] = struct{}{}
	}

	return exposedPorts
}

func (c *containerConfig) config() *enginecontainer.Config {
	genericEnvs := genericresource.EnvFormat(c.task.AssignedGenericResources, "DOCKER_RESOURCE")
	env := append(c.spec().Env, genericEnvs...)

	config := &enginecontainer.Config{
		Labels:       c.labels(),
		StopSignal:   c.spec().StopSignal,
		Tty:          c.spec().TTY,
		OpenStdin:    c.spec().OpenStdin,
		User:         c.spec().User,
		Env:          env,
		Hostname:     c.spec().Hostname,
		WorkingDir:   c.spec().Dir,
		Image:        c.image(),
		ExposedPorts: c.exposedPorts(),
		Healthcheck:  c.healthcheck(),
	}

	if len(c.spec().Command) > 0 {
		// If Command is provided, we replace the whole invocation with Command
		// by replacing Entrypoint and specifying Cmd. Args is ignored in this
		// case.
		config.Entrypoint = append(config.Entrypoint, c.spec().Command...)
		config.Cmd = append(config.Cmd, c.spec().Args...)
	} else if len(c.spec().Args) > 0 {
		// In this case, we assume the image has an Entrypoint and Args
		// specifies the arguments for that entrypoint.
		config.Cmd = c.spec().Args
	}

	return config
}

func (c *containerConfig) labels() map[string]string {
	var (
		system = map[string]string{
			"task":         "", // mark as cluster task
			"task.id":      c.task.ID,
			"task.name":    c.name(),
			"node.id":      c.task.NodeID,
			"service.id":   c.task.ServiceID,
			"service.name": c.task.ServiceAnnotations.Name,
		}
		labels = make(map[string]string)
	)

	// base labels are those defined in the spec.
	for k, v := range c.spec().Labels {
		labels[k] = v
	}

	// we then apply the overrides from the task, which may be set via the
	// orchestrator.
	for k, v := range c.task.Annotations.Labels {
		labels[k] = v
	}

	// finally, we apply the system labels, which override all labels.
	for k, v := range system {
		labels[strings.Join([]string{systemLabelPrefix, k}, ".")] = v
	}

	return labels
}

func (c *containerConfig) mounts() []enginemount.Mount {
	var r []enginemount.Mount
	for _, mount := range c.spec().Mounts {
		r = append(r, convertMount(mount))
	}
	return r
}

func convertMount(m api.Mount) enginemount.Mount {
	mount := enginemount.Mount{
		Source:   m.Source,
		Target:   m.Target,
		ReadOnly: m.ReadOnly,
	}

	switch m.Type {
	case api.MountTypeBind:
		mount.Type = enginemount.TypeBind
	case api.MountTypeVolume:
		mount.Type = enginemount.TypeVolume
	case api.MountTypeTmpfs:
		mount.Type = enginemount.TypeTmpfs
	}

	if m.BindOptions != nil {
		mount.BindOptions = &enginemount.BindOptions{}
		switch m.BindOptions.Propagation {
		case api.MountPropagationRPrivate:
			mount.BindOptions.Propagation = enginemount.PropagationRPrivate
		case api.MountPropagationPrivate:
			mount.BindOptions.Propagation = enginemount.PropagationPrivate
		case api.MountPropagationRSlave:
			mount.BindOptions.Propagation = enginemount.PropagationRSlave
		case api.MountPropagationSlave:
			mount.BindOptions.Propagation = enginemount.PropagationSlave
		case api.MountPropagationRShared:
			mount.BindOptions.Propagation = enginemount.PropagationRShared
		case api.MountPropagationShared:
			mount.BindOptions.Propagation = enginemount.PropagationShared
		}
	}

	if m.VolumeOptions != nil {
		mount.VolumeOptions = &enginemount.VolumeOptions{
			NoCopy: m.VolumeOptions.NoCopy,
		}
		if m.VolumeOptions.Labels != nil {
			mount.VolumeOptions.Labels = make(map[string]string, len(m.VolumeOptions.Labels))
			for k, v := range m.VolumeOptions.Labels {
				mount.VolumeOptions.Labels[k] = v
			}
		}
		if m.VolumeOptions.DriverConfig != nil {
			mount.VolumeOptions.DriverConfig = &enginemount.Driver{
				Name: m.VolumeOptions.DriverConfig.Name,
			}
			if m.VolumeOptions.DriverConfig.Options != nil {
				mount.VolumeOptions.DriverConfig.Options = make(map[string]string, len(m.VolumeOptions.DriverConfig.Options))
				for k, v := range m.VolumeOptions.DriverConfig.Options {
					mount.VolumeOptions.DriverConfig.Options[k] = v
				}
			}
		}
	}

	if m.TmpfsOptions != nil {
		mount.TmpfsOptions = &enginemount.TmpfsOptions{
			SizeBytes: m.TmpfsOptions.SizeBytes,
			Mode:      m.TmpfsOptions.Mode,
		}
	}

	return mount
}

func (c *containerConfig) healthcheck() *enginecontainer.HealthConfig {
	hcSpec := c.spec().Healthcheck
	if hcSpec == nil {
		return nil
	}
	interval, _ := gogotypes.DurationFromProto(hcSpec.Interval)
	timeout, _ := gogotypes.DurationFromProto(hcSpec.Timeout)
	startPeriod, _ := gogotypes.DurationFromProto(hcSpec.StartPeriod)
	return &enginecontainer.HealthConfig{
		Test:        hcSpec.Test,
		Interval:    interval,
		Timeout:     timeout,
		Retries:     int(hcSpec.Retries),
		StartPeriod: startPeriod,
	}
}

func (c *containerConfig) hostConfig() *enginecontainer.HostConfig {
	hc := &enginecontainer.HostConfig{
		Resources:      c.resources(),
		GroupAdd:       c.spec().Groups,
		PortBindings:   c.portBindings(),
		Mounts:         c.mounts(),
		ReadonlyRootfs: c.spec().ReadOnly,
	}

	if c.spec().DNSConfig != nil {
		hc.DNS = c.spec().DNSConfig.Nameservers
		hc.DNSSearch = c.spec().DNSConfig.Search
		hc.DNSOptions = c.spec().DNSConfig.Options
	}

	c.applyPrivileges(hc)

	// The format of extra hosts on swarmkit is specified in:
	// http://man7.org/linux/man-pages/man5/hosts.5.html
	//    IP_address canonical_hostname [aliases...]
	// However, the format of ExtraHosts in HostConfig is
	//    <host>:<ip>
	// We need to do the conversion here
	// (Alias is ignored for now)
	for _, entry := range c.spec().Hosts {
		parts := strings.Fields(entry)
		if len(parts) > 1 {
			hc.ExtraHosts = append(hc.ExtraHosts, fmt.Sprintf("%s:%s", parts[1], parts[0]))
		}
	}

	if c.task.LogDriver != nil {
		hc.LogConfig = enginecontainer.LogConfig{
			Type:   c.task.LogDriver.Name,
			Config: c.task.LogDriver.Options,
		}
	}

	if len(c.task.Networks) > 0 {
		labels := c.task.Networks[0].Network.Spec.Annotations.Labels
		name := c.task.Networks[0].Network.Spec.Annotations.Name
		if v, ok := labels["com.docker.swarm.predefined"]; ok && v == "true" {
			hc.NetworkMode = enginecontainer.NetworkMode(name)
		}
	}

	return hc
}

// This handles the case of volumes that are defined inside a service Mount
func (c *containerConfig) volumeCreateRequest(mount *api.Mount) *volumetypes.VolumesCreateBody {
	var (
		driverName string
		driverOpts map[string]string
		labels     map[string]string
	)

	if mount.VolumeOptions != nil && mount.VolumeOptions.DriverConfig != nil {
		driverName = mount.VolumeOptions.DriverConfig.Name
		driverOpts = mount.VolumeOptions.DriverConfig.Options
		labels = mount.VolumeOptions.Labels
	}

	if mount.VolumeOptions != nil {
		return &volumetypes.VolumesCreateBody{
			Name:       mount.Source,
			Driver:     driverName,
			DriverOpts: driverOpts,
			Labels:     labels,
		}
	}
	return nil
}

func (c *containerConfig) resources() enginecontainer.Resources {
	resources := enginecontainer.Resources{}

	// If no limits are specified let the engine use its defaults.
	//
	// TODO(aluzzardi): We might want to set some limits anyway otherwise
	// "unlimited" tasks will step over the reservation of other tasks.
	r := c.task.Spec.Resources
	if r == nil || r.Limits == nil {
		return resources
	}

	if r.Limits.MemoryBytes > 0 {
		resources.Memory = r.Limits.MemoryBytes
	}

	if r.Limits.NanoCPUs > 0 {
		// CPU Period must be set in microseconds.
		resources.CPUPeriod = int64(cpuQuotaPeriod / time.Microsecond)
		resources.CPUQuota = r.Limits.NanoCPUs * resources.CPUPeriod / 1e9
	}

	return resources
}

// Docker daemon supports just 1 network during container create.
func (c *containerConfig) createNetworkingConfig(b executorpkg.Backend) *network.NetworkingConfig {
	var networks []*api.NetworkAttachment
	if c.task.Spec.GetContainer() != nil || c.task.Spec.GetAttachment() != nil {
		networks = c.task.Networks
	}

	epConfig := make(map[string]*network.EndpointSettings)
	if len(networks) > 0 {
		epConfig[networks[0].Network.Spec.Annotations.Name] = getEndpointConfig(networks[0], b)
	}

	return &network.NetworkingConfig{EndpointsConfig: epConfig}
}

// TODO: Merge this function with createNetworkingConfig after daemon supports multiple networks in container create
func (c *containerConfig) connectNetworkingConfig(b executorpkg.Backend) *network.NetworkingConfig {
	var networks []*api.NetworkAttachment
	if c.task.Spec.GetContainer() != nil {
		networks = c.task.Networks
	}
	// First network is used during container create. Other networks are used in "docker network connect"
	if len(networks) < 2 {
		return nil
	}

	epConfig := make(map[string]*network.EndpointSettings)
	for _, na := range networks[1:] {
		epConfig[na.Network.Spec.Annotations.Name] = getEndpointConfig(na, b)
	}
	return &network.NetworkingConfig{EndpointsConfig: epConfig}
}

func getEndpointConfig(na *api.NetworkAttachment, b executorpkg.Backend) *network.EndpointSettings {
	var ipv4, ipv6 string
	for _, addr := range na.Addresses {
		ip, _, err := net.ParseCIDR(addr)
		if err != nil {
			continue
		}

		if ip.To4() != nil {
			ipv4 = ip.String()
			continue
		}

		if ip.To16() != nil {
			ipv6 = ip.String()
		}
	}

	n := &network.EndpointSettings{
		NetworkID: na.Network.ID,
		IPAMConfig: &network.EndpointIPAMConfig{
			IPv4Address: ipv4,
			IPv6Address: ipv6,
		},
		DriverOpts: na.DriverAttachmentOpts,
	}
	if v, ok := na.Network.Spec.Annotations.Labels["com.docker.swarm.predefined"]; ok && v == "true" {
		if ln, err := b.FindNetwork(na.Network.Spec.Annotations.Name); err == nil {
			n.NetworkID = ln.ID()
		}
	}
	return n
}

func (c *containerConfig) virtualIP(networkID string) string {
	if c.task.Endpoint == nil {
		return ""
	}

	for _, eVip := range c.task.Endpoint.VirtualIPs {
		// We only support IPv4 VIPs for now.
		if eVip.NetworkID == networkID {
			vip, _, err := net.ParseCIDR(eVip.Addr)
			if err != nil {
				return ""
			}

			return vip.String()
		}
	}

	return ""
}

func (c *containerConfig) serviceConfig() *clustertypes.ServiceConfig {
	if len(c.task.Networks) == 0 {
		return nil
	}

	logrus.Debugf("Creating service config in agent for t = %+v", c.task)
	svcCfg := &clustertypes.ServiceConfig{
		Name:             c.task.ServiceAnnotations.Name,
		Aliases:          make(map[string][]string),
		ID:               c.task.ServiceID,
		VirtualAddresses: make(map[string]*clustertypes.VirtualAddress),
	}

	for _, na := range c.task.Networks {
		svcCfg.VirtualAddresses[na.Network.ID] = &clustertypes.VirtualAddress{
			// We support only IPv4 virtual IP for now.
			IPv4: c.virtualIP(na.Network.ID),
		}
		if len(na.Aliases) > 0 {
			svcCfg.Aliases[na.Network.ID] = na.Aliases
		}
	}

	if c.task.Endpoint != nil {
		for _, ePort := range c.task.Endpoint.Ports {
			if ePort.PublishMode != api.PublishModeIngress {
				continue
			}

			svcCfg.ExposedPorts = append(svcCfg.ExposedPorts, &clustertypes.PortConfig{
				Name:          ePort.Name,
				Protocol:      int32(ePort.Protocol),
				TargetPort:    ePort.TargetPort,
				PublishedPort: ePort.PublishedPort,
			})
		}
	}

	return svcCfg
}

// networks returns a list of network names attached to the container. The
// returned name can be used to lookup the corresponding network create
// options.
func (c *containerConfig) networks() []string {
	var networks []string

	for name := range c.networksAttachments {
		networks = append(networks, name)
	}

	return networks
}

func (c *containerConfig) networkCreateRequest(name string) (clustertypes.NetworkCreateRequest, error) {
	na, ok := c.networksAttachments[name]
	if !ok {
		return clustertypes.NetworkCreateRequest{}, errors.New("container: unknown network referenced")
	}

	options := types.NetworkCreate{
		// ID:     na.Network.ID,
		Labels:         na.Network.Spec.Annotations.Labels,
		Internal:       na.Network.Spec.Internal,
		Attachable:     na.Network.Spec.Attachable,
		Ingress:        convert.IsIngressNetwork(na.Network),
		EnableIPv6:     na.Network.Spec.Ipv6Enabled,
		CheckDuplicate: true,
		Scope:          netconst.SwarmScope,
	}

	if na.Network.Spec.GetNetwork() != "" {
		options.ConfigFrom = &network.ConfigReference{
			Network: na.Network.Spec.GetNetwork(),
		}
	}

	if na.Network.DriverState != nil {
		options.Driver = na.Network.DriverState.Name
		options.Options = na.Network.DriverState.Options
	}
	if na.Network.IPAM != nil {
		options.IPAM = &network.IPAM{
			Driver:  na.Network.IPAM.Driver.Name,
			Options: na.Network.IPAM.Driver.Options,
		}
		for _, ic := range na.Network.IPAM.Configs {
			c := network.IPAMConfig{
				Subnet:  ic.Subnet,
				IPRange: ic.Range,
				Gateway: ic.Gateway,
			}
			options.IPAM.Config = append(options.IPAM.Config, c)
		}
	}

	return clustertypes.NetworkCreateRequest{
		ID: na.Network.ID,
		NetworkCreateRequest: types.NetworkCreateRequest{
			Name:          name,
			NetworkCreate: options,
		},
	}, nil
}

func (c *containerConfig) applyPrivileges(hc *enginecontainer.HostConfig) {
	privileges := c.spec().Privileges
	if privileges == nil {
		return
	}

	credentials := privileges.CredentialSpec
	if credentials != nil {
		switch credentials.Source.(type) {
		case *api.Privileges_CredentialSpec_File:
			hc.SecurityOpt = append(hc.SecurityOpt, "credentialspec=file://"+credentials.GetFile())
		case *api.Privileges_CredentialSpec_Registry:
			hc.SecurityOpt = append(hc.SecurityOpt, "credentialspec=registry://"+credentials.GetRegistry())
		}
	}

	selinux := privileges.SELinuxContext
	if selinux != nil {
		if selinux.Disable {
			hc.SecurityOpt = append(hc.SecurityOpt, "label=disable")
		}
		if selinux.User != "" {
			hc.SecurityOpt = append(hc.SecurityOpt, "label=user:"+selinux.User)
		}
		if selinux.Role != "" {
			hc.SecurityOpt = append(hc.SecurityOpt, "label=role:"+selinux.Role)
		}
		if selinux.Level != "" {
			hc.SecurityOpt = append(hc.SecurityOpt, "label=level:"+selinux.Level)
		}
		if selinux.Type != "" {
			hc.SecurityOpt = append(hc.SecurityOpt, "label=type:"+selinux.Type)
		}
	}
}

func (c containerConfig) eventFilter() filters.Args {
	filter := filters.NewArgs()
	filter.Add("type", events.ContainerEventType)
	filter.Add("name", c.name())
	filter.Add("label", fmt.Sprintf("%v.task.id=%v", systemLabelPrefix, c.task.ID))
	return filter
}
