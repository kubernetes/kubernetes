package daemon

import (
	"fmt"
	"path/filepath"
	"sort"
	"strconv"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/oci"
	"github.com/docker/libnetwork"
	"github.com/opencontainers/runtime-spec/specs-go"
)

func setResources(s *specs.Spec, r containertypes.Resources) error {
	mem := getMemoryResources(r)
	s.Solaris.CappedMemory = &mem

	capCPU := getCPUResources(r)
	s.Solaris.CappedCPU = &capCPU

	return nil
}

func setUser(s *specs.Spec, c *container.Container) error {
	uid, gid, additionalGids, err := getUser(c, c.Config.User)
	if err != nil {
		return err
	}
	s.Process.User.UID = uid
	s.Process.User.GID = gid
	s.Process.User.AdditionalGids = additionalGids
	return nil
}

func getUser(c *container.Container, username string) (uint32, uint32, []uint32, error) {
	return 0, 0, nil, nil
}

func (daemon *Daemon) getRunzAnet(ep libnetwork.Endpoint) (specs.Anet, error) {
	var (
		linkName  string
		lowerLink string
		defRouter string
	)

	epInfo := ep.Info()
	if epInfo == nil {
		return specs.Anet{}, fmt.Errorf("invalid endpoint")
	}

	nw, err := daemon.GetNetworkByName(ep.Network())
	if err != nil {
		return specs.Anet{}, fmt.Errorf("Failed to get network %s: %v", ep.Network(), err)
	}

	// Evaluate default router, linkname and lowerlink for interface endpoint
	switch nw.Type() {
	case "bridge":
		defRouter = epInfo.Gateway().String()
		linkName = "net0" // Should always be net0 for a container

		// TODO We construct lowerlink here exactly as done for solaris bridge
		// initialization. Need modular code to reuse.
		options := nw.Info().DriverOptions()
		nwName := options["com.docker.network.bridge.name"]
		lastChar := nwName[len(nwName)-1:]
		if _, err = strconv.Atoi(lastChar); err != nil {
			lowerLink = nwName + "_0"
		} else {
			lowerLink = nwName
		}

	case "overlay":
		defRouter = ""
		linkName = "net1"

		// TODO Follows generateVxlanName() in solaris overlay.
		id := nw.ID()
		if len(nw.ID()) > 12 {
			id = nw.ID()[:12]
		}
		lowerLink = "vx_" + id + "_0"
	}

	runzanet := specs.Anet{
		Linkname:          linkName,
		Lowerlink:         lowerLink,
		Allowedaddr:       epInfo.Iface().Address().String(),
		Configallowedaddr: "true",
		Defrouter:         defRouter,
		Linkprotection:    "mac-nospoof, ip-nospoof",
		Macaddress:        epInfo.Iface().MacAddress().String(),
	}

	return runzanet, nil
}

func (daemon *Daemon) setNetworkInterface(s *specs.Spec, c *container.Container) error {
	var anets []specs.Anet

	sb, err := daemon.netController.SandboxByID(c.NetworkSettings.SandboxID)
	if err != nil {
		return fmt.Errorf("Could not obtain sandbox for container")
	}

	// Populate interfaces required for each endpoint
	for _, ep := range sb.Endpoints() {
		runzanet, err := daemon.getRunzAnet(ep)
		if err != nil {
			return fmt.Errorf("Failed to get interface information for endpoint %d: %v", ep.ID(), err)
		}
		anets = append(anets, runzanet)
	}

	s.Solaris.Anet = anets
	if anets != nil {
		s.Solaris.Milestone = "svc:/milestone/container:default"
	}
	return nil
}

func (daemon *Daemon) populateCommonSpec(s *specs.Spec, c *container.Container) error {
	linkedEnv, err := daemon.setupLinkedContainers(c)
	if err != nil {
		return err
	}
	s.Root = specs.Root{
		Path:     filepath.Dir(c.BaseFS),
		Readonly: c.HostConfig.ReadonlyRootfs,
	}
	if err := c.SetupWorkingDirectory(daemon.idMappings.RootPair()); err != nil {
		return err
	}
	cwd := c.Config.WorkingDir
	s.Process.Args = append([]string{c.Path}, c.Args...)
	s.Process.Cwd = cwd
	s.Process.Env = c.CreateDaemonEnvironment(c.Config.Tty, linkedEnv)
	s.Process.Terminal = c.Config.Tty
	s.Hostname = c.FullHostname()

	return nil
}

func (daemon *Daemon) createSpec(c *container.Container) (*specs.Spec, error) {
	s := oci.DefaultSpec()
	if err := daemon.populateCommonSpec(&s, c); err != nil {
		return nil, err
	}

	if err := setResources(&s, c.HostConfig.Resources); err != nil {
		return nil, fmt.Errorf("runtime spec resources: %v", err)
	}

	if err := setUser(&s, c); err != nil {
		return nil, fmt.Errorf("spec user: %v", err)
	}

	if err := daemon.setNetworkInterface(&s, c); err != nil {
		return nil, err
	}

	if err := daemon.setupIpcDirs(c); err != nil {
		return nil, err
	}

	ms, err := daemon.setupMounts(c)
	if err != nil {
		return nil, err
	}
	ms = append(ms, c.IpcMounts()...)
	tmpfsMounts, err := c.TmpfsMounts()
	if err != nil {
		return nil, err
	}
	ms = append(ms, tmpfsMounts...)
	sort.Sort(mounts(ms))

	return (*specs.Spec)(&s), nil
}

// mergeUlimits merge the Ulimits from HostConfig with daemon defaults, and update HostConfig
// It will do nothing on non-Linux platform
func (daemon *Daemon) mergeUlimits(c *containertypes.HostConfig) {
	return
}
