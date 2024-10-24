package libcontainer

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/checkpoint-restore/go-criu/v6"
	criurpc "github.com/checkpoint-restore/go-criu/v6/rpc"
	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
	"google.golang.org/protobuf/proto"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/utils"
)

var criuFeatures *criurpc.CriuFeatures

var ErrCriuMissingFeatures = errors.New("criu is missing features")

func (c *Container) checkCriuFeatures(criuOpts *CriuOpts, criuFeat *criurpc.CriuFeatures) error {
	t := criurpc.CriuReqType_FEATURE_CHECK

	// make sure the features we are looking for are really not from
	// some previous check
	criuFeatures = nil

	req := &criurpc.CriuReq{
		Type:     &t,
		Features: criuFeat,
	}

	err := c.criuSwrk(nil, req, criuOpts, nil)
	if err != nil {
		return fmt.Errorf("CRIU feature check failed: %w", err)
	}

	var missingFeatures []string

	// The outer if checks if the fields actually exist
	if (criuFeat.MemTrack != nil) &&
		(criuFeatures.MemTrack != nil) {
		// The inner if checks if they are set to true
		if *criuFeat.MemTrack && !*criuFeatures.MemTrack {
			missingFeatures = append(missingFeatures, "MemTrack")
			logrus.Debugf("CRIU does not support MemTrack")
		}
	}

	// This needs to be repeated for every new feature check.
	// Is there a way to put this in a function. Reflection?
	if (criuFeat.LazyPages != nil) &&
		(criuFeatures.LazyPages != nil) {
		if *criuFeat.LazyPages && !*criuFeatures.LazyPages {
			missingFeatures = append(missingFeatures, "LazyPages")
			logrus.Debugf("CRIU does not support LazyPages")
		}
	}

	if len(missingFeatures) != 0 {
		return fmt.Errorf("%w: %v", ErrCriuMissingFeatures, missingFeatures)
	}

	return nil
}

func compareCriuVersion(criuVersion int, minVersion int) error {
	// simple function to perform the actual version compare
	if criuVersion < minVersion {
		return fmt.Errorf("CRIU version %d must be %d or higher", criuVersion, minVersion)
	}

	return nil
}

// checkCriuVersion checks CRIU version greater than or equal to minVersion.
func (c *Container) checkCriuVersion(minVersion int) error {
	// If the version of criu has already been determined there is no need
	// to ask criu for the version again. Use the value from c.criuVersion.
	if c.criuVersion != 0 {
		return compareCriuVersion(c.criuVersion, minVersion)
	}

	criu := criu.MakeCriu()
	var err error
	c.criuVersion, err = criu.GetCriuVersion()
	if err != nil {
		return fmt.Errorf("CRIU version check failed: %w", err)
	}

	return compareCriuVersion(c.criuVersion, minVersion)
}

const descriptorsFilename = "descriptors.json"

func (c *Container) addCriuDumpMount(req *criurpc.CriuReq, m *configs.Mount) {
	mountDest := strings.TrimPrefix(m.Destination, c.config.Rootfs)
	if dest, err := securejoin.SecureJoin(c.config.Rootfs, mountDest); err == nil {
		mountDest = dest[len(c.config.Rootfs):]
	}
	extMnt := &criurpc.ExtMountMap{
		Key: proto.String(mountDest),
		Val: proto.String(mountDest),
	}
	req.Opts.ExtMnt = append(req.Opts.ExtMnt, extMnt)
}

func (c *Container) addMaskPaths(req *criurpc.CriuReq) error {
	for _, path := range c.config.MaskPaths {
		fi, err := os.Stat(fmt.Sprintf("/proc/%d/root/%s", c.initProcess.pid(), path))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return err
		}
		if fi.IsDir() {
			continue
		}

		extMnt := &criurpc.ExtMountMap{
			Key: proto.String(path),
			Val: proto.String("/dev/null"),
		}
		req.Opts.ExtMnt = append(req.Opts.ExtMnt, extMnt)
	}
	return nil
}

func (c *Container) handleCriuConfigurationFile(rpcOpts *criurpc.CriuOpts) {
	// CRIU will evaluate a configuration starting with release 3.11.
	// Settings in the configuration file will overwrite RPC settings.
	// Look for annotations. The annotation 'org.criu.config'
	// specifies if CRIU should use a different, container specific
	// configuration file.
	configFile, exists := utils.SearchLabels(c.config.Labels, "org.criu.config")
	if exists {
		// If the annotation 'org.criu.config' exists and is set
		// to a non-empty string, tell CRIU to use that as a
		// configuration file. If the file does not exist, CRIU
		// will just ignore it.
		if configFile != "" {
			rpcOpts.ConfigFile = proto.String(configFile)
		}
		// If 'org.criu.config' exists and is set to an empty
		// string, a runc specific CRIU configuration file will
		// be not set at all.
	} else {
		// If the mentioned annotation has not been found, specify
		// a default CRIU configuration file.
		rpcOpts.ConfigFile = proto.String("/etc/criu/runc.conf")
	}
}

func (c *Container) criuSupportsExtNS(t configs.NamespaceType) bool {
	var minVersion int
	switch t {
	case configs.NEWNET:
		// CRIU supports different external namespace with different released CRIU versions.
		// For network namespaces to work we need at least criu 3.11.0 => 31100.
		minVersion = 31100
	case configs.NEWPID:
		// For PID namespaces criu 31500 is needed.
		minVersion = 31500
	default:
		return false
	}
	return c.checkCriuVersion(minVersion) == nil
}

func criuNsToKey(t configs.NamespaceType) string {
	return "extRoot" + strings.Title(configs.NsName(t)) + "NS" //nolint:staticcheck // SA1019: strings.Title is deprecated
}

func (c *Container) handleCheckpointingExternalNamespaces(rpcOpts *criurpc.CriuOpts, t configs.NamespaceType) error {
	if !c.criuSupportsExtNS(t) {
		return nil
	}

	nsPath := c.config.Namespaces.PathOf(t)
	if nsPath == "" {
		return nil
	}
	// CRIU expects the information about an external namespace
	// like this: --external <TYPE>[<inode>]:<key>
	// This <key> is always 'extRoot<TYPE>NS'.
	var ns unix.Stat_t
	if err := unix.Stat(nsPath, &ns); err != nil {
		return err
	}
	criuExternal := fmt.Sprintf("%s[%d]:%s", configs.NsName(t), ns.Ino, criuNsToKey(t))
	rpcOpts.External = append(rpcOpts.External, criuExternal)

	return nil
}

func (c *Container) handleRestoringNamespaces(rpcOpts *criurpc.CriuOpts, extraFiles *[]*os.File) error {
	for _, ns := range c.config.Namespaces {
		switch ns.Type {
		case configs.NEWNET, configs.NEWPID:
			// If the container is running in a network or PID namespace and has
			// a path to the network or PID namespace configured, we will dump
			// that network or PID namespace as an external namespace and we
			// will expect that the namespace exists during restore.
			// This basically means that CRIU will ignore the namespace
			// and expect it to be setup correctly.
			if err := c.handleRestoringExternalNamespaces(rpcOpts, extraFiles, ns.Type); err != nil {
				return err
			}
		default:
			// For all other namespaces except NET and PID CRIU has
			// a simpler way of joining the existing namespace if set
			nsPath := c.config.Namespaces.PathOf(ns.Type)
			if nsPath == "" {
				continue
			}
			if ns.Type == configs.NEWCGROUP {
				// CRIU has no code to handle NEWCGROUP
				return fmt.Errorf("Do not know how to handle namespace %v", ns.Type)
			}
			// CRIU has code to handle NEWTIME, but it does not seem to be defined in runc

			// CRIU will issue a warning for NEWUSER:
			// criu/namespaces.c: 'join-ns with user-namespace is not fully tested and dangerous'
			rpcOpts.JoinNs = append(rpcOpts.JoinNs, &criurpc.JoinNamespace{
				Ns:     proto.String(configs.NsName(ns.Type)),
				NsFile: proto.String(nsPath),
			})
		}
	}

	return nil
}

func (c *Container) handleRestoringExternalNamespaces(rpcOpts *criurpc.CriuOpts, extraFiles *[]*os.File, t configs.NamespaceType) error {
	if !c.criuSupportsExtNS(t) {
		return nil
	}

	nsPath := c.config.Namespaces.PathOf(t)
	if nsPath == "" {
		return nil
	}
	// CRIU wants the information about an existing namespace
	// like this: --inherit-fd fd[<fd>]:<key>
	// The <key> needs to be the same as during checkpointing.
	// We are always using 'extRoot<TYPE>NS' as the key in this.
	nsFd, err := os.Open(nsPath)
	if err != nil {
		logrus.Errorf("If a specific network namespace is defined it must exist: %s", err)
		return fmt.Errorf("Requested network namespace %v does not exist", nsPath)
	}
	inheritFd := &criurpc.InheritFd{
		Key: proto.String(criuNsToKey(t)),
		// The offset of four is necessary because 0, 1, 2 and 3 are
		// already used by stdin, stdout, stderr, 'criu swrk' socket.
		Fd: proto.Int32(int32(4 + len(*extraFiles))),
	}
	rpcOpts.InheritFd = append(rpcOpts.InheritFd, inheritFd)
	// All open FDs need to be transferred to CRIU via extraFiles
	*extraFiles = append(*extraFiles, nsFd)

	return nil
}

func (c *Container) Checkpoint(criuOpts *CriuOpts) error {
	const logFile = "dump.log"
	c.m.Lock()
	defer c.m.Unlock()

	// Checkpoint is unlikely to work if os.Geteuid() != 0 || system.RunningInUserNS().
	// (CLI prints a warning)
	// TODO(avagin): Figure out how to make this work nicely. CRIU 2.0 has
	//               support for doing unprivileged dumps, but the setup of
	//               rootless containers might make this complicated.

	// We are relying on the CRIU version RPC which was introduced with CRIU 3.0.0
	if err := c.checkCriuVersion(30000); err != nil {
		return err
	}

	if criuOpts.ImagesDirectory == "" {
		return errors.New("invalid directory to save checkpoint")
	}

	// Since a container can be C/R'ed multiple times,
	// the checkpoint directory may already exist.
	if err := os.Mkdir(criuOpts.ImagesDirectory, 0o700); err != nil && !os.IsExist(err) {
		return err
	}

	logDir := criuOpts.ImagesDirectory
	imageDir, err := os.Open(criuOpts.ImagesDirectory)
	if err != nil {
		return err
	}
	defer imageDir.Close()

	rpcOpts := criurpc.CriuOpts{
		ImagesDirFd:     proto.Int32(int32(imageDir.Fd())),
		LogLevel:        proto.Int32(4),
		LogFile:         proto.String(logFile),
		Root:            proto.String(c.config.Rootfs),
		ManageCgroups:   proto.Bool(true),
		NotifyScripts:   proto.Bool(true),
		Pid:             proto.Int32(int32(c.initProcess.pid())),
		ShellJob:        proto.Bool(criuOpts.ShellJob),
		LeaveRunning:    proto.Bool(criuOpts.LeaveRunning),
		TcpEstablished:  proto.Bool(criuOpts.TcpEstablished),
		ExtUnixSk:       proto.Bool(criuOpts.ExternalUnixConnections),
		FileLocks:       proto.Bool(criuOpts.FileLocks),
		EmptyNs:         proto.Uint32(criuOpts.EmptyNs),
		OrphanPtsMaster: proto.Bool(true),
		AutoDedup:       proto.Bool(criuOpts.AutoDedup),
		LazyPages:       proto.Bool(criuOpts.LazyPages),
	}

	// if criuOpts.WorkDirectory is not set, criu default is used.
	if criuOpts.WorkDirectory != "" {
		if err := os.Mkdir(criuOpts.WorkDirectory, 0o700); err != nil && !os.IsExist(err) {
			return err
		}
		workDir, err := os.Open(criuOpts.WorkDirectory)
		if err != nil {
			return err
		}
		defer workDir.Close()
		rpcOpts.WorkDirFd = proto.Int32(int32(workDir.Fd()))
		logDir = criuOpts.WorkDirectory
	}

	c.handleCriuConfigurationFile(&rpcOpts)

	// If the container is running in a network namespace and has
	// a path to the network namespace configured, we will dump
	// that network namespace as an external namespace and we
	// will expect that the namespace exists during restore.
	// This basically means that CRIU will ignore the namespace
	// and expect to be setup correctly.
	if err := c.handleCheckpointingExternalNamespaces(&rpcOpts, configs.NEWNET); err != nil {
		return err
	}

	// Same for possible external PID namespaces
	if err := c.handleCheckpointingExternalNamespaces(&rpcOpts, configs.NEWPID); err != nil {
		return err
	}

	// CRIU can use cgroup freezer; when rpcOpts.FreezeCgroup
	// is not set, CRIU uses ptrace() to pause the processes.
	// Note cgroup v2 freezer is only supported since CRIU release 3.14.
	if !cgroups.IsCgroup2UnifiedMode() || c.checkCriuVersion(31400) == nil {
		if fcg := c.cgroupManager.Path("freezer"); fcg != "" {
			rpcOpts.FreezeCgroup = proto.String(fcg)
		}
	}

	// append optional criu opts, e.g., page-server and port
	if criuOpts.PageServer.Address != "" && criuOpts.PageServer.Port != 0 {
		rpcOpts.Ps = &criurpc.CriuPageServerInfo{
			Address: proto.String(criuOpts.PageServer.Address),
			Port:    proto.Int32(criuOpts.PageServer.Port),
		}
	}

	// pre-dump may need parentImage param to complete iterative migration
	if criuOpts.ParentImage != "" {
		rpcOpts.ParentImg = proto.String(criuOpts.ParentImage)
		rpcOpts.TrackMem = proto.Bool(true)
	}

	// append optional manage cgroups mode
	if criuOpts.ManageCgroupsMode != 0 {
		mode := criuOpts.ManageCgroupsMode
		rpcOpts.ManageCgroupsMode = &mode
	}

	var t criurpc.CriuReqType
	if criuOpts.PreDump {
		feat := criurpc.CriuFeatures{
			MemTrack: proto.Bool(true),
		}

		if err := c.checkCriuFeatures(criuOpts, &feat); err != nil {
			return err
		}

		t = criurpc.CriuReqType_PRE_DUMP
	} else {
		t = criurpc.CriuReqType_DUMP
	}

	if criuOpts.LazyPages {
		// lazy migration requested; check if criu supports it
		feat := criurpc.CriuFeatures{
			LazyPages: proto.Bool(true),
		}
		if err := c.checkCriuFeatures(criuOpts, &feat); err != nil {
			return err
		}

		if fd := criuOpts.StatusFd; fd != -1 {
			// check that the FD is valid
			flags, err := unix.FcntlInt(uintptr(fd), unix.F_GETFL, 0)
			if err != nil {
				return fmt.Errorf("invalid --status-fd argument %d: %w", fd, err)
			}
			// and writable
			if flags&unix.O_WRONLY == 0 {
				return fmt.Errorf("invalid --status-fd argument %d: not writable", fd)
			}

			if c.checkCriuVersion(31500) != nil {
				// For criu 3.15+, use notifications (see case "status-ready"
				// in criuNotifications). Otherwise, rely on criu status fd.
				rpcOpts.StatusFd = proto.Int32(int32(fd))
			}
		}
	}

	req := &criurpc.CriuReq{
		Type: &t,
		Opts: &rpcOpts,
	}

	// no need to dump all this in pre-dump
	if !criuOpts.PreDump {
		hasCgroupns := c.config.Namespaces.Contains(configs.NEWCGROUP)
		for _, m := range c.config.Mounts {
			switch m.Device {
			case "bind":
				c.addCriuDumpMount(req, m)
			case "cgroup":
				if cgroups.IsCgroup2UnifiedMode() || hasCgroupns {
					// real mount(s)
					continue
				}
				// a set of "external" bind mounts
				binds, err := getCgroupMounts(m)
				if err != nil {
					return err
				}
				for _, b := range binds {
					c.addCriuDumpMount(req, b)
				}
			}
		}

		if err := c.addMaskPaths(req); err != nil {
			return err
		}

		for _, node := range c.config.Devices {
			m := &configs.Mount{Destination: node.Path, Source: node.Path}
			c.addCriuDumpMount(req, m)
		}

		// Write the FD info to a file in the image directory
		fdsJSON, err := json.Marshal(c.initProcess.externalDescriptors())
		if err != nil {
			return err
		}

		err = os.WriteFile(filepath.Join(criuOpts.ImagesDirectory, descriptorsFilename), fdsJSON, 0o600)
		if err != nil {
			return err
		}
	}

	err = c.criuSwrk(nil, req, criuOpts, nil)
	if err != nil {
		logCriuErrors(logDir, logFile)
		return err
	}
	return nil
}

func (c *Container) addCriuRestoreMount(req *criurpc.CriuReq, m *configs.Mount) {
	mountDest := strings.TrimPrefix(m.Destination, c.config.Rootfs)
	if dest, err := securejoin.SecureJoin(c.config.Rootfs, mountDest); err == nil {
		mountDest = dest[len(c.config.Rootfs):]
	}
	extMnt := &criurpc.ExtMountMap{
		Key: proto.String(mountDest),
		Val: proto.String(m.Source),
	}
	req.Opts.ExtMnt = append(req.Opts.ExtMnt, extMnt)
}

func (c *Container) restoreNetwork(req *criurpc.CriuReq, criuOpts *CriuOpts) {
	for _, iface := range c.config.Networks {
		switch iface.Type {
		case "veth":
			veth := new(criurpc.CriuVethPair)
			veth.IfOut = proto.String(iface.HostInterfaceName)
			veth.IfIn = proto.String(iface.Name)
			req.Opts.Veths = append(req.Opts.Veths, veth)
		case "loopback":
			// Do nothing
		}
	}
	for _, i := range criuOpts.VethPairs {
		veth := new(criurpc.CriuVethPair)
		veth.IfOut = proto.String(i.HostInterfaceName)
		veth.IfIn = proto.String(i.ContainerInterfaceName)
		req.Opts.Veths = append(req.Opts.Veths, veth)
	}
}

// makeCriuRestoreMountpoints makes the actual mountpoints for the
// restore using CRIU. This function is inspired from the code in
// rootfs_linux.go.
func (c *Container) makeCriuRestoreMountpoints(m *configs.Mount) error {
	if m.Device == "cgroup" {
		// No mount point(s) need to be created:
		//
		// * for v1, mount points are saved by CRIU because
		//   /sys/fs/cgroup is a tmpfs mount
		//
		// * for v2, /sys/fs/cgroup is a real mount, but
		//   the mountpoint appears as soon as /sys is mounted
		return nil
	}
	// TODO: pass srcFD? Not sure if criu is impacted by issue #2484.
	me := mountEntry{Mount: m}
	// For all other filesystems, just make the target.
	if _, err := createMountpoint(c.config.Rootfs, me); err != nil {
		return fmt.Errorf("create criu restore mountpoint for %s mount: %w", me.Destination, err)
	}
	return nil
}

// isPathInPrefixList is a small function for CRIU restore to make sure
// mountpoints, which are on a tmpfs, are not created in the roofs.
func isPathInPrefixList(path string, prefix []string) bool {
	for _, p := range prefix {
		if strings.HasPrefix(path, p+"/") {
			return true
		}
	}
	return false
}

// prepareCriuRestoreMounts tries to set up the rootfs of the
// container to be restored in the same way runc does it for
// initial container creation. Even for a read-only rootfs container
// runc modifies the rootfs to add mountpoints which do not exist.
// This function also creates missing mountpoints as long as they
// are not on top of a tmpfs, as CRIU will restore tmpfs content anyway.
func (c *Container) prepareCriuRestoreMounts(mounts []*configs.Mount) error {
	// First get a list of a all tmpfs mounts
	tmpfs := []string{}
	for _, m := range mounts {
		switch m.Device {
		case "tmpfs":
			tmpfs = append(tmpfs, m.Destination)
		}
	}
	// Now go through all mounts and create the mountpoints
	// if the mountpoints are not on a tmpfs, as CRIU will
	// restore the complete tmpfs content from its checkpoint.
	umounts := []string{}
	defer func() {
		for _, u := range umounts {
			_ = utils.WithProcfd(c.config.Rootfs, u, func(procfd string) error {
				if e := unix.Unmount(procfd, unix.MNT_DETACH); e != nil {
					if e != unix.EINVAL {
						// Ignore EINVAL as it means 'target is not a mount point.'
						// It probably has already been unmounted.
						logrus.Warnf("Error during cleanup unmounting of %s (%s): %v", procfd, u, e)
					}
				}
				return nil
			})
		}
	}()
	for _, m := range mounts {
		if !isPathInPrefixList(m.Destination, tmpfs) {
			if err := c.makeCriuRestoreMountpoints(m); err != nil {
				return err
			}
			// If the mount point is a bind mount, we need to mount
			// it now so that runc can create the necessary mount
			// points for mounts in bind mounts.
			// This also happens during initial container creation.
			// Without this CRIU restore will fail
			// See: https://github.com/opencontainers/runc/issues/2748
			// It is also not necessary to order the mount points
			// because during initial container creation mounts are
			// set up in the order they are configured.
			if m.Device == "bind" {
				if err := utils.WithProcfd(c.config.Rootfs, m.Destination, func(dstFd string) error {
					return mountViaFds(m.Source, nil, m.Destination, dstFd, "", unix.MS_BIND|unix.MS_REC, "")
				}); err != nil {
					return err
				}
				umounts = append(umounts, m.Destination)
			}
		}
	}
	return nil
}

// Restore restores the checkpointed container to a running state using the
// criu(8) utility.
func (c *Container) Restore(process *Process, criuOpts *CriuOpts) error {
	const logFile = "restore.log"
	c.m.Lock()
	defer c.m.Unlock()

	var extraFiles []*os.File

	// Restore is unlikely to work if os.Geteuid() != 0 || system.RunningInUserNS().
	// (CLI prints a warning)
	// TODO(avagin): Figure out how to make this work nicely. CRIU doesn't have
	//               support for unprivileged restore at the moment.

	// We are relying on the CRIU version RPC which was introduced with CRIU 3.0.0
	if err := c.checkCriuVersion(30000); err != nil {
		return err
	}
	if criuOpts.ImagesDirectory == "" {
		return errors.New("invalid directory to restore checkpoint")
	}
	logDir := criuOpts.ImagesDirectory
	imageDir, err := os.Open(criuOpts.ImagesDirectory)
	if err != nil {
		return err
	}
	defer imageDir.Close()
	// CRIU has a few requirements for a root directory:
	// * it must be a mount point
	// * its parent must not be overmounted
	// c.config.Rootfs is bind-mounted to a temporary directory
	// to satisfy these requirements.
	root := filepath.Join(c.stateDir, "criu-root")
	if err := os.Mkdir(root, 0o755); err != nil {
		return err
	}
	defer os.Remove(root)
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return err
	}
	err = mount(c.config.Rootfs, root, "", unix.MS_BIND|unix.MS_REC, "")
	if err != nil {
		return err
	}
	defer unix.Unmount(root, unix.MNT_DETACH) //nolint: errcheck
	t := criurpc.CriuReqType_RESTORE
	req := &criurpc.CriuReq{
		Type: &t,
		Opts: &criurpc.CriuOpts{
			ImagesDirFd:     proto.Int32(int32(imageDir.Fd())),
			EvasiveDevices:  proto.Bool(true),
			LogLevel:        proto.Int32(4),
			LogFile:         proto.String(logFile),
			RstSibling:      proto.Bool(true),
			Root:            proto.String(root),
			ManageCgroups:   proto.Bool(true),
			NotifyScripts:   proto.Bool(true),
			ShellJob:        proto.Bool(criuOpts.ShellJob),
			ExtUnixSk:       proto.Bool(criuOpts.ExternalUnixConnections),
			TcpEstablished:  proto.Bool(criuOpts.TcpEstablished),
			FileLocks:       proto.Bool(criuOpts.FileLocks),
			EmptyNs:         proto.Uint32(criuOpts.EmptyNs),
			OrphanPtsMaster: proto.Bool(true),
			AutoDedup:       proto.Bool(criuOpts.AutoDedup),
			LazyPages:       proto.Bool(criuOpts.LazyPages),
		},
	}

	if criuOpts.LsmProfile != "" {
		// CRIU older than 3.16 has a bug which breaks the possibility
		// to set a different LSM profile.
		if err := c.checkCriuVersion(31600); err != nil {
			return errors.New("--lsm-profile requires at least CRIU 3.16")
		}
		req.Opts.LsmProfile = proto.String(criuOpts.LsmProfile)
	}
	if criuOpts.LsmMountContext != "" {
		if err := c.checkCriuVersion(31600); err != nil {
			return errors.New("--lsm-mount-context requires at least CRIU 3.16")
		}
		req.Opts.LsmMountContext = proto.String(criuOpts.LsmMountContext)
	}

	if criuOpts.WorkDirectory != "" {
		// Since a container can be C/R'ed multiple times,
		// the work directory may already exist.
		if err := os.Mkdir(criuOpts.WorkDirectory, 0o700); err != nil && !os.IsExist(err) {
			return err
		}
		workDir, err := os.Open(criuOpts.WorkDirectory)
		if err != nil {
			return err
		}
		defer workDir.Close()
		req.Opts.WorkDirFd = proto.Int32(int32(workDir.Fd()))
		logDir = criuOpts.WorkDirectory
	}
	c.handleCriuConfigurationFile(req.Opts)

	if err := c.handleRestoringNamespaces(req.Opts, &extraFiles); err != nil {
		return err
	}

	// This will modify the rootfs of the container in the same way runc
	// modifies the container during initial creation.
	if err := c.prepareCriuRestoreMounts(c.config.Mounts); err != nil {
		return err
	}

	hasCgroupns := c.config.Namespaces.Contains(configs.NEWCGROUP)
	for _, m := range c.config.Mounts {
		switch m.Device {
		case "bind":
			c.addCriuRestoreMount(req, m)
		case "cgroup":
			if cgroups.IsCgroup2UnifiedMode() || hasCgroupns {
				continue
			}
			// cgroup v1 is a set of bind mounts, unless cgroupns is used
			binds, err := getCgroupMounts(m)
			if err != nil {
				return err
			}
			for _, b := range binds {
				c.addCriuRestoreMount(req, b)
			}
		}
	}

	if len(c.config.MaskPaths) > 0 {
		m := &configs.Mount{Destination: "/dev/null", Source: "/dev/null"}
		c.addCriuRestoreMount(req, m)
	}

	for _, node := range c.config.Devices {
		m := &configs.Mount{Destination: node.Path, Source: node.Path}
		c.addCriuRestoreMount(req, m)
	}

	if criuOpts.EmptyNs&unix.CLONE_NEWNET == 0 {
		c.restoreNetwork(req, criuOpts)
	}

	// append optional manage cgroups mode
	if criuOpts.ManageCgroupsMode != 0 {
		mode := criuOpts.ManageCgroupsMode
		req.Opts.ManageCgroupsMode = &mode
	}

	var (
		fds    []string
		fdJSON []byte
	)
	if fdJSON, err = os.ReadFile(filepath.Join(criuOpts.ImagesDirectory, descriptorsFilename)); err != nil {
		return err
	}

	if err := json.Unmarshal(fdJSON, &fds); err != nil {
		return err
	}
	for i := range fds {
		if s := fds[i]; strings.Contains(s, "pipe:") {
			inheritFd := new(criurpc.InheritFd)
			inheritFd.Key = proto.String(s)
			inheritFd.Fd = proto.Int32(int32(i))
			req.Opts.InheritFd = append(req.Opts.InheritFd, inheritFd)
		}
	}
	err = c.criuSwrk(process, req, criuOpts, extraFiles)
	if err != nil {
		logCriuErrors(logDir, logFile)
	}

	// Now that CRIU is done let's close all opened FDs CRIU needed.
	for _, fd := range extraFiles {
		fd.Close()
	}

	return err
}

// logCriuErrors tries to find and log errors from a criu log file.
// The output is similar to what "grep -n -B5 Error" does.
func logCriuErrors(dir, file string) {
	lookFor := []byte("Error") // Print the line that contains this...
	const max = 5 + 1          // ... and a few preceding lines.

	logFile := filepath.Join(dir, file)
	f, err := os.Open(logFile)
	if err != nil {
		logrus.Warn(err)
		return
	}
	defer f.Close()

	var lines [max][]byte
	var idx, lineNo, printedLineNo int
	s := bufio.NewScanner(f)
	for s.Scan() {
		lineNo++
		lines[idx] = s.Bytes()
		idx = (idx + 1) % max
		if !bytes.Contains(s.Bytes(), lookFor) {
			continue
		}
		// Found an error.
		if printedLineNo == 0 {
			logrus.Warnf("--- Quoting %q", logFile)
		} else if lineNo-max > printedLineNo {
			// Mark the gap.
			logrus.Warn("...")
		}
		// Print the last lines.
		for add := 0; add < max; add++ {
			i := (idx + add) % max
			s := lines[i]
			actLineNo := lineNo + add - max + 1
			if len(s) > 0 && actLineNo > printedLineNo {
				logrus.Warnf("%d:%s", actLineNo, s)
				printedLineNo = actLineNo
			}
		}
	}
	if printedLineNo != 0 {
		logrus.Warn("---") // End of "Quoting ...".
	}
	if err := s.Err(); err != nil {
		logrus.Warnf("read %q: %v", logFile, err)
	}
}

func (c *Container) criuApplyCgroups(pid int, req *criurpc.CriuReq) error {
	// need to apply cgroups only on restore
	if req.GetType() != criurpc.CriuReqType_RESTORE {
		return nil
	}

	// XXX: Do we need to deal with this case? AFAIK criu still requires root.
	if err := c.cgroupManager.Apply(pid); err != nil {
		return err
	}

	if err := c.cgroupManager.Set(c.config.Cgroups.Resources); err != nil {
		return err
	}

	// TODO(@kolyshkin): should we use c.cgroupManager.GetPaths()
	// instead of reading /proc/pid/cgroup?
	path := fmt.Sprintf("/proc/%d/cgroup", pid)
	cgroupsPaths, err := cgroups.ParseCgroupFile(path)
	if err != nil {
		return err
	}

	for c, p := range cgroupsPaths {
		cgroupRoot := &criurpc.CgroupRoot{
			Ctrl: proto.String(c),
			Path: proto.String(p),
		}
		req.Opts.CgRoot = append(req.Opts.CgRoot, cgroupRoot)
	}

	return nil
}

func (c *Container) criuSwrk(process *Process, req *criurpc.CriuReq, opts *CriuOpts, extraFiles []*os.File) error {
	fds, err := unix.Socketpair(unix.AF_LOCAL, unix.SOCK_SEQPACKET|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return err
	}

	criuClient := os.NewFile(uintptr(fds[0]), "criu-transport-client")
	criuClientFileCon, err := net.FileConn(criuClient)
	criuClient.Close()
	if err != nil {
		return err
	}

	criuClientCon := criuClientFileCon.(*net.UnixConn)
	defer criuClientCon.Close()

	criuServer := os.NewFile(uintptr(fds[1]), "criu-transport-server")
	defer criuServer.Close()

	if c.criuVersion != 0 {
		// If the CRIU Version is still '0' then this is probably
		// the initial CRIU run to detect the version. Skip it.
		logrus.Debugf("Using CRIU %d", c.criuVersion)
	}
	cmd := exec.Command("criu", "swrk", "3")
	if process != nil {
		cmd.Stdin = process.Stdin
		cmd.Stdout = process.Stdout
		cmd.Stderr = process.Stderr
	}
	cmd.ExtraFiles = append(cmd.ExtraFiles, criuServer)
	if extraFiles != nil {
		cmd.ExtraFiles = append(cmd.ExtraFiles, extraFiles...)
	}

	if err := cmd.Start(); err != nil {
		return err
	}
	// we close criuServer so that even if CRIU crashes or unexpectedly exits, runc will not hang.
	criuServer.Close()
	// cmd.Process will be replaced by a restored init.
	criuProcess := cmd.Process

	var criuProcessState *os.ProcessState
	defer func() {
		if criuProcessState == nil {
			criuClientCon.Close()
			_, err := criuProcess.Wait()
			if err != nil {
				logrus.Warnf("wait on criuProcess returned %v", err)
			}
		}
	}()

	if err := c.criuApplyCgroups(criuProcess.Pid, req); err != nil {
		return err
	}

	var extFds []string
	if process != nil {
		extFds, err = getPipeFds(criuProcess.Pid)
		if err != nil {
			return err
		}
	}

	logrus.Debugf("Using CRIU in %s mode", req.GetType().String())
	// In the case of criurpc.CriuReqType_FEATURE_CHECK req.GetOpts()
	// should be empty. For older CRIU versions it still will be
	// available but empty. criurpc.CriuReqType_VERSION actually
	// has no req.GetOpts().
	if logrus.GetLevel() >= logrus.DebugLevel &&
		!(req.GetType() == criurpc.CriuReqType_FEATURE_CHECK ||
			req.GetType() == criurpc.CriuReqType_VERSION) {

		val := reflect.ValueOf(req.GetOpts())
		v := reflect.Indirect(val)
		for i := 0; i < v.NumField(); i++ {
			st := v.Type()
			name := st.Field(i).Name
			if 'A' <= name[0] && name[0] <= 'Z' {
				value := val.MethodByName("Get" + name).Call([]reflect.Value{})
				logrus.Debugf("CRIU option %s with value %v", name, value[0])
			}
		}
	}
	data, err := proto.Marshal(req)
	if err != nil {
		return err
	}
	_, err = criuClientCon.Write(data)
	if err != nil {
		return err
	}

	buf := make([]byte, 10*4096)
	oob := make([]byte, 4096)
	for {
		n, oobn, _, _, err := criuClientCon.ReadMsgUnix(buf, oob)
		if req.Opts != nil && req.Opts.StatusFd != nil {
			// Close status_fd as soon as we got something back from criu,
			// assuming it has consumed (reopened) it by this time.
			// Otherwise it will might be left open forever and whoever
			// is waiting on it will wait forever.
			fd := int(*req.Opts.StatusFd)
			_ = unix.Close(fd)
			req.Opts.StatusFd = nil
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return errors.New("unexpected EOF")
		}
		if n == len(buf) {
			return errors.New("buffer is too small")
		}

		resp := new(criurpc.CriuResp)
		err = proto.Unmarshal(buf[:n], resp)
		if err != nil {
			return err
		}
		t := resp.GetType()
		if !resp.GetSuccess() {
			return fmt.Errorf("criu failed: type %s errno %d", t, resp.GetCrErrno())
		}

		switch t {
		case criurpc.CriuReqType_FEATURE_CHECK:
			logrus.Debugf("Feature check says: %s", resp)
			criuFeatures = resp.GetFeatures()
		case criurpc.CriuReqType_NOTIFY:
			if err := c.criuNotifications(resp, process, cmd, opts, extFds, oob[:oobn]); err != nil {
				return err
			}
			req = &criurpc.CriuReq{
				Type:          &t,
				NotifySuccess: proto.Bool(true),
			}
			data, err = proto.Marshal(req)
			if err != nil {
				return err
			}
			_, err = criuClientCon.Write(data)
			if err != nil {
				return err
			}
			continue
		case criurpc.CriuReqType_RESTORE:
		case criurpc.CriuReqType_DUMP:
		case criurpc.CriuReqType_PRE_DUMP:
		default:
			return fmt.Errorf("unable to parse the response %s", resp.String())
		}

		break
	}

	_ = criuClientCon.CloseWrite()
	// cmd.Wait() waits cmd.goroutines which are used for proxying file descriptors.
	// Here we want to wait only the CRIU process.
	criuProcessState, err = criuProcess.Wait()
	if err != nil {
		return err
	}

	// In pre-dump mode CRIU is in a loop and waits for
	// the final DUMP command.
	// The current runc pre-dump approach, however, is
	// start criu in PRE_DUMP once for a single pre-dump
	// and not the whole series of pre-dump, pre-dump, ...m, dump
	// If we got the message CriuReqType_PRE_DUMP it means
	// CRIU was successful and we need to forcefully stop CRIU
	if !criuProcessState.Success() && *req.Type != criurpc.CriuReqType_PRE_DUMP {
		return fmt.Errorf("criu failed: %s", criuProcessState)
	}
	return nil
}

// lockNetwork blocks any external network activity.
func lockNetwork(config *configs.Config) error {
	for _, config := range config.Networks {
		strategy, err := getStrategy(config.Type)
		if err != nil {
			return err
		}

		if err := strategy.detach(config); err != nil {
			return err
		}
	}
	return nil
}

func unlockNetwork(config *configs.Config) error {
	for _, config := range config.Networks {
		strategy, err := getStrategy(config.Type)
		if err != nil {
			return err
		}
		if err = strategy.attach(config); err != nil {
			return err
		}
	}
	return nil
}

func (c *Container) criuNotifications(resp *criurpc.CriuResp, process *Process, cmd *exec.Cmd, opts *CriuOpts, fds []string, oob []byte) error {
	notify := resp.GetNotify()
	if notify == nil {
		return fmt.Errorf("invalid response: %s", resp.String())
	}
	script := notify.GetScript()
	logrus.Debugf("notify: %s\n", script)
	switch script {
	case "post-dump":
		f, err := os.Create(filepath.Join(c.stateDir, "checkpoint"))
		if err != nil {
			return err
		}
		f.Close()
	case "network-unlock":
		if err := unlockNetwork(c.config); err != nil {
			return err
		}
	case "network-lock":
		if err := lockNetwork(c.config); err != nil {
			return err
		}
	case "setup-namespaces":
		if c.config.Hooks != nil {
			s, err := c.currentOCIState()
			if err != nil {
				return nil
			}
			s.Pid = int(notify.GetPid())

			if err := c.config.Hooks.Run(configs.Prestart, s); err != nil {
				return err
			}
			if err := c.config.Hooks.Run(configs.CreateRuntime, s); err != nil {
				return err
			}
		}
	case "post-restore":
		pid := notify.GetPid()

		p, err := os.FindProcess(int(pid))
		if err != nil {
			return err
		}
		cmd.Process = p

		r, err := newRestoredProcess(cmd, fds)
		if err != nil {
			return err
		}
		process.ops = r
		if err := c.state.transition(&restoredState{
			imageDir: opts.ImagesDirectory,
			c:        c,
		}); err != nil {
			return err
		}
		// create a timestamp indicating when the restored checkpoint was started
		c.created = time.Now().UTC()
		if _, err := c.updateState(r); err != nil {
			return err
		}
		if err := os.Remove(filepath.Join(c.stateDir, "checkpoint")); err != nil {
			if !os.IsNotExist(err) {
				logrus.Error(err)
			}
		}
	case "orphan-pts-master":
		scm, err := unix.ParseSocketControlMessage(oob)
		if err != nil {
			return err
		}
		fds, err := unix.ParseUnixRights(&scm[0])
		if err != nil {
			return err
		}

		master := os.NewFile(uintptr(fds[0]), "orphan-pts-master")
		defer master.Close()

		// While we can access console.master, using the API is a good idea.
		if err := utils.SendFile(process.ConsoleSocket, master); err != nil {
			return err
		}
	case "status-ready":
		if opts.StatusFd != -1 {
			// write \0 to status fd to notify that lazy page server is ready
			_, err := unix.Write(opts.StatusFd, []byte{0})
			if err != nil {
				logrus.Warnf("can't write \\0 to status fd: %v", err)
			}
			_ = unix.Close(opts.StatusFd)
			opts.StatusFd = -1
		}
	}
	return nil
}
