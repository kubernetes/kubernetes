// Copyright 2014 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//+build linux

package main

// this implements /init of stage1/nspawn+systemd

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"github.com/appc/goaci/proj2aci"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/util"
	"github.com/godbus/dbus"
	"github.com/godbus/dbus/introspect"
	"github.com/hashicorp/errwrap"

	stage1common "github.com/coreos/rkt/stage1/common"
	stage1commontypes "github.com/coreos/rkt/stage1/common/types"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/cgroup"
	"github.com/coreos/rkt/networking"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/coreos/rkt/stage1/init/kvm"
)

const (
	// Path to systemd-nspawn binary within the stage1 rootfs
	nspawnBin = "/usr/bin/systemd-nspawn"
	// Path to the interpreter within the stage1 rootfs
	interpBin = "/usr/lib/ld-linux-x86-64.so.2"
	// Path to the localtime file/symlink in host
	localtimePath = "/etc/localtime"
)

// mirrorLocalZoneInfo tries to reproduce the /etc/localtime target in stage1/ to satisfy systemd-nspawn
func mirrorLocalZoneInfo(root string) {
	zif, err := os.Readlink(localtimePath)
	if err != nil {
		return
	}

	// On some systems /etc/localtime is a relative symlink, make it absolute
	if !filepath.IsAbs(zif) {
		zif = filepath.Join(filepath.Dir(localtimePath), zif)
		zif = filepath.Clean(zif)
	}

	src, err := os.Open(zif)
	if err != nil {
		return
	}
	defer src.Close()

	destp := filepath.Join(common.Stage1RootfsPath(root), zif)

	if err = os.MkdirAll(filepath.Dir(destp), 0755); err != nil {
		return
	}

	dest, err := os.OpenFile(destp, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer dest.Close()

	_, _ = io.Copy(dest, src)
}

var (
	debug        bool
	netList      common.NetList
	interactive  bool
	privateUsers string
	mdsToken     string
	localhostIP  net.IP
	localConfig  string
	hostname     string
	log          *rktlog.Logger
	diag         *rktlog.Logger
)

func init() {
	flag.BoolVar(&debug, "debug", false, "Run in debug mode")
	flag.Var(&netList, "net", "Setup networking")
	flag.BoolVar(&interactive, "interactive", false, "The pod is interactive")
	flag.StringVar(&privateUsers, "private-users", "", "Run within user namespace. Can be set to [=UIDBASE[:NUIDS]]")
	flag.StringVar(&mdsToken, "mds-token", "", "MDS auth token")
	flag.StringVar(&localConfig, "local-config", common.DefaultLocalConfigDir, "Local config path")
	flag.StringVar(&hostname, "hostname", "", "Hostname of the pod")
	// this ensures that main runs only on main thread (thread group leader).
	// since namespace ops (unshare, setns) are done for a single thread, we
	// must ensure that the goroutine does not jump from OS thread to thread
	runtime.LockOSThread()

	localhostIP = net.ParseIP("127.0.0.1")
	if localhostIP == nil {
		panic("localhost IP failed to parse")
	}
}

// machinedRegister checks if nspawn should register the pod to machined
func machinedRegister() bool {
	// machined has a D-Bus interface following versioning guidelines, see:
	// http://www.freedesktop.org/wiki/Software/systemd/machined/
	// Therefore we can just check if the D-Bus method we need exists and we
	// don't need to check the signature.
	var found int

	conn, err := dbus.SystemBus()
	if err != nil {
		return false
	}
	node, err := introspect.Call(conn.Object("org.freedesktop.machine1", "/org/freedesktop/machine1"))
	if err != nil {
		return false
	}
	for _, iface := range node.Interfaces {
		if iface.Name != "org.freedesktop.machine1.Manager" {
			continue
		}
		// machined v215 supports methods "RegisterMachine" and "CreateMachine" called by nspawn v215.
		// machined v216+ (since commit 5aa4bb) additionally supports methods "CreateMachineWithNetwork"
		// and "RegisterMachineWithNetwork", called by nspawn v216+.
		for _, method := range iface.Methods {
			if method.Name == "CreateMachineWithNetwork" || method.Name == "RegisterMachineWithNetwork" {
				found++
			}
		}
		break
	}
	return found == 2
}

func installAssets() error {
	systemctlBin, err := common.LookupPath("systemctl", os.Getenv("PATH"))
	if err != nil {
		return err
	}
	bashBin, err := common.LookupPath("bash", os.Getenv("PATH"))
	if err != nil {
		return err
	}
	// More paths could be added in that list if some Linux distributions install it in a different path
	// Note that we look in /usr/lib/... first because of the merge:
	// http://www.freedesktop.org/wiki/Software/systemd/TheCaseForTheUsrMerge/
	systemdShutdownBin, err := common.LookupPath("systemd-shutdown", "/usr/lib/systemd:/lib/systemd")
	if err != nil {
		return err
	}
	systemdBin, err := common.LookupPath("systemd", "/usr/lib/systemd:/lib/systemd")
	if err != nil {
		return err
	}
	systemdJournaldBin, err := common.LookupPath("systemd-journald", "/usr/lib/systemd:/lib/systemd")
	if err != nil {
		return err
	}

	systemdUnitsPath := "/lib/systemd/system"
	assets := []string{
		proj2aci.GetAssetString("/usr/lib/systemd/systemd", systemdBin),
		proj2aci.GetAssetString("/usr/bin/systemctl", systemctlBin),
		proj2aci.GetAssetString("/usr/lib/systemd/systemd-journald", systemdJournaldBin),
		proj2aci.GetAssetString("/usr/bin/bash", bashBin),
		proj2aci.GetAssetString(fmt.Sprintf("%s/systemd-journald.service", systemdUnitsPath), fmt.Sprintf("%s/systemd-journald.service", systemdUnitsPath)),
		proj2aci.GetAssetString(fmt.Sprintf("%s/systemd-journald.socket", systemdUnitsPath), fmt.Sprintf("%s/systemd-journald.socket", systemdUnitsPath)),
		proj2aci.GetAssetString(fmt.Sprintf("%s/systemd-journald-dev-log.socket", systemdUnitsPath), fmt.Sprintf("%s/systemd-journald-dev-log.socket", systemdUnitsPath)),
		proj2aci.GetAssetString(fmt.Sprintf("%s/systemd-journald-audit.socket", systemdUnitsPath), fmt.Sprintf("%s/systemd-journald-audit.socket", systemdUnitsPath)),
		// systemd-shutdown has to be installed at the same path as on the host
		// because it depends on systemd build flag -DSYSTEMD_SHUTDOWN_BINARY_PATH=
		proj2aci.GetAssetString(systemdShutdownBin, systemdShutdownBin),
	}

	return proj2aci.PrepareAssets(assets, "./stage1/rootfs/", nil)
}

// getArgsEnv returns the nspawn or lkvm args and env according to the flavor used
func getArgsEnv(p *stage1commontypes.Pod, flavor string, debug bool, n *networking.Networking) ([]string, []string, error) {
	var args []string
	env := os.Environ()

	// We store the pod's flavor so we can later garbage collect it correctly
	if err := os.Symlink(flavor, filepath.Join(p.Root, stage1initcommon.FlavorFile)); err != nil {
		return nil, nil, errwrap.Wrap(errors.New("failed to create flavor symlink"), err)
	}

	// set hostname inside pod
	// According to systemd manual (https://www.freedesktop.org/software/systemd/man/hostname.html) :
	// "The /etc/hostname file configures the name of the local system that is set
	// during boot using the sethostname system call"
	if hostname == "" {
		hostname = stage1initcommon.GetMachineID(p)
	}
	hostnamePath := filepath.Join(common.Stage1RootfsPath(p.Root), "etc/hostname")
	if err := ioutil.WriteFile(hostnamePath, []byte(hostname), 0644); err != nil {
		return nil, nil, fmt.Errorf("error writing %s, %s", hostnamePath, err)
	}

	switch flavor {
	case "kvm":
		if privateUsers != "" {
			return nil, nil, fmt.Errorf("flag --private-users cannot be used with an lkvm stage1")
		}

		// kernel and lkvm are relative path, because init has /var/lib/rkt/..../uuid as its working directory
		// TODO: move to path.go
		kernelPath := filepath.Join(common.Stage1RootfsPath(p.Root), "bzImage")
		lkvmPath := filepath.Join(common.Stage1RootfsPath(p.Root), "lkvm")
		netDescriptions := kvm.GetNetworkDescriptions(n)
		lkvmNetArgs, err := kvm.GetKVMNetArgs(netDescriptions)
		if err != nil {
			return nil, nil, err
		}

		cpu, mem := kvm.GetAppsResources(p.Manifest.Apps)

		kernelParams := []string{
			"console=hvc0",
			"init=/usr/lib/systemd/systemd",
			"no_timer_check",
			"noreplace-smp",
			"systemd.default_standard_error=journal+console",
			"systemd.default_standard_output=journal+console",
			// "systemd.default_standard_output=tty",
			"tsc=reliable",
			"MACHINEID=" + p.UUID.String(),
		}

		if debug {
			kernelParams = append(kernelParams, []string{
				"debug",
				"systemd.log_level=debug",
				"systemd.show_status=true",
				// "systemd.confirm_spawn=true",
			}...)
		} else {
			kernelParams = append(kernelParams, "quiet")
		}

		args = append(args, []string{
			"./" + lkvmPath, // relative path
			"run",
			"--name", "rkt-" + p.UUID.String(),
			"--no-dhcp", // speed bootup
			"--cpu", strconv.FormatInt(cpu, 10),
			"--mem", strconv.FormatInt(mem, 10),
			"--console=virtio",
			"--kernel", kernelPath,
			"--disk", "stage1/rootfs", // relative to run/pods/uuid dir this is a place where systemd resides
			// MACHINEID will be available as environment variable
			"--params", strings.Join(kernelParams, " "),
		}...,
		)
		args = append(args, lkvmNetArgs...)

		if debug {
			args = append(args, "--debug")
		}

		// host volume sharing with 9p
		nsargs := stage1initcommon.VolumesToKvmDiskArgs(p.Manifest.Volumes)
		args = append(args, nsargs...)

		// lkvm requires $HOME to be defined,
		// see https://github.com/coreos/rkt/issues/1393
		if os.Getenv("HOME") == "" {
			env = append(env, "HOME=/root")
		}

		return args, env, nil

	case "coreos":
		args = append(args, filepath.Join(common.Stage1RootfsPath(p.Root), interpBin))
		args = append(args, filepath.Join(common.Stage1RootfsPath(p.Root), nspawnBin))
		args = append(args, "--boot") // Launch systemd in the pod

		if context := os.Getenv(common.EnvSELinuxContext); context != "" {
			args = append(args, fmt.Sprintf("-Z%s", context))
		}

		if context := os.Getenv(common.EnvSELinuxMountContext); context != "" {
			args = append(args, fmt.Sprintf("-L%s", context))
		}

		if machinedRegister() {
			args = append(args, fmt.Sprintf("--register=true"))
		} else {
			args = append(args, fmt.Sprintf("--register=false"))
		}

		// use only dynamic libraries provided in the image
		env = append(env, "LD_LIBRARY_PATH="+filepath.Join(common.Stage1RootfsPath(p.Root), "usr/lib"))

	case "src":
		args = append(args, filepath.Join(common.Stage1RootfsPath(p.Root), nspawnBin))
		args = append(args, "--boot") // Launch systemd in the pod

		if context := os.Getenv(common.EnvSELinuxContext); context != "" {
			args = append(args, fmt.Sprintf("-Z%s", context))
		}

		if context := os.Getenv(common.EnvSELinuxMountContext); context != "" {
			args = append(args, fmt.Sprintf("-L%s", context))
		}

		if machinedRegister() {
			args = append(args, fmt.Sprintf("--register=true"))
		} else {
			args = append(args, fmt.Sprintf("--register=false"))
		}

	case "host":
		hostNspawnBin, err := common.LookupPath("systemd-nspawn", os.Getenv("PATH"))
		if err != nil {
			return nil, nil, err
		}

		// Check dynamically which version is installed on the host
		// Support version >= 220
		versionBytes, err := exec.Command(hostNspawnBin, "--version").CombinedOutput()
		if err != nil {
			return nil, nil, errwrap.Wrap(fmt.Errorf("unable to probe %s version", hostNspawnBin), err)
		}
		versionStr := strings.SplitN(string(versionBytes), "\n", 2)[0]
		var version int
		n, err := fmt.Sscanf(versionStr, "systemd %d", &version)
		if err != nil {
			return nil, nil, fmt.Errorf("cannot parse version: %q", versionStr)
		}
		if n != 1 || version < 220 {
			return nil, nil, fmt.Errorf("rkt needs systemd-nspawn >= 220. %s version not supported: %v", hostNspawnBin, versionStr)
		}

		// Copy systemd, bash, etc. in stage1 at run-time
		if err := installAssets(); err != nil {
			return nil, nil, errwrap.Wrap(errors.New("cannot install assets from the host"), err)
		}

		args = append(args, hostNspawnBin)
		args = append(args, "--boot") // Launch systemd in the pod
		args = append(args, fmt.Sprintf("--register=true"))

		if context := os.Getenv(common.EnvSELinuxContext); context != "" {
			args = append(args, fmt.Sprintf("-Z%s", context))
		}

		if context := os.Getenv(common.EnvSELinuxMountContext); context != "" {
			args = append(args, fmt.Sprintf("-L%s", context))
		}

	default:
		return nil, nil, fmt.Errorf("unrecognized stage1 flavor: %q", flavor)
	}

	// link journal only if the host is running systemd
	if util.IsRunningSystemd() {
		// we write /etc/machine-id here because systemd-nspawn needs it to link
		// the container's journal to the host
		mPath := filepath.Join(common.Stage1RootfsPath(p.Root), "etc", "machine-id")
		mID := strings.Replace(p.UUID.String(), "-", "", -1)

		if err := ioutil.WriteFile(mPath, []byte(mID), 0644); err != nil {
			log.FatalE("error writing /etc/machine-id", err)
		}

		args = append(args, "--link-journal=try-guest")

		keepUnit, err := util.RunningFromSystemService()
		if err != nil {
			if err == util.ErrSoNotFound {
				log.Print("warning: libsystemd not found even though systemd is running. Cgroup limits set by the environment (e.g. a systemd service) won't be enforced.")
			} else {
				return nil, nil, errwrap.Wrap(errors.New("error determining if we're running from a system service"), err)
			}
		}

		if keepUnit {
			args = append(args, "--keep-unit")
		}
	}

	if !debug {
		args = append(args, "--quiet")             // silence most nspawn output (log_warning is currently not covered by this)
		env = append(env, "SYSTEMD_LOG_LEVEL=err") // silence log_warning too
	}

	env = append(env, "SYSTEMD_NSPAWN_CONTAINER_SERVICE=rkt")

	if len(privateUsers) > 0 {
		args = append(args, "--private-users="+privateUsers)
	}

	nsargs, err := stage1initcommon.PodToNspawnArgs(p)
	if err != nil {
		return nil, nil, errwrap.Wrap(errors.New("failed to generate nspawn args"), err)
	}
	args = append(args, nsargs...)

	// Arguments to systemd
	args = append(args, "--")
	args = append(args, "--default-standard-output=tty") // redirect all service logs straight to tty
	if !debug {
		args = append(args, "--log-target=null") // silence systemd output inside pod
		args = append(args, "--show-status=0")   // silence systemd initialization status output
	}

	return args, env, nil
}

func forwardedPorts(pod *stage1commontypes.Pod) ([]networking.ForwardedPort, error) {
	var fps []networking.ForwardedPort

NextPort:
	for _, ep := range pod.Manifest.Ports {
		n := ""
		fp := networking.ForwardedPort{}

		for _, a := range pod.Manifest.Apps {
			for _, p := range a.App.Ports {
				if p.Name == ep.Name {
					if n == "" {
						// skip socket-activated ports, they don't need port forwarding
						if p.SocketActivated {
							continue NextPort
						}
						fp.Protocol = p.Protocol
						fp.HostPort = ep.HostPort
						fp.PodPort = p.Port
						n = a.Name.String()
					} else {
						return nil, fmt.Errorf("ambiguous exposed port in PodManifest: %q and %q both define port %q", n, a.Name, p.Name)
					}
				}
			}
		}

		if n == "" {
			return nil, fmt.Errorf("port name %q is not defined by any apps", ep.Name)
		}

		fps = append(fps, fp)
	}

	// TODO(eyakubovich): validate that there're no conflicts

	return fps, nil
}

func stage1() int {
	uuid, err := types.NewUUID(flag.Arg(0))
	if err != nil {
		log.PrintE("UUID is missing or malformed", err)
		return 1
	}

	root := "."
	p, err := stage1commontypes.LoadPod(root, uuid)
	if err != nil {
		log.PrintE("failed to load pod", err)
		return 1
	}

	// set close-on-exec flag on RKT_LOCK_FD so it gets correctly closed when invoking
	// network plugins
	lfd, err := common.GetRktLockFD()
	if err != nil {
		log.PrintE("failed to get rkt lock fd", err)
		return 1
	}

	if err := sys.CloseOnExec(lfd, true); err != nil {
		log.PrintE("failed to set FD_CLOEXEC on rkt lock", err)
		return 1
	}

	mirrorLocalZoneInfo(p.Root)

	flavor, _, err := stage1initcommon.GetFlavor(p)
	if err != nil {
		log.PrintE("failed to get stage1 flavor", err)
		return 3
	}

	var n *networking.Networking
	if netList.Contained() {
		fps, err := forwardedPorts(p)
		if err != nil {
			log.Error(err)
			return 6
		}

		n, err = networking.Setup(root, p.UUID, fps, netList, localConfig, flavor, debug)
		if err != nil {
			log.PrintE("failed to setup network", err)
			return 6
		}

		if err = n.Save(); err != nil {
			log.PrintE("failed to save networking state", err)
			n.Teardown(flavor, debug)
			return 6
		}

		if len(mdsToken) > 0 {
			hostIP, err := n.GetDefaultHostIP()
			if err != nil {
				log.PrintE("failed to get default Host IP", err)
				return 6
			}

			p.MetadataServiceURL = common.MetadataServicePublicURL(hostIP, mdsToken)
		}
	} else {
		if flavor == "kvm" {
			log.Print("flavor kvm requires private network configuration (try --net)")
			return 6
		}
		if len(mdsToken) > 0 {
			p.MetadataServiceURL = common.MetadataServicePublicURL(localhostIP, mdsToken)
		}
	}

	if err = stage1initcommon.WriteDefaultTarget(p); err != nil {
		log.PrintE("failed to write default.target", err)
		return 2
	}

	if err = stage1initcommon.WritePrepareAppTemplate(p); err != nil {
		log.PrintE("failed to write prepare-app service template", err)
		return 2
	}

	if err := stage1initcommon.SetJournalPermissions(p); err != nil {
		log.PrintE("warning: error setting journal ACLs, you'll need root to read the pod journal", err)
	}

	if flavor == "kvm" {
		if err := KvmPodToSystemd(p, n); err != nil {
			log.PrintE("failed to configure systemd for kvm", err)
			return 2
		}
	}

	if err = stage1initcommon.PodToSystemd(p, interactive, flavor, privateUsers); err != nil {
		log.PrintE("failed to configure systemd", err)
		return 2
	}

	args, env, err := getArgsEnv(p, flavor, debug, n)
	if err != nil {
		log.Error(err)
		return 3
	}

	// create a separate mount namespace so the cgroup filesystems
	// are unmounted when exiting the pod
	if err := syscall.Unshare(syscall.CLONE_NEWNS); err != nil {
		log.FatalE("error unsharing", err)
	}

	// we recursively make / a "shared and slave" so mount events from the
	// new namespace don't propagate to the host namespace but mount events
	// from the host propagate to the new namespace and are forwarded to
	// its peer group
	// See https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt
	if err := syscall.Mount("", "/", "none", syscall.MS_REC|syscall.MS_SLAVE, ""); err != nil {
		log.FatalE("error making / a slave mount", err)
	}
	if err := syscall.Mount("", "/", "none", syscall.MS_REC|syscall.MS_SHARED, ""); err != nil {
		log.FatalE("error making / a shared and slave mount", err)
	}

	enabledCgroups, err := cgroup.GetEnabledCgroups()
	if err != nil {
		log.FatalE("error getting cgroups", err)
		return 5
	}

	// mount host cgroups in the rkt mount namespace
	if err := mountHostCgroups(enabledCgroups); err != nil {
		log.FatalE("couldn't mount the host cgroups", err)
		return 5
	}

	var serviceNames []string
	for _, app := range p.Manifest.Apps {
		serviceNames = append(serviceNames, stage1initcommon.ServiceUnitName(app.Name))
	}
	s1Root := common.Stage1RootfsPath(p.Root)
	machineID := stage1initcommon.GetMachineID(p)
	subcgroup, err := getContainerSubCgroup(machineID)
	if err == nil {
		if err := mountContainerCgroups(s1Root, enabledCgroups, subcgroup, serviceNames); err != nil {
			log.PrintE("couldn't mount the container cgroups", err)
			return 5
		}
	} else {
		log.PrintE("continuing with per-app isolators disabled", err)
	}

	if err = stage1common.WritePpid(os.Getpid()); err != nil {
		log.Error(err)
		return 4
	}

	err = stage1common.WithClearedCloExec(lfd, func() error {
		return syscall.Exec(args[0], args, env)
	})
	if err != nil {
		log.PrintE(fmt.Sprintf("failed to execute %q", args[0]), err)
		return 7
	}

	return 0
}

func areHostCgroupsMounted(enabledCgroups map[int][]string) bool {
	controllers := cgroup.GetControllerDirs(enabledCgroups)
	for _, c := range controllers {
		if !cgroup.IsControllerMounted(c) {
			return false
		}
	}

	return true
}

// mountHostCgroups mounts the host cgroup hierarchy as required by
// systemd-nspawn. We need this because some distributions don't have the
// "name=systemd" cgroup or don't mount the cgroup controllers in
// "/sys/fs/cgroup", and systemd-nspawn needs this. Since this is mounted
// inside the rkt mount namespace, it doesn't affect the host.
func mountHostCgroups(enabledCgroups map[int][]string) error {
	systemdControllerPath := "/sys/fs/cgroup/systemd"
	if !areHostCgroupsMounted(enabledCgroups) {
		mountContext := os.Getenv(common.EnvSELinuxMountContext)
		if err := cgroup.CreateCgroups("/", enabledCgroups, mountContext); err != nil {
			return errwrap.Wrap(errors.New("error creating host cgroups"), err)
		}
	}

	if !cgroup.IsControllerMounted("systemd") {
		if err := os.MkdirAll(systemdControllerPath, 0700); err != nil {
			return err
		}
		if err := syscall.Mount("cgroup", systemdControllerPath, "cgroup", 0, "none,name=systemd"); err != nil {
			return errwrap.Wrap(fmt.Errorf("error mounting name=systemd hierarchy on %q", systemdControllerPath), err)
		}
	}

	return nil
}

// mountContainerCgroups mounts the cgroup controllers hierarchy in the container's
// namespace read-only, leaving the needed knobs in the subcgroup for each-app
// read-write so systemd inside stage1 can apply isolators to them
func mountContainerCgroups(s1Root string, enabledCgroups map[int][]string, subcgroup string, serviceNames []string) error {
	mountContext := os.Getenv(common.EnvSELinuxMountContext)
	if err := cgroup.CreateCgroups(s1Root, enabledCgroups, mountContext); err != nil {
		return errwrap.Wrap(errors.New("error creating container cgroups"), err)
	}
	if err := cgroup.RemountCgroupsRO(s1Root, enabledCgroups, subcgroup, serviceNames); err != nil {
		return errwrap.Wrap(errors.New("error restricting container cgroups"), err)
	}

	return nil
}

func getContainerSubCgroup(machineID string) (string, error) {
	var subcgroup string
	fromUnit, err := util.RunningFromSystemService()
	if err != nil {
		return "", errwrap.Wrap(errors.New("could not determine if we're running from a unit file"), err)
	}
	if fromUnit {
		slice, err := util.GetRunningSlice()
		if err != nil {
			return "", errwrap.Wrap(errors.New("could not get slice name"), err)
		}
		slicePath, err := common.SliceToPath(slice)
		if err != nil {
			return "", errwrap.Wrap(errors.New("could not convert slice name to path"), err)
		}
		unit, err := util.CurrentUnitName()
		if err != nil {
			return "", errwrap.Wrap(errors.New("could not get unit name"), err)
		}
		subcgroup = filepath.Join(slicePath, unit, "system.slice")
	} else {
		escapedmID := strings.Replace(machineID, "-", "\\x2d", -1)
		machineDir := "machine-" + escapedmID + ".scope"
		if machinedRegister() {
			// we are not in the final cgroup yet: systemd-nspawn will move us
			// to the correct cgroup later during registration so we can't
			// look it up in /proc/self/cgroup
			subcgroup = filepath.Join("machine.slice", machineDir, "system.slice")
		} else {
			// when registration is disabled the container will be directly
			// under the current cgroup so we can look it up in /proc/self/cgroup
			ownCgroupPath, err := cgroup.GetOwnCgroupPath("name=systemd")
			if err != nil {
				return "", errwrap.Wrap(errors.New("could not get own cgroup path"), err)
			}
			// systemd-nspawn won't work if we are in the root cgroup. In addition,
			// we want all rkt instances to be in distinct cgroups. Create a
			// subcgroup and add ourselves to it.
			ownCgroupPath = filepath.Join(ownCgroupPath, machineDir)
			if err := cgroup.JoinSubcgroup("systemd", ownCgroupPath); err != nil {
				return "", errwrap.Wrap(fmt.Errorf("error joining %s subcgroup", ownCgroupPath), err)
			}
			subcgroup = filepath.Join(ownCgroupPath, "system.slice")
		}
	}

	return subcgroup, nil
}

func main() {
	flag.Parse()

	stage1initcommon.InitDebug(debug)

	log, diag, _ = rktlog.NewLogSet("stage1", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	// move code into stage1() helper so deferred fns get run
	os.Exit(stage1())
}
