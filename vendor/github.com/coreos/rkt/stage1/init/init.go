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
	"strings"
	"syscall"

	"github.com/appc/goaci/proj2aci"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/util"
	"github.com/coreos/pkg/dlopen"
	"github.com/godbus/dbus"
	"github.com/godbus/dbus/introspect"
	"github.com/hashicorp/errwrap"

	stage1common "github.com/coreos/rkt/stage1/common"
	stage1commontypes "github.com/coreos/rkt/stage1/common/types"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/cgroup"
	"github.com/coreos/rkt/common/cgroup/v1"
	"github.com/coreos/rkt/common/cgroup/v2"
	commonnet "github.com/coreos/rkt/common/networking"
	"github.com/coreos/rkt/networking"
	pkgflag "github.com/coreos/rkt/pkg/flag"
	"github.com/coreos/rkt/pkg/fs"
	rktlog "github.com/coreos/rkt/pkg/log"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/coreos/rkt/pkg/user"
	"github.com/coreos/rkt/stage1/init/kvm"
	"github.com/coreos/rkt/stage1/init/kvm/hypervisor/hvlkvm"
	"github.com/coreos/rkt/stage1/init/kvm/hypervisor/hvqemu"
)

const (
	// Path to systemd-nspawn binary within the stage1 rootfs
	nspawnBin = "/usr/bin/systemd-nspawn"
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
	debug       bool
	localhostIP net.IP
	localConfig string
	log         *rktlog.Logger
	diag        *rktlog.Logger
	interpBin   string // Path to the interpreter within the stage1 rootfs, set by the linker
)

func parseFlags() *stage1commontypes.RuntimePod {
	rp := stage1commontypes.RuntimePod{}

	flag.BoolVar(&debug, "debug", false, "Run in debug mode")
	flag.StringVar(&localConfig, "local-config", common.DefaultLocalConfigDir, "Local config path")

	// These flags are persisted in the PodRuntime
	flag.BoolVar(&rp.Interactive, "interactive", false, "The pod is interactive")
	flag.BoolVar(&rp.Mutable, "mutable", false, "Enable mutable operations on this pod, including starting an empty one")
	flag.Var(&rp.NetList, "net", "Setup networking")
	flag.StringVar(&rp.PrivateUsers, "private-users", "", "Run within user namespace. Can be set to [=UIDBASE[:NUIDS]]")
	flag.StringVar(&rp.MDSToken, "mds-token", "", "MDS auth token")
	flag.StringVar(&rp.Hostname, "hostname", "", "Hostname of the pod")
	flag.BoolVar(&rp.InsecureOptions.DisableCapabilities, "disable-capabilities-restriction", false, "Disable capability restrictions")
	flag.BoolVar(&rp.InsecureOptions.DisablePaths, "disable-paths", false, "Disable paths restrictions")
	flag.BoolVar(&rp.InsecureOptions.DisableSeccomp, "disable-seccomp", false, "Disable seccomp restrictions")
	dnsConfMode := pkgflag.MustNewPairList(map[string][]string{
		"resolv": {"host", "stage0", "none", "default"},
		"hosts":  {"host", "stage0", "default"},
	}, map[string]string{
		"resolv": "default",
		"hosts":  "default",
	})
	flag.Var(dnsConfMode, "dns-conf-mode", "DNS config file modes")

	flag.Parse()

	rp.Debug = debug
	rp.ResolvConfMode = dnsConfMode.Pairs["resolv"]
	rp.EtcHostsMode = dnsConfMode.Pairs["hosts"]

	return &rp
}

func init() {
	// this ensures that main runs only on main thread (thread group leader).
	// since namespace ops (unshare, setns) are done for a single thread, we
	// must ensure that the goroutine does not jump from OS thread to thread
	runtime.LockOSThread()

	// We'll need this later
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
	systemdSysusersBin, err := common.LookupPath("systemd-sysusers", os.Getenv("PATH"))
	if err != nil {
		return err
	}
	systemdTmpfilesBin, err := common.LookupPath("systemd-tmpfiles", os.Getenv("PATH"))
	if err != nil {
		return err
	}
	bashBin, err := common.LookupPath("bash", os.Getenv("PATH"))
	if err != nil {
		return err
	}
	mountBin, err := common.LookupPath("mount", os.Getenv("PATH"))
	if err != nil {
		return err
	}
	umountBin, err := common.LookupPath("umount", os.Getenv("PATH"))
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
		proj2aci.GetAssetString("/usr/bin/systemd-sysusers", systemdSysusersBin),
		proj2aci.GetAssetString("/usr/bin/systemd-tmpfiles", systemdTmpfilesBin),
		proj2aci.GetAssetString("/usr/lib/systemd/systemd-journald", systemdJournaldBin),
		proj2aci.GetAssetString("/usr/bin/bash", bashBin),
		proj2aci.GetAssetString("/bin/mount", mountBin),
		proj2aci.GetAssetString("/bin/umount", umountBin),
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

// getArgsEnv returns the nspawn or lkvm args and env according to the flavor
// as the first two return values respectively.
func getArgsEnv(p *stage1commontypes.Pod, flavor string, canMachinedRegister bool, debug bool, n *networking.Networking) ([]string, []string, error) {
	var args []string
	env := os.Environ()

	// We store the pod's flavor so we can later garbage collect it correctly
	if err := os.Symlink(flavor, filepath.Join(p.Root, stage1initcommon.FlavorFile)); err != nil {
		return nil, nil, errwrap.Wrap(errors.New("failed to create flavor symlink"), err)
	}

	// systemd-nspawn needs /etc/machine-id to link the container's journal
	// to the host. Since systemd-v230, /etc/machine-id is mandatory, see
	// https://github.com/systemd/systemd/commit/e01ff70a77e781734e1e73a2238af2e9bf7967a8
	mPath := filepath.Join(common.Stage1RootfsPath(p.Root), "etc", "machine-id")
	machineID := strings.Replace(p.UUID.String(), "-", "", -1)

	switch flavor {
	case "kvm":
		if p.PrivateUsers != "" {
			return nil, nil, fmt.Errorf("flag --private-users cannot be used with an lkvm stage1")
		}

		// kernel and hypervisor binaries are located relative to the working directory
		// of init (/var/lib/rkt/..../uuid)
		// TODO: move to path.go
		kernelPath := filepath.Join(common.Stage1RootfsPath(p.Root), "bzImage")
		netDescriptions := kvm.GetNetworkDescriptions(n)

		cpu, mem := kvm.GetAppsResources(p.Manifest.Apps)

		// Parse hypervisor
		hv, err := KvmCheckHypervisor(common.Stage1RootfsPath(p.Root))
		if err != nil {
			return nil, nil, err
		}

		// Set start command for hypervisor
		StartCmd := hvlkvm.StartCmd
		switch hv {
		case "lkvm":
			StartCmd = hvlkvm.StartCmd
		case "qemu":
			StartCmd = hvqemu.StartCmd
		default:
			return nil, nil, fmt.Errorf("unrecognized hypervisor")
		}

		hvStartCmd := StartCmd(
			common.Stage1RootfsPath(p.Root),
			p.UUID.String(),
			kernelPath,
			netDescriptions,
			cpu,
			mem,
			debug,
		)

		if hvStartCmd == nil {
			return nil, nil, fmt.Errorf("no hypervisor")
		}

		args = append(args, hvStartCmd...)

		// lkvm requires $HOME to be defined,
		// see https://github.com/coreos/rkt/issues/1393
		if os.Getenv("HOME") == "" {
			env = append(env, "HOME=/root")
		}

		if err := linkJournal(common.Stage1RootfsPath(p.Root), machineID); err != nil {
			return nil, nil, errwrap.Wrap(errors.New("error linking pod's journal"), err)
		}

		// use only dynamic libraries provided in the image
		// from systemd v231 there's a new internal libsystemd-shared-v231.so
		// which is present in /usr/lib/systemd
		env = append(env, "LD_LIBRARY_PATH="+filepath.Join(common.Stage1RootfsPath(p.Root), "usr/lib/systemd"))

		return args, env, nil

	case "coreos":
		args = append(args, filepath.Join(common.Stage1RootfsPath(p.Root), interpBin))
		args = append(args, filepath.Join(common.Stage1RootfsPath(p.Root), nspawnBin))
		args = append(args, "--boot")             // Launch systemd in the pod
		args = append(args, "--notify-ready=yes") // From systemd v231

		if context := os.Getenv(common.EnvSELinuxContext); context != "" {
			args = append(args, fmt.Sprintf("-Z%s", context))
		}

		if context := os.Getenv(common.EnvSELinuxMountContext); context != "" {
			args = append(args, fmt.Sprintf("-L%s", context))
		}

		if canMachinedRegister {
			args = append(args, fmt.Sprintf("--register=true"))
		} else {
			args = append(args, fmt.Sprintf("--register=false"))
		}

		// use only dynamic libraries provided in the image
		// from systemd v231 there's a new internal libsystemd-shared-v231.so
		// which is present in /usr/lib/systemd
		env = append(env, "LD_LIBRARY_PATH="+
			filepath.Join(common.Stage1RootfsPath(p.Root), "usr/lib")+":"+
			filepath.Join(common.Stage1RootfsPath(p.Root), "usr/lib/systemd"))

	case "src":
		args = append(args, filepath.Join(common.Stage1RootfsPath(p.Root), interpBin))
		args = append(args, filepath.Join(common.Stage1RootfsPath(p.Root), nspawnBin))
		args = append(args, "--boot")             // Launch systemd in the pod
		args = append(args, "--notify-ready=yes") // From systemd v231

		if context := os.Getenv(common.EnvSELinuxContext); context != "" {
			args = append(args, fmt.Sprintf("-Z%s", context))
		}

		if context := os.Getenv(common.EnvSELinuxMountContext); context != "" {
			args = append(args, fmt.Sprintf("-L%s", context))
		}

		if canMachinedRegister {
			args = append(args, fmt.Sprintf("--register=true"))
		} else {
			args = append(args, fmt.Sprintf("--register=false"))
		}

		// use only dynamic libraries provided in the image
		// from systemd v231 there's a new internal libsystemd-shared-v231.so
		// which is present in /usr/lib/systemd
		env = append(env, "LD_LIBRARY_PATH="+
			filepath.Join(common.Stage1RootfsPath(p.Root), "usr/lib")+":"+
			filepath.Join(common.Stage1RootfsPath(p.Root), "usr/lib/systemd"))

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

		if version >= 231 {
			args = append(args, "--notify-ready=yes") // From systemd v231
		}

		if context := os.Getenv(common.EnvSELinuxContext); context != "" {
			args = append(args, fmt.Sprintf("-Z%s", context))
		}

		if context := os.Getenv(common.EnvSELinuxMountContext); context != "" {
			args = append(args, fmt.Sprintf("-L%s", context))
		}

	default:
		return nil, nil, fmt.Errorf("unrecognized stage1 flavor: %q", flavor)
	}

	machineIDBytes := append([]byte(machineID), '\n')
	if err := ioutil.WriteFile(mPath, machineIDBytes, 0644); err != nil {
		return nil, nil, errwrap.Wrap(errors.New("error writing /etc/machine-id"), err)
	}
	if err := user.ShiftFiles([]string{mPath}, &p.UidRange); err != nil {
		return nil, nil, errwrap.Wrap(errors.New("error shifting /etc/machine-id"), err)
	}

	// link journal only if the host is running systemd
	if util.IsRunningSystemd() {
		args = append(args, "--link-journal=try-guest")

		keepUnit, err := util.RunningFromSystemService()
		if err != nil {
			if err == dlopen.ErrSoNotFound {
				log.Print("warning: libsystemd not found even though systemd is running. Cgroup limits set by the environment (e.g. a systemd service) won't be enforced.")
			} else {
				return nil, nil, errwrap.Wrap(errors.New("error determining if we're running from a system service"), err)
			}
		}

		if keepUnit {
			args = append(args, "--keep-unit")
		}
	} else {
		args = append(args, "--link-journal=no")
	}

	if !debug {
		args = append(args, "--quiet")             // silence most nspawn output (log_warning is currently not covered by this)
		env = append(env, "SYSTEMD_LOG_LEVEL=err") // silence log_warning too
	}

	env = append(env, "SYSTEMD_NSPAWN_CONTAINER_SERVICE=rkt")
	// TODO (alepuccetti) remove this line when rkt will use cgroup namespace
	// If the kernel has the cgroup namespace enabled, systemd v232 will use it by default.
	// This was introduced by https://github.com/systemd/systemd/pull/3809 and it will cause
	// problems in rkt when cgns is enabled and cgroup-v1 is used. For more information see
	// https://github.com/systemd/systemd/pull/3589#discussion_r70277625.
	// The following line tells systemd-nspawn not to use cgroup namespace using the environment variable
	// introduced by https://github.com/systemd/systemd/pull/3809.
	env = append(env, "SYSTEMD_NSPAWN_USE_CGNS=no")

	if p.InsecureOptions.DisablePaths {
		env = append(env, "SYSTEMD_NSPAWN_API_VFS_WRITABLE=yes")
	}

	if len(p.PrivateUsers) > 0 {
		args = append(args, "--private-users="+p.PrivateUsers)
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

func stage1(rp *stage1commontypes.RuntimePod) int {
	uuid, err := types.NewUUID(flag.Arg(0))
	if err != nil {
		log.FatalE("UUID is missing or malformed", err)
	}

	root := "."
	p, err := stage1commontypes.LoadPod(root, uuid, rp)
	if err != nil {
		log.FatalE("failed to load pod", err)
	}

	if err := p.SaveRuntime(); err != nil {
		log.FatalE("failed to save runtime parameters", err)
	}

	// set close-on-exec flag on RKT_LOCK_FD so it gets correctly closed when invoking
	// network plugins
	lfd, err := common.GetRktLockFD()
	if err != nil {
		log.FatalE("failed to get rkt lock fd", err)
	}

	if err := sys.CloseOnExec(lfd, true); err != nil {
		log.FatalE("failed to set FD_CLOEXEC on rkt lock", err)
	}

	mirrorLocalZoneInfo(p.Root)

	flavor, _, err := stage1initcommon.GetFlavor(p)
	if err != nil {
		log.FatalE("failed to get stage1 flavor", err)
	}

	var n *networking.Networking
	if p.NetList.Contained() {
		fps, err := commonnet.ForwardedPorts(p.Manifest)
		if err != nil {
			log.FatalE("error initializing forwarding ports", err)
		}

		noDNS := p.ResolvConfMode != "default" // force ignore CNI DNS results
		n, err = networking.Setup(root, p.UUID, fps, p.NetList, localConfig, flavor, noDNS, debug)
		if err != nil {
			log.FatalE("failed to setup network", err)
		}

		if err = n.Save(); err != nil {
			log.PrintE("failed to save networking state", err)
			n.Teardown(flavor, debug)
			return 254
		}

		if len(p.MDSToken) > 0 {
			hostIP, err := n.GetForwardableNetHostIP()
			if err != nil {
				log.FatalE("failed to get default Host IP", err)
			}

			p.MetadataServiceURL = common.MetadataServicePublicURL(hostIP, p.MDSToken)
		}
	} else {
		if flavor == "kvm" {
			log.Fatal("flavor kvm requires private network configuration (try --net)")
		}
		if len(p.MDSToken) > 0 {
			p.MetadataServiceURL = common.MetadataServicePublicURL(localhostIP, p.MDSToken)
		}
	}

	mnt := fs.NewLoggingMounter(
		fs.MounterFunc(syscall.Mount),
		fs.UnmounterFunc(syscall.Unmount),
		diag.Printf,
	)

	// set hostname inside pod
	// According to systemd manual (https://www.freedesktop.org/software/systemd/man/hostname.html) :
	// "The /etc/hostname file configures the name of the local system that is set
	// during boot using the sethostname system call"
	if p.Hostname == "" {
		p.Hostname = stage1initcommon.GetMachineID(p)
	}
	hostnamePath := filepath.Join(common.Stage1RootfsPath(p.Root), "etc/hostname")
	if err := ioutil.WriteFile(hostnamePath, []byte(p.Hostname), 0644); err != nil {
		log.PrintE("error writing "+hostnamePath, err)
		return 254
	}
	if err := user.ShiftFiles([]string{hostnamePath}, &p.UidRange); err != nil {
		log.PrintE("error shifting "+hostnamePath, err)
	}

	if p.ResolvConfMode == "host" {
		stage1initcommon.UseHostResolv(mnt, root)
	}

	// Set up the hosts file.
	// We write <stage1>/etc/rkt-hosts if we want to override each app's hosts,
	// and <stage1>/etc/hosts-fallback if we want to let the app "win"
	// Either way, we should add our hostname to it, unless the hosts's
	// /etc/hosts is bind-mounted in.
	if p.EtcHostsMode == "host" { // We should bind-mount the hosts's /etc/hosts
		stage1initcommon.UseHostHosts(mnt, root)
	} else if p.EtcHostsMode == "default" { // Create hosts-fallback
		hostsFile := filepath.Join(common.Stage1RootfsPath(p.Root), "etc", "hosts-fallback")
		if err := stage1initcommon.AddHostsEntry(hostsFile, "127.0.0.1", p.Hostname); err != nil {
			log.PrintE("Failed to write hostname to "+hostsFile, err)
			return 254
		}
	} else if p.EtcHostsMode == "stage0" { // The stage0 has created rkt-hosts
		hostsFile := filepath.Join(common.Stage1RootfsPath(p.Root), "etc", "rkt-hosts")
		if err := stage1initcommon.AddHostsEntry(hostsFile, "127.0.0.1", p.Hostname); err != nil {
			log.PrintE("Failed to write hostname to "+hostsFile, err)
			return 254
		}
	}

	if p.Mutable {
		if err = stage1initcommon.MutableEnv(p); err != nil {
			log.FatalE("cannot initialize mutable environment", err)
		}
	} else {
		if err = stage1initcommon.ImmutableEnv(p); err != nil {
			log.FatalE("cannot initialize immutable environment", err)
		}
	}

	if err := stage1initcommon.SetJournalPermissions(p); err != nil {
		log.PrintE("warning: error setting journal ACLs, you'll need root to read the pod journal", err)
	}

	if flavor == "kvm" {
		kvm.InitDebug(debug)
		if err := KvmNetworkingToSystemd(p, n); err != nil {
			log.FatalE("failed to configure systemd for kvm", err)
		}
	}

	canMachinedRegister := false
	if flavor != "kvm" {
		// kvm doesn't register with systemd right now, see #2664.
		canMachinedRegister = machinedRegister()
	}
	diag.Printf("canMachinedRegister %t", canMachinedRegister)

	args, env, err := getArgsEnv(p, flavor, canMachinedRegister, debug, n)
	if err != nil {
		log.FatalE("cannot get environment", err)
	}
	diag.Printf("args %q", args)
	diag.Printf("env %q", env)

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
	if err := mnt.Mount("", "/", "none", syscall.MS_REC|syscall.MS_SLAVE, ""); err != nil {
		log.FatalE("error making / a slave mount", err)
	}
	if err := mnt.Mount("", "/", "none", syscall.MS_REC|syscall.MS_SHARED, ""); err != nil {
		log.FatalE("error making / a shared and slave mount", err)
	}

	unifiedCgroup, err := cgroup.IsCgroupUnified("/")
	if err != nil {
		log.FatalE("error determining cgroup version", err)
	}
	diag.Printf("unifiedCgroup %t", unifiedCgroup)

	machineID := stage1initcommon.GetMachineID(p)

	subcgroup, err := getContainerSubCgroup(machineID, canMachinedRegister, unifiedCgroup)
	if err != nil {
		log.FatalE("error getting container subcgroup", err)
	}
	diag.Printf("subcgroup %q", subcgroup)

	if err := ioutil.WriteFile(filepath.Join(p.Root, "subcgroup"),
		[]byte(fmt.Sprintf("%s", subcgroup)), 0644); err != nil {
		log.FatalE("cannot write subcgroup file", err)
	}

	if !unifiedCgroup {
		enabledCgroups, err := v1.GetEnabledCgroups()
		if err != nil {
			log.FatalE("error getting v1 cgroups", err)
		}
		diag.Printf("enabledCgroups %q", enabledCgroups)

		if err := mountHostV1Cgroups(mnt, enabledCgroups); err != nil {
			log.FatalE("couldn't mount the host v1 cgroups", err)
		}

		if !canMachinedRegister {
			if err := v1.JoinSubcgroup("systemd", subcgroup); err != nil {
				log.FatalE(fmt.Sprintf("error joining subcgroup %q", subcgroup), err)
			}
		}

		var serviceNames []string
		for _, app := range p.Manifest.Apps {
			serviceNames = append(serviceNames, stage1initcommon.ServiceUnitName(app.Name))
		}
		diag.Printf("serviceNames %q", serviceNames)

		if err := mountContainerV1Cgroups(mnt, p, enabledCgroups, subcgroup, serviceNames); err != nil {
			log.FatalE("couldn't mount the container v1 cgroups", err)
		}

	}

	// KVM flavor has a bit different logic in handling pid vs ppid, for details look into #2389
	// it doesn't require the existence of a "ppid", instead it registers the current pid (which
	// will be reused by lkvm binary) as a pod process pid used during entering
	pid_filename := "ppid"
	if flavor == "kvm" {
		pid_filename = "pid"
	}

	if err = stage1common.WritePid(os.Getpid(), pid_filename); err != nil {
		log.FatalE("error writing pid", err)
	}

	if flavor == "kvm" {
		if err := KvmPrepareMounts(p); err != nil {
			log.FatalE("error preparing mounts", err)
		}
	}

	err = stage1common.WithClearedCloExec(lfd, func() error {
		return syscall.Exec(args[0], args, env)
	})

	if err != nil {
		log.FatalE(fmt.Sprintf("failed to execute %q", args[0]), err)
	}

	return 0
}

func areHostV1CgroupsMounted(enabledV1Cgroups map[int][]string) bool {
	controllers := v1.GetControllerDirs(enabledV1Cgroups)
	for _, c := range controllers {
		if mounted, _ := v1.IsControllerMounted(c); !mounted {
			return false
		}
	}

	return true
}

// mountHostV1Cgroups mounts the host v1 cgroup hierarchy as required by
// systemd-nspawn. We need this because some distributions don't have the
// "name=systemd" cgroup or don't mount the cgroup controllers in
// "/sys/fs/cgroup", and systemd-nspawn needs this. Since this is mounted
// inside the rkt mount namespace, it doesn't affect the host.
func mountHostV1Cgroups(m fs.Mounter, enabledCgroups map[int][]string) error {
	systemdControllerPath := "/sys/fs/cgroup/systemd"
	if !areHostV1CgroupsMounted(enabledCgroups) {
		mountContext := os.Getenv(common.EnvSELinuxMountContext)
		if err := v1.CreateCgroups(m, "/", enabledCgroups, mountContext); err != nil {
			return errwrap.Wrap(errors.New("error creating host cgroups"), err)
		}
	}

	mounted, err := v1.IsControllerMounted("systemd")
	if err != nil {
		return err
	}
	if !mounted {
		if err := os.MkdirAll(systemdControllerPath, 0700); err != nil {
			return err
		}
		if err := m.Mount("cgroup", systemdControllerPath, "cgroup", 0, "none,name=systemd"); err != nil {
			return errwrap.Wrap(fmt.Errorf("error mounting name=systemd hierarchy on %q", systemdControllerPath), err)
		}
	}

	return nil
}

// mountContainerV1Cgroups mounts the cgroup controllers hierarchy in the container's
// namespace read-only, leaving the needed knobs in the subcgroup for each-app
// read-write so systemd inside stage1 can apply isolators to them
func mountContainerV1Cgroups(m fs.Mounter, p *stage1commontypes.Pod, enabledCgroups map[int][]string, subcgroup string, serviceNames []string) error {
	mountContext := os.Getenv(common.EnvSELinuxMountContext)
	stage1Root := common.Stage1RootfsPath(p.Root)
	if err := v1.CreateCgroups(m, stage1Root, enabledCgroups, mountContext); err != nil {
		return errwrap.Wrap(errors.New("error creating container cgroups"), err)
	}

	if err := v1.RemountCgroups(m, stage1Root, enabledCgroups, subcgroup, p.InsecureOptions.DisablePaths); err != nil {
		return errwrap.Wrap(errors.New("error restricting container cgroups"), err)
	}

	return nil
}

func getContainerSubCgroup(machineID string, canMachinedRegister, unified bool) (string, error) {
	var fromUnit bool

	if util.IsRunningSystemd() {
		var err error
		if fromUnit, err = util.RunningFromSystemService(); err != nil {
			return "", errwrap.Wrap(errors.New("could not determine if we're running from a unit file"), err)
		}
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
		subcgroup := filepath.Join(slicePath, unit)

		if unified {
			return filepath.Join(subcgroup, "payload"), nil
		}

		return subcgroup, nil
	}

	escapedmID := strings.Replace(machineID, "-", "\\x2d", -1)
	machineDir := "machine-" + escapedmID + ".scope"

	if canMachinedRegister {
		// we are not in the final cgroup yet: systemd-nspawn will move us
		// to the correct cgroup later during registration so we can't
		// look it up in /proc/self/cgroup
		return filepath.Join("machine.slice", machineDir), nil
	}

	if unified {
		subcgroup, err := v2.GetOwnCgroupPath()
		if err != nil {
			return "", errwrap.Wrap(errors.New("could not get own v2 cgroup path"), err)
		}
		return subcgroup, nil
	}

	// when registration is disabled the container will be directly
	// under the current cgroup so we can look it up in /proc/self/cgroup
	// Try the systemd slice first, falling back to cpu if that fails (e.g. on
	// systems not running systemd). See issue #3502.
	ownV1CgroupPath, err := v1.GetOwnCgroupPath("name=systemd")
	if err != nil {
		ownV1CgroupPath, err = v1.GetOwnCgroupPath("cpu")
		if err != nil {
			return "", errwrap.Wrap(errors.New("could not get own v1 cgroup path"), err)
		}
	}

	// systemd-nspawn won't work if we are in the root cgroup. In addition,
	// we want all rkt instances to be in distinct cgroups. Create a
	// subcgroup and add ourselves to it.
	return filepath.Join(ownV1CgroupPath, machineDir), nil
}

func main() {
	rp := parseFlags()
	stage1initcommon.InitDebug(debug)

	log, diag, _ = rktlog.NewLogSet("stage1", debug)
	if !debug {
		diag.SetOutput(ioutil.Discard)
	}

	// move code into stage1() helper so deferred fns get run
	os.Exit(stage1(rp))
}
