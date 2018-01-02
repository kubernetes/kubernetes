// Copyright 2016 The rkt Authors
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

package main

import (
	"fmt"
	"net"
	"strconv"
	"strings"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/label"
	"github.com/coreos/rkt/pkg/lock"
	"github.com/coreos/rkt/pkg/pod"
	"github.com/coreos/rkt/pkg/user"
	"github.com/coreos/rkt/stage0"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"
	"github.com/spf13/cobra"
)

var (
	cmdAppSandbox = &cobra.Command{
		Use:   "sandbox",
		Short: "Create an empty pod application sandbox",
		Long:  "Initializes an empty pod having no applications.",
		Run:   runWrapper(runAppSandbox),
	}
	flagAppPorts    appPortList
	flagAnnotations kvMap
	flagLabels      kvMap
)

func init() {
	cmdApp.AddCommand(cmdAppSandbox)

	addStage1ImageFlags(cmdAppSandbox.Flags())
	cmdAppSandbox.Flags().StringVar(&flagUUIDFileSave, "uuid-file-save", "", "write out pod UUID to specified file")
	cmdAppSandbox.Flags().Var(&flagNet, "net", "configure the pod's networking. Optionally, pass a list of user-configured networks to load and set arguments to pass to each network, respectively. Syntax: --net[=n[:args], ...]")
	cmdAppSandbox.Flags().BoolVar(&flagNoOverlay, "no-overlay", false, "disable overlay filesystem")
	cmdAppSandbox.Flags().Var(&flagDNS, "dns", "name servers to write in /etc/resolv.conf")
	cmdAppSandbox.Flags().Var(&flagDNSSearch, "dns-search", "DNS search domains to write in /etc/resolv.conf")
	cmdAppSandbox.Flags().Var(&flagDNSOpt, "dns-opt", "DNS options to write in /etc/resolv.conf")
	cmdAppSandbox.Flags().StringVar(&flagDNSDomain, "dns-domain", "", "DNS domain to write in /etc/resolv.conf")
	cmdAppSandbox.Flags().Var(&flagHostsEntries, "hosts-entry", "Entries to add to the pod-wide /etc/hosts. Pass 'host' to use the host's /etc/hosts")
	cmdAppSandbox.Flags().StringVar(&flagHostname, "hostname", "", `pod's hostname. If empty, it will be "rkt-$PODUUID"`)
	cmdAppSandbox.Flags().Var(&flagAppPorts, "port", "ports to forward. format: \"name:proto:podPort:hostIP:hostPort\"")

	flagAppPorts = appPortList{}
	cmdAppSandbox.Flags().Var(&flagAnnotations, "user-annotation", "optional, set the pod's annotations in the form of key=value")
	cmdAppSandbox.Flags().Var(&flagLabels, "user-label", "optional, set the pod's label in the form of key=value")
}

func runAppSandbox(cmd *cobra.Command, args []string) int {
	s, err := imagestore.NewStore(storeDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 1
	}

	ts, err := treestore.NewStore(treeStoreDir(), s)
	if err != nil {
		stderr.PrintE("cannot open treestore", err)
		return 1
	}

	config, err := getConfig()
	if err != nil {
		stderr.PrintE("cannot get configuration", err)
		return 1
	}

	s1img, err := getStage1Hash(s, ts, config)
	if err != nil {
		stderr.Error(err)
		return 1
	}

	p, err := pod.NewPod(getDataDir())
	if err != nil {
		stderr.PrintE("error creating new pod", err)
		return 1
	}

	if flagUUIDFileSave != "" {
		if err := pod.WriteUUIDToFile(p.UUID, flagUUIDFileSave); err != nil {
			stderr.PrintE("error saving pod UUID to file", err)
			return 1
		}
	}

	processLabel, mountLabel, err := label.InitLabels("/var/run/rkt/mcs", []string{})
	if err != nil {
		stderr.PrintE("error initialising SELinux", err)
		return 1
	}

	p.MountLabel = mountLabel
	cfg := stage0.CommonConfig{
		DataDir:      getDataDir(),
		MountLabel:   mountLabel,
		ProcessLabel: processLabel,
		Store:        s,
		TreeStore:    ts,
		Stage1Image:  *s1img,
		UUID:         p.UUID,
		Debug:        globalFlags.Debug,
		Mutable:      true,
	}

	ovlOk := true
	if err := common.PathSupportsOverlay(getDataDir()); err != nil {
		if oerr, ok := err.(common.ErrOverlayUnsupported); ok {
			stderr.Printf("disabling overlay support: %q", oerr.Error())
			ovlOk = false
		} else {
			stderr.PrintE("error determining overlay support", err)
			return 1
		}
	}

	useOverlay := !flagNoOverlay && ovlOk

	pcfg := stage0.PrepareConfig{
		CommonConfig:       &cfg,
		UseOverlay:         useOverlay,
		PrivateUsers:       user.NewBlankUidRange(),
		SkipTreeStoreCheck: globalFlags.InsecureFlags.SkipOnDiskCheck(),
		Apps:               &rktApps,
		Ports:              []types.ExposedPort(flagAppPorts),
		UserAnnotations:    parseAnnotations(&flagAnnotations),
		UserLabels:         parseLabels(&flagLabels),
	}

	if globalFlags.Debug {
		stage0.InitDebug()
	}

	keyLock, err := lock.SharedKeyLock(lockDir(), common.PrepareLock)
	if err != nil {
		stderr.PrintE("cannot get shared prepare lock", err)
		return 1
	}

	err = stage0.Prepare(pcfg, p.Path(), p.UUID)
	if err != nil {
		stderr.PrintE("error setting up stage0", err)
		keyLock.Close()
		return 1
	}
	keyLock.Close()

	// get the lock fd for run
	lfd, err := p.Fd()
	if err != nil {
		stderr.PrintE("error getting pod lock fd", err)
		return 1
	}

	// skip prepared by jumping directly to run, we own this pod
	if err := p.ToRun(); err != nil {
		stderr.PrintE("unable to transition to run", err)
		return 1
	}

	rktgid, err := common.LookupGid(common.RktGroup)
	if err != nil {
		stderr.Printf("group %q not found, will use default gid when rendering images", common.RktGroup)
		rktgid = -1
	}

	DNSConfMode, DNSConfig, HostsEntries, err := parseDNSFlags(flagHostsEntries, flagDNS, flagDNSSearch, flagDNSOpt, flagDNSDomain)
	if err != nil {
		stderr.PrintE("error with dns flags", err)
		return 1
	}

	rcfg := stage0.RunConfig{
		CommonConfig:         &cfg,
		Net:                  flagNet,
		LockFd:               lfd,
		Interactive:          false,
		DNSConfMode:          DNSConfMode,
		DNSConfig:            DNSConfig,
		MDSRegister:          false,
		LocalConfig:          globalFlags.LocalConfigDir,
		RktGid:               rktgid,
		Hostname:             flagHostname,
		InsecureCapabilities: globalFlags.InsecureFlags.SkipCapabilities(),
		InsecurePaths:        globalFlags.InsecureFlags.SkipPaths(),
		InsecureSeccomp:      globalFlags.InsecureFlags.SkipSeccomp(),
		UseOverlay:           useOverlay,
		HostsEntries:         *HostsEntries,
	}

	_, manifest, err := p.PodManifest()
	if err != nil {
		stderr.PrintE("cannot get the pod manifest", err)
		return 1
	}
	rcfg.Apps = manifest.Apps
	stage0.Run(rcfg, p.Path(), getDataDir()) // execs, never returns

	return 1
}

/*
 * The sandbox uses a different style of port forwarding - instead of mapping
 * from port to app (via name), we just map ports directly.
 *
 * The format is name:proto:podPort:hostIP:hostPort
 * e.g. http:tcp:8080:0.0.0.0:80
 */
type appPortList []types.ExposedPort

func (apl *appPortList) Set(s string) error {
	parts := strings.SplitN(s, ":", 5)
	if len(parts) != 5 {
		return fmt.Errorf("--port invalid format")
	}

	// parsey parsey
	name, err := types.NewACName(parts[0])
	if err != nil {
		return err
	}

	proto := parts[1]
	switch proto {
	case "tcp", "udp":
	default:
		return fmt.Errorf("invalid protocol %q", proto)
	}

	p, err := strconv.ParseUint(parts[2], 10, 16)
	if err != nil {
		return err
	}
	podPortNo := uint(p)

	ip := net.ParseIP(parts[3])
	if ip == nil {
		return fmt.Errorf("could not parse IP %q", ip)
	}

	p, err = strconv.ParseUint(parts[4], 10, 16)
	if err != nil {
		return err
	}
	hostPortNo := uint(p)

	podSide := types.Port{
		Name:            *name,
		Protocol:        proto,
		Port:            podPortNo,
		Count:           1,
		SocketActivated: false,
	}

	hostSide := types.ExposedPort{
		Name:     *name,
		HostPort: hostPortNo,
		HostIP:   ip,
		PodPort:  &podSide,
	}

	*apl = append(*apl, hostSide)
	return nil
}

func (apl *appPortList) String() string {
	ss := make([]string, 0, len(*apl))
	for _, p := range *apl {
		ss = append(ss, fmt.Sprintf("%s:%s:%d:%s:%d",
			p.Name, p.PodPort.Protocol, p.PodPort.Port,
			p.HostIP, p.HostPort))

	}
	return strings.Join(ss, ",")
}

func (apl *appPortList) Type() string {
	return "appPortList"
}

// parseAnnotations converts the annotations set by '--user-annotation' flag,
// and returns types.UserAnnotations.
func parseAnnotations(flagAnnotations *kvMap) types.UserAnnotations {
	if flagAnnotations.IsEmpty() {
		return nil
	}
	annotations := make(types.UserAnnotations)
	for k, v := range flagAnnotations.mapping {
		annotations[k] = v
	}
	return annotations
}

// parseLabels converts the labels set by '--user-label' flag,
// and returns types.UserLabels.
func parseLabels(flagLabels *kvMap) types.UserLabels {
	if flagLabels.IsEmpty() {
		return nil
	}
	labels := make(types.UserLabels)
	for k, v := range flagLabels.mapping {
		labels[k] = v
	}
	return labels
}
