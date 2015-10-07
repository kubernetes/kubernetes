// Copyright 2015 CoreOS, Inc.
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
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"

	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/go-systemd/daemon"
	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/pkg/flagutil"
	log "github.com/coreos/flannel/Godeps/_workspace/src/github.com/golang/glog"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/network"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/remote"
	"github.com/coreos/flannel/subnet"
)

type CmdLineOpts struct {
	publicIP       string
	etcdEndpoints  string
	etcdPrefix     string
	etcdKeyfile    string
	etcdCertfile   string
	etcdCAFile     string
	help           bool
	version        bool
	ipMasq         bool
	subnetFile     string
	subnetDir      string
	iface          string
	listen         string
	remote         string
	remoteKeyfile  string
	remoteCertfile string
	remoteCAFile   string
	networks       string
}

var opts CmdLineOpts

func init() {
	flag.StringVar(&opts.publicIP, "public-ip", "", "IP accessible by other nodes for inter-host communication")
	flag.StringVar(&opts.etcdEndpoints, "etcd-endpoints", "http://127.0.0.1:4001,http://127.0.0.1:2379", "a comma-delimited list of etcd endpoints")
	flag.StringVar(&opts.etcdPrefix, "etcd-prefix", "/coreos.com/network", "etcd prefix")
	flag.StringVar(&opts.etcdKeyfile, "etcd-keyfile", "", "SSL key file used to secure etcd communication")
	flag.StringVar(&opts.etcdCertfile, "etcd-certfile", "", "SSL certification file used to secure etcd communication")
	flag.StringVar(&opts.etcdCAFile, "etcd-cafile", "", "SSL Certificate Authority file used to secure etcd communication")
	flag.StringVar(&opts.subnetFile, "subnet-file", "/run/flannel/subnet.env", "filename where env variables (subnet, MTU, ... ) will be written to")
	flag.StringVar(&opts.subnetDir, "subnet-dir", "/run/flannel/networks", "directory where files with env variables (subnet, MTU, ...) will be written to")
	flag.StringVar(&opts.iface, "iface", "", "interface to use (IP or name) for inter-host communication")
	flag.StringVar(&opts.listen, "listen", "", "run as server and listen on specified address (e.g. ':8080')")
	flag.StringVar(&opts.remote, "remote", "", "run as client and connect to server on specified address (e.g. '10.1.2.3:8080')")
	flag.StringVar(&opts.remoteKeyfile, "remote-keyfile", "", "SSL key file used to secure client/server communication")
	flag.StringVar(&opts.remoteCertfile, "remote-certfile", "", "SSL certification file used to secure client/server communication")
	flag.StringVar(&opts.remoteCAFile, "remote-cafile", "", "SSL Certificate Authority file used to secure client/server communication")
	flag.StringVar(&opts.networks, "networks", "", "run in multi-network mode and service the specified networks")
	flag.BoolVar(&opts.ipMasq, "ip-masq", false, "setup IP masquerade rule for traffic destined outside of overlay network")
	flag.BoolVar(&opts.help, "help", false, "print this message")
	flag.BoolVar(&opts.version, "version", false, "print version and exit")
}

func writeSubnetFile(path string, nw ip.IP4Net, sd *backend.SubnetDef) error {
	dir, name := filepath.Split(path)
	os.MkdirAll(dir, 0755)

	tempFile := filepath.Join(dir, "."+name)
	f, err := os.Create(tempFile)
	if err != nil {
		return err
	}

	// Write out the first usable IP by incrementing
	// sn.IP by one
	sn := sd.Lease.Subnet
	sn.IP += 1

	fmt.Fprintf(f, "FLANNEL_NETWORK=%s\n", nw)
	fmt.Fprintf(f, "FLANNEL_SUBNET=%s\n", sn)
	fmt.Fprintf(f, "FLANNEL_MTU=%d\n", sd.MTU)
	_, err = fmt.Fprintf(f, "FLANNEL_IPMASQ=%v\n", opts.ipMasq)
	f.Close()
	if err != nil {
		return err
	}

	// rename(2) the temporary file to the desired location so that it becomes
	// atomically visible with the contents
	return os.Rename(tempFile, path)
}

func lookupIface() (*net.Interface, net.IP, error) {
	var iface *net.Interface
	var iaddr net.IP
	var err error

	if len(opts.iface) > 0 {
		if iaddr = net.ParseIP(opts.iface); iaddr != nil {
			iface, err = ip.GetInterfaceByIP(iaddr)
			if err != nil {
				return nil, nil, fmt.Errorf("Error looking up interface %s: %s", opts.iface, err)
			}
		} else {
			iface, err = net.InterfaceByName(opts.iface)
			if err != nil {
				return nil, nil, fmt.Errorf("Error looking up interface %s: %s", opts.iface, err)
			}
		}
	} else {
		log.Info("Determining IP address of default interface")
		if iface, err = ip.GetDefaultGatewayIface(); err != nil {
			return nil, nil, fmt.Errorf("Failed to get default interface: %s", err)
		}
	}

	if iaddr == nil {
		iaddr, err = ip.GetIfaceIP4Addr(iface)
		if err != nil {
			return nil, nil, fmt.Errorf("Failed to find IPv4 address for interface %s", iface.Name)
		}
	}

	return iface, iaddr, nil
}

func isMultiNetwork() bool {
	return len(opts.networks) > 0
}

func newSubnetManager() (subnet.Manager, error) {
	if opts.remote != "" {
		return remote.NewRemoteManager(opts.remote, opts.remoteCAFile, opts.remoteCertfile, opts.remoteKeyfile)
	}

	cfg := &subnet.EtcdConfig{
		Endpoints: strings.Split(opts.etcdEndpoints, ","),
		Keyfile:   opts.etcdKeyfile,
		Certfile:  opts.etcdCertfile,
		CAFile:    opts.etcdCAFile,
		Prefix:    opts.etcdPrefix,
	}

	return subnet.NewEtcdManager(cfg)
}

func initAndRun(ctx context.Context, sm subnet.Manager, netnames []string) {
	iface, iaddr, err := lookupIface()
	if err != nil {
		log.Error(err)
		return
	}

	if iface.MTU == 0 {
		log.Errorf("Failed to determine MTU for %s interface", iaddr)
		return
	}

	var eaddr net.IP

	if len(opts.publicIP) > 0 {
		eaddr = net.ParseIP(opts.publicIP)
	}

	if eaddr == nil {
		eaddr = iaddr
	}

	log.Infof("Using %s as external interface", iaddr)
	log.Infof("Using %s as external endpoint", eaddr)

	nets := []*network.Network{}
	for _, n := range netnames {
		nets = append(nets, network.New(sm, n, opts.ipMasq))
	}

	wg := sync.WaitGroup{}

	for _, n := range nets {
		go func(n *network.Network) {
			wg.Add(1)
			defer wg.Done()

			sn := n.Init(ctx, iface, iaddr, eaddr)
			if sn != nil {
				if isMultiNetwork() {
					path := filepath.Join(opts.subnetDir, n.Name) + ".env"
					if err := writeSubnetFile(path, n.Config.Network, sn); err != nil {
						return
					}
				} else {
					if err := writeSubnetFile(opts.subnetFile, n.Config.Network, sn); err != nil {
						return
					}
					daemon.SdNotify("READY=1")
				}

				n.Run(ctx)
				log.Infof("%v exited", n.Name)
			}
		}(n)
	}

	wg.Wait()
}

func main() {
	// glog will log to tmp files by default. override so all entries
	// can flow into journald (if running under systemd)
	flag.Set("logtostderr", "true")

	// now parse command line args
	flag.Parse()

	if flag.NArg() > 0 || opts.help {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTION]...\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(0)
	}

	if opts.version {
		fmt.Fprintln(os.Stderr, Version)
		os.Exit(0)
	}

	flagutil.SetFlagsFromEnv(flag.CommandLine, "FLANNELD")

	sm, err := newSubnetManager()
	if err != nil {
		log.Error("Failed to create SubnetManager: ", err)
		os.Exit(1)
	}

	var runFunc func(ctx context.Context)

	if opts.listen != "" {
		if opts.remote != "" {
			log.Error("--listen and --remote are mutually exclusive")
			os.Exit(1)
		}
		log.Info("running as server")
		runFunc = func(ctx context.Context) {
			remote.RunServer(ctx, sm, opts.listen, opts.remoteCAFile, opts.remoteCertfile, opts.remoteKeyfile)
		}
	} else {
		networks := strings.Split(opts.networks, ",")
		if len(networks) == 0 {
			networks = append(networks, "")
		}
		runFunc = func(ctx context.Context) {
			initAndRun(ctx, sm, networks)
		}
	}

	// Register for SIGINT and SIGTERM
	log.Info("Installing signal handlers")
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)

	ctx, cancel := context.WithCancel(context.Background())

	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		runFunc(ctx)
		wg.Done()
	}()

	<-sigs
	// unregister to get default OS nuke behaviour in case we don't exit cleanly
	signal.Stop(sigs)

	log.Info("Exiting...")
	cancel()

	wg.Wait()
}
