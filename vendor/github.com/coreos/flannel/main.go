// Copyright 2015 flannel authors
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
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	"github.com/coreos/pkg/flagutil"
	log "github.com/golang/glog"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/network"
	"github.com/coreos/flannel/remote"
	"github.com/coreos/flannel/subnet"
	"github.com/coreos/flannel/version"

	// Backends need to be imported for their init() to get executed and them to register
	_ "github.com/coreos/flannel/backend/alloc"
	_ "github.com/coreos/flannel/backend/awsvpc"
	_ "github.com/coreos/flannel/backend/gce"
	_ "github.com/coreos/flannel/backend/hostgw"
	_ "github.com/coreos/flannel/backend/udp"
	_ "github.com/coreos/flannel/backend/vxlan"
)

type CmdLineOpts struct {
	etcdEndpoints  string
	etcdPrefix     string
	etcdKeyfile    string
	etcdCertfile   string
	etcdCAFile     string
	etcdUsername   string
	etcdPassword   string
	help           bool
	version        bool
	listen         string
	remote         string
	remoteKeyfile  string
	remoteCertfile string
	remoteCAFile   string
}

var opts CmdLineOpts

func init() {
	flag.StringVar(&opts.etcdEndpoints, "etcd-endpoints", "http://127.0.0.1:4001,http://127.0.0.1:2379", "a comma-delimited list of etcd endpoints")
	flag.StringVar(&opts.etcdPrefix, "etcd-prefix", "/coreos.com/network", "etcd prefix")
	flag.StringVar(&opts.etcdKeyfile, "etcd-keyfile", "", "SSL key file used to secure etcd communication")
	flag.StringVar(&opts.etcdCertfile, "etcd-certfile", "", "SSL certification file used to secure etcd communication")
	flag.StringVar(&opts.etcdCAFile, "etcd-cafile", "", "SSL Certificate Authority file used to secure etcd communication")
	flag.StringVar(&opts.etcdUsername, "etcd-username", "", "Username for BasicAuth to etcd")
	flag.StringVar(&opts.etcdPassword, "etcd-password", "", "Password for BasicAuth to etcd")
	flag.StringVar(&opts.listen, "listen", "", "run as server and listen on specified address (e.g. ':8080')")
	flag.StringVar(&opts.remote, "remote", "", "run as client and connect to server on specified address (e.g. '10.1.2.3:8080')")
	flag.StringVar(&opts.remoteKeyfile, "remote-keyfile", "", "SSL key file used to secure client/server communication")
	flag.StringVar(&opts.remoteCertfile, "remote-certfile", "", "SSL certification file used to secure client/server communication")
	flag.StringVar(&opts.remoteCAFile, "remote-cafile", "", "SSL Certificate Authority file used to secure client/server communication")
	flag.BoolVar(&opts.help, "help", false, "print this message")
	flag.BoolVar(&opts.version, "version", false, "print version and exit")
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
		Username:  opts.etcdUsername,
		Password:  opts.etcdPassword,
	}

	return subnet.NewLocalManager(cfg)
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
		fmt.Fprintln(os.Stderr, version.Version)
		os.Exit(0)
	}

	flagutil.SetFlagsFromEnv(flag.CommandLine, "FLANNELD")

	sm, err := newSubnetManager()
	if err != nil {
		log.Error("Failed to create SubnetManager: ", err)
		os.Exit(1)
	}

	// Register for SIGINT and SIGTERM
	log.Info("Installing signal handlers")
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)

	ctx, cancel := context.WithCancel(context.Background())

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
		nm, err := network.NewNetworkManager(ctx, sm)
		if err != nil {
			log.Error("Failed to create NetworkManager: ", err)
			os.Exit(1)
		}

		runFunc = func(ctx context.Context) {
			nm.Run(ctx)
		}
	}

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
