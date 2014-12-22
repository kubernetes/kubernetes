/*
Copyright 2014 Neil Horman <nhorman@tuxdriver.com>. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"flag"
	"net"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/onramp"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version/verflag"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

var (
	etcdServerList util.StringList
	kubeApiServer  util.StringList
	bindAddress    = util.IP(net.ParseIP("0.0.0.0"))
	eintf      string
	iintf      string
	externalAddrs  util.StringList
)

func init() {
	flag.Var(&etcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated (requried). Mutually exclusive with -etcd_config")
	flag.Var(&kubeApiServer, "master", "List of Kube api servers to watch (http://ip:port), comma separated (erquired).")
	flag.StringVar(&eintf, "external_interface", "", "Interface on which external traffic arrives. (required)")
	flag.StringVar(&iintf, "internal_interface", "", "Interface on which external traffic arrives. (required)")
	flag.Var(&externalAddrs, "external_addrs", "List of ipv4 address which to use as external, comma separated (required).")
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	verflag.PrintAndExitIfRequested()

	var etcdClient *etcd.Client
	var kubeClient *client.Client

	// Set up etcd client
	if len(etcdServerList) > 0 {
		// Set up logger for etcd client
		etcd.SetLogger(util.NewLogger("etcd "))
		etcdClient = etcd.NewClient(etcdServerList)
	}

	if eintf == "" {
		glog.Infof("Must specify an interface to act as external\n")
		return
	}

	if iintf == "" {
		glog.Infof("Must specify an interface to act as internal\n")
		return
	}

	if len(externalAddrs) == 0 {
		glog.Infof("Must specify at least one external address\n") 
		return
	}

	if len(kubeApiServer) > 0 {
		kubeClient = client.NewOrDie(&client.Config{Host: kubeApiServer[0], Version: "v1beta1"})
	}
	if kubeClient != nil {
		glog.Infof("Established api client to %s\n", kubeApiServer[0])
	} else {
		glog.Infof("Error in establishing api client\n")
		return
	}

	if etcdClient != nil {
		glog.Infof("Watching for etcd configs at %v", etcdClient.GetCluster())
	}

	Onrmp := onramp.NewOnramp(etcdClient, kubeClient, eintf, iintf, externalAddrs)
	// Start watching for work to do
	go util.Forever(func() { Onrmp.Run() }, 0)

	select {}
}
