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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version/verflag"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/onramp"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

var (
	etcdServerList util.StringList
	bindAddress    = util.IP(net.ParseIP("0.0.0.0"))
)

func init() {
	flag.Var(&etcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated (optional). Mutually exclusive with -etcd_config")
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	verflag.PrintAndExitIfRequested()

	var etcdClient *etcd.Client

	// Set up etcd client
	if len(etcdServerList) > 0 {
		// Set up logger for etcd client
		etcd.SetLogger(util.NewLogger("etcd "))
		etcdClient = etcd.NewClient(etcdServerList)
	}

	if etcdClient != nil {
		glog.Infof("Watching for etcd configs at %v", etcdClient.GetCluster())
        }

	Onrmp := onramp.NewOnramp(etcdClient)
	// Start watching for work to do
	go util.Forever(func() { Onrmp.Run() }, 0)

	select { }
}
