/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/proxy"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/proxy/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

var (
	configFile     = flag.String("configfile", "/tmp/proxy_config", "Configuration file for the proxy")
	etcdServerList util.StringList
)

func init() {
	flag.Var(&etcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated")
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	// Set up logger for etcd client
	etcd.SetLogger(util.NewLogger("etcd "))

	glog.Infof("Using configuration file %s and etcd_servers %v", *configFile, etcdServerList)

	serviceConfig := config.NewServiceConfig()
	endpointsConfig := config.NewEndpointsConfig()

	// Create a configuration source that handles configuration from etcd.
	etcdClient := etcd.NewClient(etcdServerList)
	config.NewConfigSourceEtcd(etcdClient,
		serviceConfig.Channel("etcd"),
		endpointsConfig.Channel("etcd"))

	// And create a configuration source that reads from a local file
	config.NewConfigSourceFile(*configFile,
		serviceConfig.Channel("file"),
		endpointsConfig.Channel("file"))

	loadBalancer := proxy.NewLoadBalancerRR()
	proxier := proxy.NewProxier(loadBalancer)
	// Wire proxier to handle changes to services
	serviceConfig.RegisterHandler(proxier)
	// And wire loadBalancer to handle changes to endpoints to services
	endpointsConfig.RegisterHandler(loadBalancer)

	// Just loop forever for now...
	select {}
}
