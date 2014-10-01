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

// apiserver is the main api server and master for the cluster.
// it is responsible for serving the cluster management API.
package main

import (
	"flag"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version/verflag"
	"github.com/golang/glog"
)

var (
	port                  = flag.Uint("port", 8080, "The port to listen on. Default 8080")
	address               = flag.String("address", "127.0.0.1", "The address on the local server to listen to. Default 127.0.0.1")
	apiPrefix             = flag.String("api_prefix", "/api", "The prefix for API requests on the server. Default '/api'")
	storageVersion        = flag.String("storage_version", "", "The version to store resources with. Defaults to server preferred")
	cloudProvider         = flag.String("cloud_provider", "", "The provider for cloud services.  Empty string for no provider.")
	cloudConfigFile       = flag.String("cloud_config", "", "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	minionRegexp          = flag.String("minion_regexp", "", "If non empty, and -cloud_provider is specified, a regular expression for matching minion VMs")
	minionPort            = flag.Uint("minion_port", 10250, "The port at which kubelet will be listening on the minions.")
	healthCheckMinions    = flag.Bool("health_check_minions", true, "If true, health check minions and filter unhealthy ones. Default true")
	minionCacheTTL        = flag.Duration("minion_cache_ttl", 30*time.Second, "Duration of time to cache minion information. Default 30 seconds")
	etcdServerList        util.StringList
	machineList           util.StringList
	corsAllowedOriginList util.StringList
	allowPrivileged       = flag.Bool("allow_privileged", false, "If true, allow privileged containers.")
)

func init() {
	flag.Var(&etcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated")
	flag.Var(&machineList, "machines", "List of machines to schedule onto, comma separated.")
	flag.Var(&corsAllowedOriginList, "cors_allowed_origins", "List of allowed origins for CORS, comma separated.  An allowed origin can be a regular expression to support subdomain matching.  If this list is empty CORS will not be enabled.")
}

func verifyMinionFlags() {
	if *cloudProvider == "" || *minionRegexp == "" {
		if len(machineList) == 0 {
			glog.Info("No machines specified!")
		}
		return
	}
	if len(machineList) != 0 {
		glog.Info("-machines is overwritten by -minion_regexp")
	}
}

func initCloudProvider(name string, configFilePath string) cloudprovider.Interface {
	var config *os.File

	if name == "" {
		glog.Info("No cloud provider specified.")
		return nil
	}

	if configFilePath != "" {
		var err error

		config, err = os.Open(configFilePath)
		if err != nil {
			glog.Fatalf("Couldn't open cloud provider configuration %s: %#v",
				configFilePath, err)
		}

		defer config.Close()
	}

	cloud, err := cloudprovider.GetCloudProvider(name, config)
	if err != nil {
		glog.Fatalf("Couldn't init cloud provider %q: %#v", name, err)
	}
	if cloud == nil {
		glog.Fatalf("Unknown cloud provider: %s", name)
	}

	return cloud
}

func main() {
	flag.Parse()
	util.InitLogs()
	defer util.FlushLogs()

	verflag.PrintAndExitIfRequested()
	verifyMinionFlags()

	if len(etcdServerList) == 0 {
		glog.Fatalf("-etcd_servers flag is required.")
	}

	capabilities.Initialize(capabilities.Capabilities{
		AllowPrivileged: *allowPrivileged,
	})

	cloud := initCloudProvider(*cloudProvider, *cloudConfigFile)

	podInfoGetter := &client.HTTPPodInfoGetter{
		Client: http.DefaultClient,
		Port:   *minionPort,
	}

	// TODO: expose same flags as client.BindClientConfigFlags but for a server
	clientConfig := &client.Config{
		Host:    net.JoinHostPort(*address, strconv.Itoa(int(*port))),
		Version: *storageVersion,
	}
	client, err := client.New(clientConfig)
	if err != nil {
		glog.Fatalf("Invalid server address: %v", err)
	}

	helper, err := master.NewEtcdHelper(etcdServerList, *storageVersion)
	if err != nil {
		glog.Fatalf("Invalid storage version: %v", err)
	}

	m := master.New(&master.Config{
		Client:             client,
		Cloud:              cloud,
		EtcdHelper:         helper,
		HealthCheckMinions: *healthCheckMinions,
		Minions:            machineList,
		MinionCacheTTL:     *minionCacheTTL,
		MinionRegexp:       *minionRegexp,
		PodInfoGetter:      podInfoGetter,
	})

	mux := http.NewServeMux()
	apiserver.NewAPIGroup(m.API_v1beta1()).InstallREST(mux, *apiPrefix+"/v1beta1")
	apiserver.NewAPIGroup(m.API_v1beta2()).InstallREST(mux, *apiPrefix+"/v1beta2")
	apiserver.InstallSupport(mux)

	handler := http.Handler(mux)
	if len(corsAllowedOriginList) > 0 {
		allowedOriginRegexps, err := util.CompileRegexps(corsAllowedOriginList)
		if err != nil {
			glog.Fatalf("Invalid CORS allowed origin, --cors_allowed_origins flag was set to %v - %v", strings.Join(corsAllowedOriginList, ","), err)
		}
		handler = apiserver.CORS(handler, allowedOriginRegexps, nil, nil, "true")
	}
	handler = apiserver.RecoverPanics(handler)

	s := &http.Server{
		Addr:           net.JoinHostPort(*address, strconv.Itoa(int(*port))),
		Handler:        handler,
		ReadTimeout:    5 * time.Minute,
		WriteTimeout:   5 * time.Minute,
		MaxHeaderBytes: 1 << 20,
	}
	glog.Fatal(s.ListenAndServe())
}
