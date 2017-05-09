/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"os"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/contrib/kube-haproxy/haproxy"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	kubectl_util "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

/*
Usage
  inside a Pod, just balancing TCP services:
  	kube-proxy --services=service1,service2,...,serviceN

  inside a Pod, doing path-based-routing:
    kube-proxy --http-services=/path1:service1,/path2:service2,...,/pathN:serviceN --http-port=8080

  outside a Pod, or if you want to target a different kubernetes cluster, you need to have a .kube config from kubectl.
  If 'kubectl get services' works properly, then so will kube-haproxy.
*/
func main() {
	flags := flag.NewFlagSet("", flag.ContinueOnError)

	cluster := flags.Bool("use-kubernetes-cluster-service", true, "If true, use the built in kubernetes cluster for creating the client")
	serviceSpec := flags.String("services", "", "Comma separated list of tcp services to proxy")
	httpSpec := flags.String("http-services", "", "Comma separated list of <path>:<http-service> to add to a route-based HTTP proxy")
	httpPort := flags.Int("http-port", 8080, "The port to turn up an HTTP routed service on, ignored if --http-services is empty")
	configFile := flags.String("haproxy-config-file", "/usr/local/etc/haproxy/haproxy.cfg", "The path to the haproxy config file to write")
	clientConfig := kubectl_util.DefaultClientConfig(flags)

	flags.Parse(os.Args)

	if len(*serviceSpec) == 0 && len(*httpSpec) == 0 {
		glog.Fatalf("one of --services or --http-services is required.")
	}

	var kubeClient *client.Client
	var err error
	if *cluster {
		if kubeClient, err = client.NewInCluster(); err != nil {
			glog.Fatalf("Failed to create client: %v", err)
		}
	} else {
		config, err := clientConfig.ClientConfig()
		if err != nil {
			glog.Fatalf("error connecting to the client: %v", err)
		}
		kubeClient, err = client.New(config)
	}
	mgr := &haproxy.HAProxyManager{
		Exec:       exec.New(),
		ConfigFile: *configFile,
	}
	namespace, specified, err := clientConfig.Namespace()
	if err != nil {
		glog.Fatalf("unexpected error: %v", err)
	}
	if !specified {
		namespace = "default"
	}
	if len(*serviceSpec) > 0 {
		for _, service := range strings.Split(*serviceSpec, ",") {
			mgr.AddSimpleService(namespace, service)
		}
	}
	if len(*httpSpec) > 0 {
		mgr.HTTPPort = *httpPort
		for _, httpService := range strings.Split(*httpSpec, ",") {
			parts := strings.Split(httpService, ":")
			if len(parts) != 2 {
				glog.Fatalf("unexpected format for http service, expected <path>:<service-name> saw: %s", httpService)
			}
			mgr.AddRoutedService(parts[0], namespace, parts[1])
		}
	}
	tick := time.Tick(10 * time.Second)
	for {
		if err := mgr.SyncOnce(kubeClient, namespace); err != nil {
			glog.Errorf("Error syncing: %v", err)
		}
		<-tick
	}
}
