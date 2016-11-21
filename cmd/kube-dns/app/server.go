/*
Copyright 2016 The Kubernetes Authors.

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

package app

import (
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/golang/glog"
	"github.com/skynetservices/skydns/metrics"
	"github.com/skynetservices/skydns/server"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kube-dns/app/options"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	kclientcmd "k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	kdns "k8s.io/kubernetes/pkg/dns"
	dnsConfig "k8s.io/kubernetes/pkg/dns/config"
)

type KubeDNSServer struct {
	// DNS domain name.
	domain         string
	healthzPort    int
	dnsBindAddress string
	dnsPort        int
	kd             *kdns.KubeDNS
}

func NewKubeDNSServerDefault(config *options.KubeDNSConfig) *KubeDNSServer {
	ks := KubeDNSServer{domain: config.ClusterDomain}

	kubeClient, err := newKubeClient(config)
	if err != nil {
		glog.Fatalf("Failed to create a kubernetes client: %v", err)
	}

	ks.healthzPort = config.HealthzPort
	ks.dnsBindAddress = config.DNSBindAddress
	ks.dnsPort = config.DNSPort

	var configSync dnsConfig.Sync
	if config.ConfigMap == "" {
		glog.V(0).Infof("ConfigMap not configured, using values from command line flags")
		configSync = dnsConfig.NewNopSync(
			&dnsConfig.Config{Federations: config.Federations})
	} else {
		glog.V(0).Infof("Using configuration read from ConfigMap: %v:%v",
			config.ConfigMapNs, config.ConfigMap)
		configSync = dnsConfig.NewSync(
			kubeClient, config.ConfigMapNs, config.ConfigMap)
	}

	ks.kd = kdns.NewKubeDNS(kubeClient, config.ClusterDomain, configSync)

	return &ks
}

// TODO: evaluate using pkg/client/clientcmd
func newKubeClient(dnsConfig *options.KubeDNSConfig) (clientset.Interface, error) {
	var (
		config *restclient.Config
		err    error
	)

	if dnsConfig.KubeMasterURL != "" && dnsConfig.KubeConfigFile == "" {
		// Only --kube-master-url was provided.
		config = &restclient.Config{
			Host:          dnsConfig.KubeMasterURL,
			ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}},
		}
	} else {
		// We either have:
		//  1) --kube-master-url and --kubecfg-file
		//  2) just --kubecfg-file
		//  3) neither flag
		// In any case, the logic is the same.  If (3), this will automatically
		// fall back on the service account token.
		overrides := &kclientcmd.ConfigOverrides{}
		overrides.ClusterInfo.Server = dnsConfig.KubeMasterURL                                // might be "", but that is OK
		rules := &kclientcmd.ClientConfigLoadingRules{ExplicitPath: dnsConfig.KubeConfigFile} // might be "", but that is OK
		if config, err = kclientcmd.NewNonInteractiveDeferredLoadingClientConfig(rules, overrides).ClientConfig(); err != nil {
			return nil, err
		}
	}

	glog.V(0).Infof("Using %v for kubernetes master, kubernetes API: %v",
		config.Host, config.GroupVersion)
	return clientset.NewForConfig(config)
}

func (server *KubeDNSServer) Run() {
	pflag.VisitAll(func(flag *pflag.Flag) {
		glog.V(0).Infof("FLAG: --%s=%q", flag.Name, flag.Value)
	})
	setupSignalHandlers()
	server.startSkyDNSServer()
	server.kd.Start()
	server.setupHandlers()

	glog.V(0).Infof("Status HTTP port %v", server.healthzPort)
	glog.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", server.healthzPort), nil))
}

// setupHealthzHandlers sets up a readiness and liveness endpoint for kube2sky.
func (server *KubeDNSServer) setupHandlers() {
	glog.V(0).Infof("Setting up Healthz Handler (/readiness)")
	http.HandleFunc("/readiness", func(w http.ResponseWriter, req *http.Request) {
		fmt.Fprintf(w, "ok\n")
	})

	glog.V(0).Infof("Setting up cache handler (/cache)")
	http.HandleFunc("/cache", func(w http.ResponseWriter, req *http.Request) {
		serializedJSON, err := server.kd.GetCacheAsJSON()
		if err == nil {
			fmt.Fprint(w, serializedJSON)
		} else {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, err)
		}
	})
}

// setupSignalHandlers installs signal handler to ignore SIGINT and
// SIGTERM. This daemon will be killed by SIGKILL after the grace
// period to allow for some manner of graceful shutdown.
func setupSignalHandlers() {
	sigChan := make(chan os.Signal)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		glog.V(0).Infof("Ignoring signal %v (can only be terminated by SIGKILL)", <-sigChan)
	}()
}

func (d *KubeDNSServer) startSkyDNSServer() {
	glog.V(0).Infof("Starting SkyDNS server (%v:%v)", d.dnsBindAddress, d.dnsPort)
	skydnsConfig := &server.Config{
		Domain:  d.domain,
		DnsAddr: fmt.Sprintf("%s:%d", d.dnsBindAddress, d.dnsPort),
	}
	server.SetDefaults(skydnsConfig)
	s := server.New(d.kd, skydnsConfig)
	if err := metrics.Metrics(); err != nil {
		glog.Fatalf("Skydns metrics error: %s", err)
	} else if metrics.Port != "" {
		glog.V(0).Infof("Skydns metrics enabled (%v:%v)", metrics.Path, metrics.Port)
	} else {
		glog.V(0).Infof("Skydns metrics not enabled")
	}

	go s.Run()
}
