/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// kube2sky is a bridge between Kubernetes and SkyDNS.  It watches the
// Kubernetes master for changes in Services and manifests them into etcd for
// SkyDNS to serve as DNS records.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/url"
	"os"
	"time"

	kapi "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kclient "github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	kclientcmd "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	kclientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	kcontrollerFramework "github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	kSelector "github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	tools "github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	etcd "github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	skymsg "github.com/skynetservices/skydns/msg"
)

var (
	argDomain              = flag.String("domain", "cluster.local", "domain under which to create names")
	argEtcdMutationTimeout = flag.Duration("etcd_mutation_timeout", 10*time.Second, "crash after retrying etcd mutation for a specified duration")
	argEtcdServer          = flag.String("etcd-server", "http://127.0.0.1:4001", "URL to etcd server")
	argKubecfgFile         = flag.String("kubecfg_file", "", "Location of kubecfg file for access to kubernetes service")
	argKubeMasterUrl       = flag.String("kube_master_url", "https://${KUBERNETES_SERVICE_HOST}:${KUBERNETES_SERVICE_PORT}", "Url to reach kubernetes master. Env variables in this flag will be expanded.")
)

const (
	// Maximum number of attempts to connect to etcd server.
	maxConnectAttempts = 12
	// Resync period for the kube controller loop.
	resyncPeriod = 5 * time.Second
)

type etcdClient interface {
	Set(path, value string, ttl uint64) (*etcd.Response, error)
	Delete(path string, recursive bool) (*etcd.Response, error)
}

type kube2sky struct {
	// Etcd client.
	etcdClient etcdClient
	// DNS domain name.
	domain string
	// Etcd mutation timeout.
	etcdMutationTimeout time.Duration
}

func (ks *kube2sky) removeDNS(record string) error {
	glog.V(2).Infof("Removing %s from DNS", record)
	_, err := ks.etcdClient.Delete(skymsg.Path(record), true)
	return err
}

func (ks *kube2sky) addDNS(record string, service *kapi.Service) error {
	// if PortalIP is not set, a DNS entry should not be created
	if !kapi.IsServiceIPSet(service) {
		glog.V(1).Infof("Skipping dns record for headless service: %s\n", service.Name)
		return nil
	}

	for i := range service.Spec.Ports {
		svc := skymsg.Service{
			Host:     service.Spec.PortalIP,
			Port:     service.Spec.Ports[i].Port,
			Priority: 10,
			Weight:   10,
			Ttl:      30,
		}
		b, err := json.Marshal(svc)
		if err != nil {
			return err
		}
		// Set with no TTL, and hope that kubernetes events are accurate.
		glog.V(2).Infof("Setting DNS record: %v -> %s:%d\n", record, service.Spec.PortalIP, service.Spec.Ports[i].Port)
		_, err = ks.etcdClient.Set(skymsg.Path(record), string(b), uint64(0))
		if err != nil {
			return err
		}
	}
	return nil
}

// Implements retry logic for arbitrary mutator. Crashes after retrying for
// etcd_mutation_timeout.
func (ks *kube2sky) mutateEtcdOrDie(mutator func() error) {
	timeout := time.After(ks.etcdMutationTimeout)
	for {
		select {
		case <-timeout:
			glog.Fatalf("Failed to mutate etcd for %v using mutator: %v", ks.etcdMutationTimeout, mutator)
		default:
			if err := mutator(); err != nil {
				delay := 50 * time.Millisecond
				glog.V(1).Infof("Failed to mutate etcd using mutator: %v due to: %v. Will retry in: %v", mutator, err, delay)
				time.Sleep(delay)
			} else {
				return
			}
		}
	}
}

func newEtcdClient(etcdServer string) (*etcd.Client, error) {
	var (
		client *etcd.Client
		err    error
	)
	for attempt := 1; attempt <= maxConnectAttempts; attempt++ {
		if _, err = tools.GetEtcdVersion(etcdServer); err == nil {
			break
		}
		if attempt == maxConnectAttempts {
			break
		}
		glog.Infof("[Attempt: %d] Attempting access to etcd after 5 second sleep", attempt)
		time.Sleep(5 * time.Second)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to connect to etcd server: %v, error: %v", etcdServer, err)
	}
	glog.Infof("Etcd server found: %v", etcdServer)

	// loop until we have > 0 machines && machines[0] != ""
	poll, timeout := 1*time.Second, 10*time.Second
	if err := wait.Poll(poll, timeout, func() (bool, error) {
		if client = etcd.NewClient([]string{etcdServer}); client == nil {
			return false, fmt.Errorf("etcd.NewClient returned nil")
		}
		client.SyncCluster()
		machines := client.GetCluster()
		if len(machines) == 0 || len(machines[0]) == 0 {
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, fmt.Errorf("Timed out after %s waiting for at least 1 synchronized etcd server in the cluster. Error: %v", timeout, err)
	}
	return client, nil
}

func getKubeMasterUrl() (string, error) {
	if *argKubeMasterUrl == "" {
		return "", fmt.Errorf("no --kube_master_url specified")
	}
	parsedUrl, err := url.Parse(os.ExpandEnv(*argKubeMasterUrl))
	if err != nil {
		return "", fmt.Errorf("failed to parse --kube_master_url %s - %v", *argKubeMasterUrl, err)
	}
	if parsedUrl.Scheme == "" || parsedUrl.Host == "" || parsedUrl.Host == ":" {
		return "", fmt.Errorf("invalid --kube_master_url specified %s", *argKubeMasterUrl)
	}
	return parsedUrl.String(), nil
}

// TODO: evaluate using pkg/client/clientcmd
func newKubeClient() (*kclient.Client, error) {
	var config *kclient.Config
	masterUrl, err := getKubeMasterUrl()
	if err != nil {
		return nil, err
	}
	if *argKubecfgFile == "" {
		config = &kclient.Config{
			Host:    masterUrl,
			Version: "v1beta3",
		}
	} else {
		var err error
		if config, err = kclientcmd.NewNonInteractiveDeferredLoadingClientConfig(
			&kclientcmd.ClientConfigLoadingRules{ExplicitPath: *argKubecfgFile},
			&kclientcmd.ConfigOverrides{ClusterInfo: kclientcmdapi.Cluster{Server: masterUrl, InsecureSkipTLSVerify: true}}).ClientConfig(); err != nil {
			return nil, err
		}
	}
	glog.Infof("Using %s for kubernetes master", config.Host)
	glog.Infof("Using kubernetes API %s", config.Version)
	return kclient.New(config)
}

func buildOldNameString(service, namespace, domain string) string {
	return fmt.Sprintf("%s.%s.%s.", service, namespace, domain)
}

func buildNewServiceNameString(service, namespace, domain string) string {
	return fmt.Sprintf("%s.%s.svc.%s.", service, namespace, domain)
}

// Returns a cache.ListWatch that gets all changes to services.
func createServiceLW(kubeClient *kclient.Client) *cache.ListWatch {
	return cache.NewListWatchFromClient(kubeClient, "services", kapi.NamespaceAll, kSelector.Everything())
}

func (ks *kube2sky) newService(obj interface{}) {
	if s, ok := obj.(*kapi.Service); ok {
		//TODO(artfulcoder) stop adding and deleting old-format string for service
		name := buildOldNameString(s.Name, s.Namespace, ks.domain)
		ks.mutateEtcdOrDie(func() error { return ks.addDNS(name, s) })
		name1 := buildNewServiceNameString(s.Name, s.Namespace, ks.domain)
		ks.mutateEtcdOrDie(func() error { return ks.addDNS(name1, s) })
	}
}

func (ks *kube2sky) removeService(obj interface{}) {
	if s, ok := obj.(*kapi.Service); ok {
		name := buildOldNameString(s.Name, s.Namespace, ks.domain)
		ks.mutateEtcdOrDie(func() error { return ks.removeDNS(name) })
		name1 := buildNewServiceNameString(s.Name, s.Namespace, ks.domain)
		ks.mutateEtcdOrDie(func() error { return ks.removeDNS(name1) })
	}
}

func watchForServices(kubeClient *kclient.Client, ks *kube2sky) {
	var serviceController *kcontrollerFramework.Controller
	_, serviceController = framework.NewInformer(
		createServiceLW(kubeClient),
		&kapi.Service{},
		resyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    ks.newService,
			DeleteFunc: ks.removeService,
			UpdateFunc: func(oldObj, newObj interface{}) {
				ks.newService(newObj)
			},
		},
	)
	serviceController.Run(util.NeverStop)
}

func main() {
	flag.Parse()
	var err error
	// TODO: Validate input flags.
	ks := kube2sky{
		domain:              *argDomain,
		etcdMutationTimeout: *argEtcdMutationTimeout,
	}
	if ks.etcdClient, err = newEtcdClient(*argEtcdServer); err != nil {
		glog.Fatalf("Failed to create etcd client - %v", err)
	}

	kubeClient, err := newKubeClient()
	if err != nil {
		glog.Fatalf("Failed to create a kubernetes client: %v", err)
	}

	watchForServices(kubeClient, &ks)
}
