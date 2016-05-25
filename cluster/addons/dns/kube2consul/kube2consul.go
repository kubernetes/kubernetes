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

// kube2consul is a bridge between Kubernetes and Consul.  It watches the
// Kubernetes master for changes in Services and manifests them into consul
// to serve as DNS records.
package main

import (
	json "encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/golang/glog"
	consulApi "github.com/hashicorp/consul/api"
	flag "github.com/spf13/pflag"
	bridge "k8s.io/kubernetes/cluster/addons/dns/bridge"
	kapi "k8s.io/kubernetes/pkg/api"
	kcache "k8s.io/kubernetes/pkg/client/cache"
	kclient "k8s.io/kubernetes/pkg/client/unversioned"
	kframework "k8s.io/kubernetes/pkg/controller/framework"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
	"k8s.io/kubernetes/pkg/util/wait"
	"net/url"
)

const (
	// Maximum number of attempts to connect to consul server.
	maxConnectAttempts = 12
	// Resync period for the kube controller loop.
	resyncPeriod = 30 * time.Minute
	// A subdomain added to the user specified domain for all services.
	serviceSubdomain = "svc"
	// A subdomain added to the user specified domain for all pods.
	podSubdomain = "pod"
	// A subdomain added to the user specified endpoint for all endpoints.
	endpointSubdomain = "endpoint"
)

var (
	argDomain        = flag.String("domain", "cluster.local", "domain under which to create names")
	argKubeMasterURL = flag.String("kube-master-url", "", "URL to reach kubernetes master. Env variables in this flag will be expanded.")
	argKubecfgFile   = flag.String("kubecfg-file", "", "Location of kubecfg file for access to kubernetes master service; --kube-master-url overrides the URL part of this; if neither this nor --kube-master-url are provided, defaults to service account tokens")
	argConsulAgent   = flag.String("consul-agent", "http://127.0.0.1:8500", "URL to consul agent")
	healthzPort      = flag.Int("healthz-port", 8081, "port on which to serve a kube2sky HTTP readiness probe.")
)

type consulAgent interface {
	ServiceRegister(service *consulApi.AgentServiceRegistration) error
	ServiceDeregister(serviceID string) error
}

type consulKVStorage interface {
	Get(key string, q *consulApi.QueryOptions) (*consulApi.KVPair, *consulApi.QueryMeta, error)
	Put(p *consulApi.KVPair, q *consulApi.WriteOptions) (*consulApi.WriteMeta, error)
	Delete(key string, w *consulApi.WriteOptions) (*consulApi.WriteMeta, error)
}

type kube2consul struct {
	consulKV consulKVStorage

	// Consul client agent.
	consulAgent consulAgent
	// DNS domain name.
	domain string
	// A cache that contains all the endpoints in the system.
	endpointsStore kcache.Store
	// A cache that contains all the services in the system.
	servicesStore kcache.Store
	// A cache that contains all the pods in the system.
	podsStore kcache.Store
	// Lock for controlling access to headless services.
	mlock sync.Mutex
}

func sanitizeString(ip string) string {
	return strings.Replace(ip, ".", "-", -1)
}

func buildDNSNameString(labels ...string) string {
	// Note: Consul does not support '.' chars in keys
	var res string
	for _, label := range labels {
		if res == "" {
			res = label
		} else {
			res = fmt.Sprintf("%s/%s", res, label)
		}
	}
	return res
}

func (ks *kube2consul) addDNS(record string, service *kapi.Service) error {
	glog.V(2).Infof("Attempting to add record: %v", record)
	if strings.Contains(record, ".") {
		glog.V(1).Infof("Service names containing '.' are not supported: %s\n", service.Name)
		return nil
	}

	// if PortalIP is not set, do not create a DNS records
	if !kapi.IsServiceIPSet(service) {
		glog.V(1).Infof("Skipping dns record for headless service: %s\n", service.Name)
		return nil
	}

	for i := range service.Spec.Ports {
		port := &service.Spec.Ports[i]
		asr := &consulApi.AgentServiceRegistration{
			ID:      record,
			Name:    record,
			Address: service.Spec.ClusterIP,
			Port:    int(port.Port),
		}

		glog.V(2).Infof("Setting DNS record: %v -> %d\n", record, service.Spec.Ports[i].Port)

		if err := ks.consulAgent.ServiceRegister(asr); err != nil {
			return err
		}
	}
	return nil
}

func (kc *kube2consul) removeDNS(record string) error {
	glog.V(2).Infof("Removing %s from DNS", record)
	return kc.consulAgent.ServiceDeregister(record)
}

func newConsulClient(consulAgent string) (*consulApi.Client, error) {
	var (
		client *consulApi.Client
		err    error
	)

	consulConfig := consulApi.DefaultConfig()
	consulAgentUrl, err := url.Parse(consulAgent)
	if err != nil {
		glog.Infof("Error parsing Consul url")
		return nil, err
	}

	if consulAgentUrl.Host != "" {
		consulConfig.Address = consulAgentUrl.Host
	}

	if consulAgentUrl.Scheme != "" {
		consulConfig.Scheme = consulAgentUrl.Scheme
	}

	client, err = consulApi.NewClient(consulConfig)
	if err != nil {
		glog.Infof("Error creating Consul client")
		return nil, err
	}

	for attempt := 1; attempt <= maxConnectAttempts; attempt++ {
		if _, err = client.Agent().Self(); err == nil {
			break
		}

		if attempt == maxConnectAttempts {
			break
		}

		glog.Infof("[Attempt: %d] Attempting access to Consul after 5 second sleep", attempt)
		time.Sleep(5 * time.Second)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to connect to Consul agent: %v, error: %v", consulAgent, err)
	}
	glog.Infof("Consul agent found: %v", consulAgent)

	return client, nil
}

// setupSignalHandlers runs a goroutine that waits on SIGINT or SIGTERM and logs it
// before exiting.
func setupSignalHandlers() {
	sigChan := make(chan os.Signal)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// This program should always exit gracefully logging that it received
	// either a SIGINT or SIGTERM. Since kube2sky is run in a container
	// without a liveness probe as part of the kube-dns pod, it shouldn't
	// restart unless the pod is deleted. If it restarts without logging
	// anything it means something is seriously wrong.
	// TODO: Remove once #22290 is fixed.
	go func() {
		glog.Fatalf("Received signal %s", <-sigChan)
	}()
}

func (kc *kube2consul) updateService(oldObj, newObj interface{}) {
	kc.removeService(oldObj)
	kc.newService(newObj)
}

func (kc *kube2consul) getServiceFromEndpoints(e *kapi.Endpoints) (*kapi.Service, error) {
	key, err := kcache.MetaNamespaceKeyFunc(e)
	if err != nil {
		return nil, err
	}
	obj, exists, err := kc.servicesStore.GetByKey(key)
	if err != nil {
		return nil, fmt.Errorf("failed to get service object from services store - %v", err)
	}
	if !exists {
		glog.V(1).Infof("could not find service for endpoint %q in namespace %q", e.Name, e.Namespace)
		return nil, nil
	}
	if svc, ok := obj.(*kapi.Service); ok {
		return svc, nil
	}
	return nil, fmt.Errorf("got a non service object in services store %v", obj)
}

func (kc *kube2consul) newService(obj interface{}) {
	_, err := consulApi.NewClient(consulApi.DefaultConfig())
	if err != nil {
		panic(err)
	}

	if s, ok := obj.(*kapi.Service); ok {
		name := buildDNSNameString(kc.domain, serviceSubdomain, s.Namespace, s.Name)
		glog.V(2).Infof("***Adding service %v", name)
		if err := kc.addDNS(name, s); err != nil {
			glog.V(1).Infof("Failed to add service: %v due to: %v", name, err)
		}
	}
}

func (kc *kube2consul) removeService(obj interface{}) {
	if s, ok := obj.(*kapi.Service); ok {
		name := buildDNSNameString(kc.domain, serviceSubdomain, s.Namespace, s.Name)
		glog.V(2).Infof("***Removing service %v", name)
		if err := kc.removeDNS(name); err != nil {
			glog.V(1).Infof("Failed to remove service: %v due to: %v", name, err)
		}
	}
}

func (kc *kube2consul) storeKV(subDomain string, name string, value string) {
	key := fmt.Sprintf("%v/%v/", subDomain, name)
	p := &consulApi.KVPair{Key: key, Value: []byte(value)}
	kc.consulKV.Put(p, nil)
}

func (kc *kube2consul) deleteKV(subDomain string, name string) {
	key := fmt.Sprintf("%v/%v/", subDomain, name)
	kc.consulKV.Delete(key, nil)
}

func (kc *kube2consul) handlePodCreate(obj interface{}) {
	if p, ok := obj.(*kapi.Pod); ok {
		// If the pod ip is not yet available, do not attempt to create.
		if p.Status.PodIP != "" {
			podIP := sanitizeString(p.Status.PodIP)
			volumes := p.Spec.Volumes
			volumesJson, _ := json.Marshal(volumes)
			volumesStr := fmt.Sprintf("%v", volumesJson)
			kc.storeKV(podSubdomain, podIP, volumesStr)
		}
	}
}

func (kc *kube2consul) handlePodUpdate(oldObj interface{}, newObj interface{}) {
	if np, ok := newObj.(*kapi.Pod); ok {

		if p, ok := oldObj.(*kapi.Pod); ok {
			oldPodIP := sanitizeString(p.Status.PodIP)
			kc.deleteKV(podSubdomain, oldPodIP)

			newPodIP := sanitizeString(np.Status.PodIP)
			volumes := p.Spec.Volumes
			volumesJson, _ := json.Marshal(volumes)
			volumesStr := fmt.Sprintf("%v", volumesJson)
			kc.storeKV(podSubdomain, newPodIP, volumesStr)
		}
	}
}

func (kc *kube2consul) handlePodRemove(obj interface{}) {
	if p, ok := obj.(*kapi.Pod); ok {
		podIP := sanitizeString(p.Status.PodIP)
		glog.V(2).Infof("Attempting to remove pod: %v", podIP)
		kc.deleteKV(podSubdomain, podIP)
	}
}

func (kc *kube2consul) handleEndpointAdd(obj interface{}) {
	kc.mlock.Lock()
	defer kc.mlock.Unlock()
	if e, ok := obj.(*kapi.Endpoints); ok {
		svc, err := kc.getServiceFromEndpoints(e)
		if err != nil || svc == nil || kapi.IsServiceIPSet(svc) {
			glog.V(1).Infof("Failed to get endpoint from %v", e.Name)
		}
		endpointsData, _ := json.Marshal(e)
		value := fmt.Sprintf("%s", endpointsData)
		kc.storeKV(serviceSubdomain, svc.Name, value)
	}
}

func (kc *kube2consul) handleEndpointUpdate(old interface{}, newObj interface{}) {
	kc.mlock.Lock()
	defer kc.mlock.Unlock()
	if e, ok := newObj.(*kapi.Endpoints); ok {
		svc, err := kc.getServiceFromEndpoints(e)
		if err != nil || svc == nil || kapi.IsServiceIPSet(svc) {
			// No headless service found corresponding to endpoints object.
			glog.V(1).Infof("Failed to get service from %v", e.Name)
		}
		endpointsData, _ := json.Marshal(e)
		value := fmt.Sprintf("%s", endpointsData)
		kc.storeKV(endpointSubdomain, svc.Name, value)
	}
}

func (kc *kube2consul) handleEndpointRemove(obj interface{}) {
	kc.mlock.Lock()
	defer kc.mlock.Unlock()
	if e, ok := obj.(*kapi.Endpoints); ok {
		svc, err := kc.getServiceFromEndpoints(e)
		if err != nil || svc == nil || kapi.IsServiceIPSet(svc) {
			// No headless service found corresponding to endpoints object.
			glog.V(1).Infof("Failed to remove endpoint from %v", e.Name)
		}
		if e == nil {
			kc.deleteKV(endpointSubdomain, svc.Name)
		}
	}
}

func watchEndpoints(kubeClient *kclient.Client, kc *kube2consul) kcache.Store {
	eStore, eController := kframework.NewInformer(
		bridge.CreateEndpointsLW(kubeClient),
		&kapi.Endpoints{},
		resyncPeriod,
		kframework.ResourceEventHandlerFuncs{
			AddFunc:    kc.handleEndpointAdd,
			DeleteFunc: kc.handleEndpointRemove,
			UpdateFunc: func(oldObj, newObj interface{}) {
				// TODO: Avoid unwanted updates.
				kc.handleEndpointAdd(newObj)
			},
		},
	)

	go eController.Run(wait.NeverStop)
	return eStore
}

func watchForServices(kubeClient *kclient.Client, kc *kube2consul) kcache.Store {
	serviceStore, serviceController := kframework.NewInformer(
		bridge.CreateServiceLW(kubeClient),
		&kapi.Service{},
		resyncPeriod,
		kframework.ResourceEventHandlerFuncs{
			AddFunc:    kc.newService,
			DeleteFunc: kc.removeService,
			UpdateFunc: kc.updateService,
		},
	)
	go serviceController.Run(wait.NeverStop)
	return serviceStore
}

func watchPods(kubeClient *kclient.Client, kc *kube2consul) kcache.Store {
	eStore, eController := kframework.NewInformer(
		bridge.CreateEndpointsPodLW(kubeClient),
		&kapi.Pod{},
		resyncPeriod,
		kframework.ResourceEventHandlerFuncs{
			AddFunc: kc.handlePodCreate,
			UpdateFunc: func(oldObj, newObj interface{}) {
				kc.handlePodUpdate(oldObj, newObj)
			},
			DeleteFunc: kc.handlePodRemove,
		},
	)

	go eController.Run(wait.NeverStop)
	return eStore
}

// setupHealthzHandlers sets up a readiness and liveness endpoint for kube2sky.
func setupHealthzHandlers(kc *kube2consul) {
	http.HandleFunc("/readiness", func(w http.ResponseWriter, req *http.Request) {
		fmt.Fprintf(w, "ok\n")
	})
}

func main() {
	flag.CommandLine.SetNormalizeFunc(utilflag.WarnWordSepNormalizeFunc)
	flag.Parse()
	var err error
	setupSignalHandlers()
	// TODO: Validate input flags.
	domain := sanitizeString(*argDomain)
	kc := kube2consul{
		domain: domain,
	}

	consulClient, err := newConsulClient(*argConsulAgent)
	if err != nil {
		glog.Fatalf("Failed to create Consul client - %v", err)
	}
	kc.consulAgent = consulClient.Agent()
	kc.consulKV = consulClient.KV()

	kubeClient, err := bridge.NewKubeClient(argKubeMasterURL, argKubecfgFile)
	if err != nil {
		glog.Fatalf("Failed to create a kubernetes client: %v", err)
	}
	// Wait synchronously for the Kubernetes service and add a DNS record for it.
	kc.newService(bridge.WaitForKubernetesService(kubeClient))
	glog.Infof("Successfully added DNS record for Kubernetes service.")

	kc.endpointsStore = watchEndpoints(kubeClient, &kc)
	kc.servicesStore = watchForServices(kubeClient, &kc)
	kc.podsStore = watchPods(kubeClient, &kc)

	// We declare kube2consul ready when:
	// 1. It has retrieved the Kubernetes master service from the apiserver. If this
	//    doesn't happen consul will fail its liveness probe assuming that it can't
	//    perform any cluster local DNS lookups.
	// 2. It has setup the 3 watches above.
	// Once ready this container never flips to not-ready.
	setupHealthzHandlers(&kc)
	glog.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *healthzPort), nil))
}
