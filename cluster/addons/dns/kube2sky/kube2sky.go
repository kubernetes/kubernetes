/*
Copyright 2015 Google Inc. All rights reserved.

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
// Kubernetes master for changes in Services and Pods and manifests them
// into etcd for SkyDNS to serve as DNS records.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	kapi "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kclient "github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	kfields "github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	klabels "github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	kutil "github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	kwatch "github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	etcd "github.com/coreos/go-etcd/etcd"
	skymsg "github.com/skynetservices/skydns/msg"
)

var (
	_            = flag.Duration("etcd_mutation_timeout", 0, "deprecated")
	etcdServers  = flag.String("etcd-server", "http://127.0.0.1:4001", "comma separated list of etcd server URLs")
	svcDomain    = flag.String("domain", "kubernetes.local", "domain under which to create names corresponding to services")
	podDomain    = flag.String("pod_domain", "", "domain under which to create names corresponding to running pods")
	pollInterval = flag.Duration("poll_interval", 10*time.Second, "don't rely on watch API, and poll services and pods at this interval (0 to disable)")
	verbose      = flag.Bool("verbose", false, "log extra information")
)

const (
	dnsPriority = 10
	dnsWeight   = 10
	dnsTTL      = 30
	retryWatch  = 1 * time.Second
)

func newDNSRecord(ip string, port int) string {
	svc := skymsg.Service{
		Host:     ip,
		Port:     port,
		Priority: dnsPriority,
		Weight:   dnsWeight,
		Ttl:      dnsTTL,
	}
	b, err := json.Marshal(svc)
	if err != nil {
		log.Fatalf("error marshalling skymsg.Service: %v", err)
	}
	return string(b)
}

// TODO: evaluate using pkg/client/clientcmd
func newKubeClient() (*kclient.Client, error) {
	config := &kclient.Config{}

	masterHost := os.Getenv("KUBERNETES_RO_SERVICE_HOST")
	if masterHost == "" {
		log.Fatalf("KUBERNETES_RO_SERVICE_HOST is not defined")
	}
	masterPort := os.Getenv("KUBERNETES_RO_SERVICE_PORT")
	if masterPort == "" {
		log.Fatalf("KUBERNETES_RO_SERVICE_PORT is not defined")
	}
	config.Host = fmt.Sprintf("http://%s:%s", masterHost, masterPort)
	if *verbose {
		log.Printf("Using %s for kubernetes master", config.Host)
	}

	config.Version = "v1beta1"
	if *verbose {
		log.Printf("Using kubernetes API %s", config.Version)
	}

	return kclient.New(config)
}

func buildServiceName(port, service, namespace, domain string) string {
	if port != "" {
		port += "."
	}
	return skymsg.Path(fmt.Sprintf("%s%s.%s.%s.", port, service, namespace, domain))
}

func buildPodName(pod, port, service, namespace, domain string) string {
	if port != "" {
		port += "."
	}
	return skymsg.Path(fmt.Sprintf("%s.%s%s.%s.%s.", pod, port, service, namespace, domain))
}

func buildExistingHosts(hosts map[string]string, node *etcd.Node) {
	if node.Value != "" {
		hosts[node.Key] = node.Value
	}
	for _, n := range node.Nodes {
		buildExistingHosts(hosts, n)
	}
}

type etcdInterface interface {
	Set(string, string, uint64) (*etcd.Response, error)
	Delete(string, bool) (*etcd.Response, error)
}

type updater struct {
	etcd        etcdInterface
	ch          chan interface{}
	services    map[string]*kapi.Service
	pods        map[string]*kapi.Pod
	existingDNS map[string]string
	err         chan error
	stopChan    chan struct{}
}

func newUpdater(etcd etcdInterface) *updater {
	u := &updater{
		etcd:     etcd,
		ch:       make(chan interface{}, 1024),
		err:      make(chan error, 1),
		stopChan: make(chan struct{}),
	}
	go u.process()
	return u
}

type runCmd struct {
	existingDNS map[string]string
}

type stopCmd struct{}

// process runs main updater goroutine. This goroutine is responsible for bookkeeping services,
// pods, existingDNS maps. It receives events from watchers, and when it has all necessary
// information (existing services, existing pods, existing DNS entries), it starts doing useful
// work (compare target vs existing DNS entries, and applying the difference).
func (u *updater) process() {
	for {
		select {
		case req := <-u.ch:
			// Process first request in blocking mode.
			u.processOnce(req)
			// Process everything remaining in the queue.
			for len(u.ch) > 0 {
				u.processOnce(<-u.ch)
			}
			// Commit everything at once (only after everything is loaded).
			if u.existingDNS != nil && u.services != nil && u.pods != nil {
				u.commit()
			}
		case <-u.stopChan:
			return
		}
	}
}

func (u *updater) processOnce(req interface{}) {
	switch r := req.(type) {
	case []kapi.Pod:
		u.pods = make(map[string]*kapi.Pod)
		for i, p := range r {
			u.pods[p.Name] = &r[i]
		}
	case []kapi.Service:
		u.services = make(map[string]*kapi.Service)
		for i, s := range r {
			u.services[s.Name] = &r[i]
		}
	case kwatch.Event:
		if r.Type == kwatch.Error {
			log.Printf("Watch error: %+v", r)
			return
		}
		switch o := r.Object.(type) {
		case *kapi.Service:
			if *verbose {
				log.Printf("%s service %s", r.Type, o.Name)
			}
			if r.Type == kwatch.Deleted {
				delete(u.services, o.Name)
				break
			}
			u.services[o.Name] = o
		case *kapi.Pod:
			if *verbose {
				log.Printf("%s pod %s", r.Type, o.Name)
			}
			if r.Type == kwatch.Deleted {
				delete(u.pods, o.Name)
				break
			}
			u.pods[o.Name] = o
		default:
			log.Printf("Unexpected object type in the event: %T", r)
		}
	case runCmd:
		u.existingDNS = r.existingDNS
	case stopCmd:
		u.err <- nil
		return
	case error:
		u.err <- r
		return
	default:
		log.Printf("Unexpected request of type %T", req)
	}
}

func (u *updater) watchServices(svcService kclient.ServiceInterface) {
	for {
		services, err := svcService.List(klabels.Everything())
		if err != nil {
			u.ch <- err
			return
		}
		u.ch <- services.Items
		retry, err := u.watch(svcService, services.ResourceVersion)
		if err != nil {
			u.ch <- err
			return
		}
		if !retry {
			break
		}
		time.Sleep(retryWatch)
	}
}

func (u *updater) watchPods(podService kclient.PodInterface) {
	for {
		pods, err := podService.List(klabels.Everything())
		if err != nil {
			u.ch <- err
			return
		}
		u.ch <- pods.Items
		retry, err := u.watch(podService, pods.ResourceVersion)
		if err != nil {
			u.ch <- err
			return
		}
		if !retry {
			break
		}
		time.Sleep(retryWatch)
	}
}

type watcher interface {
	Watch(klabels.Selector, kfields.Selector, string) (kwatch.Interface, error)
}

func (u *updater) watch(storage watcher, version string) (retry bool, err error) {
	watcher, err := storage.Watch(klabels.Everything(), kfields.Everything(), version)
	if err != nil {
		return false, err
	}
	var nextPoll <-chan time.Time
	if *pollInterval > 0 {
		nextPoll = time.After(*pollInterval)
	}
	for {
		select {
		case <-u.stopChan:
			watcher.Stop()
			return false, nil
		case <-nextPoll:
			watcher.Stop()
			return true, nil
		case e, ok := <-watcher.ResultChan():
			if !ok {
				return true, nil
			}
			u.ch <- e
		}
	}
}

func (u *updater) stop() {
	u.ch <- stopCmd{}
}

func (u *updater) run(existingDNS map[string]string) error {
	u.ch <- runCmd{existingDNS}
	defer close(u.stopChan)
	return <-u.err
}

func (u *updater) commit() {
	target := make(map[string]string)
	for _, s := range u.services {
		// DNS record for the service.
		if *svcDomain != "" {
			for _, port := range s.Spec.Ports {
				target[buildServiceName(port.Name, s.Name, s.Namespace, *svcDomain)] = newDNSRecord(s.Spec.PortalIP, port.Port)
			}
		}

		// FIXME: This is the workaround for builtin "kubernetes" and "kubernetes-ro" services with empty selectors.
		if len(s.Spec.Selector) == 0 {
			continue
		}

		if *podDomain == "" {
			continue
		}
		for _, p := range u.pods {
			if p.Status.Phase != kapi.PodRunning {
				continue
			}
			if p.Namespace != s.Namespace {
				continue
			}
			labels := p.ObjectMeta.Labels
			match := true
			for k, v := range s.Spec.Selector {
				if labels[k] != v {
					match = false
					break
				}
			}
			if !match {
				continue
			}
			// DNS record for the pod matching this service.
			for _, port := range s.Spec.Ports {
				target[buildPodName(p.Name, port.Name, s.Name, s.Namespace, *podDomain)] = newDNSRecord(p.Status.PodIP, port.Port)
			}
		}
	}

	for k := range u.existingDNS {
		if target[k] == "" {
			if *verbose {
				log.Printf("Deleting DNS record: %s", skymsg.Domain(k))
			}
			if _, err := u.etcd.Delete(k, false); err != nil {
				log.Print(err)
				continue
			}
			delete(u.existingDNS, k)
		}
	}
	for k, v := range target {
		if u.existingDNS[k] != v {
			if *verbose {
				log.Printf("Setting DNS record: %s = %s", skymsg.Domain(k), v)
			}
			if _, err := u.etcd.Set(k, v, uint64(0)); err != nil {
				log.Print(err)
				continue
			}
			u.existingDNS[k] = v
		}
	}
}

func main() {
	flag.Parse()

	// Connect to the cluster services.
	etcdClient := etcd.NewClient(strings.Split(*etcdServers, ","))
	etcdClient.SyncCluster()
	kubeClient, err := newKubeClient()
	if err != nil {
		log.Fatalf("Failed to create kubernetes client: %v", err)
		os.Exit(2)
	}

	// Load existing DNS records.
	existingDNS := make(map[string]string)
	for _, domain := range []string{*svcDomain, *podDomain} {
		if domain == "" {
			continue
		}
		paths, err := etcdClient.Get(skymsg.Path(domain), false, true)
		if err != nil {
			log.Printf("Couldn't get list of existing DNS entries in %s: %v", domain, err)
			continue
		}
		buildExistingHosts(existingDNS, paths.Node)
	}

	// Load initial state and register watchers.
	upd := newUpdater(etcdClient)
	svcService := kubeClient.Services(kapi.NamespaceAll)
	podService := kubeClient.Pods(kapi.NamespaceAll)
	go kutil.Forever(func() { upd.watchServices(svcService) }, retryWatch)
	go kutil.Forever(func() { upd.watchPods(podService) }, retryWatch)

	// Run.
	if err := upd.run(existingDNS); err != nil {
		log.Print(err)
		os.Exit(6)
	}
}
