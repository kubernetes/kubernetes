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
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	kubectl_util "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
)

const (
	reloadQPS    = 10.0
	resyncPeriod = 10 * time.Second
	healthzPort  = 8081
)

var (
	flags = flag.NewFlagSet("", flag.ContinueOnError)

	// keyFunc for endpoints and services.
	keyFunc = framework.DeletionHandlingMetaNamespaceKeyFunc

	// Error used to indicate that a sync is deferred because the controller isn't ready yet
	deferredSync = fmt.Errorf("Deferring sync till endpoints controller has synced.")

	config = flags.String("cfg", "loadbalancer.json", `path to load balancer json config.
		Note that this is *not* the path to the configuration file for the load balancer
		itself, but rather, the path to the json configuration of how you would like the
		load balancer to behave in the kubernetes cluster.`)

	dry = flags.Bool("dry", false, `if set, a single dry run of configuration
		parsing is executed. Results written to stdout.`)

	cluster = flags.Bool("use-kubernetes-cluster-service", true, `If true, use the built in kubernetes
		cluster for creating the client`)

	// If you have pure tcp services or https services that need L3 routing, you
	// must specify them by name. Note that you are responsible for:
	// 1. Making sure there is no collision between the service ports of these services.
	//	- You can have multiple <mysql svc name>:3306 specifications in this map, and as
	//	  long as the service ports of your mysql service don't clash, you'll get
	//	  loadbalancing for each one.
	// 2. Exposing the service ports as node ports on a pod.
	// 3. Adding firewall rules so these ports can ingress traffic.
	//
	// Any service not specified in this map is treated as an http:80 service,
	// unless TargetService dictates otherwise.

	tcpServices = flags.String("tcp-services", "", `Comma separated list of tcp/https
		serviceName:servicePort pairings. This assumes you've opened up the right
		hostPorts for each service that serves ingress traffic.`)

	targetService = flags.String(
		"target-service", "", `Restrict loadbalancing to a single target service.`)

	// ForwardServices == true:
	// The lb just forwards packets to the vip of the service and we use
	// kube-proxy's inbuilt load balancing. You get rules:
	// backend svc_p1: svc_ip:p1
	// backend svc_p2: svc_ip:p2
	//
	// ForwardServices == false:
	// The lb is configured to match up services to endpoints. So for example,
	// you have (svc:p1, p2 -> tp1, tp2) we essentially get all endpoints with
	// the same targetport and create a new svc backend for them, i.e:
	// backend svc_p1: pod1:tp1, pod2:tp1
	// backend svc_p2: pod1:tp2, pod2:tp2

	forwardServices = flags.Bool("forward-services", false, `Forward to service vip
		instead of endpoints. This will use kube-proxy's inbuilt load balancing.`)

	httpPort  = flags.Int("http-port", 80, `Port to expose http services.`)
	statsPort = flags.Int("stats-port", 1936, `Port for loadbalancer stats,
		Used in the loadbalancer liveness probe.`)
)

// service encapsulates a single backend entry in the load balancer config.
// The Ep field can contain the ips of the pods that make up a service, or the
// clusterIP of the service itself (in which case the list has a single entry,
// and kubernetes handles loadbalancing across the service endpoints).
type service struct {
	Name string
	Ep   []string

	// FrontendPort is the port that the loadbalancer listens on for traffic
	// for this service. For http, it's always :80, for each tcp service it
	// is the service port of any service matching a name in the tcpServices set.
	FrontendPort int
}

// loadBalancerConfig represents loadbalancer specific configuration. Eventually
// kubernetes will have an api for l7 loadbalancing.
type loadBalancerConfig struct {
	Name      string `json:"name" description:"Name of the load balancer, eg: haproxy."`
	ReloadCmd string `json:"reloadCmd" description:"command used to reload the load balancer."`
	Config    string `json:"config" description:"path to loadbalancers configuration file."`
	Template  string `json:"template" description:"template for the load balancer config."`
	Algorithm string `json:"algorithm" description:"loadbalancing algorithm."`
}

// write writes the configuration file, will write to stdout if dryRun == true
func (cfg *loadBalancerConfig) write(services map[string][]service, dryRun bool) (err error) {
	var w io.Writer
	if dryRun {
		w = os.Stdout
	} else {
		w, err = os.Create(cfg.Config)
		if err != nil {
			return
		}
	}
	var t *template.Template
	t, err = template.ParseFiles(cfg.Template)
	if err != nil {
		return
	}
	return t.Execute(w, services)
}

// reload reloads the loadbalancer using the reload cmd specified in the json manifest.
func (cfg *loadBalancerConfig) reload() error {
	output, err := exec.Command("sh", "-c", cfg.ReloadCmd).CombinedOutput()
	msg := fmt.Sprintf("%v -- %v", cfg.Name, string(output))
	if err != nil {
		return fmt.Errorf("Error restarting %v: %v", msg, err)
	}
	glog.Infof(msg)
	return nil
}

// loadBalancerController watches the kubernetes api and adds/removes services
// from the loadbalancer, via loadBalancerConfig.
type loadBalancerController struct {
	cfg               *loadBalancerConfig
	queue             *workqueue.Type
	client            *client.Client
	epController      *framework.Controller
	svcController     *framework.Controller
	svcLister         cache.StoreToServiceLister
	epLister          cache.StoreToEndpointsLister
	reloadRateLimiter util.RateLimiter
	template          string
	targetService     string
	forwardServices   bool
	tcpServices       map[string]int
	httpPort          int
}

// getEndpoints returns a list of <endpoint ip>:<port> for a given service/target port combination.
func (lbc *loadBalancerController) getEndpoints(
	s *api.Service, servicePort *api.ServicePort) (endpoints []string) {
	ep, err := lbc.epLister.GetServiceEndpoints(s)
	if err != nil {
		return
	}

	// The intent here is to create a union of all subsets that match a targetPort.
	// We know the endpoint already matches the service, so all pod ips that have
	// the target port are capable of service traffic for it.
	for _, ss := range ep.Subsets {
		for _, epPort := range ss.Ports {
			var targetPort int
			switch servicePort.TargetPort.Kind {
			case util.IntstrInt:
				if epPort.Port == servicePort.TargetPort.IntVal {
					targetPort = epPort.Port
				}
			case util.IntstrString:
				if epPort.Name == servicePort.TargetPort.StrVal {
					targetPort = epPort.Port
				}
			}
			if targetPort == 0 {
				continue
			}
			for _, epAddress := range ss.Addresses {
				endpoints = append(endpoints, fmt.Sprintf("%v:%v", epAddress.IP, targetPort))
			}
		}
	}
	return
}

// encapsulates all the hacky convenience type name modifications for lb rules.
// - :80 services don't need a :80 postfix
// - default ns should be accessible without /ns/name (when we have /ns support)
func getServiceNameForLBRule(s *api.Service, servicePort int) string {
	if servicePort == 80 {
		return s.Name
	}
	return fmt.Sprintf("%v:%v", s.Name, servicePort)
}

// getServices returns a list of services and their endpoints.
func (lbc *loadBalancerController) getServices() (httpSvc []service, tcpSvc []service) {
	ep := []string{}
	services, _ := lbc.svcLister.List()
	for _, s := range services.Items {
		if s.Spec.Type == api.ServiceTypeLoadBalancer {
			glog.Infof("Ignoring service %v, it already has a loadbalancer", s.Name)
			continue
		}
		for _, servicePort := range s.Spec.Ports {
			// TODO: headless services?
			sName := s.Name
			if servicePort.Protocol == api.ProtocolUDP ||
				(lbc.targetService != "" && lbc.targetService != sName) {
				glog.Infof("Ignoring %v: %+v", sName, servicePort)
				continue
			}

			if lbc.forwardServices {
				ep = []string{
					fmt.Sprintf("%v:%v", s.Spec.ClusterIP, servicePort.Port)}
			} else {
				ep = lbc.getEndpoints(&s, &servicePort)
			}
			if len(ep) == 0 {
				glog.Infof("No endpoints found for service %v, port %+v",
					sName, servicePort)
				continue
			}
			newSvc := service{
				Name: getServiceNameForLBRule(&s, servicePort.Port),
				Ep:   ep,
			}
			if port, ok := lbc.tcpServices[sName]; ok && port == servicePort.Port {
				newSvc.FrontendPort = servicePort.Port
				tcpSvc = append(tcpSvc, newSvc)
			} else {
				newSvc.FrontendPort = lbc.httpPort
				httpSvc = append(httpSvc, newSvc)
			}
			glog.Infof("Found service: %+v", newSvc)
		}
	}
	return
}

// sync all services with the loadbalancer.
func (lbc *loadBalancerController) sync(dryRun bool) error {
	if !lbc.epController.HasSynced() || !lbc.svcController.HasSynced() {
		time.Sleep(100 * time.Millisecond)
		return deferredSync
	}
	httpSvc, tcpSvc := lbc.getServices()
	if len(httpSvc) == 0 && len(tcpSvc) == 0 {
		return nil
	}
	if err := lbc.cfg.write(
		map[string][]service{
			"httpServices": httpSvc,
			"tcpServices":  tcpSvc,
		}, dryRun); err != nil {
		return err
	}
	if dryRun {
		return nil
	}
	lbc.reloadRateLimiter.Accept()
	return lbc.cfg.reload()
}

// worker handles the work queue.
func (lbc *loadBalancerController) worker() {
	for {
		key, _ := lbc.queue.Get()
		glog.Infof("Sync triggered by service %v", key)
		if err := lbc.sync(false); err != nil {
			glog.Infof("Requeuing %v because of error: %v", key, err)
			lbc.queue.Add(key)
		} else {
			lbc.queue.Done(key)
		}
	}
}

// newLoadBalancerController creates a new controller from the given config.
func newLoadBalancerController(cfg *loadBalancerConfig, kubeClient *client.Client, namespace string) *loadBalancerController {

	lbc := loadBalancerController{
		cfg:    cfg,
		client: kubeClient,
		queue:  workqueue.New(),
		reloadRateLimiter: util.NewTokenBucketRateLimiter(
			reloadQPS, int(reloadQPS)),
		targetService:   *targetService,
		forwardServices: *forwardServices,
		httpPort:        *httpPort,
		tcpServices:     map[string]int{},
	}

	for _, service := range strings.Split(*tcpServices, ",") {
		portSplit := strings.Split(service, ":")
		if len(portSplit) != 2 {
			glog.Errorf("Ignoring misconfigured TCP service %v", service)
			continue
		}
		if port, err := strconv.Atoi(portSplit[1]); err != nil {
			glog.Errorf("Ignoring misconfigured TCP service %v: %v", service, err)
			continue
		} else {
			lbc.tcpServices[portSplit[0]] = port
		}
	}
	enqueue := func(obj interface{}) {
		key, err := keyFunc(obj)
		if err != nil {
			glog.Infof("Couldn't get key for object %+v: %v", obj, err)
			return
		}
		lbc.queue.Add(key)
	}
	eventHandlers := framework.ResourceEventHandlerFuncs{
		AddFunc:    enqueue,
		DeleteFunc: enqueue,
		UpdateFunc: func(old, cur interface{}) {
			if !reflect.DeepEqual(old, cur) {
				enqueue(cur)
			}
		},
	}

	lbc.svcLister.Store, lbc.svcController = framework.NewInformer(
		cache.NewListWatchFromClient(
			lbc.client, "services", namespace, fields.Everything()),
		&api.Service{}, resyncPeriod, eventHandlers)

	lbc.epLister.Store, lbc.epController = framework.NewInformer(
		cache.NewListWatchFromClient(
			lbc.client, "endpoints", namespace, fields.Everything()),
		&api.Endpoints{}, resyncPeriod, eventHandlers)

	return &lbc
}

// parseCfg parses the given configuration file.
// cmd line params take precedence over config directives.
func parseCfg(configPath string) *loadBalancerConfig {
	jsonBlob, err := ioutil.ReadFile(configPath)
	if err != nil {
		glog.Fatalf("Could not parse lb config: %v", err)
	}
	var cfg loadBalancerConfig
	err = json.Unmarshal(jsonBlob, &cfg)
	if err != nil {
		glog.Fatalf("Unable to unmarshal json blob: %v", string(jsonBlob))
	}
	glog.Infof("Creating new loadbalancer: %+v", cfg)
	return &cfg
}

// healthzServer services liveness probes.
func healthzServer() {
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		// Delegate a check to the haproxy stats service.
		response, err := http.Get(fmt.Sprintf("http://localhost:%v", *statsPort))
		if err != nil {
			glog.Infof("Error %v", err)
			w.WriteHeader(http.StatusInternalServerError)
		} else {
			defer response.Body.Close()
			if response.StatusCode != http.StatusOK {
				contents, err := ioutil.ReadAll(response.Body)
				if err != nil {
					glog.Infof("Error reading resonse on receiving status %v: %v",
						response.StatusCode, err)
				}
				glog.Infof("%v\n", string(contents))
				w.WriteHeader(response.StatusCode)
			} else {
				w.WriteHeader(200)
				w.Write([]byte("ok"))
			}
		}
	})
	glog.Fatal(http.ListenAndServe(fmt.Sprintf(":%v", healthzPort), nil))
}

func dryRun(lbc *loadBalancerController) {
	var err error
	for err = lbc.sync(true); err == deferredSync; err = lbc.sync(true) {
	}
	if err != nil {
		glog.Infof("ERROR: %+v", err)
	}
}

func main() {
	flags.Parse(os.Args)
	cfg := parseCfg(*config)
	if len(*tcpServices) == 0 {
		glog.Infof("All tcp/https services will be ignored.")
	}
	go healthzServer()

	var kubeClient *client.Client
	var err error
	clientConfig := kubectl_util.DefaultClientConfig(flags)
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
	namespace, specified, err := clientConfig.Namespace()
	if err != nil {
		glog.Fatalf("unexpected error: %v", err)
	}
	if !specified {
		namespace = "default"
	}

	// TODO: Handle multiple namespaces
	lbc := newLoadBalancerController(cfg, kubeClient, namespace)
	go lbc.epController.Run(util.NeverStop)
	go lbc.svcController.Run(util.NeverStop)
	if *dry {
		dryRun(lbc)
	} else {
		util.Until(lbc.worker, time.Second, util.NeverStop)
	}
}
