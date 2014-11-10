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

package master

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator/bearertoken"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator/tokenfile"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/handlers"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/binding"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/controller"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/event"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/ui"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// Config is a structure used to configure a Master.
type Config struct {
	Client                *client.Client
	Cloud                 cloudprovider.Interface
	EtcdHelper            tools.EtcdHelper
	HealthCheckMinions    bool
	EventTTL              time.Duration
	MinionRegexp          string
	KubeletClient         client.KubeletClient
	PortalNet             *net.IPNet
	Mux                   apiserver.Mux
	EnableLogsSupport     bool
	EnableUISupport       bool
	APIPrefix             string
	CorsAllowedOriginList util.StringList
	TokenAuthFile         string
	Authorizer            authorizer.Authorizer

	// Number of masters running; all masters must be started with the
	// same value for this field. (Numbers > 1 currently untested.)
	MasterCount int

	// The port on PublicAddress where a read-only server will be installed.
	// Defaults to 7080 if not set.
	ReadOnlyPort int
	// The port on PublicAddress where a read-write server will be installed.
	// Defaults to 443 if not set.
	ReadWritePort int

	// If empty, the first result from net.InterfaceAddrs will be used.
	PublicAddress string
}

// Master contains state for a Kubernetes cluster master/api server.
type Master struct {
	// "Inputs", Copied from Config
	podRegistry           pod.Registry
	controllerRegistry    controller.Registry
	serviceRegistry       service.Registry
	endpointRegistry      endpoint.Registry
	minionRegistry        minion.Registry
	bindingRegistry       binding.Registry
	eventRegistry         generic.Registry
	storage               map[string]apiserver.RESTStorage
	client                *client.Client
	portalNet             *net.IPNet
	mux                   apiserver.Mux
	enableLogsSupport     bool
	enableUISupport       bool
	apiPrefix             string
	corsAllowedOriginList util.StringList
	tokenAuthFile         string
	authorizer            authorizer.Authorizer
	masterCount           int

	readOnlyServer  string
	readWriteServer string
	masterServices  *util.Runner

	// "Outputs"
	Handler         http.Handler
	InsecureHandler http.Handler
}

// NewEtcdHelper returns an EtcdHelper for the provided arguments or an error if the version
// is incorrect.
func NewEtcdHelper(client tools.EtcdGetSet, version string) (helper tools.EtcdHelper, err error) {
	if version == "" {
		version = latest.Version
	}
	versionInterfaces, err := latest.InterfacesFor(version)
	if err != nil {
		return helper, err
	}
	return tools.EtcdHelper{client, versionInterfaces.Codec, tools.RuntimeVersionAdapter{versionInterfaces.MetadataAccessor}}, nil
}

// setDefaults fills in any fields not set that are required to have valid data.
func setDefaults(c *Config) {
	if c.PortalNet == nil {
		defaultNet := "10.0.0.0/24"
		glog.Warningf("Portal net unspecified. Defaulting to %v.", defaultNet)
		_, portalNet, err := net.ParseCIDR(defaultNet)
		if err != nil {
			glog.Fatalf("Unable to parse CIDR: %v", err)
		}
		c.PortalNet = portalNet
	}
	if c.MasterCount == 0 {
		// Clearly, there will be at least one master.
		c.MasterCount = 1
	}
	if c.ReadOnlyPort == 0 {
		c.ReadOnlyPort = 7080
	}
	if c.ReadWritePort == 0 {
		c.ReadWritePort = 443
	}
	for c.PublicAddress == "" {
		// Find and use the first non-loopback address.
		// TODO: potentially it'd be useful to skip the docker interface if it
		// somehow is first in the list.
		addrs, err := net.InterfaceAddrs()
		if err != nil {
			glog.Fatalf("Unable to get network interfaces: error='%v'", err)
		}
		found := false
		for i := range addrs {
			ip, _, err := net.ParseCIDR(addrs[i].String())
			if err != nil {
				glog.Errorf("Error parsing '%v': %v", addrs[i], err)
				continue
			}
			if ip.IsLoopback() {
				glog.Infof("'%v' (%v) is a loopback address, ignoring.", ip, addrs[i])
				continue
			}
			found = true
			c.PublicAddress = ip.String()
			glog.Infof("Will report %v as public IP address.", ip)
			break
		}
		if !found {
			glog.Errorf("Unable to find suitible network address in list: '%v'\n"+
				"Will try again in 5 seconds. Set the public address directly to avoid this wait.", addrs)
			time.Sleep(5 * time.Second)
		}
	}
}

// New returns a new instance of Master from the given config.
// Certain config fields will be set to a default value if unset,
// including:
//   PortalNet
//   MasterCount
//   ReadOnlyPort
//   ReadWritePort
//   PublicAddress
// Certain config fields must be specified, including:
//   KubeletClient
// Public fields:
//   Handler -- The returned master has a field TopHandler which is an
//   http.Handler which handles all the endpoints provided by the master,
//   including the API, the UI, and miscelaneous debugging endpoints.  All
//   these are subject to authorization and authentication.
//   InsecureHandler -- an http.Handler which handles all the same
//   endpoints as Handler, but no authorization and authentication is done.
// Public methods:
//   HandleWithAuth -- Allows caller to add an http.Handler for an endpoint
//   that uses the same authentication and authorization (if any is configured)
//   as the master's built-in endpoints.
//   If the caller wants to add additional endpoints not using the master's
//   auth, then the caller should create a handler for those endpoints, which delegates the
//   any unhandled paths to "Handler".
func New(c *Config) *Master {
	setDefaults(c)
	minionRegistry := makeMinionRegistry(c)
	serviceRegistry := etcd.NewRegistry(c.EtcdHelper, nil)
	boundPodFactory := &pod.BasicBoundPodFactory{
		ServiceRegistry: serviceRegistry,
	}
	if c.KubeletClient == nil {
		glog.Fatalf("master.New() called with config.KubeletClient == nil")
	}
	m := &Master{
		podRegistry:           etcd.NewRegistry(c.EtcdHelper, boundPodFactory),
		controllerRegistry:    etcd.NewRegistry(c.EtcdHelper, nil),
		serviceRegistry:       serviceRegistry,
		endpointRegistry:      etcd.NewRegistry(c.EtcdHelper, nil),
		bindingRegistry:       etcd.NewRegistry(c.EtcdHelper, boundPodFactory),
		eventRegistry:         event.NewEtcdRegistry(c.EtcdHelper, uint64(c.EventTTL.Seconds())),
		minionRegistry:        minionRegistry,
		client:                c.Client,
		portalNet:             c.PortalNet,
		mux:                   http.NewServeMux(),
		enableLogsSupport:     c.EnableLogsSupport,
		enableUISupport:       c.EnableUISupport,
		apiPrefix:             c.APIPrefix,
		corsAllowedOriginList: c.CorsAllowedOriginList,
		tokenAuthFile:         c.TokenAuthFile,
		authorizer:            c.Authorizer,

		masterCount:     c.MasterCount,
		readOnlyServer:  net.JoinHostPort(c.PublicAddress, strconv.Itoa(int(c.ReadOnlyPort))),
		readWriteServer: net.JoinHostPort(c.PublicAddress, strconv.Itoa(int(c.ReadWritePort))),
	}
	m.masterServices = util.NewRunner(m.serviceWriterLoop, m.roServiceWriterLoop)
	m.init(c)
	return m
}

// HandleWithAuth adds an http.Handler for pattern to an http.ServeMux
// Applies the same authentication and authorization (if any is configured)
// to the request is used for the master's built-in endpoints.
func (m *Master) HandleWithAuth(pattern string, handler http.Handler) {
	// TODO: Add a way for plugged-in endpoints to translate their
	// URLs into attributes that an Authorizer can understand, and have
	// sensible policy defaults for plugged-in endpoints.  This will be different
	// for generic endpoints versus REST object endpoints.
	m.mux.Handle(pattern, handler)
}

// HandleFuncWithAuth adds an http.Handler for pattern to an http.ServeMux
// Applies the same authentication and authorization (if any is configured)
// to the request is used for the master's built-in endpoints.
func (m *Master) HandleFuncWithAuth(pattern string, handler func(http.ResponseWriter, *http.Request)) {
	m.mux.HandleFunc(pattern, handler)
}

func makeMinionRegistry(c *Config) minion.Registry {
	var minionRegistry minion.Registry = etcd.NewRegistry(c.EtcdHelper, nil)
	if c.HealthCheckMinions {
		minionRegistry = minion.NewHealthyRegistry(minionRegistry, c.KubeletClient)
	}
	return minionRegistry
}

// init initializes master.
func (m *Master) init(c *Config) {
	podCache := NewPodCache(c.KubeletClient, m.podRegistry)
	go util.Forever(func() { podCache.UpdateAllContainers() }, time.Second*30)

	var userContexts = handlers.NewUserRequestContext()
	var authenticator authenticator.Request
	if len(c.TokenAuthFile) != 0 {
		tokenAuthenticator, err := tokenfile.New(c.TokenAuthFile)
		if err != nil {
			glog.Fatalf("Unable to load the token authentication file '%s': %v", c.TokenAuthFile, err)
		}
		authenticator = bearertoken.New(tokenAuthenticator)
	}

	m.storage = map[string]apiserver.RESTStorage{
		"pods": pod.NewREST(&pod.RESTConfig{
			CloudProvider: c.Cloud,
			PodCache:      podCache,
			PodInfoGetter: c.KubeletClient,
			Registry:      m.podRegistry,
			Minions:       m.client.Minions(),
		}),
		"replicationControllers": controller.NewREST(m.controllerRegistry, m.podRegistry),
		"services":               service.NewREST(m.serviceRegistry, c.Cloud, m.minionRegistry, m.portalNet),
		"endpoints":              endpoint.NewREST(m.endpointRegistry),
		"minions":                minion.NewREST(m.minionRegistry),
		"events":                 event.NewREST(m.eventRegistry),

		// TODO: should appear only in scheduler API group.
		"bindings": binding.NewREST(m.bindingRegistry),
	}

	apiserver.NewAPIGroup(m.API_v1beta1()).InstallREST(m.mux, c.APIPrefix+"/v1beta1")
	apiserver.NewAPIGroup(m.API_v1beta2()).InstallREST(m.mux, c.APIPrefix+"/v1beta2")
	versionHandler := apiserver.APIVersionHandler("v1beta1", "v1beta2")
	m.mux.Handle(c.APIPrefix, versionHandler)
	apiserver.InstallSupport(m.mux)
	serversToValidate := m.getServersToValidate(c)

	apiserver.InstallValidator(m.mux, serversToValidate)
	if c.EnableLogsSupport {
		apiserver.InstallLogsSupport(m.mux)
	}
	if c.EnableUISupport {
		ui.InstallSupport(m.mux)
	}

	handler := http.Handler(m.mux.(*http.ServeMux))

	if len(c.CorsAllowedOriginList) > 0 {
		allowedOriginRegexps, err := util.CompileRegexps(c.CorsAllowedOriginList)
		if err != nil {
			glog.Fatalf("Invalid CORS allowed origin, --cors_allowed_origins flag was set to %v - %v", strings.Join(c.CorsAllowedOriginList, ","), err)
		}
		handler = apiserver.CORS(handler, allowedOriginRegexps, nil, nil, "true")
	}

	m.InsecureHandler = handler

	attributeGetter := apiserver.NewRequestAttributeGetter(userContexts)
	handler = apiserver.WithAuthorizationCheck(handler, attributeGetter, m.authorizer)

	// Install Authenticator
	if authenticator != nil {
		handler = handlers.NewRequestAuthenticator(userContexts, authenticator, handlers.Unauthorized, handler)
	}
	m.mux.HandleFunc("/_whoami", handleWhoAmI(authenticator))

	m.Handler = handler

	// TODO: Attempt clean shutdown?
	m.masterServices.Start()
}

func (m *Master) getServersToValidate(c *Config) map[string]apiserver.Server {
	serversToValidate := map[string]apiserver.Server{
		"controller-manager": {Addr: "127.0.0.1", Port: 10252, Path: "/healthz"},
		"scheduler":          {Addr: "127.0.0.1", Port: 10251, Path: "/healthz"},
	}
	for ix, machine := range c.EtcdHelper.Client.GetCluster() {
		etcdUrl, err := url.Parse(machine)
		if err != nil {
			glog.Errorf("Failed to parse etcd url for validation: %v", err)
			continue
		}
		var port int
		var addr string
		if strings.Contains(etcdUrl.Host, ":") {
			var portString string
			addr, portString, err = net.SplitHostPort(etcdUrl.Host)
			if err != nil {
				glog.Errorf("Failed to split host/port: %s (%v)", etcdUrl.Host, err)
				continue
			}
			port, _ = strconv.Atoi(portString)
		} else {
			addr = etcdUrl.Host
			port = 4001
		}
		serversToValidate[fmt.Sprintf("etcd-%d", ix)] = apiserver.Server{Addr: addr, Port: port, Path: "/v2/keys/"}
	}
	nodes, err := m.minionRegistry.ListMinions(api.NewDefaultContext())
	if err != nil {
		glog.Errorf("Failed to list minions: %v", err)
	}
	for ix, node := range nodes.Items {
		serversToValidate[fmt.Sprintf("node-%d", ix)] = apiserver.Server{Addr: node.HostIP, Port: 10250, Path: "/healthz"}
	}
	return serversToValidate
}

// API_v1beta1 returns the resources and codec for API version v1beta1.
func (m *Master) API_v1beta1() (map[string]apiserver.RESTStorage, runtime.Codec, string, runtime.SelfLinker) {
	storage := make(map[string]apiserver.RESTStorage)
	for k, v := range m.storage {
		storage[k] = v
	}
	return storage, v1beta1.Codec, "/api/v1beta1", latest.SelfLinker
}

// API_v1beta2 returns the resources and codec for API version v1beta2.
func (m *Master) API_v1beta2() (map[string]apiserver.RESTStorage, runtime.Codec, string, runtime.SelfLinker) {
	storage := make(map[string]apiserver.RESTStorage)
	for k, v := range m.storage {
		storage[k] = v
	}
	return storage, v1beta2.Codec, "/api/v1beta2", latest.SelfLinker
}
