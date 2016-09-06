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

package genericapiserver

import (
	"crypto/tls"
	"fmt"
	"mime"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	"github.com/golang/glog"
	"gopkg.in/natefinch/lumberjack.v2"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/apiserver/audit"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/handlers"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/genericapiserver/options"
	"k8s.io/kubernetes/pkg/genericapiserver/routes"
	genericvalidation "k8s.io/kubernetes/pkg/genericapiserver/validation"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	ipallocator "k8s.io/kubernetes/pkg/registry/service/ipallocator"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	utilnet "k8s.io/kubernetes/pkg/util/net"
)

// Config is a structure used to configure a GenericAPIServer.
type Config struct {
	// The storage factory for other objects
	StorageFactory     StorageFactory
	AuditLogPath       string
	AuditLogMaxAge     int
	AuditLogMaxBackups int
	AuditLogMaxSize    int
	// Allow downstream consumers to disable swagger.
	// This includes returning the generated swagger spec at /swaggerapi and swagger ui at /swagger-ui.
	EnableSwaggerSupport bool
	// Allow downstream consumers to disable swagger ui.
	// Note that this is ignored if EnableSwaggerSupport is false
	EnableSwaggerUI bool
	// Allows api group versions or specific resources to be conditionally enabled/disabled.
	APIResourceConfigSource APIResourceConfigSource
	// allow downstream consumers to disable the index route
	EnableIndex             bool
	EnableProfiling         bool
	EnableVersion           bool
	EnableWatchCache        bool
	EnableGarbageCollection bool
	APIPrefix               string
	APIGroupPrefix          string
	CorsAllowedOriginList   []string
	Authenticator           authenticator.Request
	// TODO(roberthbailey): Remove once the server no longer supports http basic auth.
	SupportsBasicAuth      bool
	Authorizer             authorizer.Authorizer
	AdmissionControl       admission.Interface
	MasterServiceNamespace string
	// TODO(ericchiang): Determine if policy escalation checks should be an admission controller.
	AuthorizerRBACSuperUser string

	// Map requests to contexts. Exported so downstream consumers can provider their own mappers
	RequestContextMapper api.RequestContextMapper

	// Required, the interface for serializing and converting objects to and from the wire
	Serializer runtime.NegotiatedSerializer

	// If specified, all web services will be registered into this container
	RestfulContainer *restful.Container

	// If specified, requests will be allocated a random timeout between this value, and twice this value.
	// Note that it is up to the request handlers to ignore or honor this timeout. In seconds.
	MinRequestTimeout int

	// Number of masters running; all masters must be started with the
	// same value for this field. (Numbers > 1 currently untested.)
	MasterCount int

	// The port on PublicAddress where a read-write server will be installed.
	// Defaults to 6443 if not set.
	ReadWritePort int

	// ExternalHost is the host name to use for external (public internet) facing URLs (e.g. Swagger)
	ExternalHost string

	// PublicAddress is the IP address where members of the cluster (kubelet,
	// kube-proxy, services, etc.) can reach the GenericAPIServer.
	// If nil or 0.0.0.0, the host's default interface will be used.
	PublicAddress net.IP

	// Control the interval that pod, node IP, and node heath status caches
	// expire.
	CacheTimeout time.Duration

	// The range of IPs to be assigned to services with type=ClusterIP or greater
	ServiceClusterIPRange *net.IPNet

	// The IP address for the GenericAPIServer service (must be inside ServiceClusterIPRange)
	ServiceReadWriteIP net.IP

	// Port for the apiserver service.
	ServiceReadWritePort int

	// The range of ports to be assigned to services with type=NodePort or greater
	ServiceNodePortRange utilnet.PortRange

	// Used to customize default proxy dial/tls options
	ProxyDialer          apiserver.ProxyDialerFunc
	ProxyTLSClientConfig *tls.Config

	// Additional ports to be exposed on the GenericAPIServer service
	// extraServicePorts is injectable in the event that more ports
	// (other than the default 443/tcp) are exposed on the GenericAPIServer
	// and those ports need to be load balanced by the GenericAPIServer
	// service because this pkg is linked by out-of-tree projects
	// like openshift which want to use the GenericAPIServer but also do
	// more stuff.
	ExtraServicePorts []api.ServicePort
	// Additional ports to be exposed on the GenericAPIServer endpoints
	// Port names should align with ports defined in ExtraServicePorts
	ExtraEndpointPorts []api.EndpointPort

	KubernetesServiceNodePort int

	// EnableOpenAPISupport enables OpenAPI support. Allow downstream customers to disable OpenAPI spec.
	EnableOpenAPISupport bool

	// OpenAPIInfo will be directly available as Info section of Open API spec.
	OpenAPIInfo spec.Info

	// OpenAPIDefaultResponse will be used if an web service operation does not have any responses listed.
	OpenAPIDefaultResponse spec.Response
}

func NewConfig(options *options.ServerRunOptions) *Config {
	return &Config{
		APIGroupPrefix:            options.APIGroupPrefix,
		APIPrefix:                 options.APIPrefix,
		CorsAllowedOriginList:     options.CorsAllowedOriginList,
		AuditLogPath:              options.AuditLogPath,
		AuditLogMaxAge:            options.AuditLogMaxAge,
		AuditLogMaxBackups:        options.AuditLogMaxBackups,
		AuditLogMaxSize:           options.AuditLogMaxSize,
		EnableGarbageCollection:   options.EnableGarbageCollection,
		EnableIndex:               true,
		EnableProfiling:           options.EnableProfiling,
		EnableSwaggerSupport:      true,
		EnableSwaggerUI:           options.EnableSwaggerUI,
		EnableVersion:             true,
		EnableWatchCache:          options.EnableWatchCache,
		ExternalHost:              options.ExternalHost,
		KubernetesServiceNodePort: options.KubernetesServiceNodePort,
		MasterCount:               options.MasterCount,
		MinRequestTimeout:         options.MinRequestTimeout,
		PublicAddress:             options.AdvertiseAddress,
		ReadWritePort:             options.SecurePort,
		ServiceClusterIPRange:     &options.ServiceClusterIPRange,
		ServiceNodePortRange:      options.ServiceNodePortRange,
		EnableOpenAPISupport:      true,
		OpenAPIDefaultResponse: spec.Response{
			ResponseProps: spec.ResponseProps{
				Description: "Default Response."}},
		OpenAPIInfo: spec.Info{
			InfoProps: spec.InfoProps{
				Title:   "Generic API Server",
				Version: "unversioned",
			},
		},
	}
}

// setDefaults fills in any fields not set that are required to have valid data.
func (c *Config) setDefaults() {
	if c.ServiceClusterIPRange == nil {
		defaultNet := "10.0.0.0/24"
		glog.Warningf("Network range for service cluster IPs is unspecified. Defaulting to %v.", defaultNet)
		_, serviceClusterIPRange, err := net.ParseCIDR(defaultNet)
		if err != nil {
			glog.Fatalf("Unable to parse CIDR: %v", err)
		}
		if size := ipallocator.RangeSize(serviceClusterIPRange); size < 8 {
			glog.Fatalf("The service cluster IP range must be at least %d IP addresses", 8)
		}
		c.ServiceClusterIPRange = serviceClusterIPRange
	}
	if c.ServiceReadWriteIP == nil {
		// Select the first valid IP from ServiceClusterIPRange to use as the GenericAPIServer service IP.
		serviceReadWriteIP, err := ipallocator.GetIndexedIP(c.ServiceClusterIPRange, 1)
		if err != nil {
			glog.Fatalf("Failed to generate service read-write IP for GenericAPIServer service: %v", err)
		}
		glog.V(4).Infof("Setting GenericAPIServer service IP to %q (read-write).", serviceReadWriteIP)
		c.ServiceReadWriteIP = serviceReadWriteIP
	}
	if c.ServiceReadWritePort == 0 {
		c.ServiceReadWritePort = 443
	}
	if c.ServiceNodePortRange.Size == 0 {
		// TODO: Currently no way to specify an empty range (do we need to allow this?)
		// We should probably allow this for clouds that don't require NodePort to do load-balancing (GCE)
		// but then that breaks the strict nestedness of ServiceType.
		// Review post-v1
		c.ServiceNodePortRange = options.DefaultServiceNodePortRange
		glog.Infof("Node port range unspecified. Defaulting to %v.", c.ServiceNodePortRange)
	}
	if c.MasterCount == 0 {
		// Clearly, there will be at least one GenericAPIServer.
		c.MasterCount = 1
	}
	if c.ReadWritePort == 0 {
		c.ReadWritePort = 6443
	}
	if c.CacheTimeout == 0 {
		c.CacheTimeout = 5 * time.Second
	}
	if c.RequestContextMapper == nil {
		c.RequestContextMapper = api.NewRequestContextMapper()
	}
	if len(c.ExternalHost) == 0 && c.PublicAddress != nil {
		hostAndPort := c.PublicAddress.String()
		if c.ReadWritePort != 0 {
			hostAndPort = net.JoinHostPort(hostAndPort, strconv.Itoa(c.ReadWritePort))
		}
		c.ExternalHost = hostAndPort
	}
}

// New returns a new instance of GenericAPIServer from the given config.
// Certain config fields will be set to a default value if unset,
// including:
//   ServiceClusterIPRange
//   ServiceNodePortRange
//   MasterCount
//   ReadWritePort
//   PublicAddress
// Public fields:
//   Handler -- The returned GenericAPIServer has a field TopHandler which is an
//   http.Handler which handles all the endpoints provided by the GenericAPIServer,
//   including the API, the UI, and miscellaneous debugging endpoints.  All
//   these are subject to authorization and authentication.
//   InsecureHandler -- an http.Handler which handles all the same
//   endpoints as Handler, but no authorization and authentication is done.
// Public methods:
//   HandleWithAuth -- Allows caller to add an http.Handler for an endpoint
//   that uses the same authentication and authorization (if any is configured)
//   as the GenericAPIServer's built-in endpoints.
//   If the caller wants to add additional endpoints not using the GenericAPIServer's
//   auth, then the caller should create a handler for those endpoints, which delegates the
//   any unhandled paths to "Handler".
func (c Config) New() (*GenericAPIServer, error) {
	if c.Serializer == nil {
		return nil, fmt.Errorf("Genericapiserver.New() called with config.Serializer == nil")
	}

	c.setDefaults()

	s := &GenericAPIServer{
		ServiceClusterIPRange: c.ServiceClusterIPRange,
		ServiceNodePortRange:  c.ServiceNodePortRange,
		legacyAPIPrefix:       c.APIPrefix,
		apiPrefix:             c.APIGroupPrefix,
		admissionControl:      c.AdmissionControl,
		requestContextMapper:  c.RequestContextMapper,
		Serializer:            c.Serializer,

		minRequestTimeout:    time.Duration(c.MinRequestTimeout) * time.Second,
		enableSwaggerSupport: c.EnableSwaggerSupport,

		MasterCount:          c.MasterCount,
		ExternalAddress:      c.ExternalHost,
		ClusterIP:            c.PublicAddress,
		PublicReadWritePort:  c.ReadWritePort,
		ServiceReadWriteIP:   c.ServiceReadWriteIP,
		ServiceReadWritePort: c.ServiceReadWritePort,
		ExtraServicePorts:    c.ExtraServicePorts,
		ExtraEndpointPorts:   c.ExtraEndpointPorts,

		KubernetesServiceNodePort: c.KubernetesServiceNodePort,
		apiGroupsForDiscovery:     map[string]unversioned.APIGroup{},

		enableOpenAPISupport:   c.EnableOpenAPISupport,
		openAPIInfo:            c.OpenAPIInfo,
		openAPIDefaultResponse: c.OpenAPIDefaultResponse,
	}

	if c.EnableWatchCache {
		s.storageDecorator = registry.StorageWithCacher
	} else {
		s.storageDecorator = generic.UndecoratedStorage
	}

	if c.RestfulContainer != nil {
		s.HandlerContainer = c.RestfulContainer
	} else {
		s.HandlerContainer = NewHandlerContainer(http.NewServeMux(), c.Serializer)
	}
	// Use CurlyRouter to be able to use regular expressions in paths. Regular expressions are required in paths for example for proxy (where the path is proxy/{kind}/{name}/{*})
	s.HandlerContainer.Router(restful.CurlyRouter{})
	s.Mux = apiserver.NewPathRecorderMux(s.HandlerContainer.ServeMux)

	if c.ProxyDialer != nil || c.ProxyTLSClientConfig != nil {
		s.ProxyTransport = utilnet.SetTransportDefaults(&http.Transport{
			Dial:            c.ProxyDialer,
			TLSClientConfig: c.ProxyTLSClientConfig,
		})
	}

	// Send correct mime type for .svg files.
	// TODO: remove when https://github.com/golang/go/commit/21e47d831bafb59f22b1ea8098f709677ec8ce33
	// makes it into all of our supported go versions (only in v1.7.1 now).
	mime.AddExtensionType(".svg", "image/svg+xml")

	// Register root handler.
	// We do not register this using restful Webservice since we do not want to surface this in api docs.
	// Allow GenericAPIServer to be embedded in contexts which already have something registered at the root
	if c.EnableIndex {
		routes.Index{}.Install(s.Mux, s.HandlerContainer)
	}

	if c.EnableSwaggerSupport && c.EnableSwaggerUI {
		routes.SwaggerUI{}.Install(s.Mux, s.HandlerContainer)
	}
	if c.EnableProfiling {
		routes.Profiling{}.Install(s.Mux, s.HandlerContainer)
	}
	if c.EnableVersion {
		routes.Version{}.Install(s.Mux, s.HandlerContainer)
	}

	handler := http.Handler(s.Mux.BaseMux().(*http.ServeMux))

	// TODO: handle CORS and auth using go-restful
	// See github.com/emicklei/go-restful/blob/master/examples/restful-CORS-filter.go, and
	// github.com/emicklei/go-restful/blob/master/examples/restful-basic-authentication.go

	if len(c.CorsAllowedOriginList) > 0 {
		allowedOriginRegexps, err := util.CompileRegexps(c.CorsAllowedOriginList)
		if err != nil {
			glog.Fatalf("Invalid CORS allowed origin, --cors-allowed-origins flag was set to %v - %v", strings.Join(c.CorsAllowedOriginList, ","), err)
		}
		handler = apiserver.CORS(handler, allowedOriginRegexps, nil, nil, "true")
	}

	s.InsecureHandler = handler

	attributeGetter := apiserver.NewRequestAttributeGetter(c.RequestContextMapper, s.NewRequestInfoResolver())
	handler = apiserver.WithAuthorizationCheck(handler, attributeGetter, c.Authorizer)
	if len(c.AuditLogPath) != 0 {
		// audit handler must comes before the impersonationFilter to read the original user
		writer := &lumberjack.Logger{
			Filename:   c.AuditLogPath,
			MaxAge:     c.AuditLogMaxAge,
			MaxBackups: c.AuditLogMaxBackups,
			MaxSize:    c.AuditLogMaxSize,
		}
		handler = audit.WithAudit(handler, attributeGetter, writer)
		defer writer.Close()
	}
	handler = apiserver.WithImpersonation(handler, c.RequestContextMapper, c.Authorizer)

	// Install Authenticator
	if c.Authenticator != nil {
		authenticatedHandler, err := handlers.NewRequestAuthenticator(c.RequestContextMapper, c.Authenticator, handlers.Unauthorized(c.SupportsBasicAuth), handler)
		if err != nil {
			glog.Fatalf("Could not initialize authenticator: %v", err)
		}
		handler = authenticatedHandler
	}

	// TODO: Make this optional?  Consumers of GenericAPIServer depend on this currently.
	s.Handler = handler

	// After all wrapping is done, put a context filter around both handlers
	var err error
	handler, err = api.NewRequestContextFilter(c.RequestContextMapper, s.Handler)
	if err != nil {
		glog.Fatalf("Could not initialize request context filter for s.Handler: %v", err)
	}
	s.Handler = handler

	handler, err = api.NewRequestContextFilter(c.RequestContextMapper, s.InsecureHandler)
	if err != nil {
		glog.Fatalf("Could not initialize request context filter for s.InsecureHandler: %v", err)
	}
	s.InsecureHandler = handler

	s.installGroupsDiscoveryHandler()

	return s, nil
}

func DefaultAndValidateRunOptions(options *options.ServerRunOptions) {
	genericvalidation.ValidateRunOptions(options)

	// If advertise-address is not specified, use bind-address. If bind-address
	// is not usable (unset, 0.0.0.0, or loopback), we will use the host's default
	// interface as valid public addr for master (see: util/net#ValidPublicAddrForMaster)
	if options.AdvertiseAddress == nil || options.AdvertiseAddress.IsUnspecified() {
		hostIP, err := utilnet.ChooseBindAddress(options.BindAddress)
		if err != nil {
			glog.Fatalf("Unable to find suitable network address.error='%v' . "+
				"Try to set the AdvertiseAddress directly or provide a valid BindAddress to fix this.", err)
		}
		options.AdvertiseAddress = hostIP
	}
	glog.Infof("Will report %v as public IP address.", options.AdvertiseAddress)

	// Set default value for ExternalHost if not specified.
	if len(options.ExternalHost) == 0 {
		// TODO: extend for other providers
		if options.CloudProvider == "gce" {
			cloud, err := cloudprovider.InitCloudProvider(options.CloudProvider, options.CloudConfigFile)
			if err != nil {
				glog.Fatalf("Cloud provider could not be initialized: %v", err)
			}
			instances, supported := cloud.Instances()
			if !supported {
				glog.Fatalf("GCE cloud provider has no instances.  this shouldn't happen. exiting.")
			}
			name, err := os.Hostname()
			if err != nil {
				glog.Fatalf("Failed to get hostname: %v", err)
			}
			addrs, err := instances.NodeAddresses(name)
			if err != nil {
				glog.Warningf("Unable to obtain external host address from cloud provider: %v", err)
			} else {
				for _, addr := range addrs {
					if addr.Type == api.NodeExternalIP {
						options.ExternalHost = addr.Address
					}
				}
			}
		}
	}
}
