/*
Copyright 2014 The Kubernetes Authors.

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
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	systemd "github.com/coreos/go-systemd/daemon"
	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"
	"gopkg.in/natefinch/lumberjack.v2"

	"github.com/go-openapi/spec"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/apiserver/audit"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/handlers"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/genericapiserver/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver/options"
	genericvalidation "k8s.io/kubernetes/pkg/genericapiserver/validation"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	ipallocator "k8s.io/kubernetes/pkg/registry/service/ipallocator"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/ui"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/async"
	"k8s.io/kubernetes/pkg/util/crypto"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

const globalTimeout = time.Minute

// Info about an API group.
type APIGroupInfo struct {
	GroupMeta apimachinery.GroupMeta
	// Info about the resources in this group. Its a map from version to resource to the storage.
	VersionedResourcesStorageMap map[string]map[string]rest.Storage
	// True, if this is the legacy group ("/v1").
	IsLegacyGroup bool
	// OptionsExternalVersion controls the APIVersion used for common objects in the
	// schema like api.Status, api.DeleteOptions, and api.ListOptions. Other implementors may
	// define a version "v1beta1" but want to use the Kubernetes "v1" internal objects.
	// If nil, defaults to groupMeta.GroupVersion.
	// TODO: Remove this when https://github.com/kubernetes/kubernetes/issues/19018 is fixed.
	OptionsExternalVersion *unversioned.GroupVersion

	// Scheme includes all of the types used by this group and how to convert between them (or
	// to convert objects from outside of this group that are accepted in this API).
	// TODO: replace with interfaces
	Scheme *runtime.Scheme
	// NegotiatedSerializer controls how this group encodes and decodes data
	NegotiatedSerializer runtime.NegotiatedSerializer
	// ParameterCodec performs conversions for query parameters passed to API calls
	ParameterCodec runtime.ParameterCodec

	// SubresourceGroupVersionKind contains the GroupVersionKind overrides for each subresource that is
	// accessible from this API group version. The GroupVersionKind is that of the external version of
	// the subresource. The key of this map should be the path of the subresource. The keys here should
	// match the keys in the Storage map above for subresources.
	SubresourceGroupVersionKind map[string]unversioned.GroupVersionKind
}

// Config is a structure used to configure a GenericAPIServer.
type Config struct {
	// The storage factory for other objects
	StorageFactory     StorageFactory
	AuditLogPath       string
	AuditLogMaxAge     int
	AuditLogMaxBackups int
	AuditLogMaxSize    int
	// allow downstream consumers to disable the core controller loops
	EnableLogsSupport bool
	EnableUISupport   bool
	// Allow downstream consumers to disable swagger.
	// This includes returning the generated swagger spec at /swaggerapi and swagger ui at /swagger-ui.
	EnableSwaggerSupport bool
	// Allow downstream consumers to disable swagger ui.
	// Note that this is ignored if either EnableSwaggerSupport or EnableUISupport is false.
	EnableSwaggerUI bool
	// Allows api group versions or specific resources to be conditionally enabled/disabled.
	APIResourceConfigSource APIResourceConfigSource
	// allow downstream consumers to disable the index route
	EnableIndex           bool
	EnableProfiling       bool
	EnableWatchCache      bool
	APIPrefix             string
	APIGroupPrefix        string
	CorsAllowedOriginList []string
	Authenticator         authenticator.Request
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

// GenericAPIServer contains state for a Kubernetes cluster api server.
type GenericAPIServer struct {
	// "Inputs", Copied from Config
	ServiceClusterIPRange *net.IPNet
	ServiceNodePortRange  utilnet.PortRange
	cacheTimeout          time.Duration
	MinRequestTimeout     time.Duration

	mux                   apiserver.Mux
	MuxHelper             *apiserver.MuxHelper
	HandlerContainer      *restful.Container
	RootWebService        *restful.WebService
	enableLogsSupport     bool
	enableUISupport       bool
	enableSwaggerSupport  bool
	enableSwaggerUI       bool
	enableProfiling       bool
	enableWatchCache      bool
	APIPrefix             string
	APIGroupPrefix        string
	corsAllowedOriginList []string
	authenticator         authenticator.Request
	authorizer            authorizer.Authorizer
	AdmissionControl      admission.Interface
	MasterCount           int
	RequestContextMapper  api.RequestContextMapper

	// ExternalAddress is the address (hostname or IP and port) that should be used in
	// external (public internet) URLs for this GenericAPIServer.
	ExternalAddress string
	// ClusterIP is the IP address of the GenericAPIServer within the cluster.
	ClusterIP            net.IP
	PublicReadWritePort  int
	ServiceReadWriteIP   net.IP
	ServiceReadWritePort int
	masterServices       *async.Runner
	ExtraServicePorts    []api.ServicePort
	ExtraEndpointPorts   []api.EndpointPort

	// storage contains the RESTful endpoints exposed by this GenericAPIServer
	storage map[string]rest.Storage

	// Serializer controls how common API objects not in a group/version prefix are serialized for this server.
	// Individual APIGroups may define their own serializers.
	Serializer runtime.NegotiatedSerializer

	// "Outputs"
	Handler         http.Handler
	InsecureHandler http.Handler

	// Used for custom proxy dialing, and proxy TLS options
	ProxyTransport http.RoundTripper

	KubernetesServiceNodePort int

	// Map storing information about all groups to be exposed in discovery response.
	// The map is from name to the group.
	apiGroupsForDiscovery map[string]unversioned.APIGroup

	// See Config.$name for documentation of these flags

	enableOpenAPISupport   bool
	openAPIInfo            spec.Info
	openAPIDefaultResponse spec.Response
}

func (s *GenericAPIServer) StorageDecorator() generic.StorageDecorator {
	if s.enableWatchCache {
		return registry.StorageWithCacher
	}
	return generic.UndecoratedStorage
}

// setDefaults fills in any fields not set that are required to have valid data.
func setDefaults(c *Config) {
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
func New(c *Config) (*GenericAPIServer, error) {
	if c.Serializer == nil {
		return nil, fmt.Errorf("Genericapiserver.New() called with config.Serializer == nil")
	}
	setDefaults(c)

	s := &GenericAPIServer{
		ServiceClusterIPRange: c.ServiceClusterIPRange,
		ServiceNodePortRange:  c.ServiceNodePortRange,
		RootWebService:        new(restful.WebService),
		enableLogsSupport:     c.EnableLogsSupport,
		enableUISupport:       c.EnableUISupport,
		enableSwaggerSupport:  c.EnableSwaggerSupport,
		enableSwaggerUI:       c.EnableSwaggerUI,
		enableProfiling:       c.EnableProfiling,
		enableWatchCache:      c.EnableWatchCache,
		APIPrefix:             c.APIPrefix,
		APIGroupPrefix:        c.APIGroupPrefix,
		corsAllowedOriginList: c.CorsAllowedOriginList,
		authenticator:         c.Authenticator,
		authorizer:            c.Authorizer,
		AdmissionControl:      c.AdmissionControl,
		RequestContextMapper:  c.RequestContextMapper,
		Serializer:            c.Serializer,

		cacheTimeout:      c.CacheTimeout,
		MinRequestTimeout: time.Duration(c.MinRequestTimeout) * time.Second,

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

	if c.RestfulContainer != nil {
		s.mux = c.RestfulContainer.ServeMux
		s.HandlerContainer = c.RestfulContainer
	} else {
		mux := http.NewServeMux()
		s.mux = mux
		s.HandlerContainer = NewHandlerContainer(mux, c.Serializer)
	}
	// Use CurlyRouter to be able to use regular expressions in paths. Regular expressions are required in paths for example for proxy (where the path is proxy/{kind}/{name}/{*})
	s.HandlerContainer.Router(restful.CurlyRouter{})
	s.MuxHelper = &apiserver.MuxHelper{Mux: s.mux, RegisteredPaths: []string{}}

	s.init(c)

	return s, nil
}

func (s *GenericAPIServer) NewRequestInfoResolver() *apiserver.RequestInfoResolver {
	return &apiserver.RequestInfoResolver{
		APIPrefixes:          sets.NewString(strings.Trim(s.APIPrefix, "/"), strings.Trim(s.APIGroupPrefix, "/")), // all possible API prefixes
		GrouplessAPIPrefixes: sets.NewString(strings.Trim(s.APIPrefix, "/")),                                      // APIPrefixes that won't have groups (legacy)
	}
}

// HandleWithAuth adds an http.Handler for pattern to an http.ServeMux
// Applies the same authentication and authorization (if any is configured)
// to the request is used for the GenericAPIServer's built-in endpoints.
func (s *GenericAPIServer) HandleWithAuth(pattern string, handler http.Handler) {
	// TODO: Add a way for plugged-in endpoints to translate their
	// URLs into attributes that an Authorizer can understand, and have
	// sensible policy defaults for plugged-in endpoints.  This will be different
	// for generic endpoints versus REST object endpoints.
	// TODO: convert to go-restful
	s.MuxHelper.Handle(pattern, handler)
}

// HandleFuncWithAuth adds an http.Handler for pattern to an http.ServeMux
// Applies the same authentication and authorization (if any is configured)
// to the request is used for the GenericAPIServer's built-in endpoints.
func (s *GenericAPIServer) HandleFuncWithAuth(pattern string, handler func(http.ResponseWriter, *http.Request)) {
	// TODO: convert to go-restful
	s.MuxHelper.HandleFunc(pattern, handler)
}

func NewHandlerContainer(mux *http.ServeMux, s runtime.NegotiatedSerializer) *restful.Container {
	container := restful.NewContainer()
	container.ServeMux = mux
	apiserver.InstallRecoverHandler(s, container)
	return container
}

// init initializes GenericAPIServer.
func (s *GenericAPIServer) init(c *Config) {

	if c.ProxyDialer != nil || c.ProxyTLSClientConfig != nil {
		s.ProxyTransport = utilnet.SetTransportDefaults(&http.Transport{
			Dial:            c.ProxyDialer,
			TLSClientConfig: c.ProxyTLSClientConfig,
		})
	}

	// Register root handler.
	// We do not register this using restful Webservice since we do not want to surface this in api docs.
	// Allow GenericAPIServer to be embedded in contexts which already have something registered at the root
	if c.EnableIndex {
		s.mux.HandleFunc("/", apiserver.IndexHandler(s.HandlerContainer, s.MuxHelper))
	}

	if c.EnableLogsSupport {
		apiserver.InstallLogsSupport(s.MuxHelper, s.HandlerContainer)
	}
	if c.EnableUISupport {
		ui.InstallSupport(s.MuxHelper, s.enableSwaggerSupport && s.enableSwaggerUI)
	}

	if c.EnableProfiling {
		s.mux.HandleFunc("/debug/pprof/", pprof.Index)
		s.mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		s.mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	}

	apiserver.InstallVersionHandler(s.MuxHelper, s.HandlerContainer)

	handler := http.Handler(s.mux.(*http.ServeMux))

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

	attributeGetter := apiserver.NewRequestAttributeGetter(s.RequestContextMapper, s.NewRequestInfoResolver())
	handler = apiserver.WithAuthorizationCheck(handler, attributeGetter, s.authorizer)
	if len(c.AuditLogPath) != 0 {
		// audit handler must comes before the impersonationFilter to read the original user
		writer := &lumberjack.Logger{
			Filename:   c.AuditLogPath,
			MaxAge:     c.AuditLogMaxAge,
			MaxBackups: c.AuditLogMaxBackups,
			MaxSize:    c.AuditLogMaxSize,
		}
		handler = audit.WithAudit(handler, s.RequestContextMapper, writer)
		defer writer.Close()
	}
	handler = apiserver.WithImpersonation(handler, s.RequestContextMapper, s.authorizer)

	// Install Authenticator
	if c.Authenticator != nil {
		authenticatedHandler, err := handlers.NewRequestAuthenticator(s.RequestContextMapper, c.Authenticator, handlers.Unauthorized(c.SupportsBasicAuth), handler)
		if err != nil {
			glog.Fatalf("Could not initialize authenticator: %v", err)
		}
		handler = authenticatedHandler
	}

	// TODO: Make this optional?  Consumers of GenericAPIServer depend on this currently.
	s.Handler = handler

	// After all wrapping is done, put a context filter around both handlers
	var err error
	handler, err = api.NewRequestContextFilter(s.RequestContextMapper, s.Handler)
	if err != nil {
		glog.Fatalf("Could not initialize request context filter for s.Handler: %v", err)
	}
	s.Handler = handler

	handler, err = api.NewRequestContextFilter(s.RequestContextMapper, s.InsecureHandler)
	if err != nil {
		glog.Fatalf("Could not initialize request context filter for s.InsecureHandler: %v", err)
	}
	s.InsecureHandler = handler

	s.installGroupsDiscoveryHandler()
}

// Exposes the given group versions in API. Helper method to install multiple group versions at once.
func (s *GenericAPIServer) InstallAPIGroups(groupsInfo []APIGroupInfo) error {
	for _, apiGroupInfo := range groupsInfo {
		if err := s.InstallAPIGroup(&apiGroupInfo); err != nil {
			return err
		}
	}
	return nil
}

// Installs handler at /apis to list all group versions for discovery
func (s *GenericAPIServer) installGroupsDiscoveryHandler() {
	apiserver.AddApisWebService(s.Serializer, s.HandlerContainer, s.APIGroupPrefix, func(req *restful.Request) []unversioned.APIGroup {
		// Return the list of supported groups in sorted order (to have a deterministic order).
		groups := []unversioned.APIGroup{}
		groupNames := make([]string, len(s.apiGroupsForDiscovery))
		var i int = 0
		for groupName := range s.apiGroupsForDiscovery {
			groupNames[i] = groupName
			i++
		}
		sort.Strings(groupNames)
		for _, groupName := range groupNames {
			apiGroup := s.apiGroupsForDiscovery[groupName]
			// Add ServerAddressByClientCIDRs.
			apiGroup.ServerAddressByClientCIDRs = s.getServerAddressByClientCIDRs(req.Request)
			groups = append(groups, apiGroup)
		}
		return groups
	})
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
		EnableIndex:               true,
		EnableLogsSupport:         options.EnableLogsSupport,
		EnableProfiling:           options.EnableProfiling,
		EnableSwaggerSupport:      true,
		EnableSwaggerUI:           options.EnableSwaggerUI,
		EnableUISupport:           true,
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

func (s *GenericAPIServer) Run(options *options.ServerRunOptions) {
	if s.enableSwaggerSupport {
		s.InstallSwaggerAPI()
	}
	if s.enableOpenAPISupport {
		s.InstallOpenAPI()
	}
	// We serve on 2 ports. See docs/admin/accessing-the-api.md
	secureLocation := ""
	if options.SecurePort != 0 {
		secureLocation = net.JoinHostPort(options.BindAddress.String(), strconv.Itoa(options.SecurePort))
	}
	insecureLocation := net.JoinHostPort(options.InsecureBindAddress.String(), strconv.Itoa(options.InsecurePort))

	var sem chan bool
	if options.MaxRequestsInFlight > 0 {
		sem = make(chan bool, options.MaxRequestsInFlight)
	}

	longRunningRE := regexp.MustCompile(options.LongRunningRequestRE)
	longRunningRequestCheck := apiserver.BasicLongRunningRequestCheck(longRunningRE, map[string]string{"watch": "true"})
	longRunningTimeout := func(req *http.Request) (<-chan time.Time, string) {
		// TODO unify this with apiserver.MaxInFlightLimit
		if longRunningRequestCheck(req) {
			return nil, ""
		}
		return time.After(globalTimeout), ""
	}

	if secureLocation != "" {
		handler := apiserver.TimeoutHandler(apiserver.RecoverPanics(s.Handler), longRunningTimeout)
		secureServer := &http.Server{
			Addr:           secureLocation,
			Handler:        apiserver.MaxInFlightLimit(sem, longRunningRequestCheck, handler),
			MaxHeaderBytes: 1 << 20,
			TLSConfig: &tls.Config{
				// Can't use SSLv3 because of POODLE and BEAST
				// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
				// Can't use TLSv1.1 because of RC4 cipher usage
				MinVersion: tls.VersionTLS12,
			},
		}

		if len(options.ClientCAFile) > 0 {
			clientCAs, err := crypto.CertPoolFromFile(options.ClientCAFile)
			if err != nil {
				glog.Fatalf("Unable to load client CA file: %v", err)
			}
			// Populate PeerCertificates in requests, but don't reject connections without certificates
			// This allows certificates to be validated by authenticators, while still allowing other auth types
			secureServer.TLSConfig.ClientAuth = tls.RequestClientCert
			// Specify allowed CAs for client certificates
			secureServer.TLSConfig.ClientCAs = clientCAs
		}

		glog.Infof("Serving securely on %s", secureLocation)
		if options.TLSCertFile == "" && options.TLSPrivateKeyFile == "" {
			options.TLSCertFile = path.Join(options.CertDirectory, "apiserver.crt")
			options.TLSPrivateKeyFile = path.Join(options.CertDirectory, "apiserver.key")
			// TODO (cjcullen): Is ClusterIP the right address to sign a cert with?
			alternateIPs := []net.IP{s.ServiceReadWriteIP}
			alternateDNS := []string{"kubernetes.default.svc", "kubernetes.default", "kubernetes"}
			// It would be nice to set a fqdn subject alt name, but only the kubelets know, the apiserver is clueless
			// alternateDNS = append(alternateDNS, "kubernetes.default.svc.CLUSTER.DNS.NAME")
			if !crypto.FoundCertOrKey(options.TLSCertFile, options.TLSPrivateKeyFile) {
				if err := crypto.GenerateSelfSignedCert(s.ClusterIP.String(), options.TLSCertFile, options.TLSPrivateKeyFile, alternateIPs, alternateDNS); err != nil {
					glog.Errorf("Unable to generate self signed cert: %v", err)
				} else {
					glog.Infof("Using self-signed cert (%s, %s)", options.TLSCertFile, options.TLSPrivateKeyFile)
				}
			}
		}

		go func() {
			defer utilruntime.HandleCrash()
			for {
				// err == systemd.SdNotifyNoSocket when not running on a systemd system
				if err := systemd.SdNotify("READY=1\n"); err != nil && err != systemd.SdNotifyNoSocket {
					glog.Errorf("Unable to send systemd daemon successful start message: %v\n", err)
				}
				if err := secureServer.ListenAndServeTLS(options.TLSCertFile, options.TLSPrivateKeyFile); err != nil {
					glog.Errorf("Unable to listen for secure (%v); will try again.", err)
				}
				time.Sleep(15 * time.Second)
			}
		}()
	} else {
		// err == systemd.SdNotifyNoSocket when not running on a systemd system
		if err := systemd.SdNotify("READY=1\n"); err != nil && err != systemd.SdNotifyNoSocket {
			glog.Errorf("Unable to send systemd daemon successful start message: %v\n", err)
		}
	}

	handler := apiserver.TimeoutHandler(apiserver.RecoverPanics(s.InsecureHandler), longRunningTimeout)
	http := &http.Server{
		Addr:           insecureLocation,
		Handler:        handler,
		MaxHeaderBytes: 1 << 20,
	}

	glog.Infof("Serving insecurely on %s", insecureLocation)
	go func() {
		defer utilruntime.HandleCrash()
		for {
			if err := http.ListenAndServe(); err != nil {
				glog.Errorf("Unable to listen for insecure (%v); will try again.", err)
			}
			time.Sleep(15 * time.Second)
		}
	}()
	select {}
}

// Exposes the given group version in API.
func (s *GenericAPIServer) InstallAPIGroup(apiGroupInfo *APIGroupInfo) error {
	apiPrefix := s.APIGroupPrefix
	if apiGroupInfo.IsLegacyGroup {
		apiPrefix = s.APIPrefix
	}

	// Install REST handlers for all the versions in this group.
	apiVersions := []string{}
	for _, groupVersion := range apiGroupInfo.GroupMeta.GroupVersions {
		apiVersions = append(apiVersions, groupVersion.Version)

		apiGroupVersion, err := s.getAPIGroupVersion(apiGroupInfo, groupVersion, apiPrefix)
		if err != nil {
			return err
		}
		if apiGroupInfo.OptionsExternalVersion != nil {
			apiGroupVersion.OptionsExternalVersion = apiGroupInfo.OptionsExternalVersion
		}

		if err := apiGroupVersion.InstallREST(s.HandlerContainer); err != nil {
			return fmt.Errorf("Unable to setup API %v: %v", apiGroupInfo, err)
		}
	}
	// Install the version handler.
	if apiGroupInfo.IsLegacyGroup {
		// Add a handler at /api to enumerate the supported api versions.
		apiserver.AddApiWebService(s.Serializer, s.HandlerContainer, apiPrefix, func(req *restful.Request) *unversioned.APIVersions {
			apiVersionsForDiscovery := unversioned.APIVersions{
				ServerAddressByClientCIDRs: s.getServerAddressByClientCIDRs(req.Request),
				Versions:                   apiVersions,
			}
			return &apiVersionsForDiscovery
		})
	} else {
		// Do not register empty group or empty version.  Doing so claims /apis/ for the wrong entity to be returned.
		// Catching these here places the error  much closer to its origin
		if len(apiGroupInfo.GroupMeta.GroupVersion.Group) == 0 {
			return fmt.Errorf("cannot register handler with an empty group for %#v", *apiGroupInfo)
		}
		if len(apiGroupInfo.GroupMeta.GroupVersion.Version) == 0 {
			return fmt.Errorf("cannot register handler with an empty version for %#v", *apiGroupInfo)
		}

		// Add a handler at /apis/<groupName> to enumerate all versions supported by this group.
		apiVersionsForDiscovery := []unversioned.GroupVersionForDiscovery{}
		for _, groupVersion := range apiGroupInfo.GroupMeta.GroupVersions {
			apiVersionsForDiscovery = append(apiVersionsForDiscovery, unversioned.GroupVersionForDiscovery{
				GroupVersion: groupVersion.String(),
				Version:      groupVersion.Version,
			})
		}
		preferedVersionForDiscovery := unversioned.GroupVersionForDiscovery{
			GroupVersion: apiGroupInfo.GroupMeta.GroupVersion.String(),
			Version:      apiGroupInfo.GroupMeta.GroupVersion.Version,
		}
		apiGroup := unversioned.APIGroup{
			Name:             apiGroupInfo.GroupMeta.GroupVersion.Group,
			Versions:         apiVersionsForDiscovery,
			PreferredVersion: preferedVersionForDiscovery,
		}
		s.AddAPIGroupForDiscovery(apiGroup)
		apiserver.AddGroupWebService(s.Serializer, s.HandlerContainer, apiPrefix+"/"+apiGroup.Name, apiGroup)
	}
	apiserver.InstallServiceErrorHandler(s.Serializer, s.HandlerContainer, s.NewRequestInfoResolver(), apiVersions)
	return nil
}

func (s *GenericAPIServer) AddAPIGroupForDiscovery(apiGroup unversioned.APIGroup) {
	s.apiGroupsForDiscovery[apiGroup.Name] = apiGroup
}

func (s *GenericAPIServer) RemoveAPIGroupForDiscovery(groupName string) {
	delete(s.apiGroupsForDiscovery, groupName)
}

func (s *GenericAPIServer) getServerAddressByClientCIDRs(req *http.Request) []unversioned.ServerAddressByClientCIDR {
	addressCIDRMap := []unversioned.ServerAddressByClientCIDR{
		{
			ClientCIDR: "0.0.0.0/0",

			ServerAddress: s.ExternalAddress,
		},
	}

	// Add internal CIDR if the request came from internal IP.
	clientIP := utilnet.GetClientIP(req)
	clusterCIDR := s.ServiceClusterIPRange
	if clusterCIDR.Contains(clientIP) {
		addressCIDRMap = append(addressCIDRMap, unversioned.ServerAddressByClientCIDR{
			ClientCIDR:    clusterCIDR.String(),
			ServerAddress: net.JoinHostPort(s.ServiceReadWriteIP.String(), strconv.Itoa(s.ServiceReadWritePort)),
		})
	}
	return addressCIDRMap
}

func (s *GenericAPIServer) getAPIGroupVersion(apiGroupInfo *APIGroupInfo, groupVersion unversioned.GroupVersion, apiPrefix string) (*apiserver.APIGroupVersion, error) {
	storage := make(map[string]rest.Storage)
	for k, v := range apiGroupInfo.VersionedResourcesStorageMap[groupVersion.Version] {
		storage[strings.ToLower(k)] = v
	}
	version, err := s.newAPIGroupVersion(apiGroupInfo, groupVersion)
	version.Root = apiPrefix
	version.Storage = storage
	return version, err
}

func (s *GenericAPIServer) newAPIGroupVersion(apiGroupInfo *APIGroupInfo, groupVersion unversioned.GroupVersion) (*apiserver.APIGroupVersion, error) {
	return &apiserver.APIGroupVersion{
		RequestInfoResolver: s.NewRequestInfoResolver(),

		GroupVersion: groupVersion,

		ParameterCodec: apiGroupInfo.ParameterCodec,
		Serializer:     apiGroupInfo.NegotiatedSerializer,
		Creater:        apiGroupInfo.Scheme,
		Convertor:      apiGroupInfo.Scheme,
		Copier:         apiGroupInfo.Scheme,
		Typer:          apiGroupInfo.Scheme,
		SubresourceGroupVersionKind: apiGroupInfo.SubresourceGroupVersionKind,
		Linker: apiGroupInfo.GroupMeta.SelfLinker,
		Mapper: apiGroupInfo.GroupMeta.RESTMapper,

		Admit:             s.AdmissionControl,
		Context:           s.RequestContextMapper,
		MinRequestTimeout: s.MinRequestTimeout,
	}, nil
}

// getSwaggerConfig returns swagger config shared between SwaggerAPI and OpenAPI spec generators
func (s *GenericAPIServer) getSwaggerConfig() *swagger.Config {
	hostAndPort := s.ExternalAddress
	protocol := "https://"
	webServicesUrl := protocol + hostAndPort
	return &swagger.Config{
		WebServicesUrl:  webServicesUrl,
		WebServices:     s.HandlerContainer.RegisteredWebServices(),
		ApiPath:         "/swaggerapi/",
		SwaggerPath:     "/swaggerui/",
		SwaggerFilePath: "/swagger-ui/",
		SchemaFormatHandler: func(typeName string) string {
			switch typeName {
			case "unversioned.Time", "*unversioned.Time":
				return "date-time"
			}
			return ""
		},
	}
}

// InstallSwaggerAPI installs the /swaggerapi/ endpoint to allow schema discovery
// and traversal. It is optional to allow consumers of the Kubernetes GenericAPIServer to
// register their own web services into the Kubernetes mux prior to initialization
// of swagger, so that other resource types show up in the documentation.
func (s *GenericAPIServer) InstallSwaggerAPI() {

	// Enable swagger UI and discovery API
	swagger.RegisterSwaggerService(*s.getSwaggerConfig(), s.HandlerContainer)
}

// InstallOpenAPI installs the /swagger.json endpoint to allow new OpenAPI schema discovery.
func (s *GenericAPIServer) InstallOpenAPI() {
	openAPIConfig := openapi.Config{
		SwaggerConfig:   s.getSwaggerConfig(),
		IgnorePrefixes:  []string{"/swaggerapi"},
		Info:            &s.openAPIInfo,
		DefaultResponse: &s.openAPIDefaultResponse,
	}
	err := openapi.RegisterOpenAPIService(&openAPIConfig, s.HandlerContainer)
	if err != nil {
		glog.Fatalf("Failed to generate open api spec: %v", err)
	}
}

// NewDefaultAPIGroupInfo returns an APIGroupInfo stubbed with "normal" values
// exposed for easier composition from other packages
func NewDefaultAPIGroupInfo(group string) APIGroupInfo {
	groupMeta := registered.GroupOrDie(group)

	return APIGroupInfo{
		GroupMeta:                    *groupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{},
		OptionsExternalVersion:       &registered.GroupOrDie(api.GroupName).GroupVersion,
		Scheme:                       api.Scheme,
		ParameterCodec:               api.ParameterCodec,
		NegotiatedSerializer:         api.Codecs,
	}
}
