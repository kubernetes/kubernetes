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

// Package server does all of the work necessary to create a Kubernetes
// APIServer by binding together the API, master and APIServer infrastructure.
// It can be configured and called directly or via the hyperkube framework.
package server

import (
	"crypto/tls"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/hyperkube"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// APIServer runs a kubernetes api server.
type APIServer struct {
	WideOpenPort               int
	Address                    util.IP
	PublicAddressOverride      util.IP
	ReadOnlyPort               int
	APIRate                    float32
	APIBurst                   int
	SecurePort                 int
	TLSCertFile                string
	TLSPrivateKeyFile          string
	APIPrefix                  string
	StorageVersion             string
	CloudProvider              string
	CloudConfigFile            string
	EventTTL                   time.Duration
	TokenAuthFile              string
	AuthorizationMode          string
	AuthorizationPolicyFile    string
	AdmissionControl           string
	AdmissionControlConfigFile string
	EtcdServerList             util.StringList
	EtcdConfigFile             string
	CorsAllowedOriginList      util.StringList
	AllowPrivileged            bool
	PortalNet                  util.IPNet // TODO: make this a list
	EnableLogsSupport          bool
	MasterServiceNamespace     string
	RuntimeConfig              util.ConfigurationMap
	KubeletConfig              client.KubeletConfig
}

// NewAPIServer creates a new APIServer object with default parameters
func NewAPIServer() *APIServer {
	s := APIServer{
		WideOpenPort:           8080,
		Address:                util.IP(net.ParseIP("127.0.0.1")),
		PublicAddressOverride:  util.IP(net.ParseIP("")),
		ReadOnlyPort:           7080,
		APIRate:                10.0,
		APIBurst:               200,
		SecurePort:             6443,
		APIPrefix:              "/api",
		EventTTL:               48 * time.Hour,
		AuthorizationMode:      "AlwaysAllow",
		AdmissionControl:       "AlwaysAdmit",
		EnableLogsSupport:      true,
		MasterServiceNamespace: api.NamespaceDefault,

		RuntimeConfig: make(util.ConfigurationMap),
		KubeletConfig: client.KubeletConfig{
			Port:        10250,
			EnableHttps: false,
		},
	}

	return &s
}

// NewHyperkubeServer creates a new hyperkube Server object that includes the
// description and flags.
func NewHyperkubeServer() *hyperkube.Server {
	s := NewAPIServer()

	hks := hyperkube.Server{
		SimpleUsage: "apiserver",
		Long:        "The main API entrypoint and interface to the storage system.  The API server is also the focal point for all authorization decisions.",
		Run: func(_ *hyperkube.Server, args []string) error {
			return s.Run(args)
		},
	}
	s.AddFlags(hks.Flags())
	return &hks
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *APIServer) AddFlags(fs *pflag.FlagSet) {
	// Note: the weird ""+ in below lines seems to be the only way to get gofmt to
	// arrange these text blocks sensibly. Grrr.
	fs.IntVar(&s.WideOpenPort, "port", s.WideOpenPort, ""+
		"The port to listen on. Default 8080. It is assumed that firewall rules are "+
		"set up such that this port is not reachable from outside of the cluster. It is "+
		"further assumed that port 443 on the cluster's public address is proxied to this "+
		"port. This is performed by nginx in the default setup.")
	fs.Var(&s.Address, "address", "The IP address on to serve on (set to 0.0.0.0 for all interfaces)")
	fs.Var(&s.PublicAddressOverride, "public_address_override", "Public serving address."+
		"Read only port will be opened on this address, and it is assumed that port "+
		"443 at this address will be proxied/redirected to '-address':'-port'. If "+
		"blank, the address in the first listed interface will be used.")
	fs.IntVar(&s.ReadOnlyPort, "read_only_port", s.ReadOnlyPort, ""+
		"The port from which to serve read-only resources. If 0, don't serve on a "+
		"read-only address. It is assumed that firewall rules are set up such that "+
		"this port is not reachable from outside of the cluster.")
	fs.Float32Var(&s.APIRate, "api_rate", s.APIRate, "API rate limit as QPS for the read only port")
	fs.IntVar(&s.APIBurst, "api_burst", s.APIBurst, "API burst amount for the read only port")
	fs.IntVar(&s.SecurePort, "secure_port", s.SecurePort,
		"The port from which to serve HTTPS with authentication and authorization. If 0, don't serve HTTPS ")
	fs.StringVar(&s.TLSCertFile, "tls_cert_file", s.TLSCertFile, ""+
		"File containing x509 Certificate for HTTPS.  (CA cert, if any, concatenated after server cert). "+
		"If HTTPS serving is enabled, and --tls_cert_file and --tls_private_key_file are not provided, "+
		"a self-signed certificate and key are generated for the public address and saved to /var/run/kubernetes.")
	fs.StringVar(&s.TLSPrivateKeyFile, "tls_private_key_file", s.TLSPrivateKeyFile, "File containing x509 private key matching --tls_cert_file.")
	fs.StringVar(&s.APIPrefix, "api_prefix", s.APIPrefix, "The prefix for API requests on the server. Default '/api'.")
	fs.StringVar(&s.StorageVersion, "storage_version", s.StorageVersion, "The version to store resources with. Defaults to server preferred")
	fs.StringVar(&s.CloudProvider, "cloud_provider", s.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.CloudConfigFile, "cloud_config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.DurationVar(&s.EventTTL, "event_ttl", s.EventTTL, "Amount of time to retain events. Default 2 days.")
	fs.StringVar(&s.TokenAuthFile, "token_auth_file", s.TokenAuthFile, "If set, the file that will be used to secure the secure port of the API server via token authentication.")
	fs.StringVar(&s.AuthorizationMode, "authorization_mode", s.AuthorizationMode, "Selects how to do authorization on the secure port.  One of: "+strings.Join(apiserver.AuthorizationModeChoices, ","))
	fs.StringVar(&s.AuthorizationPolicyFile, "authorization_policy_file", s.AuthorizationPolicyFile, "File with authorization policy in csv format, used with --authorization_mode=ABAC, on the secure port.")
	fs.StringVar(&s.AdmissionControl, "admission_control", s.AdmissionControl, "Ordered list of plug-ins to do admission control of resources into cluster. Comma-delimited list of: "+strings.Join(admission.GetPlugins(), ", "))
	fs.StringVar(&s.AdmissionControlConfigFile, "admission_control_config_file", s.AdmissionControlConfigFile, "File with admission control configuration.")
	fs.Var(&s.EtcdServerList, "etcd_servers", "List of etcd servers to watch (http://ip:port), comma separated. Mutually exclusive with -etcd_config")
	fs.StringVar(&s.EtcdConfigFile, "etcd_config", s.EtcdConfigFile, "The config file for the etcd client. Mutually exclusive with -etcd_servers.")
	fs.Var(&s.CorsAllowedOriginList, "cors_allowed_origins", "List of allowed origins for CORS, comma separated.  An allowed origin can be a regular expression to support subdomain matching.  If this list is empty CORS will not be enabled.")
	fs.BoolVar(&s.AllowPrivileged, "allow_privileged", s.AllowPrivileged, "If true, allow privileged containers.")
	fs.Var(&s.PortalNet, "portal_net", "A CIDR notation IP range from which to assign portal IPs. This must not overlap with any IP ranges assigned to nodes for pods.")
	fs.StringVar(&s.MasterServiceNamespace, "master_service_namespace", s.MasterServiceNamespace, "The namespace from which the kubernetes master services should be injected into pods")
	fs.Var(&s.RuntimeConfig, "runtime_config", "A set of key=value pairs that describe runtime configuration that may be passed to the apiserver.")
	client.BindKubeletClientConfigFlags(fs, &s.KubeletConfig)
}

// TODO: Longer term we should read this from some config store, rather than a flag.
func (s *APIServer) verifyPortalFlags() {
	if s.PortalNet.IP == nil {
		glog.Fatal("No --portal_net specified")
	}
}

func newEtcd(etcdConfigFile string, etcdServerList util.StringList, storageVersion string) (helper tools.EtcdHelper, err error) {
	var client tools.EtcdGetSet
	if etcdConfigFile != "" {
		client, err = etcd.NewClientFromFile(etcdConfigFile)
		if err != nil {
			return helper, err
		}
	} else {
		client = etcd.NewClient(etcdServerList)
	}

	return master.NewEtcdHelper(client, storageVersion)
}

// Run runs the specified APIServer.  This should never exit.
func (s *APIServer) Run(_ []string) error {
	s.verifyPortalFlags()

	if (s.EtcdConfigFile != "" && len(s.EtcdServerList) != 0) || (s.EtcdConfigFile == "" && len(s.EtcdServerList) == 0) {
		glog.Fatalf("specify either --etcd_servers or --etcd_config")
	}

	capabilities.Initialize(capabilities.Capabilities{
		AllowPrivileged: s.AllowPrivileged,
	})

	cloud := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)

	kubeletClient, err := client.NewKubeletClient(&s.KubeletConfig)
	if err != nil {
		glog.Fatalf("Failure to start kubelet client: %v", err)
	}

	_, v1beta3 := s.RuntimeConfig["api/v1beta3"]

	// TODO: expose same flags as client.BindClientConfigFlags but for a server
	clientConfig := &client.Config{
		Host:    net.JoinHostPort(s.Address.String(), strconv.Itoa(s.WideOpenPort)),
		Version: s.StorageVersion,
	}
	client, err := client.New(clientConfig)
	if err != nil {
		glog.Fatalf("Invalid server address: %v", err)
	}

	helper, err := newEtcd(s.EtcdConfigFile, s.EtcdServerList, s.StorageVersion)
	if err != nil {
		glog.Fatalf("Invalid storage version or misconfigured etcd: %v", err)
	}

	n := net.IPNet(s.PortalNet)

	authenticator, err := apiserver.NewAuthenticatorFromTokenFile(s.TokenAuthFile)
	if err != nil {
		glog.Fatalf("Invalid Authentication Config: %v", err)
	}

	authorizer, err := apiserver.NewAuthorizerFromAuthorizationConfig(s.AuthorizationMode, s.AuthorizationPolicyFile)
	if err != nil {
		glog.Fatalf("Invalid Authorization Config: %v", err)
	}

	admissionControlPluginNames := strings.Split(s.AdmissionControl, ",")
	admissionController := admission.NewFromPlugins(client, admissionControlPluginNames, s.AdmissionControlConfigFile)

	config := &master.Config{
		Client:                 client,
		Cloud:                  cloud,
		EtcdHelper:             helper,
		EventTTL:               s.EventTTL,
		KubeletClient:          kubeletClient,
		PortalNet:              &n,
		EnableLogsSupport:      s.EnableLogsSupport,
		EnableUISupport:        true,
		EnableOAuthSupport:     true,
		EnableSwaggerSupport:   true,
		EnableIndex:            true,
		APIPrefix:              s.APIPrefix,
		CorsAllowedOriginList:  s.CorsAllowedOriginList,
		ReadOnlyPort:           s.ReadOnlyPort,
		ReadWritePort:          s.SecurePort,
		PublicAddress:          net.IP(s.PublicAddressOverride),
		Authenticator:          authenticator,
		Authorizer:             authorizer,
		AdmissionControl:       admissionController,
		EnableV1Beta3:          v1beta3,
		MasterServiceNamespace: s.MasterServiceNamespace,
	}
	m := master.New(config)

	// We serve on 3 ports.  See docs/accessing_the_api.md
	roLocation := ""
	if s.ReadOnlyPort != 0 {
		roLocation = net.JoinHostPort(config.PublicAddress.String(), strconv.Itoa(s.ReadOnlyPort))
	}
	secureLocation := ""
	if s.SecurePort != 0 {
		secureLocation = net.JoinHostPort(config.PublicAddress.String(), strconv.Itoa(s.SecurePort))
	}
	wideOpenLocation := net.JoinHostPort(s.Address.String(), strconv.Itoa(s.WideOpenPort))

	// See the flag commentary to understand our assumptions when opening the read-only and read-write ports.

	if roLocation != "" {
		// Default settings allow 1 read-only request per second, allow up to 20 in a burst before enforcing.
		rl := util.NewTokenBucketRateLimiter(s.APIRate, s.APIBurst)
		readOnlyServer := &http.Server{
			Addr:           roLocation,
			Handler:        apiserver.RecoverPanics(apiserver.ReadOnly(apiserver.RateLimit(rl, m.InsecureHandler))),
			ReadTimeout:    5 * time.Minute,
			WriteTimeout:   5 * time.Minute,
			MaxHeaderBytes: 1 << 20,
		}
		glog.Infof("Serving read-only insecurely on %s", roLocation)
		go func() {
			defer util.HandleCrash()
			for {
				if err := readOnlyServer.ListenAndServe(); err != nil {
					glog.Errorf("Unable to listen for read only traffic (%v); will try again.", err)
				}
				time.Sleep(15 * time.Second)
			}
		}()
	}

	if secureLocation != "" {
		secureServer := &http.Server{
			Addr:           secureLocation,
			Handler:        apiserver.RecoverPanics(m.Handler),
			ReadTimeout:    5 * time.Minute,
			WriteTimeout:   5 * time.Minute,
			MaxHeaderBytes: 1 << 20,
			TLSConfig: &tls.Config{
				// Change default from SSLv3 to TLSv1.0 (because of POODLE vulnerability)
				MinVersion: tls.VersionTLS10,
				// Populate PeerCertificates in requests, but don't reject connections without certificates
				// This allows certificates to be validated by authenticators, while still allowing other auth types
				ClientAuth: tls.RequestClientCert,
			},
		}
		glog.Infof("Serving securely on %s", secureLocation)
		go func() {
			defer util.HandleCrash()
			for {
				if s.TLSCertFile == "" && s.TLSPrivateKeyFile == "" {
					s.TLSCertFile = "/var/run/kubernetes/apiserver.crt"
					s.TLSPrivateKeyFile = "/var/run/kubernetes/apiserver.key"
					if err := util.GenerateSelfSignedCert(config.PublicAddress.String(), s.TLSCertFile, s.TLSPrivateKeyFile); err != nil {
						glog.Errorf("Unable to generate self signed cert: %v", err)
					} else {
						glog.Infof("Using self-signed cert (%s, %s)", s.TLSCertFile, s.TLSPrivateKeyFile)
					}
				}
				if err := secureServer.ListenAndServeTLS(s.TLSCertFile, s.TLSPrivateKeyFile); err != nil {
					glog.Errorf("Unable to listen for secure (%v); will try again.", err)
				}
				time.Sleep(15 * time.Second)
			}
		}()
	}

	http := &http.Server{
		Addr:           wideOpenLocation,
		Handler:        apiserver.RecoverPanics(m.InsecureHandler),
		ReadTimeout:    5 * time.Minute,
		WriteTimeout:   5 * time.Minute,
		MaxHeaderBytes: 1 << 20,
	}
	glog.Infof("Serving insecurely on %s", wideOpenLocation)
	glog.Fatal(http.ListenAndServe())
	return nil
}
