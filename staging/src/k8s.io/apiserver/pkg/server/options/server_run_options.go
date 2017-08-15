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

package options

import (
	"fmt"
	"net"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"

	// add the generic feature gates
	_ "k8s.io/apiserver/pkg/features"

	"github.com/spf13/pflag"
)

// ServerRunOptions contains the options while running a generic api server.
type ServerRunOptions struct {
	AdvertiseAddress net.IP

	CorsAllowedOriginList       []string
	ExternalHost                string
	MaxRequestsInFlight         int
	MaxMutatingRequestsInFlight int
	MinRequestTimeout           int
	TargetRAMMB                 int
	WatchCacheSizes             []string
}

func NewServerRunOptions() *ServerRunOptions {
	defaults := server.NewConfig(serializer.CodecFactory{})
	return &ServerRunOptions{
		MaxRequestsInFlight:         defaults.MaxRequestsInFlight,
		MaxMutatingRequestsInFlight: defaults.MaxMutatingRequestsInFlight,
		MinRequestTimeout:           defaults.MinRequestTimeout,
	}
}

// ApplyOptions applies the run options to the method receiver and returns self
func (s *ServerRunOptions) ApplyTo(c *server.Config) error {
	c.CorsAllowedOriginList = s.CorsAllowedOriginList
	c.ExternalAddress = s.ExternalHost
	c.MaxRequestsInFlight = s.MaxRequestsInFlight
	c.MaxMutatingRequestsInFlight = s.MaxMutatingRequestsInFlight
	c.MinRequestTimeout = s.MinRequestTimeout
	c.PublicAddress = s.AdvertiseAddress

	return nil
}

// DefaultAdvertiseAddress sets the field AdvertiseAddress if unset. The field will be set based on the SecureServingOptions.
func (s *ServerRunOptions) DefaultAdvertiseAddress(secure *SecureServingOptions) error {
	if secure == nil {
		return nil
	}

	if s.AdvertiseAddress == nil || s.AdvertiseAddress.IsUnspecified() {
		hostIP, err := secure.DefaultExternalAddress()
		if err != nil {
			return fmt.Errorf("Unable to find suitable network address.error='%v'. "+
				"Try to set the AdvertiseAddress directly or provide a valid BindAddress to fix this.", err)
		}
		s.AdvertiseAddress = hostIP
	}

	return nil
}

// Validate checks validation of ServerRunOptions
func (s *ServerRunOptions) Validate() []error {
	errors := []error{}
	if s.TargetRAMMB < 0 {
		errors = append(errors, fmt.Errorf("--target-ram-mb can not be negative value"))
	}
	if s.MaxRequestsInFlight < 0 {
		errors = append(errors, fmt.Errorf("--max-requests-inflight can not be negative value"))
	}
	if s.MaxMutatingRequestsInFlight < 0 {
		errors = append(errors, fmt.Errorf("--min-request-timeout can not be negative value"))
	}

	return errors
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *ServerRunOptions) AddUniversalFlags(fs *pflag.FlagSet) {
	// Note: the weird ""+ in below lines seems to be the only way to get gofmt to
	// arrange these text blocks sensibly. Grrr.

	fs.IPVar(&s.AdvertiseAddress, "advertise-address", s.AdvertiseAddress, ""+
		"The IP address on which to advertise the apiserver to members of the cluster. This "+
		"address must be reachable by the rest of the cluster. If blank, the --bind-address "+
		"will be used. If --bind-address is unspecified, the host's default interface will "+
		"be used.")

	fs.StringSliceVar(&s.CorsAllowedOriginList, "cors-allowed-origins", s.CorsAllowedOriginList, ""+
		"List of allowed origins for CORS, comma separated.  An allowed origin can be a regular "+
		"expression to support subdomain matching. If this list is empty CORS will not be enabled.")

	fs.IntVar(&s.TargetRAMMB, "target-ram-mb", s.TargetRAMMB,
		"Memory limit for apiserver in MB (used to configure sizes of caches, etc.)")

	fs.StringVar(&s.ExternalHost, "external-hostname", s.ExternalHost,
		"The hostname to use when generating externalized URLs for this master (e.g. Swagger API Docs).")

	// TODO: remove post-1.6
	fs.String("long-running-request-regexp", "", ""+
		"A regular expression matching long running requests which should "+
		"be excluded from maximum inflight request handling.")
	fs.MarkDeprecated("long-running-request-regexp", "regular expression matching of long-running requests is no longer supported")

	deprecatedMasterServiceNamespace := metav1.NamespaceDefault
	fs.StringVar(&deprecatedMasterServiceNamespace, "master-service-namespace", deprecatedMasterServiceNamespace, ""+
		"DEPRECATED: the namespace from which the kubernetes master services should be injected into pods.")

	fs.IntVar(&s.MaxRequestsInFlight, "max-requests-inflight", s.MaxRequestsInFlight, ""+
		"The maximum number of non-mutating requests in flight at a given time. When the server exceeds this, "+
		"it rejects requests. Zero for no limit.")

	fs.IntVar(&s.MaxMutatingRequestsInFlight, "max-mutating-requests-inflight", s.MaxMutatingRequestsInFlight, ""+
		"The maximum number of mutating requests in flight at a given time. When the server exceeds this, "+
		"it rejects requests. Zero for no limit.")

	fs.IntVar(&s.MinRequestTimeout, "min-request-timeout", s.MinRequestTimeout, ""+
		"An optional field indicating the minimum number of seconds a handler must keep "+
		"a request open before timing it out. Currently only honored by the watch request "+
		"handler, which picks a randomized value above this number as the connection timeout, "+
		"to spread out load.")

	fs.StringSliceVar(&s.WatchCacheSizes, "watch-cache-sizes", s.WatchCacheSizes, ""+
		"List of watch cache sizes for every resource (pods, nodes, etc.), comma separated. "+
		"The individual override format: resource#size, where size is a number. It takes effect "+
		"when watch-cache is enabled.")

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}
