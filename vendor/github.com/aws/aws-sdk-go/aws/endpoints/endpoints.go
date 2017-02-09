package endpoints

import (
	"fmt"
	"regexp"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

// Options provide the configuration needed to direct how the
// endpoints will be resolved.
type Options struct {
	// DisableSSL forces the endpoint to be resolved as HTTP.
	// instead of HTTPS if the service supports it.
	DisableSSL bool

	// Sets the resolver to resolve the endpoint as a dualstack endpoint
	// for the service. If dualstack support for a service is not known and
	// StrictMatching is not enabled a dualstack endpoint for the service will
	// be returned. This endpoint may not be valid. If StrictMatching is
	// enabled only services that are known to support dualstack will return
	// dualstack endpoints.
	UseDualStack bool

	// Enables strict matching of services and regions resolved endpoints.
	// If the partition doesn't enumerate the exact service and region an
	// error will be returned. This option will prevent returning endpoints
	// that look valid, but may not resolve to any real endpoint.
	StrictMatching bool
}

// Set combines all of the option functions together.
func (o *Options) Set(optFns ...func(*Options)) {
	for _, fn := range optFns {
		fn(o)
	}
}

// DisableSSLOption sets the DisableSSL options. Can be used as a functional
// option when resolving endpoints.
func DisableSSLOption(o *Options) {
	o.DisableSSL = true
}

// UseDualStackOption sets the UseDualStack option. Can be used as a functional
// option when resolving endpoints.
func UseDualStackOption(o *Options) {
	o.UseDualStack = true
}

// StrictMatchingOption sets the StrictMatching option. Can be used as a functional
// option when resolving endpoints.
func StrictMatchingOption(o *Options) {
	o.StrictMatching = true
}

// A Resolver provides the interface for functionality to resolve endpoints.
// The build in Partition and DefaultResolver return value satisfy this interface.
type Resolver interface {
	EndpointFor(service, region string, opts ...func(*Options)) (ResolvedEndpoint, error)
}

// ResolverFunc is a helper utility that wraps a function so it satisfies the
// Resolver interface. This is useful when you want to add additional endpoint
// resolving logic, or stub out specific endpoints with custom values.
type ResolverFunc func(service, region string, opts ...func(*Options)) (ResolvedEndpoint, error)

// EndpointFor wraps the ResolverFunc function to satisfy the Resolver interface.
func (fn ResolverFunc) EndpointFor(service, region string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	return fn(service, region, opts...)
}

var schemeRE = regexp.MustCompile("^([^:]+)://")

// AddScheme adds the HTTP or HTTPS schemes to a endpoint URL if there is no
// scheme. If disableSSL is true HTTP will set HTTP instead of the default HTTPS.
//
// If disableSSL is set, it will only set the URL's scheme if the URL does not
// contain a scheme.
func AddScheme(endpoint string, disableSSL bool) string {
	if !schemeRE.MatchString(endpoint) {
		scheme := "https"
		if disableSSL {
			scheme = "http"
		}
		endpoint = fmt.Sprintf("%s://%s", scheme, endpoint)
	}

	return endpoint
}

// EnumPartitions a provides a way to retrieve the underlying partitions that
// make up the SDK's default Resolver, or any resolver decoded from a model
// file.
//
// Use this interface with DefaultResolver and DecodeModels to get the list of
// Partitions.
type EnumPartitions interface {
	Partitions() []Partition
}

// A Partition provides the ability to enumerate the partition's regions
// and services.
type Partition struct {
	id string
	p  *partition
}

// ID returns the identifier of the partition.
func (p *Partition) ID() string { return p.id }

// EndpointFor attempts to resolve the endpoint based on service and region.
// See Options for information on configuring how the endpoint is resolved.
//
// If the service cannot be found in the metadata the UnknownServiceError
// error will be returned. This validation will occur regardless if
// StrictMatching is enabled.
//
// When resolving endpoints you can choose to enable StrictMatching. This will
// require the provided service and region to be known by the partition.
// If the endpoint cannot be strictly resolved an error will be returned. This
// mode is useful to ensure the endpoint resolved is valid. Without
// StrictMatching enabled the enpoint returned my look valid but may not work.
// StrictMatching requires the SDK to be updated if you want to take advantage
// of new regions and services expantions.
//
// Errors that can be returned.
//   * UnknownServiceError
//   * UnknownEndpointError
func (p *Partition) EndpointFor(service, region string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	return p.p.EndpointFor(service, region, opts...)
}

// Regions returns a map of Regions indexed by their ID. This is useful for
// enumerating over the regions in a partition.
func (p *Partition) Regions() map[string]Region {
	rs := map[string]Region{}
	for id := range p.p.Regions {
		rs[id] = Region{
			id: id,
			p:  p.p,
		}
	}

	return rs
}

// Services returns a map of Service indexed by their ID. This is useful for
// enumerating over the services in a partition.
func (p *Partition) Services() map[string]Service {
	ss := map[string]Service{}
	for id := range p.p.Services {
		ss[id] = Service{
			id: id,
			p:  p.p,
		}
	}

	return ss
}

// A Region provides information about a region, and ability to resolve an
// endpoint from the context of a region, given a service.
type Region struct {
	id, desc string
	p        *partition
}

// ID returns the region's identifier.
func (r *Region) ID() string { return r.id }

// ResolveEndpoint resolves an endpoint from the context of the region given
// a service. See Partition.EndpointFor for usage and errors that can be returned.
func (r *Region) ResolveEndpoint(service string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	return r.p.EndpointFor(service, r.id, opts...)
}

// Services returns a list of all services that are known to be in this region.
func (r *Region) Services() map[string]Service {
	ss := map[string]Service{}
	for id, s := range r.p.Services {
		if _, ok := s.Endpoints[r.id]; ok {
			ss[id] = Service{
				id: id,
				p:  r.p,
			}
		}
	}

	return ss
}

// A Service provides information about a service, and ability to resolve an
// endpoint from the context of a service, given a region.
type Service struct {
	id string
	p  *partition
}

// ID returns the identifier for the service.
func (s *Service) ID() string { return s.id }

// ResolveEndpoint resolves an endpoint from the context of a service given
// a region. See Partition.EndpointFor for usage and errors that can be returned.
func (s *Service) ResolveEndpoint(region string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	return s.p.EndpointFor(s.id, region, opts...)
}

// Endpoints returns a map of Endpoints indexed by their ID for all known
// endpoints for a service.
func (s *Service) Endpoints() map[string]Endpoint {
	es := map[string]Endpoint{}
	for id := range s.p.Services[s.id].Endpoints {
		es[id] = Endpoint{
			id:        id,
			serviceID: s.id,
			p:         s.p,
		}
	}

	return es
}

// A Endpoint provides information about endpoints, and provides the ability
// to resolve that endpoint for the service, and the region the endpoint
// represents.
type Endpoint struct {
	id        string
	serviceID string
	p         *partition
}

// ID returns the identifier for an endpoint.
func (e *Endpoint) ID() string { return e.id }

// ServiceID returns the identifier the endpoint belongs to.
func (e *Endpoint) ServiceID() string { return e.serviceID }

// ResolveEndpoint resolves an endpoint from the context of a service and
// region the endpoint represents. See Partition.EndpointFor for usage and
// errors that can be returned.
func (e *Endpoint) ResolveEndpoint(opts ...func(*Options)) (ResolvedEndpoint, error) {
	return e.p.EndpointFor(e.serviceID, e.id, opts...)
}

// A ResolvedEndpoint is an endpoint that has been resolved based on a partition
// service, and region.
type ResolvedEndpoint struct {
	// The endpoint URL
	URL string

	// The region that should be used for signing requests.
	SigningRegion string

	// The service name that should be used for signing requests.
	SigningName string

	// The signing method that should be used for signing requests.
	SigningMethod string
}

// So that the Error interface type can be included as an anonymous field
// in the requestError struct and not conflict with the error.Error() method.
type awsError awserr.Error

// A EndpointNotFoundError is returned when in StrictMatching mode, and the
// endpoint for the service and region cannot be found in any of the partitions.
type EndpointNotFoundError struct {
	awsError
	Partition string
	Service   string
	Region    string
}

//// NewEndpointNotFoundError builds and returns NewEndpointNotFoundError.
//func NewEndpointNotFoundError(p, s, r string) EndpointNotFoundError {
//	return EndpointNotFoundError{
//		awsError:  awserr.New("EndpointNotFoundError", "unable to find endpoint", nil),
//		Partition: p,
//		Service:   s,
//		Region:    r,
//	}
//}
//
//// Error returns string representation of the error.
//func (e EndpointNotFoundError) Error() string {
//	extra := fmt.Sprintf("partition: %q, service: %q, region: %q",
//		e.Partition, e.Service, e.Region)
//	return awserr.SprintError(e.Code(), e.Message(), extra, e.OrigErr())
//}
//
//// String returns the string representation of the error.
//func (e EndpointNotFoundError) String() string {
//	return e.Error()
//}

// A UnknownServiceError is returned when the service does not resolve to an
// endpoint. Includes a list of all known services for the partition. Returned
// when a partition does not support the service.
type UnknownServiceError struct {
	awsError
	Partition string
	Service   string
	Known     []string
}

// NewUnknownServiceError builds and returns UnknownServiceError.
func NewUnknownServiceError(p, s string, known []string) UnknownServiceError {
	return UnknownServiceError{
		awsError: awserr.New("UnknownServiceError",
			"could not resolve endpoint for unknown service", nil),
		Partition: p,
		Service:   s,
		Known:     known,
	}
}

// String returns the string representation of the error.
func (e UnknownServiceError) Error() string {
	extra := fmt.Sprintf("partition: %q, service: %q",
		e.Partition, e.Service)
	if len(e.Known) > 0 {
		extra += fmt.Sprintf(", known: %v", e.Known)
	}
	return awserr.SprintError(e.Code(), e.Message(), extra, e.OrigErr())
}

// String returns the string representation of the error.
func (e UnknownServiceError) String() string {
	return e.Error()
}

// A UnknownEndpointError is returned when in StrictMatching mode and the
// service is valid, but the region does not resolve to an endpoint. Includes
// a list of all known endpoints for the service.
type UnknownEndpointError struct {
	awsError
	Partition string
	Service   string
	Region    string
	Known     []string
}

// NewUnknownEndpointError builds and returns UnknownEndpointError.
func NewUnknownEndpointError(p, s, r string, known []string) UnknownEndpointError {
	return UnknownEndpointError{
		awsError: awserr.New("UnknownEndpointError",
			"could not resolve endpoint", nil),
		Partition: p,
		Service:   s,
		Region:    r,
		Known:     known,
	}
}

// String returns the string representation of the error.
func (e UnknownEndpointError) Error() string {
	extra := fmt.Sprintf("partition: %q, service: %q, region: %q",
		e.Partition, e.Service, e.Region)
	if len(e.Known) > 0 {
		extra += fmt.Sprintf(", known: %v", e.Known)
	}
	return awserr.SprintError(e.Code(), e.Message(), extra, e.OrigErr())
}

// String returns the string representation of the error.
func (e UnknownEndpointError) String() string {
	return e.Error()
}
