package endpoints

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

// A Logger is a minimalistic interface for the SDK to log messages to.
type Logger interface {
	Log(...interface{})
}

// DualStackEndpointState is a constant to describe the dual-stack endpoint resolution
// behavior.
type DualStackEndpointState uint

const (
	// DualStackEndpointStateUnset is the default value behavior for dual-stack endpoint
	// resolution.
	DualStackEndpointStateUnset DualStackEndpointState = iota

	// DualStackEndpointStateEnabled enable dual-stack endpoint resolution for endpoints.
	DualStackEndpointStateEnabled

	// DualStackEndpointStateDisabled disables dual-stack endpoint resolution for endpoints.
	DualStackEndpointStateDisabled
)

// FIPSEndpointState is a constant to describe the FIPS endpoint resolution behavior.
type FIPSEndpointState uint

const (
	// FIPSEndpointStateUnset is the default value behavior for FIPS endpoint resolution.
	FIPSEndpointStateUnset FIPSEndpointState = iota

	// FIPSEndpointStateEnabled enables FIPS endpoint resolution for service endpoints.
	FIPSEndpointStateEnabled

	// FIPSEndpointStateDisabled disables FIPS endpoint resolution for endpoints.
	FIPSEndpointStateDisabled
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
	//
	// Deprecated: This option will continue to function for S3 and S3 Control for backwards compatibility.
	// UseDualStackEndpoint should be used to enable usage of a service's dual-stack endpoint for all service clients
	// moving forward. For S3 and S3 Control, when UseDualStackEndpoint is set to a non-zero value it takes higher
	// precedence then this option.
	UseDualStack bool

	// Sets the resolver to resolve a dual-stack endpoint for the service.
	UseDualStackEndpoint DualStackEndpointState

	// UseFIPSEndpoint specifies the resolver must resolve a FIPS endpoint.
	UseFIPSEndpoint FIPSEndpointState

	// Enables strict matching of services and regions resolved endpoints.
	// If the partition doesn't enumerate the exact service and region an
	// error will be returned. This option will prevent returning endpoints
	// that look valid, but may not resolve to any real endpoint.
	StrictMatching bool

	// Enables resolving a service endpoint based on the region provided if the
	// service does not exist. The service endpoint ID will be used as the service
	// domain name prefix. By default the endpoint resolver requires the service
	// to be known when resolving endpoints.
	//
	// If resolving an endpoint on the partition list the provided region will
	// be used to determine which partition's domain name pattern to the service
	// endpoint ID with. If both the service and region are unknown and resolving
	// the endpoint on partition list an UnknownEndpointError error will be returned.
	//
	// If resolving and endpoint on a partition specific resolver that partition's
	// domain name pattern will be used with the service endpoint ID. If both
	// region and service do not exist when resolving an endpoint on a specific
	// partition the partition's domain pattern will be used to combine the
	// endpoint and region together.
	//
	// This option is ignored if StrictMatching is enabled.
	ResolveUnknownService bool

	// Specifies the EC2 Instance Metadata Service default endpoint selection mode (IPv4 or IPv6)
	EC2MetadataEndpointMode EC2IMDSEndpointModeState

	// STS Regional Endpoint flag helps with resolving the STS endpoint
	STSRegionalEndpoint STSRegionalEndpoint

	// S3 Regional Endpoint flag helps with resolving the S3 endpoint
	S3UsEast1RegionalEndpoint S3UsEast1RegionalEndpoint

	// ResolvedRegion is the resolved region string. If provided (non-zero length) it takes priority
	// over the region name passed to the ResolveEndpoint call.
	ResolvedRegion string

	// Logger is the logger that will be used to log messages.
	Logger Logger

	// Determines whether logging of deprecated endpoints usage is enabled.
	LogDeprecated bool
}

func (o Options) getEndpointVariant(service string) (v endpointVariant) {
	const s3 = "s3"
	const s3Control = "s3-control"

	if (o.UseDualStackEndpoint == DualStackEndpointStateEnabled) ||
		((service == s3 || service == s3Control) && (o.UseDualStackEndpoint == DualStackEndpointStateUnset && o.UseDualStack)) {
		v |= dualStackVariant
	}
	if o.UseFIPSEndpoint == FIPSEndpointStateEnabled {
		v |= fipsVariant
	}
	return v
}

// EC2IMDSEndpointModeState is an enum configuration variable describing the client endpoint mode.
type EC2IMDSEndpointModeState uint

// Enumeration values for EC2IMDSEndpointModeState
const (
	EC2IMDSEndpointModeStateUnset EC2IMDSEndpointModeState = iota
	EC2IMDSEndpointModeStateIPv4
	EC2IMDSEndpointModeStateIPv6
)

// SetFromString sets the EC2IMDSEndpointModeState based on the provided string value. Unknown values will default to EC2IMDSEndpointModeStateUnset
func (e *EC2IMDSEndpointModeState) SetFromString(v string) error {
	v = strings.TrimSpace(v)

	switch {
	case len(v) == 0:
		*e = EC2IMDSEndpointModeStateUnset
	case strings.EqualFold(v, "IPv6"):
		*e = EC2IMDSEndpointModeStateIPv6
	case strings.EqualFold(v, "IPv4"):
		*e = EC2IMDSEndpointModeStateIPv4
	default:
		return fmt.Errorf("unknown EC2 IMDS endpoint mode, must be either IPv6 or IPv4")
	}
	return nil
}

// STSRegionalEndpoint is an enum for the states of the STS Regional Endpoint
// options.
type STSRegionalEndpoint int

func (e STSRegionalEndpoint) String() string {
	switch e {
	case LegacySTSEndpoint:
		return "legacy"
	case RegionalSTSEndpoint:
		return "regional"
	case UnsetSTSEndpoint:
		return ""
	default:
		return "unknown"
	}
}

const (

	// UnsetSTSEndpoint represents that STS Regional Endpoint flag is not specified.
	UnsetSTSEndpoint STSRegionalEndpoint = iota

	// LegacySTSEndpoint represents when STS Regional Endpoint flag is specified
	// to use legacy endpoints.
	LegacySTSEndpoint

	// RegionalSTSEndpoint represents when STS Regional Endpoint flag is specified
	// to use regional endpoints.
	RegionalSTSEndpoint
)

// GetSTSRegionalEndpoint function returns the STSRegionalEndpointFlag based
// on the input string provided in env config or shared config by the user.
//
// `legacy`, `regional` are the only case-insensitive valid strings for
// resolving the STS regional Endpoint flag.
func GetSTSRegionalEndpoint(s string) (STSRegionalEndpoint, error) {
	switch {
	case strings.EqualFold(s, "legacy"):
		return LegacySTSEndpoint, nil
	case strings.EqualFold(s, "regional"):
		return RegionalSTSEndpoint, nil
	default:
		return UnsetSTSEndpoint, fmt.Errorf("unable to resolve the value of STSRegionalEndpoint for %v", s)
	}
}

// S3UsEast1RegionalEndpoint is an enum for the states of the S3 us-east-1
// Regional Endpoint options.
type S3UsEast1RegionalEndpoint int

func (e S3UsEast1RegionalEndpoint) String() string {
	switch e {
	case LegacyS3UsEast1Endpoint:
		return "legacy"
	case RegionalS3UsEast1Endpoint:
		return "regional"
	case UnsetS3UsEast1Endpoint:
		return ""
	default:
		return "unknown"
	}
}

const (

	// UnsetS3UsEast1Endpoint represents that S3 Regional Endpoint flag is not
	// specified.
	UnsetS3UsEast1Endpoint S3UsEast1RegionalEndpoint = iota

	// LegacyS3UsEast1Endpoint represents when S3 Regional Endpoint flag is
	// specified to use legacy endpoints.
	LegacyS3UsEast1Endpoint

	// RegionalS3UsEast1Endpoint represents when S3 Regional Endpoint flag is
	// specified to use regional endpoints.
	RegionalS3UsEast1Endpoint
)

// GetS3UsEast1RegionalEndpoint function returns the S3UsEast1RegionalEndpointFlag based
// on the input string provided in env config or shared config by the user.
//
// `legacy`, `regional` are the only case-insensitive valid strings for
// resolving the S3 regional Endpoint flag.
func GetS3UsEast1RegionalEndpoint(s string) (S3UsEast1RegionalEndpoint, error) {
	switch {
	case strings.EqualFold(s, "legacy"):
		return LegacyS3UsEast1Endpoint, nil
	case strings.EqualFold(s, "regional"):
		return RegionalS3UsEast1Endpoint, nil
	default:
		return UnsetS3UsEast1Endpoint,
			fmt.Errorf("unable to resolve the value of S3UsEast1RegionalEndpoint for %v", s)
	}
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
//
// Deprecated: UseDualStackEndpointOption should be used to enable usage of a service's dual-stack endpoint.
// When DualStackEndpointState is set to a non-zero value it takes higher precedence then this option.
func UseDualStackOption(o *Options) {
	o.UseDualStack = true
}

// UseDualStackEndpointOption sets the UseDualStackEndpoint option to enabled. Can be used as a functional
// option when resolving endpoints.
func UseDualStackEndpointOption(o *Options) {
	o.UseDualStackEndpoint = DualStackEndpointStateEnabled
}

// UseFIPSEndpointOption sets the UseFIPSEndpoint option to enabled. Can be used as a functional
// option when resolving endpoints.
func UseFIPSEndpointOption(o *Options) {
	o.UseFIPSEndpoint = FIPSEndpointStateEnabled
}

// StrictMatchingOption sets the StrictMatching option. Can be used as a functional
// option when resolving endpoints.
func StrictMatchingOption(o *Options) {
	o.StrictMatching = true
}

// ResolveUnknownServiceOption sets the ResolveUnknownService option. Can be used
// as a functional option when resolving endpoints.
func ResolveUnknownServiceOption(o *Options) {
	o.ResolveUnknownService = true
}

// STSRegionalEndpointOption enables the STS endpoint resolver behavior to resolve
// STS endpoint to their regional endpoint, instead of the global endpoint.
func STSRegionalEndpointOption(o *Options) {
	o.STSRegionalEndpoint = RegionalSTSEndpoint
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

// RegionsForService returns a map of regions for the partition and service.
// If either the partition or service does not exist false will be returned
// as the second parameter.
//
// This example shows how  to get the regions for DynamoDB in the AWS partition.
//
//	rs, exists := endpoints.RegionsForService(endpoints.DefaultPartitions(), endpoints.AwsPartitionID, endpoints.DynamodbServiceID)
//
// This is equivalent to using the partition directly.
//
//	rs := endpoints.AwsPartition().Services()[endpoints.DynamodbServiceID].Regions()
func RegionsForService(ps []Partition, partitionID, serviceID string) (map[string]Region, bool) {
	for _, p := range ps {
		if p.ID() != partitionID {
			continue
		}
		if _, ok := p.p.Services[serviceID]; !(ok || serviceID == Ec2metadataServiceID) {
			break
		}

		s := Service{
			id: serviceID,
			p:  p.p,
		}
		return s.Regions(), true
	}

	return map[string]Region{}, false
}

// PartitionForRegion returns the first partition which includes the region
// passed in. This includes both known regions and regions which match
// a pattern supported by the partition which may include regions that are
// not explicitly known by the partition. Use the Regions method of the
// returned Partition if explicit support is needed.
func PartitionForRegion(ps []Partition, regionID string) (Partition, bool) {
	for _, p := range ps {
		if _, ok := p.p.Regions[regionID]; ok || p.p.RegionRegex.MatchString(regionID) {
			return p, true
		}
	}

	return Partition{}, false
}

// A Partition provides the ability to enumerate the partition's regions
// and services.
type Partition struct {
	id, dnsSuffix string
	p             *partition
}

// DNSSuffix returns the base domain name of the partition.
func (p Partition) DNSSuffix() string { return p.dnsSuffix }

// ID returns the identifier of the partition.
func (p Partition) ID() string { return p.id }

// EndpointFor attempts to resolve the endpoint based on service and region.
// See Options for information on configuring how the endpoint is resolved.
//
// If the service cannot be found in the metadata the UnknownServiceError
// error will be returned. This validation will occur regardless if
// StrictMatching is enabled. To enable resolving unknown services set the
// "ResolveUnknownService" option to true. When StrictMatching is disabled
// this option allows the partition resolver to resolve a endpoint based on
// the service endpoint ID provided.
//
// When resolving endpoints you can choose to enable StrictMatching. This will
// require the provided service and region to be known by the partition.
// If the endpoint cannot be strictly resolved an error will be returned. This
// mode is useful to ensure the endpoint resolved is valid. Without
// StrictMatching enabled the endpoint returned may look valid but may not work.
// StrictMatching requires the SDK to be updated if you want to take advantage
// of new regions and services expansions.
//
// Errors that can be returned.
//   - UnknownServiceError
//   - UnknownEndpointError
func (p Partition) EndpointFor(service, region string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	return p.p.EndpointFor(service, region, opts...)
}

// Regions returns a map of Regions indexed by their ID. This is useful for
// enumerating over the regions in a partition.
func (p Partition) Regions() map[string]Region {
	rs := make(map[string]Region, len(p.p.Regions))
	for id, r := range p.p.Regions {
		rs[id] = Region{
			id:   id,
			desc: r.Description,
			p:    p.p,
		}
	}

	return rs
}

// Services returns a map of Service indexed by their ID. This is useful for
// enumerating over the services in a partition.
func (p Partition) Services() map[string]Service {
	ss := make(map[string]Service, len(p.p.Services))

	for id := range p.p.Services {
		ss[id] = Service{
			id: id,
			p:  p.p,
		}
	}

	// Since we have removed the customization that injected this into the model
	// we still need to pretend that this is a modeled service.
	if _, ok := ss[Ec2metadataServiceID]; !ok {
		ss[Ec2metadataServiceID] = Service{
			id: Ec2metadataServiceID,
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
func (r Region) ID() string { return r.id }

// Description returns the region's description. The region description
// is free text, it can be empty, and it may change between SDK releases.
func (r Region) Description() string { return r.desc }

// ResolveEndpoint resolves an endpoint from the context of the region given
// a service. See Partition.EndpointFor for usage and errors that can be returned.
func (r Region) ResolveEndpoint(service string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	return r.p.EndpointFor(service, r.id, opts...)
}

// Services returns a list of all services that are known to be in this region.
func (r Region) Services() map[string]Service {
	ss := map[string]Service{}
	for id, s := range r.p.Services {
		if _, ok := s.Endpoints[endpointKey{Region: r.id}]; ok {
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
func (s Service) ID() string { return s.id }

// ResolveEndpoint resolves an endpoint from the context of a service given
// a region. See Partition.EndpointFor for usage and errors that can be returned.
func (s Service) ResolveEndpoint(region string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	return s.p.EndpointFor(s.id, region, opts...)
}

// Regions returns a map of Regions that the service is present in.
//
// A region is the AWS region the service exists in. Whereas a Endpoint is
// an URL that can be resolved to a instance of a service.
func (s Service) Regions() map[string]Region {
	rs := map[string]Region{}

	service, ok := s.p.Services[s.id]

	// Since ec2metadata customization has been removed we need to check
	// if it was defined in non-standard endpoints.json file. If it's not
	// then we can return the empty map as there is no regional-endpoints for IMDS.
	// Otherwise, we iterate need to iterate the non-standard model.
	if s.id == Ec2metadataServiceID && !ok {
		return rs
	}

	for id := range service.Endpoints {
		if id.Variant != 0 {
			continue
		}
		if r, ok := s.p.Regions[id.Region]; ok {
			rs[id.Region] = Region{
				id:   id.Region,
				desc: r.Description,
				p:    s.p,
			}
		}
	}

	return rs
}

// Endpoints returns a map of Endpoints indexed by their ID for all known
// endpoints for a service.
//
// A region is the AWS region the service exists in. Whereas a Endpoint is
// an URL that can be resolved to a instance of a service.
func (s Service) Endpoints() map[string]Endpoint {
	es := make(map[string]Endpoint, len(s.p.Services[s.id].Endpoints))
	for id := range s.p.Services[s.id].Endpoints {
		if id.Variant != 0 {
			continue
		}
		es[id.Region] = Endpoint{
			id:        id.Region,
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
func (e Endpoint) ID() string { return e.id }

// ServiceID returns the identifier the endpoint belongs to.
func (e Endpoint) ServiceID() string { return e.serviceID }

// ResolveEndpoint resolves an endpoint from the context of a service and
// region the endpoint represents. See Partition.EndpointFor for usage and
// errors that can be returned.
func (e Endpoint) ResolveEndpoint(opts ...func(*Options)) (ResolvedEndpoint, error) {
	return e.p.EndpointFor(e.serviceID, e.id, opts...)
}

// A ResolvedEndpoint is an endpoint that has been resolved based on a partition
// service, and region.
type ResolvedEndpoint struct {
	// The endpoint URL
	URL string

	// The endpoint partition
	PartitionID string

	// The region that should be used for signing requests.
	SigningRegion string

	// The service name that should be used for signing requests.
	SigningName string

	// States that the signing name for this endpoint was derived from metadata
	// passed in, but was not explicitly modeled.
	SigningNameDerived bool

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
