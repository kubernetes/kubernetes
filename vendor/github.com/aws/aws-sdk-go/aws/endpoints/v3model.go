package endpoints

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var regionValidationRegex = regexp.MustCompile(`^[[:alnum:]]([[:alnum:]\-]*[[:alnum:]])?$`)

type partitions []partition

func (ps partitions) EndpointFor(service, region string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	var opt Options
	opt.Set(opts...)

	for i := 0; i < len(ps); i++ {
		if !ps[i].canResolveEndpoint(service, region, opt.StrictMatching) {
			continue
		}

		return ps[i].EndpointFor(service, region, opts...)
	}

	// If loose matching fallback to first partition format to use
	// when resolving the endpoint.
	if !opt.StrictMatching && len(ps) > 0 {
		return ps[0].EndpointFor(service, region, opts...)
	}

	return ResolvedEndpoint{}, NewUnknownEndpointError("all partitions", service, region, []string{})
}

// Partitions satisfies the EnumPartitions interface and returns a list
// of Partitions representing each partition represented in the SDK's
// endpoints model.
func (ps partitions) Partitions() []Partition {
	parts := make([]Partition, 0, len(ps))
	for i := 0; i < len(ps); i++ {
		parts = append(parts, ps[i].Partition())
	}

	return parts
}

type partition struct {
	ID          string      `json:"partition"`
	Name        string      `json:"partitionName"`
	DNSSuffix   string      `json:"dnsSuffix"`
	RegionRegex regionRegex `json:"regionRegex"`
	Defaults    endpoint    `json:"defaults"`
	Regions     regions     `json:"regions"`
	Services    services    `json:"services"`
}

func (p partition) Partition() Partition {
	return Partition{
		dnsSuffix: p.DNSSuffix,
		id:        p.ID,
		p:         &p,
	}
}

func (p partition) canResolveEndpoint(service, region string, strictMatch bool) bool {
	s, hasService := p.Services[service]
	_, hasEndpoint := s.Endpoints[region]

	if hasEndpoint && hasService {
		return true
	}

	if strictMatch {
		return false
	}

	return p.RegionRegex.MatchString(region)
}

func allowLegacyEmptyRegion(service string) bool {
	legacy := map[string]struct{}{
		"budgets":       {},
		"ce":            {},
		"chime":         {},
		"cloudfront":    {},
		"ec2metadata":   {},
		"iam":           {},
		"importexport":  {},
		"organizations": {},
		"route53":       {},
		"sts":           {},
		"support":       {},
		"waf":           {},
	}

	_, allowed := legacy[service]
	return allowed
}

func (p partition) EndpointFor(service, region string, opts ...func(*Options)) (resolved ResolvedEndpoint, err error) {
	var opt Options
	opt.Set(opts...)

	s, hasService := p.Services[service]
	if len(service) == 0 || !(hasService || opt.ResolveUnknownService) {
		// Only return error if the resolver will not fallback to creating
		// endpoint based on service endpoint ID passed in.
		return resolved, NewUnknownServiceError(p.ID, service, serviceList(p.Services))
	}

	if len(region) == 0 && allowLegacyEmptyRegion(service) && len(s.PartitionEndpoint) != 0 {
		region = s.PartitionEndpoint
	}

	if (service == "sts" && opt.STSRegionalEndpoint != RegionalSTSEndpoint) ||
		(service == "s3" && opt.S3UsEast1RegionalEndpoint != RegionalS3UsEast1Endpoint) {
		if _, ok := legacyGlobalRegions[service][region]; ok {
			region = "aws-global"
		}
	}

	e, hasEndpoint := s.endpointForRegion(region)
	if len(region) == 0 || (!hasEndpoint && opt.StrictMatching) {
		return resolved, NewUnknownEndpointError(p.ID, service, region, endpointList(s.Endpoints))
	}

	defs := []endpoint{p.Defaults, s.Defaults}

	return e.resolve(service, p.ID, region, p.DNSSuffix, defs, opt)
}

func serviceList(ss services) []string {
	list := make([]string, 0, len(ss))
	for k := range ss {
		list = append(list, k)
	}
	return list
}
func endpointList(es endpoints) []string {
	list := make([]string, 0, len(es))
	for k := range es {
		list = append(list, k)
	}
	return list
}

type regionRegex struct {
	*regexp.Regexp
}

func (rr *regionRegex) UnmarshalJSON(b []byte) (err error) {
	// Strip leading and trailing quotes
	regex, err := strconv.Unquote(string(b))
	if err != nil {
		return fmt.Errorf("unable to strip quotes from regex, %v", err)
	}

	rr.Regexp, err = regexp.Compile(regex)
	if err != nil {
		return fmt.Errorf("unable to unmarshal region regex, %v", err)
	}
	return nil
}

type regions map[string]region

type region struct {
	Description string `json:"description"`
}

type services map[string]service

type service struct {
	PartitionEndpoint string    `json:"partitionEndpoint"`
	IsRegionalized    boxedBool `json:"isRegionalized,omitempty"`
	Defaults          endpoint  `json:"defaults"`
	Endpoints         endpoints `json:"endpoints"`
}

func (s *service) endpointForRegion(region string) (endpoint, bool) {
	if e, ok := s.Endpoints[region]; ok {
		return e, true
	}

	if s.IsRegionalized == boxedFalse {
		return s.Endpoints[s.PartitionEndpoint], region == s.PartitionEndpoint
	}

	// Unable to find any matching endpoint, return
	// blank that will be used for generic endpoint creation.
	return endpoint{}, false
}

type endpoints map[string]endpoint

type endpoint struct {
	Hostname        string          `json:"hostname"`
	Protocols       []string        `json:"protocols"`
	CredentialScope credentialScope `json:"credentialScope"`

	// Custom fields not modeled
	HasDualStack      boxedBool `json:"-"`
	DualStackHostname string    `json:"-"`

	// Signature Version not used
	SignatureVersions []string `json:"signatureVersions"`

	// SSLCommonName not used.
	SSLCommonName string `json:"sslCommonName"`
}

const (
	defaultProtocol = "https"
	defaultSigner   = "v4"
)

var (
	protocolPriority = []string{"https", "http"}
	signerPriority   = []string{"v4", "v2"}
)

func getByPriority(s []string, p []string, def string) string {
	if len(s) == 0 {
		return def
	}

	for i := 0; i < len(p); i++ {
		for j := 0; j < len(s); j++ {
			if s[j] == p[i] {
				return s[j]
			}
		}
	}

	return s[0]
}

func (e endpoint) resolve(service, partitionID, region, dnsSuffix string, defs []endpoint, opts Options) (ResolvedEndpoint, error) {
	var merged endpoint
	for _, def := range defs {
		merged.mergeIn(def)
	}
	merged.mergeIn(e)
	e = merged

	signingRegion := e.CredentialScope.Region
	if len(signingRegion) == 0 {
		signingRegion = region
	}

	signingName := e.CredentialScope.Service
	var signingNameDerived bool
	if len(signingName) == 0 {
		signingName = service
		signingNameDerived = true
	}

	hostname := e.Hostname
	// Offset the hostname for dualstack if enabled
	if opts.UseDualStack && e.HasDualStack == boxedTrue {
		hostname = e.DualStackHostname
		region = signingRegion
	}

	if !validateInputRegion(region) {
		return ResolvedEndpoint{}, fmt.Errorf("invalid region identifier format provided")
	}

	u := strings.Replace(hostname, "{service}", service, 1)
	u = strings.Replace(u, "{region}", region, 1)
	u = strings.Replace(u, "{dnsSuffix}", dnsSuffix, 1)

	scheme := getEndpointScheme(e.Protocols, opts.DisableSSL)
	u = fmt.Sprintf("%s://%s", scheme, u)

	return ResolvedEndpoint{
		URL:                u,
		PartitionID:        partitionID,
		SigningRegion:      signingRegion,
		SigningName:        signingName,
		SigningNameDerived: signingNameDerived,
		SigningMethod:      getByPriority(e.SignatureVersions, signerPriority, defaultSigner),
	}, nil
}

func getEndpointScheme(protocols []string, disableSSL bool) string {
	if disableSSL {
		return "http"
	}

	return getByPriority(protocols, protocolPriority, defaultProtocol)
}

func (e *endpoint) mergeIn(other endpoint) {
	if len(other.Hostname) > 0 {
		e.Hostname = other.Hostname
	}
	if len(other.Protocols) > 0 {
		e.Protocols = other.Protocols
	}
	if len(other.SignatureVersions) > 0 {
		e.SignatureVersions = other.SignatureVersions
	}
	if len(other.CredentialScope.Region) > 0 {
		e.CredentialScope.Region = other.CredentialScope.Region
	}
	if len(other.CredentialScope.Service) > 0 {
		e.CredentialScope.Service = other.CredentialScope.Service
	}
	if len(other.SSLCommonName) > 0 {
		e.SSLCommonName = other.SSLCommonName
	}
	if other.HasDualStack != boxedBoolUnset {
		e.HasDualStack = other.HasDualStack
	}
	if len(other.DualStackHostname) > 0 {
		e.DualStackHostname = other.DualStackHostname
	}
}

type credentialScope struct {
	Region  string `json:"region"`
	Service string `json:"service"`
}

type boxedBool int

func (b *boxedBool) UnmarshalJSON(buf []byte) error {
	v, err := strconv.ParseBool(string(buf))
	if err != nil {
		return err
	}

	if v {
		*b = boxedTrue
	} else {
		*b = boxedFalse
	}

	return nil
}

const (
	boxedBoolUnset boxedBool = iota
	boxedFalse
	boxedTrue
)

func validateInputRegion(region string) bool {
	return regionValidationRegex.MatchString(region)
}
