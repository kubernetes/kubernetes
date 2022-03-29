package endpoints

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

const (
	ec2MetadataEndpointIPv6 = "http://[fd00:ec2::254]/latest"
	ec2MetadataEndpointIPv4 = "http://169.254.169.254/latest"
)

const dnsSuffixTemplateKey = "{dnsSuffix}"

// defaultKey is a compound map key of a variant and other values.
type defaultKey struct {
	Variant        endpointVariant
	ServiceVariant serviceVariant
}

// endpointKey is a compound map key of a region and associated variant value.
type endpointKey struct {
	Region  string
	Variant endpointVariant
}

// endpointVariant is a bit field to describe the endpoints attributes.
type endpointVariant uint64

// serviceVariant is a bit field to describe the service endpoint attributes.
type serviceVariant uint64

const (
	// fipsVariant indicates that the endpoint is FIPS capable.
	fipsVariant endpointVariant = 1 << (64 - 1 - iota)

	// dualStackVariant indicates that the endpoint is DualStack capable.
	dualStackVariant
)

var regionValidationRegex = regexp.MustCompile(`^[[:alnum:]]([[:alnum:]\-]*[[:alnum:]])?$`)

type partitions []partition

func (ps partitions) EndpointFor(service, region string, opts ...func(*Options)) (ResolvedEndpoint, error) {
	var opt Options
	opt.Set(opts...)

	if len(opt.ResolvedRegion) > 0 {
		region = opt.ResolvedRegion
	}

	for i := 0; i < len(ps); i++ {
		if !ps[i].canResolveEndpoint(service, region, opt) {
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

type endpointWithVariants struct {
	endpoint
	Variants []endpointWithTags `json:"variants"`
}

type endpointWithTags struct {
	endpoint
	Tags []string `json:"tags"`
}

type endpointDefaults map[defaultKey]endpoint

func (p *endpointDefaults) UnmarshalJSON(data []byte) error {
	if *p == nil {
		*p = make(endpointDefaults)
	}

	var e endpointWithVariants
	if err := json.Unmarshal(data, &e); err != nil {
		return err
	}

	(*p)[defaultKey{Variant: 0}] = e.endpoint

	e.Hostname = ""
	e.DNSSuffix = ""

	for _, variant := range e.Variants {
		endpointVariant, unknown := parseVariantTags(variant.Tags)
		if unknown {
			continue
		}

		var ve endpoint
		ve.mergeIn(e.endpoint)
		ve.mergeIn(variant.endpoint)

		(*p)[defaultKey{Variant: endpointVariant}] = ve
	}

	return nil
}

func parseVariantTags(tags []string) (ev endpointVariant, unknown bool) {
	if len(tags) == 0 {
		unknown = true
		return
	}

	for _, tag := range tags {
		switch {
		case strings.EqualFold("fips", tag):
			ev |= fipsVariant
		case strings.EqualFold("dualstack", tag):
			ev |= dualStackVariant
		default:
			unknown = true
		}
	}
	return ev, unknown
}

type partition struct {
	ID          string           `json:"partition"`
	Name        string           `json:"partitionName"`
	DNSSuffix   string           `json:"dnsSuffix"`
	RegionRegex regionRegex      `json:"regionRegex"`
	Defaults    endpointDefaults `json:"defaults"`
	Regions     regions          `json:"regions"`
	Services    services         `json:"services"`
}

func (p partition) Partition() Partition {
	return Partition{
		dnsSuffix: p.DNSSuffix,
		id:        p.ID,
		p:         &p,
	}
}

func (p partition) canResolveEndpoint(service, region string, options Options) bool {
	s, hasService := p.Services[service]
	_, hasEndpoint := s.Endpoints[endpointKey{
		Region:  region,
		Variant: options.getEndpointVariant(service),
	}]

	if hasEndpoint && hasService {
		return true
	}

	if options.StrictMatching {
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

	if len(opt.ResolvedRegion) > 0 {
		region = opt.ResolvedRegion
	}

	s, hasService := p.Services[service]

	if service == Ec2metadataServiceID && !hasService {
		endpoint := getEC2MetadataEndpoint(p.ID, service, opt.EC2MetadataEndpointMode)
		return endpoint, nil
	}

	if len(service) == 0 || !(hasService || opt.ResolveUnknownService) {
		// Only return error if the resolver will not fallback to creating
		// endpoint based on service endpoint ID passed in.
		return resolved, NewUnknownServiceError(p.ID, service, serviceList(p.Services))
	}

	if len(region) == 0 && allowLegacyEmptyRegion(service) && len(s.PartitionEndpoint) != 0 {
		region = s.PartitionEndpoint
	}

	if r, ok := isLegacyGlobalRegion(service, region, opt); ok {
		region = r
	}

	variant := opt.getEndpointVariant(service)

	endpoints := s.Endpoints

	serviceDefaults, hasServiceDefault := s.Defaults[defaultKey{Variant: variant}]
	// If we searched for a variant which may have no explicit service defaults,
	// then we need to inherit the standard service defaults except the hostname and dnsSuffix
	if variant != 0 && !hasServiceDefault {
		serviceDefaults = s.Defaults[defaultKey{}]
		serviceDefaults.Hostname = ""
		serviceDefaults.DNSSuffix = ""
	}

	partitionDefaults, hasPartitionDefault := p.Defaults[defaultKey{Variant: variant}]

	var dnsSuffix string
	if len(serviceDefaults.DNSSuffix) > 0 {
		dnsSuffix = serviceDefaults.DNSSuffix
	} else if variant == 0 {
		// For legacy reasons the partition dnsSuffix is not in the defaults, so if we looked for
		// a non-variant endpoint then we need to set the dnsSuffix.
		dnsSuffix = p.DNSSuffix
	}

	noDefaults := !hasServiceDefault && !hasPartitionDefault

	e, hasEndpoint := s.endpointForRegion(region, endpoints, variant)
	if len(region) == 0 || (!hasEndpoint && (opt.StrictMatching || noDefaults)) {
		return resolved, NewUnknownEndpointError(p.ID, service, region, endpointList(endpoints, variant))
	}

	defs := []endpoint{partitionDefaults, serviceDefaults}

	return e.resolve(service, p.ID, region, dnsSuffixTemplateKey, dnsSuffix, defs, opt)
}

func getEC2MetadataEndpoint(partitionID, service string, mode EC2IMDSEndpointModeState) ResolvedEndpoint {
	switch mode {
	case EC2IMDSEndpointModeStateIPv6:
		return ResolvedEndpoint{
			URL:                ec2MetadataEndpointIPv6,
			PartitionID:        partitionID,
			SigningRegion:      "aws-global",
			SigningName:        service,
			SigningNameDerived: true,
			SigningMethod:      "v4",
		}
	case EC2IMDSEndpointModeStateIPv4:
		fallthrough
	default:
		return ResolvedEndpoint{
			URL:                ec2MetadataEndpointIPv4,
			PartitionID:        partitionID,
			SigningRegion:      "aws-global",
			SigningName:        service,
			SigningNameDerived: true,
			SigningMethod:      "v4",
		}
	}
}

func isLegacyGlobalRegion(service string, region string, opt Options) (string, bool) {
	if opt.getEndpointVariant(service) != 0 {
		return "", false
	}

	const (
		sts       = "sts"
		s3        = "s3"
		awsGlobal = "aws-global"
	)

	switch {
	case service == sts && opt.STSRegionalEndpoint == RegionalSTSEndpoint:
		return region, false
	case service == s3 && opt.S3UsEast1RegionalEndpoint == RegionalS3UsEast1Endpoint:
		return region, false
	default:
		if _, ok := legacyGlobalRegions[service][region]; ok {
			return awsGlobal, true
		}
	}

	return region, false
}

func serviceList(ss services) []string {
	list := make([]string, 0, len(ss))
	for k := range ss {
		list = append(list, k)
	}
	return list
}
func endpointList(es serviceEndpoints, variant endpointVariant) []string {
	list := make([]string, 0, len(es))
	for k := range es {
		if k.Variant != variant {
			continue
		}
		list = append(list, k.Region)
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
	PartitionEndpoint string           `json:"partitionEndpoint"`
	IsRegionalized    boxedBool        `json:"isRegionalized,omitempty"`
	Defaults          endpointDefaults `json:"defaults"`
	Endpoints         serviceEndpoints `json:"endpoints"`
}

func (s *service) endpointForRegion(region string, endpoints serviceEndpoints, variant endpointVariant) (endpoint, bool) {
	if e, ok := endpoints[endpointKey{Region: region, Variant: variant}]; ok {
		return e, true
	}

	if s.IsRegionalized == boxedFalse {
		return endpoints[endpointKey{Region: s.PartitionEndpoint, Variant: variant}], region == s.PartitionEndpoint
	}

	// Unable to find any matching endpoint, return
	// blank that will be used for generic endpoint creation.
	return endpoint{}, false
}

type serviceEndpoints map[endpointKey]endpoint

func (s *serviceEndpoints) UnmarshalJSON(data []byte) error {
	if *s == nil {
		*s = make(serviceEndpoints)
	}

	var regionToEndpoint map[string]endpointWithVariants

	if err := json.Unmarshal(data, &regionToEndpoint); err != nil {
		return err
	}

	for region, e := range regionToEndpoint {
		(*s)[endpointKey{Region: region}] = e.endpoint

		e.Hostname = ""
		e.DNSSuffix = ""

		for _, variant := range e.Variants {
			endpointVariant, unknown := parseVariantTags(variant.Tags)
			if unknown {
				continue
			}

			var ve endpoint
			ve.mergeIn(e.endpoint)
			ve.mergeIn(variant.endpoint)

			(*s)[endpointKey{Region: region, Variant: endpointVariant}] = ve
		}
	}

	return nil
}

type endpoint struct {
	Hostname        string          `json:"hostname"`
	Protocols       []string        `json:"protocols"`
	CredentialScope credentialScope `json:"credentialScope"`

	DNSSuffix string `json:"dnsSuffix"`

	// Signature Version not used
	SignatureVersions []string `json:"signatureVersions"`

	// SSLCommonName not used.
	SSLCommonName string `json:"sslCommonName"`

	Deprecated boxedBool `json:"deprecated"`
}

// isZero returns whether the endpoint structure is an empty (zero) value.
func (e endpoint) isZero() bool {
	switch {
	case len(e.Hostname) != 0:
		return false
	case len(e.Protocols) != 0:
		return false
	case e.CredentialScope != (credentialScope{}):
		return false
	case len(e.SignatureVersions) != 0:
		return false
	case len(e.SSLCommonName) != 0:
		return false
	}
	return true
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

func (e endpoint) resolve(service, partitionID, region, dnsSuffixTemplateVariable, dnsSuffix string, defs []endpoint, opts Options) (ResolvedEndpoint, error) {
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

	if !validateInputRegion(region) {
		return ResolvedEndpoint{}, fmt.Errorf("invalid region identifier format provided")
	}

	if len(merged.DNSSuffix) > 0 {
		dnsSuffix = merged.DNSSuffix
	}

	u := strings.Replace(hostname, "{service}", service, 1)
	u = strings.Replace(u, "{region}", region, 1)
	u = strings.Replace(u, dnsSuffixTemplateVariable, dnsSuffix, 1)

	scheme := getEndpointScheme(e.Protocols, opts.DisableSSL)
	u = fmt.Sprintf("%s://%s", scheme, u)

	if e.Deprecated == boxedTrue && opts.LogDeprecated && opts.Logger != nil {
		opts.Logger.Log(fmt.Sprintf("endpoint identifier %q, url %q marked as deprecated", region, u))
	}

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
	if len(other.DNSSuffix) > 0 {
		e.DNSSuffix = other.DNSSuffix
	}
	if other.Deprecated != boxedBoolUnset {
		e.Deprecated = other.Deprecated
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
