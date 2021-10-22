package s3

import (
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/s3shared"
	"github.com/aws/aws-sdk-go/internal/s3shared/arn"
	"github.com/aws/aws-sdk-go/private/protocol"
)

const (
	accessPointPrefixLabel    = "accesspoint"
	accountIDPrefixLabel      = "accountID"
	accessPointPrefixTemplate = "{" + accessPointPrefixLabel + "}-{" + accountIDPrefixLabel + "}."

	outpostPrefixLabel               = "outpost"
	outpostAccessPointPrefixTemplate = accessPointPrefixTemplate + "{" + outpostPrefixLabel + "}."
)

// accessPointEndpointBuilder represents the endpoint builder for access point arn
type accessPointEndpointBuilder arn.AccessPointARN

// build builds the endpoint for corresponding access point arn
//
// For building an endpoint from access point arn, format used is:
// - Access point endpoint format : {accesspointName}-{accountId}.s3-accesspoint.{region}.{dnsSuffix}
// - example : myaccesspoint-012345678901.s3-accesspoint.us-west-2.amazonaws.com
//
// Access Point Endpoint requests are signed using "s3" as signing name.
//
func (a accessPointEndpointBuilder) build(req *request.Request) error {
	resolveService := arn.AccessPointARN(a).Service
	resolveRegion := arn.AccessPointARN(a).Region
	cfgRegion := aws.StringValue(req.Config.Region)

	if s3shared.IsFIPS(cfgRegion) {
		if aws.BoolValue(req.Config.S3UseARNRegion) && s3shared.IsCrossRegion(req, resolveRegion) {
			// FIPS with cross region is not supported, the SDK must fail
			// because there is no well defined method for SDK to construct a
			// correct FIPS endpoint.
			return s3shared.NewClientConfiguredForCrossRegionFIPSError(arn.AccessPointARN(a),
				req.ClientInfo.PartitionID, cfgRegion, nil)
		}
		resolveRegion = cfgRegion
	}

	endpoint, err := resolveRegionalEndpoint(req, resolveRegion, resolveService)
	if err != nil {
		return s3shared.NewFailedToResolveEndpointError(arn.AccessPointARN(a),
			req.ClientInfo.PartitionID, cfgRegion, err)
	}

	if err = updateRequestEndpoint(req, endpoint.URL); err != nil {
		return err
	}

	const serviceEndpointLabel = "s3-accesspoint"

	// dual stack provided by endpoint resolver
	cfgHost := req.HTTPRequest.URL.Host
	if strings.HasPrefix(cfgHost, "s3") {
		req.HTTPRequest.URL.Host = serviceEndpointLabel + cfgHost[2:]
	}

	protocol.HostPrefixBuilder{
		Prefix:   accessPointPrefixTemplate,
		LabelsFn: a.hostPrefixLabelValues,
	}.Build(req)

	// signer redirection
	redirectSigner(req, endpoint.SigningName, endpoint.SigningRegion)

	err = protocol.ValidateEndpointHost(req.Operation.Name, req.HTTPRequest.URL.Host)
	if err != nil {
		return s3shared.NewInvalidARNError(arn.AccessPointARN(a), err)
	}

	return nil
}

func (a accessPointEndpointBuilder) hostPrefixLabelValues() map[string]string {
	return map[string]string{
		accessPointPrefixLabel: arn.AccessPointARN(a).AccessPointName,
		accountIDPrefixLabel:   arn.AccessPointARN(a).AccountID,
	}
}

// outpostAccessPointEndpointBuilder represents the Endpoint builder for outpost access point arn.
type outpostAccessPointEndpointBuilder arn.OutpostAccessPointARN

// build builds an endpoint corresponding to the outpost access point arn.
//
// For building an endpoint from outpost access point arn, format used is:
// - Outpost access point endpoint format : {accesspointName}-{accountId}.{outpostId}.s3-outposts.{region}.{dnsSuffix}
// - example : myaccesspoint-012345678901.op-01234567890123456.s3-outposts.us-west-2.amazonaws.com
//
// Outpost AccessPoint Endpoint request are signed using "s3-outposts" as signing name.
//
func (o outpostAccessPointEndpointBuilder) build(req *request.Request) error {
	resolveRegion := o.Region
	resolveService := o.Service

	endpointsID := resolveService
	if resolveService == "s3-outposts" {
		endpointsID = "s3"
	}

	endpoint, err := resolveRegionalEndpoint(req, resolveRegion, endpointsID)
	if err != nil {
		return s3shared.NewFailedToResolveEndpointError(o,
			req.ClientInfo.PartitionID, resolveRegion, err)
	}

	if err = updateRequestEndpoint(req, endpoint.URL); err != nil {
		return err
	}

	// add url host as s3-outposts
	cfgHost := req.HTTPRequest.URL.Host
	if strings.HasPrefix(cfgHost, endpointsID) {
		req.HTTPRequest.URL.Host = resolveService + cfgHost[len(endpointsID):]
	}

	protocol.HostPrefixBuilder{
		Prefix:   outpostAccessPointPrefixTemplate,
		LabelsFn: o.hostPrefixLabelValues,
	}.Build(req)

	// set the signing region, name to resolved names from ARN
	redirectSigner(req, resolveService, resolveRegion)

	err = protocol.ValidateEndpointHost(req.Operation.Name, req.HTTPRequest.URL.Host)
	if err != nil {
		return s3shared.NewInvalidARNError(o, err)
	}

	return nil
}

func (o outpostAccessPointEndpointBuilder) hostPrefixLabelValues() map[string]string {
	return map[string]string{
		accessPointPrefixLabel: o.AccessPointName,
		accountIDPrefixLabel:   o.AccountID,
		outpostPrefixLabel:     o.OutpostID,
	}
}

func resolveRegionalEndpoint(r *request.Request, region string, endpointsID string) (endpoints.ResolvedEndpoint, error) {
	return r.Config.EndpointResolver.EndpointFor(endpointsID, region, func(opts *endpoints.Options) {
		opts.DisableSSL = aws.BoolValue(r.Config.DisableSSL)
		opts.UseDualStack = aws.BoolValue(r.Config.UseDualStack)
		opts.S3UsEast1RegionalEndpoint = endpoints.RegionalS3UsEast1Endpoint
	})
}

func updateRequestEndpoint(r *request.Request, endpoint string) (err error) {
	endpoint = endpoints.AddScheme(endpoint, aws.BoolValue(r.Config.DisableSSL))

	r.HTTPRequest.URL, err = url.Parse(endpoint + r.Operation.HTTPPath)
	if err != nil {
		return awserr.New(request.ErrCodeSerialization,
			"failed to parse endpoint URL", err)
	}

	return nil
}

// redirectSigner sets signing name, signing region for a request
func redirectSigner(req *request.Request, signingName string, signingRegion string) {
	req.ClientInfo.SigningName = signingName
	req.ClientInfo.SigningRegion = signingRegion
}
