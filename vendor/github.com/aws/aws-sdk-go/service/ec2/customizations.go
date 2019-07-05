package ec2

import (
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/sdkrand"
)

type retryer struct {
	client.DefaultRetryer
}

func (d retryer) RetryRules(r *request.Request) time.Duration {
	switch r.Operation.Name {
	case opModifyNetworkInterfaceAttribute:
		fallthrough
	case opAssignPrivateIpAddresses:
		return customRetryRule(r)
	default:
		return d.DefaultRetryer.RetryRules(r)
	}
}

func customRetryRule(r *request.Request) time.Duration {
	retryTimes := []time.Duration{
		time.Second,
		3 * time.Second,
		5 * time.Second,
	}

	count := r.RetryCount
	if count >= len(retryTimes) {
		count = len(retryTimes) - 1
	}

	minTime := int(retryTimes[count])
	return time.Duration(sdkrand.SeededRand.Intn(minTime) + minTime)
}

func setCustomRetryer(c *client.Client) {
	maxRetries := aws.IntValue(c.Config.MaxRetries)
	if c.Config.MaxRetries == nil || maxRetries == aws.UseServiceDefaultRetries {
		maxRetries = 3
	}

	c.Retryer = retryer{
		DefaultRetryer: client.DefaultRetryer{
			NumMaxRetries: maxRetries,
		},
	}
}

func init() {
	initClient = func(c *client.Client) {
		if c.Config.Retryer == nil {
			// Only override the retryer with a custom one if the config
			// does not already contain a retryer
			setCustomRetryer(c)
		}
	}
	initRequest = func(r *request.Request) {
		if r.Operation.Name == opCopySnapshot { // fill the PresignedURL parameter
			r.Handlers.Build.PushFront(fillPresignedURL)
		}
	}
}

func fillPresignedURL(r *request.Request) {
	if !r.ParamsFilled() {
		return
	}

	origParams := r.Params.(*CopySnapshotInput)

	// Stop if PresignedURL/DestinationRegion is set
	if origParams.PresignedUrl != nil || origParams.DestinationRegion != nil {
		return
	}

	origParams.DestinationRegion = r.Config.Region
	newParams := awsutil.CopyOf(r.Params).(*CopySnapshotInput)

	// Create a new request based on the existing request. We will use this to
	// presign the CopySnapshot request against the source region.
	cfg := r.Config.Copy(aws.NewConfig().
		WithEndpoint("").
		WithRegion(aws.StringValue(origParams.SourceRegion)))

	clientInfo := r.ClientInfo
	resolved, err := r.Config.EndpointResolver.EndpointFor(
		clientInfo.ServiceName, aws.StringValue(cfg.Region),
		func(opt *endpoints.Options) {
			opt.DisableSSL = aws.BoolValue(cfg.DisableSSL)
			opt.UseDualStack = aws.BoolValue(cfg.UseDualStack)
		},
	)
	if err != nil {
		r.Error = err
		return
	}

	clientInfo.Endpoint = resolved.URL
	clientInfo.SigningRegion = resolved.SigningRegion

	// Presign a CopySnapshot request with modified params
	req := request.New(*cfg, clientInfo, r.Handlers, r.Retryer, r.Operation, newParams, r.Data)
	url, err := req.Presign(5 * time.Minute) // 5 minutes should be enough.
	if err != nil {                          // bubble error back up to original request
		r.Error = err
		return
	}

	// We have our URL, set it on params
	origParams.PresignedUrl = &url
}
