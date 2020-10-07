package ec2

import (
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
)

const (
	// customRetryerMinRetryDelay sets min retry delay
	customRetryerMinRetryDelay = 1 * time.Second

	// customRetryerMaxRetryDelay sets max retry delay
	customRetryerMaxRetryDelay = 8 * time.Second
)

func init() {
	initRequest = func(r *request.Request) {
		if r.Operation.Name == opCopySnapshot { // fill the PresignedURL parameter
			r.Handlers.Build.PushFront(fillPresignedURL)
		}

		// only set the retryer on request if config doesn't have a retryer
		if r.Config.Retryer == nil && (r.Operation.Name == opModifyNetworkInterfaceAttribute || r.Operation.Name == opAssignPrivateIpAddresses) {
			maxRetries := client.DefaultRetryerMaxNumRetries
			if m := r.Config.MaxRetries; m != nil && *m != aws.UseServiceDefaultRetries {
				maxRetries = *m
			}
			r.Retryer = client.DefaultRetryer{
				NumMaxRetries:    maxRetries,
				MinRetryDelay:    customRetryerMinRetryDelay,
				MinThrottleDelay: customRetryerMinRetryDelay,
				MaxRetryDelay:    customRetryerMaxRetryDelay,
				MaxThrottleDelay: customRetryerMaxRetryDelay,
			}
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
