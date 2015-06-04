package ec2

import (
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awsutil"
)

func init() {
	initRequest = func(r *aws.Request) {
		if r.Operation == opCopySnapshot { // fill the PresignedURL parameter
			r.Handlers.Build.PushFront(fillPresignedURL)
		}
	}
}

func fillPresignedURL(r *aws.Request) {
	if !r.ParamsFilled() {
		return
	}

	params := r.Params.(*CopySnapshotInput)

	// Stop if PresignedURL/DestinationRegion is set
	if params.PresignedURL != nil || params.DestinationRegion != nil {
		return
	}

	// First generate a copy of parameters
	r.Params = awsutil.CopyOf(r.Params)
	params = r.Params.(*CopySnapshotInput)

	// Set destination region. Avoids infinite handler loop.
	// Also needed to sign sub-request.
	params.DestinationRegion = &r.Service.Config.Region

	// Create a new client pointing at source region.
	// We will use this to presign the CopySnapshot request against
	// the source region
	config := r.Service.Config.Copy()

	config.Endpoint = ""
	config.Region = *params.SourceRegion
	client := New(&config)

	// Presign a CopySnapshot request with modified params
	req, _ := client.CopySnapshotRequest(params)
	url, err := req.Presign(300 * time.Second) // 5 minutes should be enough.

	if err != nil { // bubble error back up to original request
		r.Error = err
	}

	// We have our URL, set it on params
	params.PresignedURL = &url
}
