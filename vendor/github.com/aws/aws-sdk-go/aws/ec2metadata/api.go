package ec2metadata

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/sdkuri"
)

// getToken uses the duration to return a token for EC2 metadata service,
// or an error if the request failed.
func (c *EC2Metadata) getToken(ctx aws.Context, duration time.Duration) (tokenOutput, error) {
	op := &request.Operation{
		Name:       "GetToken",
		HTTPMethod: "PUT",
		HTTPPath:   "/latest/api/token",
	}

	var output tokenOutput
	req := c.NewRequest(op, nil, &output)
	req.SetContext(ctx)

	// remove the fetch token handler from the request handlers to avoid infinite recursion
	req.Handlers.Sign.RemoveByName(fetchTokenHandlerName)

	// Swap the unmarshalMetadataHandler with unmarshalTokenHandler on this request.
	req.Handlers.Unmarshal.Swap(unmarshalMetadataHandlerName, unmarshalTokenHandler)

	ttl := strconv.FormatInt(int64(duration/time.Second), 10)
	req.HTTPRequest.Header.Set(ttlHeader, ttl)

	err := req.Send()

	// Errors with bad request status should be returned.
	if err != nil {
		err = awserr.NewRequestFailure(
			awserr.New(req.HTTPResponse.Status, http.StatusText(req.HTTPResponse.StatusCode), err),
			req.HTTPResponse.StatusCode, req.RequestID)
	}

	return output, err
}

// GetMetadata uses the path provided to request information from the EC2
// instance metadata service. The content will be returned as a string, or
// error if the request failed.
func (c *EC2Metadata) GetMetadata(p string) (string, error) {
	return c.GetMetadataWithContext(aws.BackgroundContext(), p)
}

// GetMetadataWithContext uses the path provided to request information from the EC2
// instance metadata service. The content will be returned as a string, or
// error if the request failed.
func (c *EC2Metadata) GetMetadataWithContext(ctx aws.Context, p string) (string, error) {
	op := &request.Operation{
		Name:       "GetMetadata",
		HTTPMethod: "GET",
		HTTPPath:   sdkuri.PathJoin("/latest/meta-data", p),
	}
	output := &metadataOutput{}

	req := c.NewRequest(op, nil, output)

	req.SetContext(ctx)

	err := req.Send()
	return output.Content, err
}

// GetUserData returns the userdata that was configured for the service. If
// there is no user-data setup for the EC2 instance a "NotFoundError" error
// code will be returned.
func (c *EC2Metadata) GetUserData() (string, error) {
	return c.GetUserDataWithContext(aws.BackgroundContext())
}

// GetUserDataWithContext returns the userdata that was configured for the service. If
// there is no user-data setup for the EC2 instance a "NotFoundError" error
// code will be returned.
func (c *EC2Metadata) GetUserDataWithContext(ctx aws.Context) (string, error) {
	op := &request.Operation{
		Name:       "GetUserData",
		HTTPMethod: "GET",
		HTTPPath:   "/latest/user-data",
	}

	output := &metadataOutput{}
	req := c.NewRequest(op, nil, output)
	req.SetContext(ctx)

	err := req.Send()
	return output.Content, err
}

// GetDynamicData uses the path provided to request information from the EC2
// instance metadata service for dynamic data. The content will be returned
// as a string, or error if the request failed.
func (c *EC2Metadata) GetDynamicData(p string) (string, error) {
	return c.GetDynamicDataWithContext(aws.BackgroundContext(), p)
}

// GetDynamicDataWithContext uses the path provided to request information from the EC2
// instance metadata service for dynamic data. The content will be returned
// as a string, or error if the request failed.
func (c *EC2Metadata) GetDynamicDataWithContext(ctx aws.Context, p string) (string, error) {
	op := &request.Operation{
		Name:       "GetDynamicData",
		HTTPMethod: "GET",
		HTTPPath:   sdkuri.PathJoin("/latest/dynamic", p),
	}

	output := &metadataOutput{}
	req := c.NewRequest(op, nil, output)
	req.SetContext(ctx)

	err := req.Send()
	return output.Content, err
}

// GetInstanceIdentityDocument retrieves an identity document describing an
// instance. Error is returned if the request fails or is unable to parse
// the response.
func (c *EC2Metadata) GetInstanceIdentityDocument() (EC2InstanceIdentityDocument, error) {
	return c.GetInstanceIdentityDocumentWithContext(aws.BackgroundContext())
}

// GetInstanceIdentityDocumentWithContext retrieves an identity document describing an
// instance. Error is returned if the request fails or is unable to parse
// the response.
func (c *EC2Metadata) GetInstanceIdentityDocumentWithContext(ctx aws.Context) (EC2InstanceIdentityDocument, error) {
	resp, err := c.GetDynamicDataWithContext(ctx, "instance-identity/document")
	if err != nil {
		return EC2InstanceIdentityDocument{},
			awserr.New("EC2MetadataRequestError",
				"failed to get EC2 instance identity document", err)
	}

	doc := EC2InstanceIdentityDocument{}
	if err := json.NewDecoder(strings.NewReader(resp)).Decode(&doc); err != nil {
		return EC2InstanceIdentityDocument{},
			awserr.New(request.ErrCodeSerialization,
				"failed to decode EC2 instance identity document", err)
	}

	return doc, nil
}

// IAMInfo retrieves IAM info from the metadata API
func (c *EC2Metadata) IAMInfo() (EC2IAMInfo, error) {
	return c.IAMInfoWithContext(aws.BackgroundContext())
}

// IAMInfoWithContext retrieves IAM info from the metadata API
func (c *EC2Metadata) IAMInfoWithContext(ctx aws.Context) (EC2IAMInfo, error) {
	resp, err := c.GetMetadataWithContext(ctx, "iam/info")
	if err != nil {
		return EC2IAMInfo{},
			awserr.New("EC2MetadataRequestError",
				"failed to get EC2 IAM info", err)
	}

	info := EC2IAMInfo{}
	if err := json.NewDecoder(strings.NewReader(resp)).Decode(&info); err != nil {
		return EC2IAMInfo{},
			awserr.New(request.ErrCodeSerialization,
				"failed to decode EC2 IAM info", err)
	}

	if info.Code != "Success" {
		errMsg := fmt.Sprintf("failed to get EC2 IAM Info (%s)", info.Code)
		return EC2IAMInfo{},
			awserr.New("EC2MetadataError", errMsg, nil)
	}

	return info, nil
}

// Region returns the region the instance is running in.
func (c *EC2Metadata) Region() (string, error) {
	return c.RegionWithContext(aws.BackgroundContext())
}

// RegionWithContext returns the region the instance is running in.
func (c *EC2Metadata) RegionWithContext(ctx aws.Context) (string, error) {
	ec2InstanceIdentityDocument, err := c.GetInstanceIdentityDocumentWithContext(ctx)
	if err != nil {
		return "", err
	}
	// extract region from the ec2InstanceIdentityDocument
	region := ec2InstanceIdentityDocument.Region
	if len(region) == 0 {
		return "", awserr.New("EC2MetadataError", "invalid region received for ec2metadata instance", nil)
	}
	// returns region
	return region, nil
}

// Available returns if the application has access to the EC2 Metadata service.
// Can be used to determine if application is running within an EC2 Instance and
// the metadata service is available.
func (c *EC2Metadata) Available() bool {
	return c.AvailableWithContext(aws.BackgroundContext())
}

// AvailableWithContext returns if the application has access to the EC2 Metadata service.
// Can be used to determine if application is running within an EC2 Instance and
// the metadata service is available.
func (c *EC2Metadata) AvailableWithContext(ctx aws.Context) bool {
	if _, err := c.GetMetadataWithContext(ctx, "instance-id"); err != nil {
		return false
	}

	return true
}

// An EC2IAMInfo provides the shape for unmarshaling
// an IAM info from the metadata API
type EC2IAMInfo struct {
	Code               string
	LastUpdated        time.Time
	InstanceProfileArn string
	InstanceProfileID  string
}

// An EC2InstanceIdentityDocument provides the shape for unmarshaling
// an instance identity document
type EC2InstanceIdentityDocument struct {
	DevpayProductCodes      []string  `json:"devpayProductCodes"`
	MarketplaceProductCodes []string  `json:"marketplaceProductCodes"`
	AvailabilityZone        string    `json:"availabilityZone"`
	PrivateIP               string    `json:"privateIp"`
	Version                 string    `json:"version"`
	Region                  string    `json:"region"`
	InstanceID              string    `json:"instanceId"`
	BillingProducts         []string  `json:"billingProducts"`
	InstanceType            string    `json:"instanceType"`
	AccountID               string    `json:"accountId"`
	PendingTime             time.Time `json:"pendingTime"`
	ImageID                 string    `json:"imageId"`
	KernelID                string    `json:"kernelId"`
	RamdiskID               string    `json:"ramdiskId"`
	Architecture            string    `json:"architecture"`
}
