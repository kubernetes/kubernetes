package s3shared

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/internal/s3shared/arn"
)

const (
	invalidARNErrorErrCode    = "InvalidARNError"
	configurationErrorErrCode = "ConfigurationError"
)

// InvalidARNError denotes the error for Invalid ARN
type InvalidARNError struct {
	message  string
	resource arn.Resource
	origErr  error
}

// Error returns the InvalidARNError
func (e InvalidARNError) Error() string {
	var extra string
	if e.resource != nil {
		extra = "ARN: " + e.resource.String()
	}
	return awserr.SprintError(e.Code(), e.Message(), extra, e.origErr)
}

// Code returns the invalid ARN error code
func (e InvalidARNError) Code() string {
	return invalidARNErrorErrCode
}

// Message returns the message for Invalid ARN error
func (e InvalidARNError) Message() string {
	return e.message
}

// OrigErr is the original error wrapped by Invalid ARN Error
func (e InvalidARNError) OrigErr() error {
	return e.origErr
}

// NewInvalidARNError denotes invalid arn error
func NewInvalidARNError(resource arn.Resource, err error) InvalidARNError {
	return InvalidARNError{
		message:  "invalid ARN",
		origErr:  err,
		resource: resource,
	}
}

// NewInvalidARNWithCustomEndpointError ARN not supported for custom clients endpoints
func NewInvalidARNWithCustomEndpointError(resource arn.Resource, err error) InvalidARNError {
	return InvalidARNError{
		message:  "resource ARN not supported with custom client endpoints",
		origErr:  err,
		resource: resource,
	}
}

// NewInvalidARNWithUnsupportedPartitionError ARN not supported for the target partition
func NewInvalidARNWithUnsupportedPartitionError(resource arn.Resource, err error) InvalidARNError {
	return InvalidARNError{
		message:  "resource ARN not supported for the target ARN partition",
		origErr:  err,
		resource: resource,
	}
}

// NewInvalidARNWithFIPSError ARN not supported for FIPS region
func NewInvalidARNWithFIPSError(resource arn.Resource, err error) InvalidARNError {
	return InvalidARNError{
		message:  "resource ARN not supported for FIPS region",
		resource: resource,
		origErr:  err,
	}
}

// ConfigurationError is used to denote a client configuration error
type ConfigurationError struct {
	message           string
	resource          arn.Resource
	clientPartitionID string
	clientRegion      string
	origErr           error
}

// Error returns the Configuration error string
func (e ConfigurationError) Error() string {
	extra := fmt.Sprintf("ARN: %s, client partition: %s, client region: %s",
		e.resource, e.clientPartitionID, e.clientRegion)

	return awserr.SprintError(e.Code(), e.Message(), extra, e.origErr)
}

// Code returns configuration error's error-code
func (e ConfigurationError) Code() string {
	return configurationErrorErrCode
}

// Message returns the configuration error message
func (e ConfigurationError) Message() string {
	return e.message
}

// OrigErr is the original error wrapped by Configuration Error
func (e ConfigurationError) OrigErr() error {
	return e.origErr
}

// NewClientPartitionMismatchError  stub
func NewClientPartitionMismatchError(resource arn.Resource, clientPartitionID, clientRegion string, err error) ConfigurationError {
	return ConfigurationError{
		message:           "client partition does not match provided ARN partition",
		origErr:           err,
		resource:          resource,
		clientPartitionID: clientPartitionID,
		clientRegion:      clientRegion,
	}
}

// NewClientRegionMismatchError denotes cross region access error
func NewClientRegionMismatchError(resource arn.Resource, clientPartitionID, clientRegion string, err error) ConfigurationError {
	return ConfigurationError{
		message:           "client region does not match provided ARN region",
		origErr:           err,
		resource:          resource,
		clientPartitionID: clientPartitionID,
		clientRegion:      clientRegion,
	}
}

// NewFailedToResolveEndpointError denotes endpoint resolving error
func NewFailedToResolveEndpointError(resource arn.Resource, clientPartitionID, clientRegion string, err error) ConfigurationError {
	return ConfigurationError{
		message:           "endpoint resolver failed to find an endpoint for the provided ARN region",
		origErr:           err,
		resource:          resource,
		clientPartitionID: clientPartitionID,
		clientRegion:      clientRegion,
	}
}

// NewClientConfiguredForFIPSError denotes client config error for unsupported cross region FIPS access
func NewClientConfiguredForFIPSError(resource arn.Resource, clientPartitionID, clientRegion string, err error) ConfigurationError {
	return ConfigurationError{
		message:           "client configured for fips but cross-region resource ARN provided",
		origErr:           err,
		resource:          resource,
		clientPartitionID: clientPartitionID,
		clientRegion:      clientRegion,
	}
}

// NewClientConfiguredForAccelerateError denotes client config error for unsupported S3 accelerate
func NewClientConfiguredForAccelerateError(resource arn.Resource, clientPartitionID, clientRegion string, err error) ConfigurationError {
	return ConfigurationError{
		message:           "client configured for S3 Accelerate but is not supported with resource ARN",
		origErr:           err,
		resource:          resource,
		clientPartitionID: clientPartitionID,
		clientRegion:      clientRegion,
	}
}

// NewClientConfiguredForCrossRegionFIPSError denotes client config error for unsupported cross region FIPS request
func NewClientConfiguredForCrossRegionFIPSError(resource arn.Resource, clientPartitionID, clientRegion string, err error) ConfigurationError {
	return ConfigurationError{
		message:           "client configured for FIPS with cross-region enabled but is supported with cross-region resource ARN",
		origErr:           err,
		resource:          resource,
		clientPartitionID: clientPartitionID,
		clientRegion:      clientRegion,
	}
}

// NewClientConfiguredForDualStackError denotes client config error for unsupported S3 Dual-stack
func NewClientConfiguredForDualStackError(resource arn.Resource, clientPartitionID, clientRegion string, err error) ConfigurationError {
	return ConfigurationError{
		message:           "client configured for S3 Dual-stack but is not supported with resource ARN",
		origErr:           err,
		resource:          resource,
		clientPartitionID: clientPartitionID,
		clientRegion:      clientRegion,
	}
}
