// +build go1.7

// Package management provides the main API client to construct other clients
// and make requests to the Microsoft Azure Service Management REST API.
package management

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"errors"
	"fmt"
	"runtime"
	"time"
)

var (
	DefaultUserAgent = userAgent()
)

const (
	DefaultAzureManagementURL    = "https://management.core.windows.net"
	DefaultOperationPollInterval = time.Second * 30
	DefaultAPIVersion            = "2014-10-01"

	errPublishSettingsConfiguration       = "PublishSettingsFilePath is set. Consequently ManagementCertificatePath and SubscriptionId must not be set."
	errManagementCertificateConfiguration = "Both ManagementCertificatePath and SubscriptionId should be set, and PublishSettingsFilePath must not be set."
	errParamNotSpecified                  = "Parameter %s is not specified."
)

type client struct {
	publishSettings publishSettings
	config          ClientConfig
}

// Client is the base Azure Service Management API client instance that
// can be used to construct client instances for various services.
type Client interface {
	// SendAzureGetRequest sends a request to the management API using the HTTP GET method
	// and returns the response body or an error.
	SendAzureGetRequest(url string) ([]byte, error)

	// SendAzurePostRequest sends a request to the management API using the HTTP POST method
	// and returns the request ID or an error.
	SendAzurePostRequest(url string, data []byte) (OperationID, error)

	// SendAzurePostRequestWithReturnedResponse sends a request to the management API using
	// the HTTP POST method and returns the response body or an error.
	SendAzurePostRequestWithReturnedResponse(url string, data []byte) ([]byte, error)

	// SendAzurePutRequest sends a request to the management API using the HTTP PUT method
	// and returns the request ID or an error. The content type can be specified, however
	// if an empty string is passed, the default of "application/xml" will be used.
	SendAzurePutRequest(url, contentType string, data []byte) (OperationID, error)

	// SendAzureDeleteRequest sends a request to the management API using the HTTP DELETE method
	// and returns the request ID or an error.
	SendAzureDeleteRequest(url string) (OperationID, error)

	// GetOperationStatus gets the status of operation with given Operation ID.
	// WaitForOperation utility method can be used for polling for operation status.
	GetOperationStatus(operationID OperationID) (GetOperationStatusResponse, error)

	// WaitForOperation polls the Azure API for given operation ID indefinitely
	// until the operation is completed with either success or failure.
	// It is meant to be used for waiting for the result of the methods that
	// return an OperationID value (meaning a long running operation has started).
	//
	// Cancellation of the polling loop (for instance, timing out) is done through
	// cancel channel. If the user does not want to cancel, a nil chan can be provided.
	// To cancel the method, it is recommended to close the channel provided to this
	// method.
	//
	// If the operation was not successful or cancelling is signaled, an error
	// is returned.
	WaitForOperation(operationID OperationID, cancel chan struct{}) error
}

// ClientConfig provides a configuration for use by a Client.
type ClientConfig struct {
	ManagementURL         string
	OperationPollInterval time.Duration
	UserAgent             string
	APIVersion            string
}

// NewAnonymousClient creates a new azure.Client with no credentials set.
func NewAnonymousClient() Client {
	return client{}
}

// DefaultConfig returns the default client configuration used to construct
// a client. This value can be used to make modifications on the default API
// configuration.
func DefaultConfig() ClientConfig {
	return ClientConfig{
		ManagementURL:         DefaultAzureManagementURL,
		OperationPollInterval: DefaultOperationPollInterval,
		APIVersion:            DefaultAPIVersion,
		UserAgent:             DefaultUserAgent,
	}
}

// NewClient creates a new Client using the given subscription ID and
// management certificate.
func NewClient(subscriptionID string, managementCert []byte) (Client, error) {
	return NewClientFromConfig(subscriptionID, managementCert, DefaultConfig())
}

// NewClientFromConfig creates a new Client using a given ClientConfig.
func NewClientFromConfig(subscriptionID string, managementCert []byte, config ClientConfig) (Client, error) {
	return makeClient(subscriptionID, managementCert, config)
}

func makeClient(subscriptionID string, managementCert []byte, config ClientConfig) (Client, error) {
	var c client

	if subscriptionID == "" {
		return c, errors.New("azure: subscription ID required")
	}

	if len(managementCert) == 0 {
		return c, errors.New("azure: management certificate required")
	}

	publishSettings := publishSettings{
		SubscriptionID:   subscriptionID,
		SubscriptionCert: managementCert,
		SubscriptionKey:  managementCert,
	}

	// Validate client configuration
	switch {
	case config.ManagementURL == "":
		return c, errors.New("azure: base URL required")
	case config.OperationPollInterval <= 0:
		return c, errors.New("azure: operation polling interval must be a positive duration")
	case config.APIVersion == "":
		return c, errors.New("azure: client configuration must specify an API version")
	case config.UserAgent == "":
		config.UserAgent = DefaultUserAgent
	}

	return client{
		publishSettings: publishSettings,
		config:          config,
	}, nil
}

func userAgent() string {
	return fmt.Sprintf("Go/%s (%s-%s) Azure-SDK-For-Go/%s asm/%s",
		runtime.Version(),
		runtime.GOARCH,
		runtime.GOOS,
		sdkVersion,
		DefaultAPIVersion)
}
