package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"net/http"
	"net/url"
	"strconv"
)

// ServiceProperties represents the storage account service properties
type ServiceProperties struct {
	Logging               *Logging
	HourMetrics           *Metrics
	MinuteMetrics         *Metrics
	Cors                  *Cors
	DeleteRetentionPolicy *RetentionPolicy // blob storage only
	StaticWebsite         *StaticWebsite   // blob storage only
}

// Logging represents the Azure Analytics Logging settings
type Logging struct {
	Version         string
	Delete          bool
	Read            bool
	Write           bool
	RetentionPolicy *RetentionPolicy
}

// RetentionPolicy indicates if retention is enabled and for how many days
type RetentionPolicy struct {
	Enabled bool
	Days    *int
}

// Metrics provide request statistics.
type Metrics struct {
	Version         string
	Enabled         bool
	IncludeAPIs     *bool
	RetentionPolicy *RetentionPolicy
}

// Cors includes all the CORS rules
type Cors struct {
	CorsRule []CorsRule
}

// CorsRule includes all settings for a Cors rule
type CorsRule struct {
	AllowedOrigins  string
	AllowedMethods  string
	MaxAgeInSeconds int
	ExposedHeaders  string
	AllowedHeaders  string
}

// StaticWebsite - The properties that enable an account to host a static website
type StaticWebsite struct {
	// Enabled - Indicates whether this account is hosting a static website
	Enabled bool
	// IndexDocument - The default name of the index page under each directory
	IndexDocument *string
	// ErrorDocument404Path - The absolute path of the custom 404 page
	ErrorDocument404Path *string
}

func (c Client) getServiceProperties(service string, auth authentication) (*ServiceProperties, error) {
	query := url.Values{
		"restype": {"service"},
		"comp":    {"properties"},
	}
	uri := c.getEndpoint(service, "", query)
	headers := c.getStandardHeaders()

	resp, err := c.exec(http.MethodGet, uri, headers, nil, auth)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if err := checkRespCode(resp, []int{http.StatusOK}); err != nil {
		return nil, err
	}

	var out ServiceProperties
	err = xmlUnmarshal(resp.Body, &out)
	if err != nil {
		return nil, err
	}

	return &out, nil
}

func (c Client) setServiceProperties(props ServiceProperties, service string, auth authentication) error {
	query := url.Values{
		"restype": {"service"},
		"comp":    {"properties"},
	}
	uri := c.getEndpoint(service, "", query)

	// Ideally, StorageServiceProperties would be the output struct
	// This is to avoid golint stuttering, while generating the correct XML
	type StorageServiceProperties struct {
		Logging               *Logging
		HourMetrics           *Metrics
		MinuteMetrics         *Metrics
		Cors                  *Cors
		DeleteRetentionPolicy *RetentionPolicy
		StaticWebsite         *StaticWebsite
	}
	input := StorageServiceProperties{
		Logging:       props.Logging,
		HourMetrics:   props.HourMetrics,
		MinuteMetrics: props.MinuteMetrics,
		Cors:          props.Cors,
	}
	// only set these fields for blob storage else it's invalid XML
	if service == blobServiceName {
		input.DeleteRetentionPolicy = props.DeleteRetentionPolicy
		input.StaticWebsite = props.StaticWebsite
	}

	body, length, err := xmlMarshal(input)
	if err != nil {
		return err
	}

	headers := c.getStandardHeaders()
	headers["Content-Length"] = strconv.Itoa(length)

	resp, err := c.exec(http.MethodPut, uri, headers, body, auth)
	if err != nil {
		return err
	}
	defer drainRespBody(resp)
	return checkRespCode(resp, []int{http.StatusAccepted})
}
