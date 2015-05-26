package aws

import (
	"io"
	"net/http"
	"os"

	"github.com/awslabs/aws-sdk-go/aws/credentials"
)

// DefaultChainCredentials is a Credentials which will find the first available
// credentials Value from the list of Providers.
//
// This should be used in the default case. Once the type of credentials are
// known switching to the specific Credentials will be more efficient.
var DefaultChainCredentials = credentials.NewChainCredentials(
	[]credentials.Provider{
		&credentials.EnvProvider{},
		&credentials.SharedCredentialsProvider{Filename: "", Profile: ""},
		&credentials.EC2RoleProvider{},
	})

// The default number of retries for a service. The value of -1 indicates that
// the service specific retry default will be used.
const DefaultRetries = -1

// DefaultConfig is the default all service configuration will be based off of.
var DefaultConfig = &Config{
	Credentials:             DefaultChainCredentials,
	Endpoint:                "",
	Region:                  os.Getenv("AWS_REGION"),
	DisableSSL:              false,
	ManualSend:              false,
	HTTPClient:              http.DefaultClient,
	LogHTTPBody:             false,
	LogLevel:                0,
	Logger:                  os.Stdout,
	MaxRetries:              DefaultRetries,
	DisableParamValidation:  false,
	DisableComputeChecksums: false,
	S3ForcePathStyle:        false,
}

// A Config provides service configuration
type Config struct {
	Credentials             *credentials.Credentials
	Endpoint                string
	Region                  string
	DisableSSL              bool
	ManualSend              bool
	HTTPClient              *http.Client
	LogHTTPBody             bool
	LogLevel                uint
	Logger                  io.Writer
	MaxRetries              int
	DisableParamValidation  bool
	DisableComputeChecksums bool
	S3ForcePathStyle        bool
}

// Copy will return a shallow copy of the Config object.
func (c Config) Copy() Config {
	dst := Config{}
	dst.Credentials = c.Credentials
	dst.Endpoint = c.Endpoint
	dst.Region = c.Region
	dst.DisableSSL = c.DisableSSL
	dst.ManualSend = c.ManualSend
	dst.HTTPClient = c.HTTPClient
	dst.LogLevel = c.LogLevel
	dst.Logger = c.Logger
	dst.MaxRetries = c.MaxRetries
	dst.DisableParamValidation = c.DisableParamValidation
	dst.DisableComputeChecksums = c.DisableComputeChecksums
	dst.S3ForcePathStyle = c.S3ForcePathStyle

	return dst
}

// Merge merges the newcfg attribute values into this Config. Each attribute
// will be merged into this config if the newcfg attribute's value is non-zero.
// Due to this, newcfg attributes with zero values cannot be merged in. For
// example bool attributes cannot be cleared using Merge, and must be explicitly
// set on the Config structure.
func (c Config) Merge(newcfg *Config) *Config {
	cfg := Config{}

	if newcfg != nil && newcfg.Credentials != nil {
		cfg.Credentials = newcfg.Credentials
	} else {
		cfg.Credentials = c.Credentials
	}

	if newcfg != nil && newcfg.Endpoint != "" {
		cfg.Endpoint = newcfg.Endpoint
	} else {
		cfg.Endpoint = c.Endpoint
	}

	if newcfg != nil && newcfg.Region != "" {
		cfg.Region = newcfg.Region
	} else {
		cfg.Region = c.Region
	}

	if newcfg != nil && newcfg.DisableSSL {
		cfg.DisableSSL = newcfg.DisableSSL
	} else {
		cfg.DisableSSL = c.DisableSSL
	}

	if newcfg != nil && newcfg.ManualSend {
		cfg.ManualSend = newcfg.ManualSend
	} else {
		cfg.ManualSend = c.ManualSend
	}

	if newcfg != nil && newcfg.HTTPClient != nil {
		cfg.HTTPClient = newcfg.HTTPClient
	} else {
		cfg.HTTPClient = c.HTTPClient
	}

	if newcfg != nil && newcfg.LogHTTPBody {
		cfg.LogHTTPBody = newcfg.LogHTTPBody
	} else {
		cfg.LogHTTPBody = c.LogHTTPBody
	}

	if newcfg != nil && newcfg.LogLevel != 0 {
		cfg.LogLevel = newcfg.LogLevel
	} else {
		cfg.LogLevel = c.LogLevel
	}

	if newcfg != nil && newcfg.Logger != nil {
		cfg.Logger = newcfg.Logger
	} else {
		cfg.Logger = c.Logger
	}

	if newcfg != nil && newcfg.MaxRetries != DefaultRetries {
		cfg.MaxRetries = newcfg.MaxRetries
	} else {
		cfg.MaxRetries = c.MaxRetries
	}

	if newcfg != nil && newcfg.DisableParamValidation {
		cfg.DisableParamValidation = newcfg.DisableParamValidation
	} else {
		cfg.DisableParamValidation = c.DisableParamValidation
	}

	if newcfg != nil && newcfg.DisableComputeChecksums {
		cfg.DisableComputeChecksums = newcfg.DisableComputeChecksums
	} else {
		cfg.DisableComputeChecksums = c.DisableComputeChecksums
	}

	if newcfg != nil && newcfg.S3ForcePathStyle {
		cfg.S3ForcePathStyle = newcfg.S3ForcePathStyle
	} else {
		cfg.S3ForcePathStyle = c.S3ForcePathStyle
	}

	return &cfg
}
