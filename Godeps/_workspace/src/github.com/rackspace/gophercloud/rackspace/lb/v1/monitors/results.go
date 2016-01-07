package monitors

import (
	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
)

// Type represents the type of Monitor.
type Type string

// Useful constants.
const (
	CONNECT Type = "CONNECT"
	HTTP    Type = "HTTP"
	HTTPS   Type = "HTTPS"
)

// Monitor represents a health monitor API resource. A monitor comes in three
// forms: CONNECT, HTTP or HTTPS.
//
// A CONNECT monitor establishes a basic connection to each node on its defined
// port to ensure that the service is listening properly. The connect monitor
// is the most basic type of health check and does no post-processing or
// protocol-specific health checks.
//
// HTTP and HTTPS health monitors are generally considered more intelligent and
// powerful than CONNECT. It is capable of processing an HTTP or HTTPS response
// to determine the condition of a node. It supports the same basic properties
// as CONNECT and includes additional attributes that are used to evaluate the
// HTTP response.
type Monitor struct {
	// Number of permissible monitor failures before removing a node from
	// rotation.
	AttemptLimit int `mapstructure:"attemptsBeforeDeactivation"`

	// The minimum number of seconds to wait before executing the health monitor.
	Delay int

	// Maximum number of seconds to wait for a connection to be established
	// before timing out.
	Timeout int

	// Type of the health monitor.
	Type Type

	// A regular expression that will be used to evaluate the contents of the
	// body of the response.
	BodyRegex string

	// The name of a host for which the health monitors will check.
	HostHeader string

	// The HTTP path that will be used in the sample request.
	Path string

	// A regular expression that will be used to evaluate the HTTP status code
	// returned in the response.
	StatusRegex string
}

// UpdateResult represents the result of an Update operation.
type UpdateResult struct {
	gophercloud.ErrResult
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// DeleteResult represents the result of an Delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Extract interprets any GetResult as a Monitor.
func (r GetResult) Extract() (*Monitor, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		M Monitor `mapstructure:"healthMonitor"`
	}

	err := mapstructure.Decode(r.Body, &response)

	return &response.M, err
}
