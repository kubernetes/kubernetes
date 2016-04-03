package sessions

import (
	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
)

// Type represents the type of session persistence being used.
type Type string

const (
	// HTTPCOOKIE is a session persistence mechanism that inserts an HTTP cookie
	// and is used to determine the destination back-end node. This is supported
	// for HTTP load balancing only.
	HTTPCOOKIE Type = "HTTP_COOKIE"

	// SOURCEIP is a session persistence mechanism that keeps track of the source
	// IP address that is mapped and is able to determine the destination
	// back-end node. This is supported for HTTPS pass-through and non-HTTP load
	// balancing only.
	SOURCEIP Type = "SOURCE_IP"
)

// SessionPersistence indicates how a load balancer is using session persistence
type SessionPersistence struct {
	Type Type `mapstructure:"persistenceType"`
}

// EnableResult represents the result of an enable operation.
type EnableResult struct {
	gophercloud.ErrResult
}

// DisableResult represents the result of a disable operation.
type DisableResult struct {
	gophercloud.ErrResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract interprets a GetResult as an SP, if possible.
func (r GetResult) Extract() (*SessionPersistence, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		SP SessionPersistence `mapstructure:"sessionPersistence"`
	}

	err := mapstructure.Decode(r.Body, &response)

	return &response.SP, err
}
