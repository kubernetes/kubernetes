package monitors

import (
	"errors"

	"github.com/rackspace/gophercloud"
)

var (
	errAttemptLimit = errors.New("AttemptLimit field must be an int greater than 1 and less than 10")
	errDelay        = errors.New("Delay field must be an int greater than 1 and less than 10")
	errTimeout      = errors.New("Timeout field must be an int greater than 1 and less than 10")
)

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Update operation in this package.
type UpdateOptsBuilder interface {
	ToMonitorUpdateMap() (map[string]interface{}, error)
}

// UpdateConnectMonitorOpts represents the options needed to update a CONNECT
// monitor.
type UpdateConnectMonitorOpts struct {
	// Required - number of permissible monitor failures before removing a node
	// from rotation. Must be a number between 1 and 10.
	AttemptLimit int

	// Required - the minimum number of seconds to wait before executing the
	// health monitor. Must be a number between 1 and 3600.
	Delay int

	// Required - maximum number of seconds to wait for a connection to be
	// established before timing out. Must be a number between 1 and 300.
	Timeout int
}

// ToMonitorUpdateMap produces a map for updating CONNECT monitors.
func (opts UpdateConnectMonitorOpts) ToMonitorUpdateMap() (map[string]interface{}, error) {
	type m map[string]interface{}

	if !gophercloud.IntWithinRange(opts.AttemptLimit, 1, 10) {
		return m{}, errAttemptLimit
	}
	if !gophercloud.IntWithinRange(opts.Delay, 1, 3600) {
		return m{}, errDelay
	}
	if !gophercloud.IntWithinRange(opts.Timeout, 1, 300) {
		return m{}, errTimeout
	}

	return m{"healthMonitor": m{
		"attemptsBeforeDeactivation": opts.AttemptLimit,
		"delay":   opts.Delay,
		"timeout": opts.Timeout,
		"type":    CONNECT,
	}}, nil
}

// UpdateHTTPMonitorOpts represents the options needed to update a HTTP monitor.
type UpdateHTTPMonitorOpts struct {
	// Required - number of permissible monitor failures before removing a node
	// from rotation. Must be a number between 1 and 10.
	AttemptLimit int `mapstructure:"attemptsBeforeDeactivation"`

	// Required - the minimum number of seconds to wait before executing the
	// health monitor. Must be a number between 1 and 3600.
	Delay int

	// Required - maximum number of seconds to wait for a connection to be
	// established before timing out. Must be a number between 1 and 300.
	Timeout int

	// Required - a regular expression that will be used to evaluate the contents
	// of the body of the response.
	BodyRegex string

	// Required - the HTTP path that will be used in the sample request.
	Path string

	// Required - a regular expression that will be used to evaluate the HTTP
	// status code returned in the response.
	StatusRegex string

	// Optional - the name of a host for which the health monitors will check.
	HostHeader string

	// Required - either HTTP or HTTPS
	Type Type
}

// ToMonitorUpdateMap produces a map for updating HTTP(S) monitors.
func (opts UpdateHTTPMonitorOpts) ToMonitorUpdateMap() (map[string]interface{}, error) {
	type m map[string]interface{}

	if !gophercloud.IntWithinRange(opts.AttemptLimit, 1, 10) {
		return m{}, errAttemptLimit
	}
	if !gophercloud.IntWithinRange(opts.Delay, 1, 3600) {
		return m{}, errDelay
	}
	if !gophercloud.IntWithinRange(opts.Timeout, 1, 300) {
		return m{}, errTimeout
	}
	if opts.Type != HTTP && opts.Type != HTTPS {
		return m{}, errors.New("Type must either by HTTP or HTTPS")
	}
	if opts.BodyRegex == "" {
		return m{}, errors.New("BodyRegex is a required field")
	}
	if opts.Path == "" {
		return m{}, errors.New("Path is a required field")
	}
	if opts.StatusRegex == "" {
		return m{}, errors.New("StatusRegex is a required field")
	}

	json := m{
		"attemptsBeforeDeactivation": opts.AttemptLimit,
		"delay":       opts.Delay,
		"timeout":     opts.Timeout,
		"type":        opts.Type,
		"bodyRegex":   opts.BodyRegex,
		"path":        opts.Path,
		"statusRegex": opts.StatusRegex,
	}

	if opts.HostHeader != "" {
		json["hostHeader"] = opts.HostHeader
	}

	return m{"healthMonitor": json}, nil
}

// Update is the operation responsible for updating a health monitor.
func Update(c *gophercloud.ServiceClient, id int, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToMonitorUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Put(rootURL(c, id), reqBody, nil, nil)
	return res
}

// Get is the operation responsible for showing details of a health monitor.
func Get(c *gophercloud.ServiceClient, id int) GetResult {
	var res GetResult
	_, res.Err = c.Get(rootURL(c, id), &res.Body, nil)
	return res
}

// Delete is the operation responsible for deleting a health monitor.
func Delete(c *gophercloud.ServiceClient, id int) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(rootURL(c, id), nil)
	return res
}
