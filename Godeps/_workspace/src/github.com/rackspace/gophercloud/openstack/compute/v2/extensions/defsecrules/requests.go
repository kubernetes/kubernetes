package defsecrules

import (
	"errors"
	"strings"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List will return a collection of default rules.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return DefaultRulePage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, rootURL(client), createPage)
}

// CreateOpts represents the configuration for adding a new default rule.
type CreateOpts struct {
	// Required - the lower bound of the port range that will be opened.
	FromPort int `json:"from_port"`

	// Required - the upper bound of the port range that will be opened.
	ToPort int `json:"to_port"`

	// Required - the protocol type that will be allowed, e.g. TCP.
	IPProtocol string `json:"ip_protocol"`

	// ONLY required if FromGroupID is blank. This represents the IP range that
	// will be the source of network traffic to your security group. Use
	// 0.0.0.0/0 to allow all IP addresses.
	CIDR string `json:"cidr,omitempty"`
}

// CreateOptsBuilder builds the create rule options into a serializable format.
type CreateOptsBuilder interface {
	ToRuleCreateMap() (map[string]interface{}, error)
}

// ToRuleCreateMap builds the create rule options into a serializable format.
func (opts CreateOpts) ToRuleCreateMap() (map[string]interface{}, error) {
	rule := make(map[string]interface{})

	if opts.FromPort == 0 && strings.ToUpper(opts.IPProtocol) != "ICMP" {
		return rule, errors.New("A FromPort must be set")
	}
	if opts.ToPort == 0 && strings.ToUpper(opts.IPProtocol) != "ICMP" {
		return rule, errors.New("A ToPort must be set")
	}
	if opts.IPProtocol == "" {
		return rule, errors.New("A IPProtocol must be set")
	}
	if opts.CIDR == "" {
		return rule, errors.New("A CIDR must be set")
	}

	rule["from_port"] = opts.FromPort
	rule["to_port"] = opts.ToPort
	rule["ip_protocol"] = opts.IPProtocol
	rule["cidr"] = opts.CIDR

	return map[string]interface{}{"security_group_default_rule": rule}, nil
}

// Create is the operation responsible for creating a new default rule.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var result CreateResult

	reqBody, err := opts.ToRuleCreateMap()
	if err != nil {
		result.Err = err
		return result
	}

	_, result.Err = client.Post(rootURL(client), reqBody, &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return result
}

// Get will return details for a particular default rule.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var result GetResult
	_, result.Err = client.Get(resourceURL(client, id), &result.Body, nil)
	return result
}

// Delete will permanently delete a default rule from the project.
func Delete(client *gophercloud.ServiceClient, id string) gophercloud.ErrResult {
	var result gophercloud.ErrResult
	_, result.Err = client.Delete(resourceURL(client, id), nil)
	return result
}
