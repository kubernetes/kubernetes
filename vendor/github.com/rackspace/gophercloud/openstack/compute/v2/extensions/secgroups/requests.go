package secgroups

import (
	"errors"
	"strings"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

func commonList(client *gophercloud.ServiceClient, url string) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return SecurityGroupPage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, url, createPage)
}

// List will return a collection of all the security groups for a particular
// tenant.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return commonList(client, rootURL(client))
}

// ListByServer will return a collection of all the security groups which are
// associated with a particular server.
func ListByServer(client *gophercloud.ServiceClient, serverID string) pagination.Pager {
	return commonList(client, listByServerURL(client, serverID))
}

// GroupOpts is the underlying struct responsible for creating or updating
// security groups. It therefore represents the mutable attributes of a
// security group.
type GroupOpts struct {
	// Required - the name of your security group.
	Name string `json:"name"`

	// Required - the description of your security group.
	Description string `json:"description"`
}

// CreateOpts is the struct responsible for creating a security group.
type CreateOpts GroupOpts

// CreateOptsBuilder builds the create options into a serializable format.
type CreateOptsBuilder interface {
	ToSecGroupCreateMap() (map[string]interface{}, error)
}

var (
	errName = errors.New("Name is a required field")
	errDesc = errors.New("Description is a required field")
)

// ToSecGroupCreateMap builds the create options into a serializable format.
func (opts CreateOpts) ToSecGroupCreateMap() (map[string]interface{}, error) {
	sg := make(map[string]interface{})

	if opts.Name == "" {
		return sg, errName
	}
	if opts.Description == "" {
		return sg, errDesc
	}

	sg["name"] = opts.Name
	sg["description"] = opts.Description

	return map[string]interface{}{"security_group": sg}, nil
}

// Create will create a new security group.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var result CreateResult

	reqBody, err := opts.ToSecGroupCreateMap()
	if err != nil {
		result.Err = err
		return result
	}

	_, result.Err = client.Post(rootURL(client), reqBody, &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return result
}

// UpdateOpts is the struct responsible for updating an existing security group.
type UpdateOpts GroupOpts

// UpdateOptsBuilder builds the update options into a serializable format.
type UpdateOptsBuilder interface {
	ToSecGroupUpdateMap() (map[string]interface{}, error)
}

// ToSecGroupUpdateMap builds the update options into a serializable format.
func (opts UpdateOpts) ToSecGroupUpdateMap() (map[string]interface{}, error) {
	sg := make(map[string]interface{})

	if opts.Name == "" {
		return sg, errName
	}
	if opts.Description == "" {
		return sg, errDesc
	}

	sg["name"] = opts.Name
	sg["description"] = opts.Description

	return map[string]interface{}{"security_group": sg}, nil
}

// Update will modify the mutable properties of a security group, notably its
// name and description.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var result UpdateResult

	reqBody, err := opts.ToSecGroupUpdateMap()
	if err != nil {
		result.Err = err
		return result
	}

	_, result.Err = client.Put(resourceURL(client, id), reqBody, &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return result
}

// Get will return details for a particular security group.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var result GetResult
	_, result.Err = client.Get(resourceURL(client, id), &result.Body, nil)
	return result
}

// Delete will permanently delete a security group from the project.
func Delete(client *gophercloud.ServiceClient, id string) gophercloud.ErrResult {
	var result gophercloud.ErrResult
	_, result.Err = client.Delete(resourceURL(client, id), nil)
	return result
}

// CreateRuleOpts represents the configuration for adding a new rule to an
// existing security group.
type CreateRuleOpts struct {
	// Required - the ID of the group that this rule will be added to.
	ParentGroupID string `json:"parent_group_id"`

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

	// ONLY required if CIDR is blank. This value represents the ID of a group
	// that forwards traffic to the parent group. So, instead of accepting
	// network traffic from an entire IP range, you can instead refine the
	// inbound source by an existing security group.
	FromGroupID string `json:"group_id,omitempty"`
}

// CreateRuleOptsBuilder builds the create rule options into a serializable format.
type CreateRuleOptsBuilder interface {
	ToRuleCreateMap() (map[string]interface{}, error)
}

// ToRuleCreateMap builds the create rule options into a serializable format.
func (opts CreateRuleOpts) ToRuleCreateMap() (map[string]interface{}, error) {
	rule := make(map[string]interface{})

	if opts.ParentGroupID == "" {
		return rule, errors.New("A ParentGroupID must be set")
	}
	if opts.FromPort == 0 && strings.ToUpper(opts.IPProtocol) != "ICMP" {
		return rule, errors.New("A FromPort must be set")
	}
	if opts.ToPort == 0 && strings.ToUpper(opts.IPProtocol) != "ICMP" {
		return rule, errors.New("A ToPort must be set")
	}
	if opts.IPProtocol == "" {
		return rule, errors.New("A IPProtocol must be set")
	}
	if opts.CIDR == "" && opts.FromGroupID == "" {
		return rule, errors.New("A CIDR or FromGroupID must be set")
	}

	rule["parent_group_id"] = opts.ParentGroupID
	rule["from_port"] = opts.FromPort
	rule["to_port"] = opts.ToPort
	rule["ip_protocol"] = opts.IPProtocol

	if opts.CIDR != "" {
		rule["cidr"] = opts.CIDR
	}
	if opts.FromGroupID != "" {
		rule["group_id"] = opts.FromGroupID
	}

	return map[string]interface{}{"security_group_rule": rule}, nil
}

// CreateRule will add a new rule to an existing security group (whose ID is
// specified in CreateRuleOpts). You have the option of controlling inbound
// traffic from either an IP range (CIDR) or from another security group.
func CreateRule(client *gophercloud.ServiceClient, opts CreateRuleOptsBuilder) CreateRuleResult {
	var result CreateRuleResult

	reqBody, err := opts.ToRuleCreateMap()
	if err != nil {
		result.Err = err
		return result
	}

	_, result.Err = client.Post(rootRuleURL(client), reqBody, &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return result
}

// DeleteRule will permanently delete a rule from a security group.
func DeleteRule(client *gophercloud.ServiceClient, id string) gophercloud.ErrResult {
	var result gophercloud.ErrResult
	_, result.Err = client.Delete(resourceRuleURL(client, id), nil)
	return result
}

func actionMap(prefix, groupName string) map[string]map[string]string {
	return map[string]map[string]string{
		prefix + "SecurityGroup": map[string]string{"name": groupName},
	}
}

// AddServerToGroup will associate a server and a security group, enforcing the
// rules of the group on the server.
func AddServerToGroup(client *gophercloud.ServiceClient, serverID, groupName string) gophercloud.ErrResult {
	var result gophercloud.ErrResult
	_, result.Err = client.Post(serverActionURL(client, serverID), actionMap("add", groupName), &result.Body, nil)
	return result
}

// RemoveServerFromGroup will disassociate a server from a security group.
func RemoveServerFromGroup(client *gophercloud.ServiceClient, serverID, groupName string) gophercloud.ErrResult {
	var result gophercloud.ErrResult
	_, result.Err = client.Post(serverActionURL(client, serverID), actionMap("remove", groupName), &result.Body, nil)
	return result
}
