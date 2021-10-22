package acls

import (
	"github.com/gophercloud/gophercloud"
)

// GetContainerACL retrieves the ACL of a container.
func GetContainerACL(client *gophercloud.ServiceClient, containerID string) (r ACLResult) {
	_, r.Err = client.Get(containerURL(client, containerID), &r.Body, nil)
	return
}

// GetSecretACL retrieves the ACL of a secret.
func GetSecretACL(client *gophercloud.ServiceClient, secretID string) (r ACLResult) {
	_, r.Err = client.Get(secretURL(client, secretID), &r.Body, nil)
	return
}

// SetOptsBuilder allows extensions to add additional parameters to the
// Set request.
type SetOptsBuilder interface {
	ToACLSetMap() (map[string]interface{}, error)
}

// SetOpts represents options to set an ACL on a resource.
type SetOpts struct {
	// Type is the type of ACL to set. ie: read.
	Type string `json:"-" required:"true"`

	// Users are the list of Keystone user UUIDs.
	Users *[]string `json:"users,omitempty"`

	// ProjectAccess toggles if all users in a project can access the resource.
	ProjectAccess *bool `json:"project-access,omitempty"`
}

// ToACLSetMap formats a SetOpts into a set request.
func (opts SetOpts) ToACLSetMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, opts.Type)
}

// SetContainerACL will set an ACL on a container.
func SetContainerACL(client *gophercloud.ServiceClient, containerID string, opts SetOptsBuilder) (r ACLRefResult) {
	b, err := opts.ToACLSetMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Put(containerURL(client, containerID), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// SetSecretACL will set an ACL on a secret.
func SetSecretACL(client *gophercloud.ServiceClient, secretID string, opts SetOptsBuilder) (r ACLRefResult) {
	b, err := opts.ToACLSetMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Put(secretURL(client, secretID), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// UpdateContainerACL will update an ACL on a container.
func UpdateContainerACL(client *gophercloud.ServiceClient, containerID string, opts SetOptsBuilder) (r ACLRefResult) {
	b, err := opts.ToACLSetMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Patch(containerURL(client, containerID), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// UpdateSecretACL will update an ACL on a secret.
func UpdateSecretACL(client *gophercloud.ServiceClient, secretID string, opts SetOptsBuilder) (r ACLRefResult) {
	b, err := opts.ToACLSetMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Patch(secretURL(client, secretID), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// DeleteContainerACL will delete an ACL from a conatiner.
func DeleteContainerACL(client *gophercloud.ServiceClient, containerID string) (r DeleteResult) {
	_, r.Err = client.Delete(containerURL(client, containerID), &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// DeleteSecretACL will delete an ACL from a secret.
func DeleteSecretACL(client *gophercloud.ServiceClient, secretID string) (r DeleteResult) {
	_, r.Err = client.Delete(secretURL(client, secretID), &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
