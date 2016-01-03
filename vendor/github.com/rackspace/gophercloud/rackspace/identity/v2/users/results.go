package users

import (
	"strconv"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/identity/v2/users"

	"github.com/mitchellh/mapstructure"
)

// User represents a user resource that exists on the API.
type User struct {
	// The UUID for this user.
	ID string

	// The human name for this user.
	Name string

	// The username for this user.
	Username string

	// Indicates whether the user is enabled (true) or disabled (false).
	Enabled bool

	// The email address for this user.
	Email string

	// The ID of the tenant to which this user belongs.
	TenantID string `mapstructure:"tenant_id"`

	// Specifies the default region for the user account. This value is inherited
	// from the user administrator when the account is created.
	DefaultRegion string `mapstructure:"RAX-AUTH:defaultRegion"`

	// Identifies the domain that contains the user account. This value is
	// inherited from the user administrator when the account is created.
	DomainID string `mapstructure:"RAX-AUTH:domainId"`

	// The password value that the user needs for authentication. If the Add user
	// request included a password value, this attribute is not included in the
	// response.
	Password string `mapstructure:"OS-KSADM:password"`

	// Indicates whether the user has enabled multi-factor authentication.
	MultiFactorEnabled bool `mapstructure:"RAX-AUTH:multiFactorEnabled"`
}

// CreateResult represents the result of a Create operation
type CreateResult struct {
	os.CreateResult
}

// GetResult represents the result of a Get operation
type GetResult struct {
	os.GetResult
}

// UpdateResult represents the result of an Update operation
type UpdateResult struct {
	os.UpdateResult
}

func commonExtract(resp interface{}, err error) (*User, error) {
	if err != nil {
		return nil, err
	}

	var respStruct struct {
		User *User `json:"user"`
	}

	// Since the API returns a string instead of a bool, we need to hack the JSON
	json := resp.(map[string]interface{})
	user := json["user"].(map[string]interface{})
	if s, ok := user["RAX-AUTH:multiFactorEnabled"].(string); ok && s != "" {
		if b, err := strconv.ParseBool(s); err == nil {
			user["RAX-AUTH:multiFactorEnabled"] = b
		}
	}

	err = mapstructure.Decode(json, &respStruct)

	return respStruct.User, err
}

// Extract will get the Snapshot object out of the GetResult object.
func (r GetResult) Extract() (*User, error) {
	return commonExtract(r.Body, r.Err)
}

// Extract will get the Snapshot object out of the CreateResult object.
func (r CreateResult) Extract() (*User, error) {
	return commonExtract(r.Body, r.Err)
}

// Extract will get the Snapshot object out of the UpdateResult object.
func (r UpdateResult) Extract() (*User, error) {
	return commonExtract(r.Body, r.Err)
}

// ResetAPIKeyResult represents the server response to the ResetAPIKey method.
type ResetAPIKeyResult struct {
	gophercloud.Result
}

// ResetAPIKeyValue represents an API Key that has been reset.
type ResetAPIKeyValue struct {
	// The Username for this API Key reset.
	Username string `mapstructure:"username"`

	// The new API Key for this user.
	APIKey string `mapstructure:"apiKey"`
}

// Extract will get the Error or ResetAPIKeyValue object out of the ResetAPIKeyResult object.
func (r ResetAPIKeyResult) Extract() (*ResetAPIKeyValue, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		ResetAPIKeyValue ResetAPIKeyValue `mapstructure:"RAX-KSKEY:apiKeyCredentials"`
	}

	err := mapstructure.Decode(r.Body, &response)

	return &response.ResetAPIKeyValue, err
}
