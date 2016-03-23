package users

import (
	"errors"

	"github.com/rackspace/gophercloud"
	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	os "github.com/rackspace/gophercloud/openstack/db/v1/users"
	"github.com/rackspace/gophercloud/pagination"
)

// List will list all available users for a specified database instance.
func List(client *gophercloud.ServiceClient, instanceID string) pagination.Pager {
	createPageFn := func(r pagination.PageResult) pagination.Page {
		return UserPage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, baseURL(client, instanceID), createPageFn)
}

/*
ChangePassword changes the password for one or more users. For example, to
change the respective passwords for two users:

	opts := os.BatchCreateOpts{
		os.CreateOpts{Name: "db_user_1", Password: "new_password_1"},
		os.CreateOpts{Name: "db_user_2", Password: "new_password_2"},
	}

	ChangePassword(client, "instance_id", opts)
*/
func ChangePassword(client *gophercloud.ServiceClient, instanceID string, opts os.CreateOptsBuilder) UpdatePasswordsResult {
	var res UpdatePasswordsResult

	reqBody, err := opts.ToUserCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Request("PUT", baseURL(client, instanceID), gophercloud.RequestOpts{
		JSONBody: &reqBody,
		OkCodes:  []int{202},
	})

	return res
}

// UpdateOpts is the struct responsible for updating an existing user.
type UpdateOpts struct {
	// [OPTIONAL] Specifies a name for the user. Valid names can be composed
	// of the following characters: letters (either case); numbers; these
	// characters '@', '?', '#', ' ' but NEVER beginning a name string; '_' is
	// permitted anywhere. Prohibited characters that are forbidden include:
	// single quotes, double quotes, back quotes, semicolons, commas, backslashes,
	// and forward slashes. Spaces at the front or end of a user name are also
	// not permitted.
	Name string

	// [OPTIONAL] Specifies a password for the user.
	Password string

	// [OPTIONAL] Specifies the host from which a user is allowed to connect to
	// the database. Possible values are a string containing an IPv4 address or
	// "%" to allow connecting from any host. Optional; the default is "%".
	Host string
}

// ToMap is a convenience function for creating sub-maps for individual users.
func (opts UpdateOpts) ToMap() (map[string]interface{}, error) {
	if opts.Name == "root" {
		return nil, errors.New("root is a reserved user name and cannot be used")
	}

	user := map[string]interface{}{}

	if opts.Name != "" {
		user["name"] = opts.Name
	}

	if opts.Password != "" {
		user["password"] = opts.Password
	}

	if opts.Host != "" {
		user["host"] = opts.Host
	}

	return user, nil
}

// Update will modify the attributes of a specified user. Attributes that can
// be updated are: user name, password, and host.
func Update(client *gophercloud.ServiceClient, instanceID, userName string, opts UpdateOpts) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToMap()
	if err != nil {
		res.Err = err
		return res
	}
	reqBody = map[string]interface{}{"user": reqBody}

	_, res.Err = client.Request("PUT", userURL(client, instanceID, userName), gophercloud.RequestOpts{
		JSONBody: &reqBody,
		OkCodes:  []int{202},
	})

	return res
}

// Get will retrieve the details for a particular user.
func Get(client *gophercloud.ServiceClient, instanceID, userName string) GetResult {
	var res GetResult

	_, res.Err = client.Request("GET", userURL(client, instanceID, userName), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})

	return res
}

// ListAccess will list all of the databases a user has access to.
func ListAccess(client *gophercloud.ServiceClient, instanceID, userName string) pagination.Pager {
	pageFn := func(r pagination.PageResult) pagination.Page {
		return AccessPage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, dbsURL(client, instanceID, userName), pageFn)
}

/*
GrantAccess for the specified user to one or more databases on a specified
instance. For example, to add a user to multiple databases:

	opts := db.BatchCreateOpts{
		db.CreateOpts{Name: "database_1"},
		db.CreateOpts{Name: "database_3"},
		db.CreateOpts{Name: "database_19"},
	}

	GrantAccess(client, "instance_id", "user_name", opts)
*/
func GrantAccess(client *gophercloud.ServiceClient, instanceID, userName string, opts db.CreateOptsBuilder) GrantAccessResult {
	var res GrantAccessResult

	reqBody, err := opts.ToDBCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Request("PUT", dbsURL(client, instanceID, userName), gophercloud.RequestOpts{
		JSONBody: &reqBody,
		OkCodes:  []int{202},
	})

	return res
}

/*
RevokeAccess will revoke access for the specified user to one or more databases
on a specified instance. For example:

	RevokeAccess(client, "instance_id", "user_name", "db_name")
*/
func RevokeAccess(client *gophercloud.ServiceClient, instanceID, userName, dbName string) RevokeAccessResult {
	var res RevokeAccessResult

	_, res.Err = client.Request("DELETE", dbURL(client, instanceID, userName, dbName), gophercloud.RequestOpts{
		OkCodes: []int{202},
	})

	return res
}
