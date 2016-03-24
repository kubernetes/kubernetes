package databases

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// CreateOptsBuilder builds create options
type CreateOptsBuilder interface {
	ToDBCreateMap() (map[string]interface{}, error)
}

// DatabaseOpts is the struct responsible for configuring a database; often in
// the context of an instance.
type CreateOpts struct {
	// [REQUIRED] Specifies the name of the database. Valid names can be composed
	// of the following characters: letters (either case); numbers; these
	// characters '@', '?', '#', ' ' but NEVER beginning a name string; '_' is
	// permitted anywhere. Prohibited characters that are forbidden include:
	// single quotes, double quotes, back quotes, semicolons, commas, backslashes,
	// and forward slashes.
	Name string

	// [OPTIONAL] Set of symbols and encodings. The default character set is
	// "utf8". See http://dev.mysql.com/doc/refman/5.1/en/charset-mysql.html for
	// supported character sets.
	CharSet string

	// [OPTIONAL] Set of rules for comparing characters in a character set. The
	// default value for collate is "utf8_general_ci". See
	// http://dev.mysql.com/doc/refman/5.1/en/charset-mysql.html for supported
	// collations.
	Collate string
}

// ToMap is a helper function to convert individual DB create opt structures
// into sub-maps.
func (opts CreateOpts) ToMap() (map[string]string, error) {
	if opts.Name == "" {
		return nil, fmt.Errorf("Name is a required field")
	}
	if len(opts.Name) > 64 {
		return nil, fmt.Errorf("Name must be less than 64 chars long")
	}

	db := map[string]string{"name": opts.Name}

	if opts.CharSet != "" {
		db["character_set"] = opts.CharSet
	}
	if opts.Collate != "" {
		db["collate"] = opts.Collate
	}
	return db, nil
}

// BatchCreateOpts allows for multiple databases to created and modified.
type BatchCreateOpts []CreateOpts

// ToDBCreateMap renders a JSON map for creating DBs.
func (opts BatchCreateOpts) ToDBCreateMap() (map[string]interface{}, error) {
	dbs := make([]map[string]string, len(opts))
	for i, db := range opts {
		dbMap, err := db.ToMap()
		if err != nil {
			return nil, err
		}
		dbs[i] = dbMap
	}
	return map[string]interface{}{"databases": dbs}, nil
}

// Create will create a new database within the specified instance. If the
// specified instance does not exist, a 404 error will be returned.
func Create(client *gophercloud.ServiceClient, instanceID string, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToDBCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Request("POST", baseURL(client, instanceID), gophercloud.RequestOpts{
		JSONBody: &reqBody,
		OkCodes:  []int{202},
	})

	return res
}

// List will list all of the databases for a specified instance. Note: this
// operation will only return user-defined databases; it will exclude system
// databases like "mysql", "information_schema", "lost+found" etc.
func List(client *gophercloud.ServiceClient, instanceID string) pagination.Pager {
	createPageFn := func(r pagination.PageResult) pagination.Page {
		return DBPage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, baseURL(client, instanceID), createPageFn)
}

// Delete will permanently delete the database within a specified instance.
// All contained data inside the database will also be permanently deleted.
func Delete(client *gophercloud.ServiceClient, instanceID, dbName string) DeleteResult {
	var res DeleteResult

	_, res.Err = client.Request("DELETE", dbURL(client, instanceID, dbName), gophercloud.RequestOpts{
		OkCodes: []int{202},
	})

	return res
}
