package databases

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder builds create options
type CreateOptsBuilder interface {
	ToDBCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the struct responsible for configuring a database; often in
// the context of an instance.
type CreateOpts struct {
	// Specifies the name of the database. Valid names can be composed
	// of the following characters: letters (either case); numbers; these
	// characters '@', '?', '#', ' ' but NEVER beginning a name string; '_' is
	// permitted anywhere. Prohibited characters that are forbidden include:
	// single quotes, double quotes, back quotes, semicolons, commas, backslashes,
	// and forward slashes.
	Name string `json:"name" required:"true"`
	// Set of symbols and encodings. The default character set is
	// "utf8". See http://dev.mysql.com/doc/refman/5.1/en/charset-mysql.html for
	// supported character sets.
	CharSet string `json:"character_set,omitempty"`
	// Set of rules for comparing characters in a character set. The
	// default value for collate is "utf8_general_ci". See
	// http://dev.mysql.com/doc/refman/5.1/en/charset-mysql.html for supported
	// collations.
	Collate string `json:"collate,omitempty"`
}

// ToMap is a helper function to convert individual DB create opt structures
// into sub-maps.
func (opts CreateOpts) ToMap() (map[string]interface{}, error) {
	if len(opts.Name) > 64 {
		err := gophercloud.ErrInvalidInput{}
		err.Argument = "databases.CreateOpts.Name"
		err.Value = opts.Name
		err.Info = "Must be less than 64 chars long"
		return nil, err
	}
	return gophercloud.BuildRequestBody(opts, "")
}

// BatchCreateOpts allows for multiple databases to created and modified.
type BatchCreateOpts []CreateOpts

// ToDBCreateMap renders a JSON map for creating DBs.
func (opts BatchCreateOpts) ToDBCreateMap() (map[string]interface{}, error) {
	dbs := make([]map[string]interface{}, len(opts))
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
func Create(client *gophercloud.ServiceClient, instanceID string, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToDBCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(baseURL(client, instanceID), &b, nil, nil)
	return
}

// List will list all of the databases for a specified instance. Note: this
// operation will only return user-defined databases; it will exclude system
// databases like "mysql", "information_schema", "lost+found" etc.
func List(client *gophercloud.ServiceClient, instanceID string) pagination.Pager {
	return pagination.NewPager(client, baseURL(client, instanceID), func(r pagination.PageResult) pagination.Page {
		return DBPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Delete will permanently delete the database within a specified instance.
// All contained data inside the database will also be permanently deleted.
func Delete(client *gophercloud.ServiceClient, instanceID, dbName string) (r DeleteResult) {
	_, r.Err = client.Delete(dbURL(client, instanceID, dbName), nil)
	return
}
