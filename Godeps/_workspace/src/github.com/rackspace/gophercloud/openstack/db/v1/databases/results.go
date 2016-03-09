package databases

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Database represents a Database API resource.
type Database struct {
	// Specifies the name of the MySQL database.
	Name string

	// Set of symbols and encodings. The default character set is utf8.
	CharSet string

	// Set of rules for comparing characters in a character set. The default
	// value for collate is utf8_general_ci.
	Collate string
}

// CreateResult represents the result of a Create operation.
type CreateResult struct {
	gophercloud.ErrResult
}

// DeleteResult represents the result of a Delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// DBPage represents a single page of a paginated DB collection.
type DBPage struct {
	pagination.LinkedPageBase
}

// IsEmpty checks to see whether the collection is empty.
func (page DBPage) IsEmpty() (bool, error) {
	dbs, err := ExtractDBs(page)
	if err != nil {
		return true, err
	}
	return len(dbs) == 0, nil
}

// NextPageURL will retrieve the next page URL.
func (page DBPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"databases_links"`
	}

	var r resp
	err := mapstructure.Decode(page.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// ExtractDBs will convert a generic pagination struct into a more
// relevant slice of DB structs.
func ExtractDBs(page pagination.Page) ([]Database, error) {
	casted := page.(DBPage).Body

	var response struct {
		Databases []Database `mapstructure:"databases"`
	}

	err := mapstructure.Decode(casted, &response)
	return response.Databases, err
}
