package databases

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
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
	return len(dbs) == 0, err
}

// NextPageURL will retrieve the next page URL.
func (page DBPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"databases_links"`
	}
	err := page.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// ExtractDBs will convert a generic pagination struct into a more
// relevant slice of DB structs.
func ExtractDBs(page pagination.Page) ([]Database, error) {
	r := page.(DBPage)
	var s struct {
		Databases []Database `json:"databases"`
	}
	err := r.ExtractInto(&s)
	return s.Databases, err
}
