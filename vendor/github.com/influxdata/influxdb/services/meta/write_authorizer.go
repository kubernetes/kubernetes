package meta

import (
	"fmt"

	"github.com/influxdata/influxdb/influxql"
)

type WriteAuthorizer struct {
	Client *Client
}

func NewWriteAuthorizer(c *Client) *WriteAuthorizer {
	return &WriteAuthorizer{Client: c}
}

// AuthorizeWrite returns nil if the user has permission to write to the database.
func (a WriteAuthorizer) AuthorizeWrite(username, database string) error {
	u, err := a.Client.User(username)
	if err != nil || u == nil || !u.Authorize(influxql.WritePrivilege, database) {
		return &ErrAuthorize{
			Database: database,
			Message:  fmt.Sprintf("%s not authorized to write to %s", username, database),
		}
	}
	return nil
}
