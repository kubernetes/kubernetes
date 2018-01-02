package meta

import (
	"fmt"

	"github.com/influxdata/influxdb/influxql"
)

type QueryAuthorizer struct {
	Client *Client
}

func NewQueryAuthorizer(c *Client) *QueryAuthorizer {
	return &QueryAuthorizer{
		Client: c,
	}
}

// AuthorizeQuery authorizes u to execute q on database.
// Database can be "" for queries that do not require a database.
// If no user is provided it will return an error unless the query's first statement is to create
// a root user.
func (a *QueryAuthorizer) AuthorizeQuery(u *UserInfo, query *influxql.Query, database string) error {
	// Special case if no users exist.
	if n := a.Client.UserCount(); n == 0 {
		// Ensure there is at least one statement.
		if len(query.Statements) > 0 {
			// First statement in the query must create a user with admin privilege.
			cu, ok := query.Statements[0].(*influxql.CreateUserStatement)
			if ok && cu.Admin == true {
				return nil
			}
		}
		return &ErrAuthorize{
			Query:    query,
			Database: database,
			Message:  "create admin user first or disable authentication",
		}
	}

	if u == nil {
		return &ErrAuthorize{
			Query:    query,
			Database: database,
			Message:  "no user provided",
		}
	}

	// Admin privilege allows the user to execute all statements.
	if u.Admin {
		return nil
	}

	// Check each statement in the query.
	for _, stmt := range query.Statements {
		// Get the privileges required to execute the statement.
		privs, err := stmt.RequiredPrivileges()
		if err != nil {
			return err
		}

		// Make sure the user has the privileges required to execute
		// each statement.
		for _, p := range privs {
			if p.Admin {
				// Admin privilege already checked so statement requiring admin
				// privilege cannot be run.
				return &ErrAuthorize{
					Query:    query,
					User:     u.Name,
					Database: database,
					Message:  fmt.Sprintf("statement '%s', requires admin privilege", stmt),
				}
			}

			// Use the db name specified by the statement or the db
			// name passed by the caller if one wasn't specified by
			// the statement.
			db := p.Name
			if db == "" {
				db = database
			}
			if !u.Authorize(p.Privilege, db) {
				return &ErrAuthorize{
					Query:    query,
					User:     u.Name,
					Database: database,
					Message:  fmt.Sprintf("statement '%s', requires %s on %s", stmt, p.Privilege.String(), db),
				}
			}
		}
	}
	return nil
}

// ErrAuthorize represents an authorization error.
type ErrAuthorize struct {
	Query    *influxql.Query
	User     string
	Database string
	Message  string
}

// Error returns the text of the error.
func (e ErrAuthorize) Error() string {
	if e.User == "" {
		return fmt.Sprint(e.Message)
	}
	return fmt.Sprintf("%s not authorized to execute %s", e.User, e.Message)
}
