package meta

import (
	"fmt"

	"github.com/influxdb/influxdb/influxql"
)

// StatementExecutor translates InfluxQL queries to meta store methods.
type StatementExecutor struct {
	Store interface {
		Nodes() ([]NodeInfo, error)

		Database(name string) (*DatabaseInfo, error)
		Databases() ([]DatabaseInfo, error)
		CreateDatabase(name string) (*DatabaseInfo, error)
		DropDatabase(name string) error

		DefaultRetentionPolicy(database string) (*RetentionPolicyInfo, error)
		CreateRetentionPolicy(database string, rpi *RetentionPolicyInfo) (*RetentionPolicyInfo, error)
		UpdateRetentionPolicy(database, name string, rpu *RetentionPolicyUpdate) error
		SetDefaultRetentionPolicy(database, name string) error
		DropRetentionPolicy(database, name string) error

		Users() ([]UserInfo, error)
		CreateUser(name, password string, admin bool) (*UserInfo, error)
		UpdateUser(name, password string) error
		DropUser(name string) error
		SetPrivilege(username, database string, p influxql.Privilege) error
		SetAdminPrivilege(username string, admin bool) error
		UserPrivileges(username string) (map[string]influxql.Privilege, error)
		UserPrivilege(username, database string) (*influxql.Privilege, error)

		CreateContinuousQuery(database, name, query string) error
		DropContinuousQuery(database, name string) error
	}
}

// ExecuteStatement executes stmt against the meta store as user.
func (e *StatementExecutor) ExecuteStatement(stmt influxql.Statement) *influxql.Result {
	switch stmt := stmt.(type) {
	case *influxql.CreateDatabaseStatement:
		return e.executeCreateDatabaseStatement(stmt)
	case *influxql.DropDatabaseStatement:
		return e.executeDropDatabaseStatement(stmt)
	case *influxql.ShowDatabasesStatement:
		return e.executeShowDatabasesStatement(stmt)
	case *influxql.ShowGrantsForUserStatement:
		return e.executeShowGrantsForUserStatement(stmt)
	case *influxql.ShowServersStatement:
		return e.executeShowServersStatement(stmt)
	case *influxql.CreateUserStatement:
		return e.executeCreateUserStatement(stmt)
	case *influxql.SetPasswordUserStatement:
		return e.executeSetPasswordUserStatement(stmt)
	case *influxql.DropUserStatement:
		return e.executeDropUserStatement(stmt)
	case *influxql.ShowUsersStatement:
		return e.executeShowUsersStatement(stmt)
	case *influxql.GrantStatement:
		return e.executeGrantStatement(stmt)
	case *influxql.GrantAdminStatement:
		return e.executeGrantAdminStatement(stmt)
	case *influxql.RevokeStatement:
		return e.executeRevokeStatement(stmt)
	case *influxql.RevokeAdminStatement:
		return e.executeRevokeAdminStatement(stmt)
	case *influxql.CreateRetentionPolicyStatement:
		return e.executeCreateRetentionPolicyStatement(stmt)
	case *influxql.AlterRetentionPolicyStatement:
		return e.executeAlterRetentionPolicyStatement(stmt)
	case *influxql.DropRetentionPolicyStatement:
		return e.executeDropRetentionPolicyStatement(stmt)
	case *influxql.ShowRetentionPoliciesStatement:
		return e.executeShowRetentionPoliciesStatement(stmt)
	case *influxql.CreateContinuousQueryStatement:
		return e.executeCreateContinuousQueryStatement(stmt)
	case *influxql.DropContinuousQueryStatement:
		return e.executeDropContinuousQueryStatement(stmt)
	case *influxql.ShowContinuousQueriesStatement:
		return e.executeShowContinuousQueriesStatement(stmt)
	case *influxql.ShowStatsStatement:
		return e.executeShowStatsStatement(stmt)
	default:
		panic(fmt.Sprintf("unsupported statement type: %T", stmt))
	}
}

func (e *StatementExecutor) executeCreateDatabaseStatement(q *influxql.CreateDatabaseStatement) *influxql.Result {
	_, err := e.Store.CreateDatabase(q.Name)
	return &influxql.Result{Err: err}
}

func (e *StatementExecutor) executeDropDatabaseStatement(q *influxql.DropDatabaseStatement) *influxql.Result {
	return &influxql.Result{Err: e.Store.DropDatabase(q.Name)}
}

func (e *StatementExecutor) executeShowDatabasesStatement(q *influxql.ShowDatabasesStatement) *influxql.Result {
	dis, err := e.Store.Databases()
	if err != nil {
		return &influxql.Result{Err: err}
	}

	row := &influxql.Row{Name: "databases", Columns: []string{"name"}}
	for _, di := range dis {
		row.Values = append(row.Values, []interface{}{di.Name})
	}
	return &influxql.Result{Series: []*influxql.Row{row}}
}

func (e *StatementExecutor) executeShowGrantsForUserStatement(q *influxql.ShowGrantsForUserStatement) *influxql.Result {
	priv, err := e.Store.UserPrivileges(q.Name)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	row := &influxql.Row{Columns: []string{"database", "privilege"}}
	for d, p := range priv {
		row.Values = append(row.Values, []interface{}{d, p.String()})
	}
	return &influxql.Result{Series: []*influxql.Row{row}}
}

func (e *StatementExecutor) executeShowServersStatement(q *influxql.ShowServersStatement) *influxql.Result {
	nis, err := e.Store.Nodes()
	if err != nil {
		return &influxql.Result{Err: err}
	}

	row := &influxql.Row{Columns: []string{"id", "url"}}
	for _, ni := range nis {
		row.Values = append(row.Values, []interface{}{ni.ID, "http://" + ni.Host})
	}
	return &influxql.Result{Series: []*influxql.Row{row}}
}

func (e *StatementExecutor) executeCreateUserStatement(q *influxql.CreateUserStatement) *influxql.Result {
	_, err := e.Store.CreateUser(q.Name, q.Password, q.Admin)
	return &influxql.Result{Err: err}
}

func (e *StatementExecutor) executeSetPasswordUserStatement(q *influxql.SetPasswordUserStatement) *influxql.Result {
	return &influxql.Result{Err: e.Store.UpdateUser(q.Name, q.Password)}
}

func (e *StatementExecutor) executeDropUserStatement(q *influxql.DropUserStatement) *influxql.Result {
	return &influxql.Result{Err: e.Store.DropUser(q.Name)}
}

func (e *StatementExecutor) executeShowUsersStatement(q *influxql.ShowUsersStatement) *influxql.Result {
	uis, err := e.Store.Users()
	if err != nil {
		return &influxql.Result{Err: err}
	}

	row := &influxql.Row{Columns: []string{"user", "admin"}}
	for _, ui := range uis {
		row.Values = append(row.Values, []interface{}{ui.Name, ui.Admin})
	}
	return &influxql.Result{Series: []*influxql.Row{row}}
}

func (e *StatementExecutor) executeGrantStatement(stmt *influxql.GrantStatement) *influxql.Result {
	return &influxql.Result{Err: e.Store.SetPrivilege(stmt.User, stmt.On, stmt.Privilege)}
}

func (e *StatementExecutor) executeGrantAdminStatement(stmt *influxql.GrantAdminStatement) *influxql.Result {
	return &influxql.Result{Err: e.Store.SetAdminPrivilege(stmt.User, true)}
}

func (e *StatementExecutor) executeRevokeStatement(stmt *influxql.RevokeStatement) *influxql.Result {
	priv := influxql.NoPrivileges

	// Revoking all privileges means there's no need to look at existing user privileges.
	if stmt.Privilege != influxql.AllPrivileges {
		p, err := e.Store.UserPrivilege(stmt.User, stmt.On)
		if err != nil {
			return &influxql.Result{Err: err}
		}
		// Bit clear (AND NOT) the user's privilege with the revoked privilege.
		priv = *p &^ stmt.Privilege
	}

	return &influxql.Result{Err: e.Store.SetPrivilege(stmt.User, stmt.On, priv)}
}

func (e *StatementExecutor) executeRevokeAdminStatement(stmt *influxql.RevokeAdminStatement) *influxql.Result {
	return &influxql.Result{Err: e.Store.SetAdminPrivilege(stmt.User, false)}
}

func (e *StatementExecutor) executeCreateRetentionPolicyStatement(stmt *influxql.CreateRetentionPolicyStatement) *influxql.Result {
	rpi := NewRetentionPolicyInfo(stmt.Name)
	rpi.Duration = stmt.Duration
	rpi.ReplicaN = stmt.Replication

	// Create new retention policy.
	_, err := e.Store.CreateRetentionPolicy(stmt.Database, rpi)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// If requested, set new policy as the default.
	if stmt.Default {
		err = e.Store.SetDefaultRetentionPolicy(stmt.Database, stmt.Name)
	}

	return &influxql.Result{Err: err}
}

func (e *StatementExecutor) executeAlterRetentionPolicyStatement(stmt *influxql.AlterRetentionPolicyStatement) *influxql.Result {
	rpu := &RetentionPolicyUpdate{
		Duration: stmt.Duration,
		ReplicaN: stmt.Replication,
	}

	// Update the retention policy.
	err := e.Store.UpdateRetentionPolicy(stmt.Database, stmt.Name, rpu)
	if err != nil {
		return &influxql.Result{Err: err}
	}

	// If requested, set as default retention policy.
	if stmt.Default {
		err = e.Store.SetDefaultRetentionPolicy(stmt.Database, stmt.Name)
	}

	return &influxql.Result{Err: err}
}

func (e *StatementExecutor) executeDropRetentionPolicyStatement(q *influxql.DropRetentionPolicyStatement) *influxql.Result {
	return &influxql.Result{Err: e.Store.DropRetentionPolicy(q.Database, q.Name)}
}

func (e *StatementExecutor) executeShowRetentionPoliciesStatement(q *influxql.ShowRetentionPoliciesStatement) *influxql.Result {
	di, err := e.Store.Database(q.Database)
	if err != nil {
		return &influxql.Result{Err: err}
	} else if di == nil {
		return &influxql.Result{Err: ErrDatabaseNotFound}
	}

	row := &influxql.Row{Columns: []string{"name", "duration", "replicaN", "default"}}
	for _, rpi := range di.RetentionPolicies {
		row.Values = append(row.Values, []interface{}{rpi.Name, rpi.Duration.String(), rpi.ReplicaN, di.DefaultRetentionPolicy == rpi.Name})
	}
	return &influxql.Result{Series: []*influxql.Row{row}}
}

func (e *StatementExecutor) executeCreateContinuousQueryStatement(q *influxql.CreateContinuousQueryStatement) *influxql.Result {
	return &influxql.Result{
		Err: e.Store.CreateContinuousQuery(q.Database, q.Name, q.String()),
	}
}

func (e *StatementExecutor) executeDropContinuousQueryStatement(q *influxql.DropContinuousQueryStatement) *influxql.Result {
	return &influxql.Result{
		Err: e.Store.DropContinuousQuery(q.Database, q.Name),
	}
}

func (e *StatementExecutor) executeShowContinuousQueriesStatement(stmt *influxql.ShowContinuousQueriesStatement) *influxql.Result {
	dis, err := e.Store.Databases()
	if err != nil {
		return &influxql.Result{Err: err}
	}

	rows := []*influxql.Row{}
	for _, di := range dis {
		row := &influxql.Row{Columns: []string{"name", "query"}, Name: di.Name}
		for _, cqi := range di.ContinuousQueries {
			row.Values = append(row.Values, []interface{}{cqi.Name, cqi.Query})
		}
		rows = append(rows, row)
	}
	return &influxql.Result{Series: rows}
}

func (e *StatementExecutor) executeShowStatsStatement(stmt *influxql.ShowStatsStatement) *influxql.Result {
	return &influxql.Result{Err: fmt.Errorf("SHOW STATS is not implemented yet")}
}
