package meta_test

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
)

// Ensure a CREATE DATABASE statement can be executed.
func TestStatementExecutor_ExecuteStatement_CreateDatabase(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateDatabaseFn = func(name string) (*meta.DatabaseInfo, error) {
		if name != "foo" {
			t.Fatalf("unexpected name: %s", name)
		}
		return &meta.DatabaseInfo{Name: name}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`CREATE DATABASE foo`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a DROP DATABASE statement can be executed.
func TestStatementExecutor_ExecuteStatement_DropDatabase(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropDatabaseFn = func(name string) error {
		if name != "foo" {
			t.Fatalf("unexpected name: %s", name)
		}
		return nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`DROP DATABASE foo`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a SHOW DATABASES statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowDatabases(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{Name: "foo"},
			{Name: "bar"},
		}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW DATABASES`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, influxql.Rows{
		{
			Name:    "databases",
			Columns: []string{"name"},
			Values: [][]interface{}{
				{"foo"},
				{"bar"},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a SHOW DATABASES statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_ShowDatabases_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return nil, errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW DATABASES`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a SHOW GRANTS FOR statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowGrantsFor(t *testing.T) {
	t.Skip("Intermittent test failure: issue 3028")
	e := NewStatementExecutor()
	e.Store.UserPrivilegesFn = func(username string) (map[string]influxql.Privilege, error) {
		if username != "dejan" {
			t.Fatalf("unexpected username: %s", username)
		}
		return map[string]influxql.Privilege{
			"dejan": influxql.ReadPrivilege,
			"golja": influxql.WritePrivilege,
		}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW GRANTS FOR dejan`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, influxql.Rows{
		{
			Columns: []string{"database", "privilege"},
			Values: [][]interface{}{
				{"dejan", "READ"},
				{"golja", "WRITE"},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a SHOW SERVERS statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowServers(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.NodesFn = func() ([]meta.NodeInfo, error) {
		return []meta.NodeInfo{
			{ID: 1, Host: "node0"},
			{ID: 2, Host: "node1"},
		}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW SERVERS`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, influxql.Rows{
		{
			Columns: []string{"id", "url"},
			Values: [][]interface{}{
				{uint64(1), "http://node0"},
				{uint64(2), "http://node1"},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a SHOW SERVERS statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_ShowServers_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.NodesFn = func() ([]meta.NodeInfo, error) {
		return nil, errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW SERVERS`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a CREATE USER statement can be executed.
func TestStatementExecutor_ExecuteStatement_CreateUser(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateUserFn = func(name, password string, admin bool) (*meta.UserInfo, error) {
		if name != "susy" {
			t.Fatalf("unexpected name: %s", name)
		} else if password != "pass" {
			t.Fatalf("unexpected password: %s", password)
		} else if admin != true {
			t.Fatalf("unexpected admin: %v", admin)
		}
		return &meta.UserInfo{Name: name, Admin: admin}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`CREATE USER susy WITH PASSWORD 'pass' WITH ALL PRIVILEGES`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a CREATE USER statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_CreateUser_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateUserFn = func(name, password string, admin bool) (*meta.UserInfo, error) {
		return nil, errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`CREATE USER susy WITH PASSWORD 'pass'`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a SET PASSWORD statement can be executed.
func TestStatementExecutor_ExecuteStatement_SetPassword(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.UpdateUserFn = func(name, password string) error {
		if name != "susy" {
			t.Fatalf("unexpected name: %s", name)
		} else if password != "pass" {
			t.Fatalf("unexpected password: %s", password)
		}
		return nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SET PASSWORD FOR susy = 'pass' WITH ALL PRIVILEGES`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a SET PASSWORD statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_SetPassword_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.UpdateUserFn = func(name, password string) error {
		return errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SET PASSWORD FOR susy = 'pass'`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a DROP USER statement can be executed.
func TestStatementExecutor_ExecuteStatement_DropUser(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropUserFn = func(name string) error {
		if name != "susy" {
			t.Fatalf("unexpected name: %s", name)
		}
		return nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`DROP USER susy`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a DROP USER statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_DropUser_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropUserFn = func(name string) error {
		return errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`DROP USER susy`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a SHOW USERS statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowUsers(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.UsersFn = func() ([]meta.UserInfo, error) {
		return []meta.UserInfo{
			{Name: "susy", Admin: true},
			{Name: "bob", Admin: false},
		}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW USERS`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, influxql.Rows{
		{
			Columns: []string{"user", "admin"},
			Values: [][]interface{}{
				{"susy", true},
				{"bob", false},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a SHOW USERS statement returns an error from the store.
func TestStatementExecutor_ExecuteStatement_ShowUsers_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.UsersFn = func() ([]meta.UserInfo, error) {
		return nil, errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW USERS`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a GRANT statement can be executed.
func TestStatementExecutor_ExecuteStatement_Grant(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.SetPrivilegeFn = func(username, database string, p influxql.Privilege) error {
		if username != "susy" {
			t.Fatalf("unexpected username: %s", username)
		} else if database != "foo" {
			t.Fatalf("unexpected database: %s", database)
		} else if p != influxql.WritePrivilege {
			t.Fatalf("unexpected privilege: %s", p)
		}
		return nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`GRANT WRITE ON foo TO susy`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a GRANT statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_Grant_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.SetPrivilegeFn = func(username, database string, p influxql.Privilege) error {
		return errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`GRANT READ ON foo TO susy`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a GRANT statement for admin privilege can be executed.
func TestStatementExecutor_ExecuteStatement_GrantAdmin(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.SetAdminPrivilegeFn = func(username string, admin bool) error {
		if username != "susy" {
			t.Fatalf("unexpected username: %s", username)
		} else if admin != true {
			t.Fatalf("unexpected admin privilege: %t", admin)
		}
		return nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`GRANT ALL TO susy`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a GRANT statement for admin privilege returns errors from the store.
func TestStatementExecutor_ExecuteStatement_GrantAdmin_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.SetAdminPrivilegeFn = func(username string, admin bool) error {
		return errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`GRANT ALL PRIVILEGES TO susy`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a REVOKE statement can be executed.
func TestStatementExecutor_ExecuteStatement_Revoke(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.SetPrivilegeFn = func(username, database string, p influxql.Privilege) error {
		if username != "susy" {
			t.Fatalf("unexpected username: %s", username)
		} else if database != "foo" {
			t.Fatalf("unexpected database: %s", database)
		} else if p != influxql.NoPrivileges {
			t.Fatalf("unexpected privilege: %s", p)
		}
		return nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`REVOKE ALL PRIVILEGES ON foo FROM susy`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a REVOKE statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_Revoke_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.SetPrivilegeFn = func(username, database string, p influxql.Privilege) error {
		return errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`REVOKE ALL PRIVILEGES ON foo FROM susy`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a REVOKE statement for admin privilege can be executed.
func TestStatementExecutor_ExecuteStatement_RevokeAdmin(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.SetAdminPrivilegeFn = func(username string, admin bool) error {
		if username != "susy" {
			t.Fatalf("unexpected username: %s", username)
		} else if admin != false {
			t.Fatalf("unexpected admin privilege: %t", admin)
		}
		return nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`REVOKE ALL PRIVILEGES FROM susy`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a REVOKE statement for admin privilege returns errors from the store.
func TestStatementExecutor_ExecuteStatement_RevokeAdmin_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.SetAdminPrivilegeFn = func(username string, admin bool) error {
		return errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`REVOKE ALL PRIVILEGES FROM susy`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a CREATE RETENTION POLICY statement can be executed.
func TestStatementExecutor_ExecuteStatement_CreateRetentionPolicy(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateRetentionPolicyFn = func(database string, rpi *meta.RetentionPolicyInfo) (*meta.RetentionPolicyInfo, error) {
		if database != "foo" {
			t.Fatalf("unexpected database: %s", database)
		} else if rpi.Name != "rp0" {
			t.Fatalf("unexpected name: %s", rpi.Name)
		} else if rpi.Duration != 2*time.Hour {
			t.Fatalf("unexpected duration: %v", rpi.Duration)
		} else if rpi.ReplicaN != 3 {
			t.Fatalf("unexpected replication factor: %v", rpi.ReplicaN)
		}
		return nil, nil
	}
	e.Store.SetDefaultRetentionPolicyFn = func(database, name string) error {
		if database != "foo" {
			t.Fatalf("unexpected database: %s", database)
		} else if name != "rp0" {
			t.Fatalf("unexpected name: %s", name)
		}
		return nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`CREATE RETENTION POLICY rp0 ON foo DURATION 2h REPLICATION 3 DEFAULT`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a CREATE RETENTION POLICY statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_CreateRetentionPolicy_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateRetentionPolicyFn = func(database string, rpi *meta.RetentionPolicyInfo) (*meta.RetentionPolicyInfo, error) {
		return nil, errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`CREATE RETENTION POLICY rp0 ON foo DURATION 2h REPLICATION 1`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure an ALTER RETENTION POLICY statement can execute.
func TestStatementExecutor_ExecuteStatement_AlterRetentionPolicy(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.UpdateRetentionPolicyFn = func(database, name string, rpu *meta.RetentionPolicyUpdate) error {
		if database != "foo" {
			t.Fatalf("unexpected database: %s", database)
		} else if name != "rp0" {
			t.Fatalf("unexpected name: %s", name)
		} else if rpu.Duration != nil && *rpu.Duration != 7*24*time.Hour {
			t.Fatalf("unexpected duration: %v", *rpu.Duration)
		} else if rpu.ReplicaN != nil && *rpu.ReplicaN != 2 {
			t.Fatalf("unexpected replication factor: %v", *rpu.ReplicaN)
		}
		return nil
	}
	e.Store.SetDefaultRetentionPolicyFn = func(database, name string) error {
		if database != "foo" {
			t.Fatalf("unexpected database: %s", database)
		} else if name != "rp0" {
			t.Fatalf("unexpected name: %s", name)
		}
		return nil
	}

	stmt := influxql.MustParseStatement(`ALTER RETENTION POLICY rp0 ON foo DURATION 7d REPLICATION 2 DEFAULT`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatalf("unexpected error: %s", res.Err)
	}

	stmt = influxql.MustParseStatement(`ALTER RETENTION POLICY rp0 ON foo DURATION 7d`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatalf("unexpected error: %s", res.Err)
	}

	stmt = influxql.MustParseStatement(`ALTER RETENTION POLICY rp0 ON foo REPLICATION 2`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a ALTER RETENTION POLICY statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_AlterRetentionPolicy_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.UpdateRetentionPolicyFn = func(database, name string, rpu *meta.RetentionPolicyUpdate) error {
		return errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`ALTER RETENTION POLICY rp0 ON foo DURATION 1m REPLICATION 4 DEFAULT`)
	if res := e.ExecuteStatement(stmt); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a ALTER RETENTION POLICY statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_AlterRetentionPolicy_ErrSetDefault(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.UpdateRetentionPolicyFn = func(database, name string, rpu *meta.RetentionPolicyUpdate) error {
		return nil
	}
	e.Store.SetDefaultRetentionPolicyFn = func(database, name string) error {
		return errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`ALTER RETENTION POLICY rp0 ON foo DURATION 1m REPLICATION 4 DEFAULT`)
	if res := e.ExecuteStatement(stmt); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a DROP RETENTION POLICY statement can execute.
func TestStatementExecutor_ExecuteStatement_DropRetentionPolicy(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropRetentionPolicyFn = func(database, name string) error {
		if database != "foo" {
			t.Fatalf("unexpected database: %s", database)
		} else if name != "rp0" {
			t.Fatalf("unexpected name: %s", name)
		}
		return nil
	}

	stmt := influxql.MustParseStatement(`DROP RETENTION POLICY rp0 ON foo`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a DROP RETENTION POLICY statement returns errors from the store.
func TestStatementExecutor_ExecuteStatement_DropRetentionPolicy_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropRetentionPolicyFn = func(database, name string) error {
		return errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`DROP RETENTION POLICY rp0 ON foo`)
	if res := e.ExecuteStatement(stmt); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a SHOW RETENTION POLICIES statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowRetentionPolicies(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabaseFn = func(name string) (*meta.DatabaseInfo, error) {
		if name != "db0" {
			t.Fatalf("unexpected name: %s", name)
		}
		return &meta.DatabaseInfo{
			Name: name,
			DefaultRetentionPolicy: "rp1",
			RetentionPolicies: []meta.RetentionPolicyInfo{
				{
					Name:     "rp0",
					Duration: 2 * time.Hour,
					ReplicaN: 3,
				},
				{
					Name:     "rp1",
					Duration: 24 * time.Hour,
					ReplicaN: 1,
				},
			},
		}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW RETENTION POLICIES ON db0`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, influxql.Rows{
		{
			Columns: []string{"name", "duration", "replicaN", "default"},
			Values: [][]interface{}{
				{"rp0", "2h0m0s", 3, false},
				{"rp1", "24h0m0s", 1, true},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a SHOW RETENTION POLICIES statement can return an error from the store.
func TestStatementExecutor_ExecuteStatement_ShowRetentionPolicies_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabaseFn = func(name string) (*meta.DatabaseInfo, error) {
		return nil, errors.New("marker")
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW RETENTION POLICIES ON db0`)); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a SHOW RETENTION POLICIES statement can return an error if the database doesn't exist.
func TestStatementExecutor_ExecuteStatement_ShowRetentionPolicies_ErrDatabaseNotFound(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabaseFn = func(name string) (*meta.DatabaseInfo, error) {
		return nil, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW RETENTION POLICIES ON db0`)); res.Err != meta.ErrDatabaseNotFound {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a CREATE CONTINUOUS QUERY statement can be executed.
func TestStatementExecutor_ExecuteStatement_CreateContinuousQuery(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateContinuousQueryFn = func(database, name, query string) error {
		if database != "db0" {
			t.Fatalf("unexpected database: %s", database)
		} else if name != "cq0" {
			t.Fatalf("unexpected name: %s", name)
		} else if query != `CREATE CONTINUOUS QUERY cq0 ON db0 BEGIN SELECT count(*) INTO db1 FROM db0 GROUP BY time(1h) END` {
			t.Fatalf("unexpected query: %s", query)
		}
		return nil
	}

	stmt := influxql.MustParseStatement(`CREATE CONTINUOUS QUERY cq0 ON db0 BEGIN SELECT count(*) INTO db1 FROM db0 GROUP BY time(1h) END`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a CREATE CONTINUOUS QUERY statement can return an error from the store.
func TestStatementExecutor_ExecuteStatement_CreateContinuousQuery_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateContinuousQueryFn = func(database, name, query string) error {
		return errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`CREATE CONTINUOUS QUERY cq0 ON db0 BEGIN SELECT count(*) INTO db1 FROM db0 GROUP BY time(1h) END`)
	if res := e.ExecuteStatement(stmt); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a DROP CONTINUOUS QUERY statement can be executed.
func TestStatementExecutor_ExecuteStatement_DropContinuousQuery(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropContinuousQueryFn = func(database, name string) error {
		if database != "db0" {
			t.Fatalf("unexpected database: %s", database)
		} else if name != "cq0" {
			t.Fatalf("unexpected name: %s", name)
		}
		return nil
	}

	stmt := influxql.MustParseStatement(`DROP CONTINUOUS QUERY cq0 ON db0`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a DROP CONTINUOUS QUERY statement can return an error from the store.
func TestStatementExecutor_ExecuteStatement_DropContinuousQuery_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropContinuousQueryFn = func(database, name string) error {
		return errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`DROP CONTINUOUS QUERY cq0 ON db0`)
	if res := e.ExecuteStatement(stmt); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a SHOW CONTINUOUS QUERIES statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowContinuousQueries(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{
				Name: "db0",
				ContinuousQueries: []meta.ContinuousQueryInfo{
					{Name: "cq0", Query: "SELECT count(*) INTO db1 FROM db0"},
					{Name: "cq1", Query: "SELECT count(*) INTO db2 FROM db0"},
				},
			},
			{
				Name: "db1",
				ContinuousQueries: []meta.ContinuousQueryInfo{
					{Name: "cq2", Query: "SELECT count(*) INTO db3 FROM db1"},
				},
			},
		}, nil
	}

	stmt := influxql.MustParseStatement(`SHOW CONTINUOUS QUERIES`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, influxql.Rows{
		{
			Name:    "db0",
			Columns: []string{"name", "query"},
			Values: [][]interface{}{
				{"cq0", "SELECT count(*) INTO db1 FROM db0"},
				{"cq1", "SELECT count(*) INTO db2 FROM db0"},
			},
		},
		{
			Name:    "db1",
			Columns: []string{"name", "query"},
			Values: [][]interface{}{
				{"cq2", "SELECT count(*) INTO db3 FROM db1"},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a SHOW CONTINUOUS QUERIES statement can return an error from the store.
func TestStatementExecutor_ExecuteStatement_ShowContinuousQueries_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return nil, errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`SHOW CONTINUOUS QUERIES`)
	if res := e.ExecuteStatement(stmt); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatal(res.Err)
	}
}

// Ensure that executing an unsupported statement will panic.
func TestStatementExecutor_ExecuteStatement_Unsupported(t *testing.T) {
	var panicked bool
	func() {
		defer func() {
			if r := recover(); r != nil {
				panicked = true
			}
		}()

		// Execute a SELECT statement.
		NewStatementExecutor().ExecuteStatement(
			influxql.MustParseStatement(`SELECT count(*) FROM db0`),
		)
	}()

	// Ensure that the executor panicked.
	if !panicked {
		t.Fatal("executor did not panic")
	}
}

// StatementExecutor represents a test wrapper for meta.StatementExecutor.
type StatementExecutor struct {
	*meta.StatementExecutor
	Store StatementExecutorStore
}

// NewStatementExecutor returns a new instance of StatementExecutor with a mock store.
func NewStatementExecutor() *StatementExecutor {
	e := &StatementExecutor{}
	e.StatementExecutor = &meta.StatementExecutor{Store: &e.Store}
	return e
}

// StatementExecutorStore represents a mock implementation of StatementExecutor.Store.
type StatementExecutorStore struct {
	NodesFn                     func() ([]meta.NodeInfo, error)
	DatabaseFn                  func(name string) (*meta.DatabaseInfo, error)
	DatabasesFn                 func() ([]meta.DatabaseInfo, error)
	CreateDatabaseFn            func(name string) (*meta.DatabaseInfo, error)
	DropDatabaseFn              func(name string) error
	DefaultRetentionPolicyFn    func(database string) (*meta.RetentionPolicyInfo, error)
	CreateRetentionPolicyFn     func(database string, rpi *meta.RetentionPolicyInfo) (*meta.RetentionPolicyInfo, error)
	UpdateRetentionPolicyFn     func(database, name string, rpu *meta.RetentionPolicyUpdate) error
	SetDefaultRetentionPolicyFn func(database, name string) error
	DropRetentionPolicyFn       func(database, name string) error
	UsersFn                     func() ([]meta.UserInfo, error)
	CreateUserFn                func(name, password string, admin bool) (*meta.UserInfo, error)
	UpdateUserFn                func(name, password string) error
	DropUserFn                  func(name string) error
	SetPrivilegeFn              func(username, database string, p influxql.Privilege) error
	SetAdminPrivilegeFn         func(username string, admin bool) error
	UserPrivilegesFn            func(username string) (map[string]influxql.Privilege, error)
	UserPrivilegeFn             func(username, database string) (*influxql.Privilege, error)
	ContinuousQueriesFn         func() ([]meta.ContinuousQueryInfo, error)
	CreateContinuousQueryFn     func(database, name, query string) error
	DropContinuousQueryFn       func(database, name string) error
}

func (s *StatementExecutorStore) Nodes() ([]meta.NodeInfo, error) {
	return s.NodesFn()
}

func (s *StatementExecutorStore) Database(name string) (*meta.DatabaseInfo, error) {
	return s.DatabaseFn(name)
}

func (s *StatementExecutorStore) Databases() ([]meta.DatabaseInfo, error) {
	return s.DatabasesFn()
}

func (s *StatementExecutorStore) CreateDatabase(name string) (*meta.DatabaseInfo, error) {
	return s.CreateDatabaseFn(name)
}

func (s *StatementExecutorStore) DropDatabase(name string) error {
	return s.DropDatabaseFn(name)
}

func (s *StatementExecutorStore) DefaultRetentionPolicy(database string) (*meta.RetentionPolicyInfo, error) {
	return s.DefaultRetentionPolicyFn(database)
}

func (s *StatementExecutorStore) CreateRetentionPolicy(database string, rpi *meta.RetentionPolicyInfo) (*meta.RetentionPolicyInfo, error) {
	return s.CreateRetentionPolicyFn(database, rpi)
}

func (s *StatementExecutorStore) UpdateRetentionPolicy(database, name string, rpu *meta.RetentionPolicyUpdate) error {
	return s.UpdateRetentionPolicyFn(database, name, rpu)
}

func (s *StatementExecutorStore) SetDefaultRetentionPolicy(database, name string) error {
	return s.SetDefaultRetentionPolicyFn(database, name)
}

func (s *StatementExecutorStore) DropRetentionPolicy(database, name string) error {
	return s.DropRetentionPolicyFn(database, name)
}

func (s *StatementExecutorStore) Users() ([]meta.UserInfo, error) {
	return s.UsersFn()
}

func (s *StatementExecutorStore) CreateUser(name, password string, admin bool) (*meta.UserInfo, error) {
	return s.CreateUserFn(name, password, admin)
}

func (s *StatementExecutorStore) UpdateUser(name, password string) error {
	return s.UpdateUserFn(name, password)
}

func (s *StatementExecutorStore) DropUser(name string) error {
	return s.DropUserFn(name)
}

func (s *StatementExecutorStore) SetPrivilege(username, database string, p influxql.Privilege) error {
	return s.SetPrivilegeFn(username, database, p)
}

func (s *StatementExecutorStore) SetAdminPrivilege(username string, admin bool) error {
	return s.SetAdminPrivilegeFn(username, admin)
}

func (s *StatementExecutorStore) UserPrivileges(username string) (map[string]influxql.Privilege, error) {
	return s.UserPrivilegesFn(username)
}

func (s *StatementExecutorStore) UserPrivilege(username, database string) (*influxql.Privilege, error) {
	return s.UserPrivilegeFn(username, database)
}

func (s *StatementExecutorStore) ContinuousQueries() ([]meta.ContinuousQueryInfo, error) {
	return s.ContinuousQueriesFn()
}

func (s *StatementExecutorStore) CreateContinuousQuery(database, name, query string) error {
	return s.CreateContinuousQueryFn(database, name, query)
}

func (s *StatementExecutorStore) DropContinuousQuery(database, name string) error {
	return s.DropContinuousQueryFn(database, name)
}
