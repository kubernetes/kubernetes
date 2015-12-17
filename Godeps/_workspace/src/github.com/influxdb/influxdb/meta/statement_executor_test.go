package meta_test

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
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
	} else if !reflect.DeepEqual(res.Series, models.Rows{
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
	} else if !reflect.DeepEqual(res.Series, models.Rows{
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
	e.Store.PeersFn = func() ([]string, error) {
		return []string{"node0"}, nil
	}
	e.Store.LeaderFn = func() string {
		return "node0"
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW SERVERS`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, models.Rows{
		{
			Columns: []string{"id", "cluster_addr", "raft", "raft-leader"},
			Values: [][]interface{}{
				{uint64(1), "node0", true, true},
				{uint64(2), "node1", false, false},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a DROP SERVER statement can be executed.
func TestStatementExecutor_ExecuteStatement_DropServer(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.PeersFn = func() ([]string, error) {
		return []string{"node1"}, nil
	}

	// Ensure non-existent nodes do not cause a problem.
	e.Store.NodeFn = func(id uint64) (*meta.NodeInfo, error) {
		return nil, nil
	}
	if res := e.ExecuteStatement(influxql.MustParseStatement(`DROP SERVER 666`)); res.Err != meta.ErrNodeNotFound {
		t.Fatalf("unexpected error: %s", res.Err)
	}

	// Make a node exist.
	e.Store.NodeFn = func(id uint64) (*meta.NodeInfo, error) {
		return &meta.NodeInfo{
			ID: 1, Host: "node1",
		}, nil
	}

	e.Store.DeleteNodeFn = func(id uint64, force bool) error {
		return nil
	}

	// Ensure Raft nodes can be dropped.
	if res := e.ExecuteStatement(influxql.MustParseStatement(`DROP SERVER 1`)); res.Err != nil {
		t.Fatalf("unexpected error: %s", res.Err)
	}

	// Ensure non-Raft nodes can be dropped.
	e.Store.PeersFn = func() ([]string, error) {
		return []string{"node2"}, nil
	}
	e.Store.DeleteNodeFn = func(id uint64, force bool) error {
		return nil
	}
	if res := e.ExecuteStatement(influxql.MustParseStatement(`DROP SERVER 1`)); res.Err != nil {
		t.Fatalf("unexpected error: %s", res.Err)
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
	} else if !reflect.DeepEqual(res.Series, models.Rows{
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
	} else if !reflect.DeepEqual(res.Series, models.Rows{
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

	expErr := influxdb.ErrDatabaseNotFound("db0")
	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW RETENTION POLICIES ON db0`)); res.Err.Error() != expErr.Error() {
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
		} else if query != `CREATE CONTINUOUS QUERY cq0 ON db0 BEGIN SELECT count(field1) INTO db1 FROM db0 GROUP BY time(1h) END` {
			t.Fatalf("unexpected query: %s", query)
		}
		return nil
	}

	stmt := influxql.MustParseStatement(`CREATE CONTINUOUS QUERY cq0 ON db0 BEGIN SELECT count(field1) INTO db1 FROM db0 GROUP BY time(1h) END`)
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

	stmt := influxql.MustParseStatement(`CREATE CONTINUOUS QUERY cq0 ON db0 BEGIN SELECT count(field1) INTO db1 FROM db0 GROUP BY time(1h) END`)
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
					{Name: "cq0", Query: "SELECT count(field1) INTO db1 FROM db0"},
					{Name: "cq1", Query: "SELECT count(field1) INTO db2 FROM db0"},
				},
			},
			{
				Name: "db1",
				ContinuousQueries: []meta.ContinuousQueryInfo{
					{Name: "cq2", Query: "SELECT count(field1) INTO db3 FROM db1"},
				},
			},
		}, nil
	}

	stmt := influxql.MustParseStatement(`SHOW CONTINUOUS QUERIES`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, models.Rows{
		{
			Name:    "db0",
			Columns: []string{"name", "query"},
			Values: [][]interface{}{
				{"cq0", "SELECT count(field1) INTO db1 FROM db0"},
				{"cq1", "SELECT count(field1) INTO db2 FROM db0"},
			},
		},
		{
			Name:    "db1",
			Columns: []string{"name", "query"},
			Values: [][]interface{}{
				{"cq2", "SELECT count(field1) INTO db3 FROM db1"},
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

// Ensure a CREATE SUBSCRIPTION statement can be executed.
func TestStatementExecutor_ExecuteStatement_CreateSubscription(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateSubscriptionFn = func(database, rp, name, mode string, destinations []string) error {
		if database != "db0" {
			t.Fatalf("unexpected database: %s", database)
		} else if rp != "rp0" {
			t.Fatalf("unexpected rp: %s", rp)
		} else if name != "s0" {
			t.Fatalf("unexpected name: %s", name)
		} else if mode != "ANY" {
			t.Fatalf("unexpected mode: %s", mode)
		} else if len(destinations) != 2 {
			t.Fatalf("unexpected destinations: %s", destinations)
		} else if destinations[0] != "udp://h0:1234" {
			t.Fatalf("unexpected destinations[0]: %s", destinations[0])
		} else if destinations[1] != "udp://h1:1234" {
			t.Fatalf("unexpected destinations[1]: %s", destinations[1])
		}
		return nil
	}

	stmt := influxql.MustParseStatement(`CREATE SUBSCRIPTION s0 ON db0.rp0 DESTINATIONS ANY 'udp://h0:1234', 'udp://h1:1234'`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a CREATE SUBSCRIPTION statement can return an error from the store.
func TestStatementExecutor_ExecuteStatement_CreateSubscription_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.CreateSubscriptionFn = func(database, rp, name, mode string, destinations []string) error {
		return errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`CREATE SUBSCRIPTION s0 ON db0.rp0 DESTINATIONS ANY 'udp://h0:1234', 'udp://h1:1234'`)
	if res := e.ExecuteStatement(stmt); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a DROP SUBSCRIPTION statement can be executed.
func TestStatementExecutor_ExecuteStatement_DropSubscription(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropSubscriptionFn = func(database, rp, name string) error {
		if database != "db0" {
			t.Fatalf("unexpected database: %s", database)
		} else if rp != "rp0" {
			t.Fatalf("unexpected rp: %s", rp)
		} else if name != "s0" {
			t.Fatalf("unexpected name: %s", name)
		}
		return nil
	}

	stmt := influxql.MustParseStatement(`DROP SUBSCRIPTION s0 ON db0.rp0`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatal(res.Err)
	} else if res.Series != nil {
		t.Fatalf("unexpected rows: %#v", res.Series)
	}
}

// Ensure a DROP SUBSCRIPTION statement can return an error from the store.
func TestStatementExecutor_ExecuteStatement_DropSubscription_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DropSubscriptionFn = func(database, rp, name string) error {
		return errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`DROP SUBSCRIPTION s0 ON db0.rp0`)
	if res := e.ExecuteStatement(stmt); res.Err == nil || res.Err.Error() != "marker" {
		t.Fatalf("unexpected error: %s", res.Err)
	}
}

// Ensure a SHOW SUBSCRIPTIONS statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowSubscriptions(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{
				Name: "db0",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name: "rp0",
						Subscriptions: []meta.SubscriptionInfo{
							{Name: "s0", Mode: "ALL", Destinations: []string{"udp://h0:1234", "udp://h1:1234"}},
							{Name: "s1", Mode: "ANY", Destinations: []string{"udp://h2:1234", "udp://h3:1234"}},
						},
					},
					{
						Name: "rp1",
						Subscriptions: []meta.SubscriptionInfo{
							{Name: "s2", Mode: "ALL", Destinations: []string{"udp://h4:1234", "udp://h5:1234"}},
						},
					},
				},
			},
			{
				Name: "db1",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name: "rp2",
						Subscriptions: []meta.SubscriptionInfo{
							{Name: "s3", Mode: "ANY", Destinations: []string{"udp://h6:1234", "udp://h7:1234"}},
						},
					},
				},
			},
		}, nil
	}

	stmt := influxql.MustParseStatement(`SHOW SUBSCRIPTIONS`)
	if res := e.ExecuteStatement(stmt); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, models.Rows{
		{
			Name:    "db0",
			Columns: []string{"retention_policy", "name", "mode", "destinations"},
			Values: [][]interface{}{
				{"rp0", "s0", "ALL", []string{"udp://h0:1234", "udp://h1:1234"}},
				{"rp0", "s1", "ANY", []string{"udp://h2:1234", "udp://h3:1234"}},
				{"rp1", "s2", "ALL", []string{"udp://h4:1234", "udp://h5:1234"}},
			},
		},
		{
			Name:    "db1",
			Columns: []string{"retention_policy", "name", "mode", "destinations"},
			Values: [][]interface{}{
				{"rp2", "s3", "ANY", []string{"udp://h6:1234", "udp://h7:1234"}},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a SHOW SUBSCRIPTIONS statement can return an error from the store.
func TestStatementExecutor_ExecuteStatement_ShowSubscriptions_Err(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return nil, errors.New("marker")
	}

	stmt := influxql.MustParseStatement(`SHOW SUBSCRIPTIONS`)
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
			influxql.MustParseStatement(`SELECT count(field1) FROM db0`),
		)
	}()

	// Ensure that the executor panicked.
	if !panicked {
		t.Fatal("executor did not panic")
	}
}

// Ensure a SHOW SHARD GROUPS statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowShardGroups(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{
				Name: "foo",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name:     "rpi_foo",
						Duration: time.Second,
						ShardGroups: []meta.ShardGroupInfo{
							{
								ID:        66,
								StartTime: time.Unix(0, 0),
								EndTime:   time.Unix(1, 0),
							},
						},
					},
				},
			},
			{
				Name: "foo",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name:     "rpi_foo",
						Duration: time.Second,
						ShardGroups: []meta.ShardGroupInfo{
							{
								ID:        77,
								StartTime: time.Unix(2, 0),
								EndTime:   time.Unix(3, 0),
							},
						},
					},
				},
			},
		}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW SHARD GROUPS`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, models.Rows{
		{
			Name:    "shard groups",
			Columns: []string{"id", "database", "retention_policy", "start_time", "end_time", "expiry_time"},
			Values: [][]interface{}{
				{uint64(66), "foo", "rpi_foo", "1970-01-01T00:00:00Z", "1970-01-01T00:00:01Z", "1970-01-01T00:00:02Z"},
				{uint64(77), "foo", "rpi_foo", "1970-01-01T00:00:02Z", "1970-01-01T00:00:03Z", "1970-01-01T00:00:04Z"},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
	}
}

// Ensure a SHOW SHARDS statement can be executed.
func TestStatementExecutor_ExecuteStatement_ShowShards(t *testing.T) {
	e := NewStatementExecutor()
	e.Store.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{
				Name: "foo",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name:     "rpi_foo",
						Duration: time.Second,
						ShardGroups: []meta.ShardGroupInfo{
							{
								ID:        66,
								StartTime: time.Unix(0, 0),
								EndTime:   time.Unix(1, 0),
								Shards: []meta.ShardInfo{
									{
										ID: 1,
										Owners: []meta.ShardOwner{
											{NodeID: 1},
											{NodeID: 2},
											{NodeID: 3},
										},
									},
									{
										ID: 2,
									},
								},
							},
						},
					},
				},
			},
		}, nil
	}

	if res := e.ExecuteStatement(influxql.MustParseStatement(`SHOW SHARDS`)); res.Err != nil {
		t.Fatal(res.Err)
	} else if !reflect.DeepEqual(res.Series, models.Rows{
		{
			Name:    "foo",
			Columns: []string{"id", "database", "retention_policy", "shard_group", "start_time", "end_time", "expiry_time", "owners"},
			Values: [][]interface{}{
				{uint64(1), "foo", "rpi_foo", uint64(66), "1970-01-01T00:00:00Z", "1970-01-01T00:00:01Z", "1970-01-01T00:00:02Z", "1,2,3"},
				{uint64(2), "foo", "rpi_foo", uint64(66), "1970-01-01T00:00:00Z", "1970-01-01T00:00:01Z", "1970-01-01T00:00:02Z", ""},
			},
		},
	}) {
		t.Fatalf("unexpected rows: %s", spew.Sdump(res.Series))
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
	NodeFn                              func(id uint64) (*meta.NodeInfo, error)
	NodesFn                             func() ([]meta.NodeInfo, error)
	PeersFn                             func() ([]string, error)
	LeaderFn                            func() string
	DatabaseFn                          func(name string) (*meta.DatabaseInfo, error)
	DatabasesFn                         func() ([]meta.DatabaseInfo, error)
	CreateDatabaseFn                    func(name string) (*meta.DatabaseInfo, error)
	CreateDatabaseWithRetentionPolicyFn func(name string, rpi *meta.RetentionPolicyInfo) (*meta.DatabaseInfo, error)
	DropDatabaseFn                      func(name string) error
	DeleteNodeFn                        func(nodeID uint64, force bool) error
	DefaultRetentionPolicyFn            func(database string) (*meta.RetentionPolicyInfo, error)
	CreateRetentionPolicyFn             func(database string, rpi *meta.RetentionPolicyInfo) (*meta.RetentionPolicyInfo, error)
	UpdateRetentionPolicyFn             func(database, name string, rpu *meta.RetentionPolicyUpdate) error
	SetDefaultRetentionPolicyFn         func(database, name string) error
	DropRetentionPolicyFn               func(database, name string) error
	UsersFn                             func() ([]meta.UserInfo, error)
	CreateUserFn                        func(name, password string, admin bool) (*meta.UserInfo, error)
	UpdateUserFn                        func(name, password string) error
	DropUserFn                          func(name string) error
	SetPrivilegeFn                      func(username, database string, p influxql.Privilege) error
	SetAdminPrivilegeFn                 func(username string, admin bool) error
	UserPrivilegesFn                    func(username string) (map[string]influxql.Privilege, error)
	UserPrivilegeFn                     func(username, database string) (*influxql.Privilege, error)
	ContinuousQueriesFn                 func() ([]meta.ContinuousQueryInfo, error)
	CreateContinuousQueryFn             func(database, name, query string) error
	DropContinuousQueryFn               func(database, name string) error
	CreateSubscriptionFn                func(database, rp, name, typ string, hosts []string) error
	DropSubscriptionFn                  func(database, rp, name string) error
}

func (s *StatementExecutorStore) Node(id uint64) (*meta.NodeInfo, error) {
	return s.NodeFn(id)
}

func (s *StatementExecutorStore) Nodes() ([]meta.NodeInfo, error) {
	return s.NodesFn()
}

func (s *StatementExecutorStore) Peers() ([]string, error) {
	return s.PeersFn()
}

func (s *StatementExecutorStore) Leader() string {
	if s.LeaderFn != nil {
		return s.LeaderFn()
	}
	return ""
}

func (s *StatementExecutorStore) DeleteNode(nodeID uint64, force bool) error {
	return s.DeleteNodeFn(nodeID, force)
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

func (s *StatementExecutorStore) CreateDatabaseWithRetentionPolicy(name string, rpi *meta.RetentionPolicyInfo) (*meta.DatabaseInfo, error) {
	return s.CreateDatabaseWithRetentionPolicy(name, rpi)
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

func (s *StatementExecutorStore) CreateSubscription(database, rp, name, typ string, hosts []string) error {
	return s.CreateSubscriptionFn(database, rp, name, typ, hosts)
}

func (s *StatementExecutorStore) DropSubscription(database, rp, name string) error {
	return s.DropSubscriptionFn(database, rp, name)
}
