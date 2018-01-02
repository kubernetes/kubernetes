package meta_test

import (
	"encoding/json"
	"io/ioutil"
	"net"
	"os"
	"path"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/influxdata/influxdb"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/services/meta"
)

func TestMetaClient_CreateDatabaseOnly(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	if db, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if db.Name != "db0" {
		t.Fatalf("database name mismatch.  exp: db0, got %s", db.Name)
	}

	db := c.Database("db0")
	if db == nil {
		t.Fatal("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	// Make sure a default retention policy was created.
	rp, err := c.RetentionPolicy("db0", "autogen")
	if err != nil {
		t.Fatal(err)
	} else if rp == nil {
		t.Fatal("failed to create rp")
	} else if exp, got := "autogen", rp.Name; exp != got {
		t.Fatalf("rp name wrong:\n\texp: %s\n\tgot: %s", exp, got)
	}
}

func TestMetaClient_CreateDatabaseIfNotExists(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	db := c.Database("db0")
	if db == nil {
		t.Fatal("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}
}

func TestMetaClient_CreateDatabaseWithRetentionPolicy(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	duration := 1 * time.Hour
	replicaN := 1
	spec := meta.RetentionPolicySpec{
		Name:               "rp0",
		Duration:           &duration,
		ReplicaN:           &replicaN,
		ShardGroupDuration: 30 * time.Minute,
	}
	if _, err := c.CreateDatabaseWithRetentionPolicy("db0", &spec); err != nil {
		t.Fatal(err)
	}

	db := c.Database("db0")
	if db == nil {
		t.Fatal("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	rp := db.RetentionPolicy("rp0")
	if rp.Name != "rp0" {
		t.Fatalf("rp name wrong: %s", rp.Name)
	} else if rp.Duration != time.Hour {
		t.Fatalf("rp duration wrong: %v", rp.Duration)
	} else if rp.ReplicaN != 1 {
		t.Fatalf("rp replication wrong: %d", rp.ReplicaN)
	} else if rp.ShardGroupDuration != 30*time.Minute {
		t.Fatalf("rp shard duration wrong: %v", rp.ShardGroupDuration)
	}

	// Recreating the exact same database with retention policy is not
	// an error.
	if _, err := c.CreateDatabaseWithRetentionPolicy("db0", &spec); err != nil {
		t.Fatal(err)
	}

	// If the rp's duration is different, an error should be returned.
	spec2 := spec
	duration2 := *spec.Duration + time.Minute
	spec2.Duration = &duration2
	if _, err := c.CreateDatabaseWithRetentionPolicy("db0", &spec2); err != meta.ErrRetentionPolicyConflict {
		t.Fatalf("got %v, but expected %v", err, meta.ErrRetentionPolicyConflict)
	}

	// If the rp's replica is different, an error should be returned.
	spec2 = spec
	replica2 := *spec.ReplicaN + 1
	spec2.ReplicaN = &replica2
	if _, err := c.CreateDatabaseWithRetentionPolicy("db0", &spec2); err != meta.ErrRetentionPolicyConflict {
		t.Fatalf("got %v, but expected %v", err, meta.ErrRetentionPolicyConflict)
	}

	// If the rp's shard group duration is different, an error should be returned.
	spec2 = spec
	spec2.ShardGroupDuration = spec.ShardGroupDuration + time.Minute
	if _, err := c.CreateDatabaseWithRetentionPolicy("db0", &spec2); err != meta.ErrRetentionPolicyConflict {
		t.Fatalf("got %v, but expected %v", err, meta.ErrRetentionPolicyConflict)
	}

	// If create database is used by itself, no error should be returned and
	// the default retention policy should not be changed.
	if dbi, err := c.CreateDatabase("db0"); err != nil {
		t.Fatalf("got %v, but expected %v", err, nil)
	} else if dbi.DefaultRetentionPolicy != "rp0" {
		t.Fatalf("got %v, but expected %v", dbi.DefaultRetentionPolicy, "rp0")
	} else if got, exp := len(dbi.RetentionPolicies), 1; got != exp {
		// Ensure no additional retention policies were created.
		t.Fatalf("got %v, but expected %v", got, exp)
	}
}

func TestMetaClient_Databases(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	// Create two databases.
	db, err := c.CreateDatabase("db0")
	if err != nil {
		t.Fatal(err)
	} else if db == nil {
		t.Fatal("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	db, err = c.CreateDatabase("db1")
	if err != nil {
		t.Fatal(err)
	} else if db.Name != "db1" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	dbs := c.Databases()
	if err != nil {
		t.Fatal(err)
	}
	if len(dbs) != 2 {
		t.Fatalf("expected 2 databases but got %d", len(dbs))
	} else if dbs[0].Name != "db0" {
		t.Fatalf("db name wrong: %s", dbs[0].Name)
	} else if dbs[1].Name != "db1" {
		t.Fatalf("db name wrong: %s", dbs[1].Name)
	}
}

func TestMetaClient_DropDatabase(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	db := c.Database("db0")
	if db == nil {
		t.Fatalf("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	if err := c.DropDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	if db = c.Database("db0"); db != nil {
		t.Fatalf("expected database to not return: %v", db)
	}

	// Dropping a database that does not exist is not an error.
	if err := c.DropDatabase("db foo"); err != nil {
		t.Fatalf("got %v error, but expected no error", err)
	}
}

func TestMetaClient_CreateRetentionPolicy(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	db := c.Database("db0")
	if db == nil {
		t.Fatal("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	rp0 := meta.RetentionPolicyInfo{
		Name:               "rp0",
		ReplicaN:           1,
		Duration:           time.Hour,
		ShardGroupDuration: time.Hour,
	}

	if _, err := c.CreateRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:               rp0.Name,
		ReplicaN:           &rp0.ReplicaN,
		Duration:           &rp0.Duration,
		ShardGroupDuration: rp0.ShardGroupDuration,
	}); err != nil {
		t.Fatal(err)
	}

	actual, err := c.RetentionPolicy("db0", "rp0")
	if err != nil {
		t.Fatal(err)
	} else if got, exp := actual, &rp0; !reflect.DeepEqual(got, exp) {
		t.Fatalf("got %#v, expected %#v", got, exp)
	}

	// Create the same policy.  Should not error.
	if _, err := c.CreateRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:               rp0.Name,
		ReplicaN:           &rp0.ReplicaN,
		Duration:           &rp0.Duration,
		ShardGroupDuration: rp0.ShardGroupDuration,
	}); err != nil {
		t.Fatal(err)
	} else if actual, err = c.RetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	} else if got, exp := actual, &rp0; !reflect.DeepEqual(got, exp) {
		t.Fatalf("got %#v, expected %#v", got, exp)
	}

	// Creating the same policy, but with a different duration should
	// result in an error.
	rp1 := rp0
	rp1.Duration = 2 * rp0.Duration

	_, got := c.CreateRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:               rp1.Name,
		ReplicaN:           &rp1.ReplicaN,
		Duration:           &rp1.Duration,
		ShardGroupDuration: rp1.ShardGroupDuration,
	})
	if exp := meta.ErrRetentionPolicyExists; got != exp {
		t.Fatalf("got error %v, expected error %v", got, exp)
	}

	// Creating the same policy, but with a different replica factor
	// should also result in an error.
	rp1 = rp0
	rp1.ReplicaN = rp0.ReplicaN + 1

	_, got = c.CreateRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:               rp1.Name,
		ReplicaN:           &rp1.ReplicaN,
		Duration:           &rp1.Duration,
		ShardGroupDuration: rp1.ShardGroupDuration,
	})
	if exp := meta.ErrRetentionPolicyExists; got != exp {
		t.Fatalf("got error %v, expected error %v", got, exp)
	}

	// Creating the same policy, but with a different shard group
	// duration should also result in an error.
	rp1 = rp0
	rp1.ShardGroupDuration = rp0.ShardGroupDuration / 2

	_, got = c.CreateRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:               rp1.Name,
		ReplicaN:           &rp1.ReplicaN,
		Duration:           &rp1.Duration,
		ShardGroupDuration: rp1.ShardGroupDuration,
	})
	if exp := meta.ErrRetentionPolicyExists; got != exp {
		t.Fatalf("got error %v, expected error %v", got, exp)
	}

	// Creating a policy with the shard duration being greater than the
	// duration should also be an error.
	rp1 = rp0
	rp1.Duration = 1 * time.Hour
	rp1.ShardGroupDuration = 2 * time.Hour

	_, got = c.CreateRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:               rp1.Name,
		ReplicaN:           &rp1.ReplicaN,
		Duration:           &rp1.Duration,
		ShardGroupDuration: rp1.ShardGroupDuration,
	})
	if exp := meta.ErrIncompatibleDurations; got != exp {
		t.Fatalf("got error %v, expected error %v", got, exp)
	}
}

func TestMetaClient_SetDefaultRetentionPolicy(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	duration := 1 * time.Hour
	replicaN := 1
	if _, err := c.CreateDatabaseWithRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:     "rp0",
		Duration: &duration,
		ReplicaN: &replicaN,
	}); err != nil {
		t.Fatal(err)
	}

	db := c.Database("db0")
	if db == nil {
		t.Fatal("datbase not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	rp, err := c.RetentionPolicy("db0", "rp0")
	if err != nil {
		t.Fatal(err)
	} else if rp.Name != "rp0" {
		t.Fatalf("rp name wrong: %s", rp.Name)
	} else if rp.Duration != time.Hour {
		t.Fatalf("rp duration wrong: %s", rp.Duration.String())
	} else if rp.ReplicaN != 1 {
		t.Fatalf("rp replication wrong: %d", rp.ReplicaN)
	}

	// Make sure default retention policy is now rp0
	if exp, got := "rp0", db.DefaultRetentionPolicy; exp != got {
		t.Fatalf("rp name wrong: \n\texp: %s\n\tgot: %s", exp, db.DefaultRetentionPolicy)
	}
}

func TestMetaClient_UpdateRetentionPolicy(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	if _, err := c.CreateDatabaseWithRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:               "rp0",
		ShardGroupDuration: 4 * time.Hour,
	}); err != nil {
		t.Fatal(err)
	}

	rpi, err := c.RetentionPolicy("db0", "rp0")
	if err != nil {
		t.Fatal(err)
	}

	// Set the duration to another value and ensure that the shard group duration
	// doesn't change.
	duration := 2 * rpi.ShardGroupDuration
	replicaN := 1
	if err := c.UpdateRetentionPolicy("db0", "rp0", &meta.RetentionPolicyUpdate{
		Duration: &duration,
		ReplicaN: &replicaN,
	}); err != nil {
		t.Fatal(err)
	}

	rpi, err = c.RetentionPolicy("db0", "rp0")
	if err != nil {
		t.Fatal(err)
	}
	if exp, got := 4*time.Hour, rpi.ShardGroupDuration; exp != got {
		t.Fatalf("shard group duration wrong: \n\texp: %s\n\tgot: %s", exp, got)
	}

	// Set the duration to below the shard group duration. This should return an error.
	duration = rpi.ShardGroupDuration / 2
	if err := c.UpdateRetentionPolicy("db0", "rp0", &meta.RetentionPolicyUpdate{
		Duration: &duration,
	}); err == nil {
		t.Fatal("expected error")
	} else if err != meta.ErrIncompatibleDurations {
		t.Fatalf("expected error '%s', got '%s'", meta.ErrIncompatibleDurations, err)
	}

	// Set the shard duration longer than the overall duration. This should also return an error.
	sgDuration := rpi.Duration * 2
	if err := c.UpdateRetentionPolicy("db0", "rp0", &meta.RetentionPolicyUpdate{
		ShardGroupDuration: &sgDuration,
	}); err == nil {
		t.Fatal("expected error")
	} else if err != meta.ErrIncompatibleDurations {
		t.Fatalf("expected error '%s', got '%s'", meta.ErrIncompatibleDurations, err)
	}

	// Set both values to incompatible values and ensure an error is returned.
	duration = rpi.ShardGroupDuration
	sgDuration = rpi.Duration
	if err := c.UpdateRetentionPolicy("db0", "rp0", &meta.RetentionPolicyUpdate{
		Duration:           &duration,
		ShardGroupDuration: &sgDuration,
	}); err == nil {
		t.Fatal("expected error")
	} else if err != meta.ErrIncompatibleDurations {
		t.Fatalf("expected error '%s', got '%s'", meta.ErrIncompatibleDurations, err)
	}

	// Allow any shard duration if the duration is set to zero.
	duration = time.Duration(0)
	sgDuration = 168 * time.Hour
	if err := c.UpdateRetentionPolicy("db0", "rp0", &meta.RetentionPolicyUpdate{
		Duration:           &duration,
		ShardGroupDuration: &sgDuration,
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestMetaClient_DropRetentionPolicy(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	db := c.Database("db0")
	if db == nil {
		t.Fatal("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	duration := 1 * time.Hour
	replicaN := 1
	if _, err := c.CreateRetentionPolicy("db0", &meta.RetentionPolicySpec{
		Name:     "rp0",
		Duration: &duration,
		ReplicaN: &replicaN,
	}); err != nil {
		t.Fatal(err)
	}

	rp, err := c.RetentionPolicy("db0", "rp0")
	if err != nil {
		t.Fatal(err)
	} else if rp.Name != "rp0" {
		t.Fatalf("rp name wrong: %s", rp.Name)
	} else if rp.Duration != time.Hour {
		t.Fatalf("rp duration wrong: %s", rp.Duration.String())
	} else if rp.ReplicaN != 1 {
		t.Fatalf("rp replication wrong: %d", rp.ReplicaN)
	}

	if err := c.DropRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	rp, err = c.RetentionPolicy("db0", "rp0")
	if err != nil {
		t.Fatal(err)
	} else if rp != nil {
		t.Fatalf("rp should have been dropped")
	}
}

func TestMetaClient_CreateUser(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	// Create an admin user
	if _, err := c.CreateUser("fred", "supersecure", true); err != nil {
		t.Fatal(err)
	}

	// Create a non-admin user
	if _, err := c.CreateUser("wilma", "password", false); err != nil {
		t.Fatal(err)
	}

	u, err := c.User("fred")
	if err != nil {
		t.Fatal(err)
	}
	if exp, got := "fred", u.Name; exp != got {
		t.Fatalf("unexpected user name: exp: %s got: %s", exp, got)
	}
	if !u.Admin {
		t.Fatalf("expected user to be admin")
	}

	u, err = c.Authenticate("fred", "supersecure")
	if u == nil || err != nil || u.Name != "fred" {
		t.Fatalf("failed to authenticate")
	}

	// Auth for bad password should fail
	u, err = c.Authenticate("fred", "badpassword")
	if u != nil || err != meta.ErrAuthenticate {
		t.Fatalf("authentication should fail with %s", meta.ErrAuthenticate)
	}

	// Auth for no password should fail
	u, err = c.Authenticate("fred", "")
	if u != nil || err != meta.ErrAuthenticate {
		t.Fatalf("authentication should fail with %s", meta.ErrAuthenticate)
	}

	// Change password should succeed.
	if err := c.UpdateUser("fred", "moresupersecure"); err != nil {
		t.Fatal(err)
	}

	// Auth for old password should fail
	u, err = c.Authenticate("fred", "supersecure")
	if u != nil || err != meta.ErrAuthenticate {
		t.Fatalf("authentication should fail with %s", meta.ErrAuthenticate)
	}

	// Auth for new password should succeed.
	u, err = c.Authenticate("fred", "moresupersecure")
	if u == nil || err != nil || u.Name != "fred" {
		t.Fatalf("failed to authenticate")
	}

	// Auth for unkonwn user should fail
	u, err = c.Authenticate("foo", "")
	if u != nil || err != meta.ErrUserNotFound {
		t.Fatalf("authentication should fail with %s", meta.ErrUserNotFound)
	}

	u, err = c.User("wilma")
	if err != nil {
		t.Fatal(err)
	}
	if exp, got := "wilma", u.Name; exp != got {
		t.Fatalf("unexpected user name: exp: %s got: %s", exp, got)
	}
	if u.Admin {
		t.Fatalf("expected user not to be an admin")
	}

	if exp, got := 2, c.UserCount(); exp != got {
		t.Fatalf("unexpected user count.  got: %d exp: %d", got, exp)
	}

	// Grant privilidges to a non-admin user
	if err := c.SetAdminPrivilege("wilma", true); err != nil {
		t.Fatal(err)
	}

	u, err = c.User("wilma")
	if err != nil {
		t.Fatal(err)
	}
	if exp, got := "wilma", u.Name; exp != got {
		t.Fatalf("unexpected user name: exp: %s got: %s", exp, got)
	}
	if !u.Admin {
		t.Fatalf("expected user to be an admin")
	}

	// Revoke privilidges from user
	if err := c.SetAdminPrivilege("wilma", false); err != nil {
		t.Fatal(err)
	}

	u, err = c.User("wilma")
	if err != nil {
		t.Fatal(err)
	}
	if exp, got := "wilma", u.Name; exp != got {
		t.Fatalf("unexpected user name: exp: %s got: %s", exp, got)
	}
	if u.Admin {
		t.Fatalf("expected user not to be an admin")
	}

	// Create a database to use for assiging privileges to.
	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	db := c.Database("db0")
	if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	// Assign a single privilege at the database level
	if err := c.SetPrivilege("wilma", "db0", influxql.ReadPrivilege); err != nil {
		t.Fatal(err)
	}

	p, err := c.UserPrivilege("wilma", "db0")
	if err != nil {
		t.Fatal(err)
	}
	if p == nil {
		t.Fatal("expected privilege but was nil")
	}
	if exp, got := influxql.ReadPrivilege, *p; exp != got {
		t.Fatalf("unexpected privilege.  exp: %d, got: %d", exp, got)
	}

	// Remove a single privilege at the database level
	if err := c.SetPrivilege("wilma", "db0", influxql.NoPrivileges); err != nil {
		t.Fatal(err)
	}
	p, err = c.UserPrivilege("wilma", "db0")
	if err != nil {
		t.Fatal(err)
	}
	if p == nil {
		t.Fatal("expected privilege but was nil")
	}
	if exp, got := influxql.NoPrivileges, *p; exp != got {
		t.Fatalf("unexpected privilege.  exp: %d, got: %d", exp, got)
	}

	// Drop a user
	if err := c.DropUser("wilma"); err != nil {
		t.Fatal(err)
	}

	u, err = c.User("wilma")
	if err != meta.ErrUserNotFound {
		t.Fatalf("user lookup should fail with %s", meta.ErrUserNotFound)
	}

	if exp, got := 1, c.UserCount(); exp != got {
		t.Fatalf("unexpected user count.  got: %d exp: %d", got, exp)
	}
}

func TestMetaClient_ContinuousQueries(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	// Create a database to use
	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}
	db := c.Database("db0")
	if db == nil {
		t.Fatalf("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	// Create a CQ
	if err := c.CreateContinuousQuery("db0", "cq0", `SELECT count(value) INTO foo_count FROM foo GROUP BY time(10m)`); err != nil {
		t.Fatal(err)
	}

	// Recreating an existing CQ with the exact same query should not
	// return an error.
	if err := c.CreateContinuousQuery("db0", "cq0", `SELECT count(value) INTO foo_count FROM foo GROUP BY time(10m)`); err != nil {
		t.Fatalf("got error %q, but didn't expect one", err)
	}

	// Recreating an existing CQ with a different query should return
	// an error.
	if err := c.CreateContinuousQuery("db0", "cq0", `SELECT min(value) INTO foo_max FROM foo GROUP BY time(20m)`); err == nil {
		t.Fatal("didn't get and error, but expected one")
	} else if got, exp := err, meta.ErrContinuousQueryExists; got.Error() != exp.Error() {
		t.Fatalf("got %v, expected %v", got, exp)
	}

	// Create a few more CQ's
	if err := c.CreateContinuousQuery("db0", "cq1", `SELECT max(value) INTO foo_max FROM foo GROUP BY time(10m)`); err != nil {
		t.Fatal(err)
	}
	if err := c.CreateContinuousQuery("db0", "cq2", `SELECT min(value) INTO foo_min FROM foo GROUP BY time(10m)`); err != nil {
		t.Fatal(err)
	}

	// Drop a single CQ
	if err := c.DropContinuousQuery("db0", "cq1"); err != nil {
		t.Fatal(err)
	}
}

func TestMetaClient_Subscriptions_Create(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	// Create a database to use
	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}
	db := c.Database("db0")
	if db == nil {
		t.Fatal("database not found")
	} else if db.Name != "db0" {
		t.Fatalf("db name wrong: %s", db.Name)
	}

	// Create a subscription
	if err := c.CreateSubscription("db0", "autogen", "sub0", "ALL", []string{"udp://example.com:9090"}); err != nil {
		t.Fatal(err)
	}

	// Re-create a subscription
	if err := c.CreateSubscription("db0", "autogen", "sub0", "ALL", []string{"udp://example.com:9090"}); err == nil || err.Error() != `subscription already exists` {
		t.Fatalf("unexpected error: %s", err)
	}

	// Create another subscription.
	if err := c.CreateSubscription("db0", "autogen", "sub1", "ALL", []string{"udp://example.com:6060"}); err != nil {
		t.Fatal(err)
	}
}

func TestMetaClient_Subscriptions_Drop(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	// Create a database to use
	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	// DROP SUBSCRIPTION returns ErrSubscriptionNotFound when the
	// subscription is unknown.
	err := c.DropSubscription("db0", "autogen", "foo")
	if got, exp := err, meta.ErrSubscriptionNotFound; got == nil || got.Error() != exp.Error() {
		t.Fatalf("got: %s, exp: %s", got, exp)
	}

	// Create a subscription.
	if err := c.CreateSubscription("db0", "autogen", "sub0", "ALL", []string{"udp://example.com:9090"}); err != nil {
		t.Fatal(err)
	}

	// DROP SUBSCRIPTION returns an influxdb.ErrDatabaseNotFound when
	// the database is unknown.
	err = c.DropSubscription("foo", "autogen", "sub0")
	if got, exp := err, influxdb.ErrDatabaseNotFound("foo"); got.Error() != exp.Error() {
		t.Fatalf("got: %s, exp: %s", got, exp)
	}

	// DROP SUBSCRIPTION returns an influxdb.ErrRetentionPolicyNotFound
	// when the retention policy is unknown.
	err = c.DropSubscription("db0", "foo_policy", "sub0")
	if got, exp := err, influxdb.ErrRetentionPolicyNotFound("foo_policy"); got.Error() != exp.Error() {
		t.Fatalf("got: %s, exp: %s", got, exp)
	}

	// DROP SUBSCRIPTION drops the subsciption if it can find it.
	err = c.DropSubscription("db0", "autogen", "sub0")
	if got := err; got != nil {
		t.Fatalf("got: %s, exp: %v", got, nil)
	}
}

func TestMetaClient_Shards(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	// Test creating a shard group.
	tmin := time.Now()
	sg, err := c.CreateShardGroup("db0", "autogen", tmin)
	if err != nil {
		t.Fatal(err)
	} else if sg == nil {
		t.Fatalf("expected ShardGroup")
	}

	// Test pre-creating shard groups.
	dur := sg.EndTime.Sub(sg.StartTime) + time.Nanosecond
	tmax := tmin.Add(dur)
	if err := c.PrecreateShardGroups(tmin, tmax); err != nil {
		t.Fatal(err)
	}

	// Test finding shard groups by time range.
	groups, err := c.ShardGroupsByTimeRange("db0", "autogen", tmin, tmax)
	if err != nil {
		t.Fatal(err)
	} else if len(groups) != 2 {
		t.Fatalf("wrong number of shard groups: %d", len(groups))
	}

	// Test finding shard owner.
	db, rp, owner := c.ShardOwner(groups[0].Shards[0].ID)
	if db != "db0" {
		t.Fatalf("wrong db name: %s", db)
	} else if rp != "autogen" {
		t.Fatalf("wrong rp name: %s", rp)
	} else if owner.ID != groups[0].ID {
		t.Fatalf("wrong owner: exp %d got %d", groups[0].ID, owner.ID)
	}

	// Test deleting a shard group.
	if err := c.DeleteShardGroup("db0", "autogen", groups[0].ID); err != nil {
		t.Fatal(err)
	} else if groups, err = c.ShardGroupsByTimeRange("db0", "autogen", tmin, tmax); err != nil {
		t.Fatal(err)
	} else if len(groups) != 1 {
		t.Fatalf("wrong number of shard groups after delete: %d", len(groups))
	}
}

// Tests that calling CreateShardGroup for the same time range doesn't increment the data.Index
func TestMetaClient_CreateShardGroupIdempotent(t *testing.T) {
	t.Parallel()

	d, c := newClient()
	defer os.RemoveAll(d)
	defer c.Close()

	if _, err := c.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	// create a shard group.
	tmin := time.Now()
	sg, err := c.CreateShardGroup("db0", "autogen", tmin)
	if err != nil {
		t.Fatal(err)
	} else if sg == nil {
		t.Fatalf("expected ShardGroup")
	}

	i := c.Data().Index
	t.Log("index: ", i)

	// create the same shard group.
	sg, err = c.CreateShardGroup("db0", "autogen", tmin)
	if err != nil {
		t.Fatal(err)
	} else if sg == nil {
		t.Fatalf("expected ShardGroup")
	}

	t.Log("index: ", i)
	if got, exp := c.Data().Index, i; got != exp {
		t.Fatalf("PrecreateShardGroups failed: invalid index, got %d, exp %d", got, exp)
	}

	// make sure pre-creating is also idempotent
	// Test pre-creating shard groups.
	dur := sg.EndTime.Sub(sg.StartTime) + time.Nanosecond
	tmax := tmin.Add(dur)
	if err := c.PrecreateShardGroups(tmin, tmax); err != nil {
		t.Fatal(err)
	}
	i = c.Data().Index
	t.Log("index: ", i)
	if err := c.PrecreateShardGroups(tmin, tmax); err != nil {
		t.Fatal(err)
	}
	t.Log("index: ", i)
	if got, exp := c.Data().Index, i; got != exp {
		t.Fatalf("PrecreateShardGroups failed: invalid index, got %d, exp %d", got, exp)
	}
}

func TestMetaClient_PersistClusterIDAfterRestart(t *testing.T) {
	t.Parallel()

	cfg := newConfig()
	defer os.RemoveAll(cfg.Dir)

	c := meta.NewClient(cfg)
	if err := c.Open(); err != nil {
		t.Fatal(err)
	}
	id := c.ClusterID()
	if id == 0 {
		t.Fatal("cluster ID can't be zero")
	}

	c = meta.NewClient(cfg)
	if err := c.Open(); err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	idAfter := c.ClusterID()
	if idAfter == 0 {
		t.Fatal("cluster ID can't be zero")
	} else if idAfter != id {
		t.Fatalf("cluster id not the same: %d, %d", idAfter, id)
	}
}

func newClient() (string, *meta.Client) {
	cfg := newConfig()
	c := meta.NewClient(cfg)
	if err := c.Open(); err != nil {
		panic(err)
	}
	return cfg.Dir, c
}

func newConfig() *meta.Config {
	cfg := meta.NewConfig()
	cfg.Dir = testTempDir(2)
	return cfg
}

func testTempDir(skip int) string {
	// Get name of the calling function.
	pc, _, _, ok := runtime.Caller(skip)
	if !ok {
		panic("failed to get name of test function")
	}
	_, prefix := path.Split(runtime.FuncForPC(pc).Name())
	// Make a temp dir prefixed with calling function's name.
	dir, err := ioutil.TempDir(os.TempDir(), prefix)
	if err != nil {
		panic(err)
	}
	return dir
}

func mustParseStatement(s string) influxql.Statement {
	stmt, err := influxql.ParseStatement(s)
	if err != nil {
		panic(err)
	}
	return stmt
}

func mustMarshalJSON(v interface{}) string {
	b, e := json.Marshal(v)
	if e != nil {
		panic(e)
	}
	return string(b)
}

func freePort() string {
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	defer l.Close()
	return l.Addr().String()
}

func freePorts(i int) []string {
	var ports []string
	for j := 0; j < i; j++ {
		ports = append(ports, freePort())
	}
	return ports
}
