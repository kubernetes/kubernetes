package meta_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/tcp"
	"github.com/influxdb/influxdb/toml"
	"golang.org/x/crypto/bcrypt"
)

// Ensure the store returns an error
func TestStore_Open_ErrStoreOpen(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	if err := s.Store.Open(); err != meta.ErrStoreOpen {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that opening a store with more than 3 peers returns an error.
func TestStore_Open_ErrTooManyPeers(t *testing.T) {
	t.Parallel()
	config := NewConfig(MustTempFile())
	config.Peers = []string{"localhost:9000", "localhost:9001", "localhost:9002", "localhost:9003"}
	s := NewStore(config)
	if err := s.Open(); err != meta.ErrTooManyPeers {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure the store can create a new node.
func TestStore_CreateNode(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create node.
	if ni, err := s.CreateNode("host0"); err != nil {
		t.Fatal(err)
	} else if *ni != (meta.NodeInfo{ID: 2, Host: "host0"}) {
		t.Fatalf("unexpected node: %#v", ni)
	}

	// Ensure cluster id is set.
	clusterID, err := s.ClusterID()
	if err != nil {
		t.Fatal(err)
	} else if clusterID == 0 {
		t.Fatal("expected cluster id to be set")
	}

	// Create another node.
	if ni, err := s.CreateNode("host1"); err != nil {
		t.Fatal(err)
	} else if *ni != (meta.NodeInfo{ID: 3, Host: "host1"}) {
		t.Fatalf("unexpected node: %#v", ni)
	}

	// Ensure cluster id remains the same.
	if id, err := s.ClusterID(); err != nil {
		t.Fatal(err)
	} else if id != clusterID {
		t.Fatalf("cluster id changed: %d", id)
	}
}

// Ensure that creating an existing node returns an error.
func TestStore_CreateNode_ErrNodeExists(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create node.
	if _, err := s.CreateNode("host0"); err != nil {
		t.Fatal(err)
	}

	// Create it again.
	if _, err := s.CreateNode("host0"); err != meta.ErrNodeExists {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure the store can find a node by ID.
func TestStore_Node(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create nodes.
	for i := 0; i < 3; i++ {
		if _, err := s.CreateNode(fmt.Sprintf("host%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	// Find second node.
	if ni, err := s.Node(3); err != nil {
		t.Fatal(err)
	} else if *ni != (meta.NodeInfo{ID: 3, Host: "host1"}) {
		t.Fatalf("unexpected node: %#v", ni)
	}
}

// Ensure the store can find a node by host.
func TestStore_NodeByHost(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create nodes.
	for i := 0; i < 3; i++ {
		if _, err := s.CreateNode(fmt.Sprintf("host%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	// Find second node.
	if ni, err := s.NodeByHost("host1"); err != nil {
		t.Fatal(err)
	} else if *ni != (meta.NodeInfo{ID: 3, Host: "host1"}) {
		t.Fatalf("unexpected node: %#v", ni)
	}
}

// Ensure the store can delete an existing node.
func TestStore_DeleteNode(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create nodes.
	for i := 0; i < 3; i++ {
		if _, err := s.CreateNode(fmt.Sprintf("host%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	// Remove second node.
	if err := s.DeleteNode(3); err != nil {
		t.Fatal(err)
	}

	// Ensure remaining nodes are correct.
	if ni, _ := s.Node(2); *ni != (meta.NodeInfo{ID: 2, Host: "host0"}) {
		t.Fatalf("unexpected node(1): %#v", ni)
	}
	if ni, _ := s.Node(3); ni != nil {
		t.Fatalf("unexpected node(2): %#v", ni)
	}
	if ni, _ := s.Node(4); *ni != (meta.NodeInfo{ID: 4, Host: "host2"}) {
		t.Fatalf("unexpected node(3): %#v", ni)
	}
}

// Ensure the store returns an error when deleting a node that doesn't exist.
func TestStore_DeleteNode_ErrNodeNotFound(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	if err := s.DeleteNode(2); err != meta.ErrNodeNotFound {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure the store can create a new database.
func TestStore_CreateDatabase(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create database.
	if di, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(di, &meta.DatabaseInfo{Name: "db0"}) {
		t.Fatalf("unexpected database: %#v", di)
	}

	// Create another database.
	if di, err := s.CreateDatabase("db1"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(di, &meta.DatabaseInfo{Name: "db1"}) {
		t.Fatalf("unexpected database: %#v", di)
	}
}

// Ensure the store can delete an existing database.
func TestStore_DropDatabase(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create databases.
	for i := 0; i < 3; i++ {
		if _, err := s.CreateDatabase(fmt.Sprintf("db%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	// Remove a database.
	if err := s.DropDatabase("db1"); err != nil {
		t.Fatal(err)
	}

	// Ensure remaining nodes are correct.
	if di, _ := s.Database("db0"); !reflect.DeepEqual(di, &meta.DatabaseInfo{Name: "db0"}) {
		t.Fatalf("unexpected database(0): %#v", di)
	}
	if di, _ := s.Database("db1"); di != nil {
		t.Fatalf("unexpected database(1): %#v", di)
	}
	if di, _ := s.Database("db2"); !reflect.DeepEqual(di, &meta.DatabaseInfo{Name: "db2"}) {
		t.Fatalf("unexpected database(2): %#v", di)
	}
}

// Ensure the store returns an error when dropping a database that doesn't exist.
func TestStore_DropDatabase_ErrDatabaseNotFound(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	if err := s.DropDatabase("no_such_database"); err != meta.ErrDatabaseNotFound {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure the store can create a retention policy on a database.
func TestStore_CreateRetentionPolicy(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create an additional nodes and database.
	if _, err := s.CreateNode("hostX"); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	// Create policy on database.
	if rpi, err := s.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{
		Name:     "rp0",
		ReplicaN: 2,
		Duration: 48 * time.Hour,
	}); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(rpi, &meta.RetentionPolicyInfo{
		Name:               "rp0",
		ReplicaN:           2,
		Duration:           48 * time.Hour,
		ShardGroupDuration: 24 * time.Hour,
	}) {
		t.Fatalf("unexpected policy: %#v", rpi)
	}
}

// Ensure the store can delete a retention policy.
func TestStore_DropRetentionPolicy(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create database.
	if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	// Create policies.
	for i := 0; i < 3; i++ {
		if _, err := s.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: fmt.Sprintf("rp%d", i), ReplicaN: 1}); err != nil {
			t.Fatal(err)
		}
	}

	// Remove a policy.
	if err := s.DropRetentionPolicy("db0", "rp1"); err != nil {
		t.Fatal(err)
	}

	// Ensure remaining policies are correct.
	if rpi, _ := s.RetentionPolicy("db0", "rp0"); !reflect.DeepEqual(rpi, &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1, ShardGroupDuration: 7 * 24 * time.Hour}) {
		t.Fatalf("unexpected policy(0): %#v", rpi)
	}
	if rpi, _ := s.RetentionPolicy("db0", "rp1"); rpi != nil {
		t.Fatalf("unexpected policy(1): %#v", rpi)
	}
	if rpi, _ := s.RetentionPolicy("db0", "rp2"); !reflect.DeepEqual(rpi, &meta.RetentionPolicyInfo{Name: "rp2", ReplicaN: 1, ShardGroupDuration: 7 * 24 * time.Hour}) {
		t.Fatalf("unexpected policy(2): %#v", rpi)
	}
}

// Ensure the store can set the default retention policy on a database.
func TestStore_SetDefaultRetentionPolicy(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create database.
	if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	}

	// Set default policy.
	if err := s.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	// Ensure default policy is set.
	if di, _ := s.Database("db0"); di.DefaultRetentionPolicy != "rp0" {
		t.Fatalf("unexpected default retention policy: %s", di.DefaultRetentionPolicy)
	}
}

// Ensure the store can update a retention policy.
func TestStore_UpdateRetentionPolicy(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create database.
	if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	}

	// Update policy.
	var rpu meta.RetentionPolicyUpdate
	rpu.SetName("rp1")
	rpu.SetDuration(10 * time.Hour)
	if err := s.UpdateRetentionPolicy("db0", "rp0", &rpu); err != nil {
		t.Fatal(err)
	}

	// Ensure policy is updated.
	if rpi, err := s.RetentionPolicy("db0", "rp1"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(rpi, &meta.RetentionPolicyInfo{
		Name:               "rp1",
		Duration:           10 * time.Hour,
		ShardGroupDuration: 7 * 24 * time.Hour,
		ReplicaN:           1,
	}) {
		t.Fatalf("unexpected policy: %#v", rpi)
	}
}

// Ensure the store can create a shard group on a retention policy.
func TestStore_CreateShardGroup(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create node & database.
	if _, err := s.CreateNode("host0"); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if _, err = s.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 2, Duration: 1 * time.Hour}); err != nil {
		t.Fatal(err)
	}

	// Create policy on database.
	if sgi, err := s.CreateShardGroup("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	} else if sgi.ID != 1 {
		t.Fatalf("unexpected shard group: %#v", sgi)
	}

}

func TestStore_ShardGroupsRetrieval(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create resources for testing.
	if _, err := s.CreateNode("host0"); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if _, err = s.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 2, Duration: 1 * time.Hour}); err != nil {
		t.Fatal(err)
	}
	if sgi, err := s.CreateShardGroup("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	} else if sgi.ID != 1 {
		t.Fatalf("unexpected shard group: %#v", sgi)
	}

	// Function to compare actual and expected. Works on integers only, as results require sorting.
	assertShardGroupsInTimeRange := func(database, policy string, actualGroups []meta.ShardGroupInfo, expectedGroupIDs []int) {
		if len(actualGroups) != len(expectedGroupIDs) {
			t.Fatalf(("number of actual groups (%d) does not equal number expected groups (%d)"), len(actualGroups), len(expectedGroupIDs))
		}

		actualGroupIDs := []int{}
		for i := range actualGroups {
			actualGroupIDs = append(actualGroupIDs, int(actualGroups[i].ID))
		}

		sort.Ints(actualGroupIDs)
		sort.Ints(expectedGroupIDs)
		for i := range actualGroupIDs {
			if actualGroupIDs[i] != expectedGroupIDs[i] {
				t.Fatalf("actual group IDs (%v) does not match expected group IDs (%v)", actualGroupIDs, expectedGroupIDs)
			}
		}
	}

	// Check that it is returned correctly when requested.
	if sgs, err := s.ShardGroups("db0", "rp0"); err != nil {
		t.Fatal(err)
	} else {
		assertShardGroupsInTimeRange("db0", "rp0", sgs, []int{1})
	}

	if sgs, err := s.ShardGroupsByTimeRange("db0", "rp0", time.Date(1999, time.January, 1, 0, 0, 0, 0, time.UTC), time.Date(1999, time.January, 2, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	} else {
		assertShardGroupsInTimeRange("db0", "rp0", sgs, []int{})
	}
	if sgs, err := s.ShardGroupsByTimeRange("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC), time.Date(2001, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	} else {
		assertShardGroupsInTimeRange("db0", "rp0", sgs, []int{1})
	}
	if sgs, err := s.ShardGroupsByTimeRange("db0", "rp0", time.Date(1999, time.January, 1, 0, 0, 0, 0, time.UTC), time.Date(2001, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	} else {
		assertShardGroupsInTimeRange("db0", "rp0", sgs, []int{1})
	}
	if sgs, err := s.ShardGroupsByTimeRange("db0", "rp0", time.Date(2002, time.January, 1, 0, 0, 0, 0, time.UTC), time.Date(2002, time.January, 2, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	} else {
		assertShardGroupsInTimeRange("db0", "rp0", sgs, []int{})
	}
}

// Ensure the store can delete an existing shard group.
func TestStore_DeleteShardGroup(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create node, database, policy, & group.
	if _, err := s.CreateNode("host0"); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if _, err = s.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 2, Duration: 1 * time.Hour}); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateShardGroup("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	}

	// Remove policy from database.
	if err := s.DeleteShardGroup("db0", "rp0", 1); err != nil {
		t.Fatal(err)
	}
}

// Ensure the store correctly precreates shard groups.
func TestStore_PrecreateShardGroup(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create node, database, policy, & group.
	if _, err := s.CreateNode("host0"); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if _, err = s.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 2, Duration: 1 * time.Hour}); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateShardGroup("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	} else if err := s.PrecreateShardGroups(time.Date(2001, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	}

	groups, err := s.ShardGroups("db0", "rp0")
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 {
		t.Fatalf("shard group precreation failed to create new shard group")
	}
	if groups[1].StartTime != time.Date(2000, time.January, 1, 1, 0, 0, 0, time.UTC) {
		t.Fatalf("precreated shard group has wrong start time, exp %s, got %s",
			time.Date(2000, time.January, 1, 1, 0, 0, 0, time.UTC), groups[1].StartTime)
	}
}

// Ensure the store can create a new continuous query.
func TestStore_CreateContinuousQuery(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create query.
	if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err := s.CreateContinuousQuery("db0", "cq0", "SELECT count() FROM foo"); err != nil {
		t.Fatal(err)
	}
}

// Ensure that creating an existing continuous query returns an error.
func TestStore_CreateContinuousQuery_ErrContinuousQueryExists(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create continuous query.
	if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err := s.CreateContinuousQuery("db0", "cq0", "SELECT count() FROM foo"); err != nil {
		t.Fatal(err)
	}

	// Create it again.
	if err := s.CreateContinuousQuery("db0", "cq0", "SELECT count() FROM foo"); err != meta.ErrContinuousQueryExists {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure the store can delete a continuous query.
func TestStore_DropContinuousQuery(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create queries.
	if _, err := s.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err := s.CreateContinuousQuery("db0", "cq0", "SELECT count() FROM foo"); err != nil {
		t.Fatal(err)
	} else if err = s.CreateContinuousQuery("db0", "cq1", "SELECT count() FROM bar"); err != nil {
		t.Fatal(err)
	} else if err = s.CreateContinuousQuery("db0", "cq2", "SELECT count() FROM baz"); err != nil {
		t.Fatal(err)
	}

	// Remove one of the queries.
	if err := s.DropContinuousQuery("db0", "cq1"); err != nil {
		t.Fatal(err)
	}

	// Ensure the resulting queries are correct.
	if di, err := s.Database("db0"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(di.ContinuousQueries, []meta.ContinuousQueryInfo{
		{Name: "cq0", Query: "SELECT count() FROM foo"},
		{Name: "cq2", Query: "SELECT count() FROM baz"},
	}) {
		t.Fatalf("unexpected queries: %#v", di.ContinuousQueries)
	}
}

// Ensure the store can create a user.
func TestStore_CreateUser(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create user.
	if ui, err := s.CreateUser("susy", "pass", true); err != nil {
		t.Fatal(err)
	} else if ui.Name != "susy" || ui.Hash == "" || ui.Admin != true {
		t.Fatalf("unexpected user: %#v", ui)
	}
}

// Ensure the store can remove a user.
func TestStore_DropUser(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create users.
	if _, err := s.CreateUser("susy", "pass", true); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateUser("bob", "pass", true); err != nil {
		t.Fatal(err)
	}

	// Remove user.
	if err := s.DropUser("bob"); err != nil {
		t.Fatal(err)
	}

	// Verify user was removed.
	if a, err := s.Users(); err != nil {
		t.Fatal(err)
	} else if len(a) != 1 {
		t.Fatalf("unexpected user count: %d", len(a))
	} else if a[0].Name != "susy" {
		t.Fatalf("unexpected user: %s", a[0].Name)
	}
}

// Ensure the store can update a user.
func TestStore_UpdateUser(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	// Create users.
	if _, err := s.CreateUser("susy", "pass", true); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateUser("bob", "pass", true); err != nil {
		t.Fatal(err)
	}

	// Store password hash for bob.
	ui, err := s.User("bob")
	if err != nil {
		t.Fatal(err)
	}

	// Update user.
	if err := s.UpdateUser("bob", "XXX"); err != nil {
		t.Fatal(err)
	}

	// Verify password hash was updated.
	if other, err := s.User("bob"); err != nil {
		t.Fatal(err)
	} else if ui.Hash == other.Hash {
		t.Fatal("password hash did not change")
	}
}

// Ensure Authentication works.
func TestStore_Authentication(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.SkipNow()
	}

	s := MustOpenStore()
	defer s.Close()

	// Set the password hash function to the real thing for this test.
	s.SetHashPasswordFn(func(password string) ([]byte, error) {
		return bcrypt.GenerateFromPassword([]byte(password), 4)
	})

	// Create user.
	s.CreateUser("susy", "pass", true)

	// Authenticate user.
	if ui, err := s.Authenticate("susy", "pass"); err != nil {
		t.Fatal(err)
	} else if ui.Name != "susy" {
		t.Fatalf(`expected "susy", got "%s"`, ui.Name)
	}

	// Update user's password.
	s.UpdateUser("susy", "pass2")

	// Make sure authentication with old password does NOT work.
	if _, err := s.Authenticate("susy", "pass"); err == nil {
		t.Fatal("expected authentication error")
	}

	// Authenticate user with new password
	if ui, err := s.Authenticate("susy", "pass2"); err != nil {
		t.Fatal(err)
	} else if ui.Name != "susy" {
		t.Fatalf(`expected "susy", got "%s"`, ui.Name)
	}

	// Drop user.
	s.DropUser("susy")

	// Make sure authentication with both old passwords does NOT work.
	if _, err := s.Authenticate("susy", "pass"); err == nil {
		t.Fatal("expected authentication error")
	} else if _, err := s.Authenticate("susy", "pass2"); err == nil {
		t.Fatal("expected authentication error")
	}
}

// Ensure the store can return the count of users in it.
func TestStore_UserCount(t *testing.T) {
	t.Parallel()
	s := MustOpenStore()
	defer s.Close()

	if count, err := s.UserCount(); count != 0 && err != nil {
		t.Fatalf("expected user count to be 0 but was %d", count)
	}

	// Create users.
	if _, err := s.CreateUser("susy", "pass", true); err != nil {
		t.Fatal(err)
	} else if _, err := s.CreateUser("bob", "pass", true); err != nil {
		t.Fatal(err)
	}

	if count, err := s.UserCount(); count != 2 && err != nil {
		t.Fatalf("expected user count to be 2 but was %d", count)
	}
}

// Ensure the store can take a snapshot.
func TestStore_Snapshot_And_Restore(t *testing.T) {
	t.Parallel()

	s := MustOpenStore()
	s.LeaveFiles = true

	// Create a bunch of databases in the Store
	nDatabases := 5
	for n := 0; n < nDatabases; n++ {
		s.CreateDatabase(fmt.Sprintf("db%d", n))
	}

	// Test taking a snapshot.
	if err := s.Store.Snapshot(); err != nil {
		t.Fatal(err)
	}

	s.Close()

	// Test restoring the snapshot taken above.
	existingDataPath := s.Path()
	s = NewStore(NewConfig(existingDataPath))
	if err := s.Open(); err != nil {
		panic(err)
	}
	defer s.Close()

	// Wait until the server is ready.
	select {
	case err := <-s.Err():
		panic(err)
	case <-s.Ready():
	}

	// Make sure all the data we added to the Store is still there.
	for n := 0; n < nDatabases; n++ {
		name := fmt.Sprintf("db%d", n)
		if dbi, err := s.Database(name); err != nil {
			t.Fatal(err)
		} else if dbi == nil {
			t.Fatalf("database not found: %s", name)
		} else if dbi.Name != name {
			t.Fatal(name)
		}
	}
}

// Ensure a multi-node cluster can start, join the cluster, and replicate commands.
func TestCluster_Open(t *testing.T) {
	c := MustOpenCluster(3)
	defer c.Close()

	// Check that one node is leader and two are followers.
	if s := c.Leader(); s == nil {
		t.Fatal("no leader found")
	}

	// Add a database to each node.
	for i, s := range c.Stores {
		if di, err := s.CreateDatabase(fmt.Sprintf("db%d", i)); err != nil {
			t.Fatal(err)
		} else if di == nil {
			t.Fatal("expected database")
		}
	}

	// Verify that each store has all databases.
	for i := 0; i < len(c.Stores); i++ {
		for _, s := range c.Stores {
			if di, err := s.Database(fmt.Sprintf("db%d", i)); err != nil {
				t.Fatal(err)
			} else if di == nil {
				t.Fatal("expected database")
			}
		}
	}
}

// Store is a test wrapper for meta.Store.
type Store struct {
	*meta.Store
	Listener   net.Listener
	Stderr     bytes.Buffer
	LeaveFiles bool // set to true to leave temporary files on close
}

// NewStore returns a new test wrapper for Store.
func NewStore(c meta.Config) *Store {
	s := &Store{
		Store: meta.NewStore(c),
	}
	s.Logger = log.New(&s.Stderr, "", log.LstdFlags)
	s.SetHashPasswordFn(mockHashPassword)
	return s
}

// MustOpenStore opens a store in a temporary path. Panic on error.
func MustOpenStore() *Store {
	s := NewStore(NewConfig(MustTempFile()))
	if err := s.Open(); err != nil {
		panic(err)
	}

	// Wait until the server is ready.
	select {
	case err := <-s.Err():
		panic(err)
	case <-s.Ready():
	}

	return s
}

// Open opens the store on a random TCP port.
func (s *Store) Open() error {
	// Open a TCP port.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("listen: %s", err)
	}
	s.Addr = ln.Addr()
	s.Listener = ln

	// Wrap listener in a muxer.
	mux := tcp.NewMux()
	s.RaftListener = mux.Listen(meta.MuxRaftHeader)
	s.ExecListener = mux.Listen(meta.MuxExecHeader)
	go mux.Serve(ln)

	// Open store.
	if err := s.Store.Open(); err != nil {
		return err
	}

	return nil
}

// Close shuts down the store and removes all data from the path.
func (s *Store) Close() error {
	if s.Listener != nil {
		s.Listener.Close()
	}
	if !s.LeaveFiles {
		defer os.RemoveAll(s.Path())
	}
	return s.Store.Close()
}

// NewConfig returns the default test configuration.
func NewConfig(path string) meta.Config {
	return meta.Config{
		Dir:                path,
		Hostname:           "localhost",
		BindAddress:        "127.0.0.1:0",
		HeartbeatTimeout:   toml.Duration(500 * time.Millisecond),
		ElectionTimeout:    toml.Duration(500 * time.Millisecond),
		LeaderLeaseTimeout: toml.Duration(500 * time.Millisecond),
		CommitTimeout:      toml.Duration(5 * time.Millisecond),
	}
}

// Cluster represents a group of stores joined as a raft cluster.
type Cluster struct {
	Stores []*Store
}

// NewCluster returns a cluster of n stores within path.
func NewCluster(path string, n int) *Cluster {
	c := &Cluster{}

	// Construct a list of temporary peers.
	peers := make([]string, n)
	for i := range peers {
		peers[i] = "127.0.0.1:0"
	}

	// Create new stores with temporary peers.
	for i := 0; i < n; i++ {
		config := NewConfig(filepath.Join(path, strconv.Itoa(i)))
		config.Peers = peers
		s := NewStore(config)
		c.Stores = append(c.Stores, s)
	}

	return c
}

// MustOpenCluster opens a cluster in a temporary path. Panic on error.
func MustOpenCluster(n int) *Cluster {
	c := NewCluster(MustTempFile(), n)
	if err := c.Open(); err != nil {
		panic(err.Error())
	}

	// Wait for all stores to be ready.
	for i, s := range c.Stores {
		select {
		case err := <-s.Err():
			panic(fmt.Sprintf("store: i=%d, addr=%s, err=%s", i, s.Addr.String(), err))
		case <-s.Ready():
		}
	}
	return c
}

// Open opens and initializes all stores in the cluster.
func (c *Cluster) Open() error {
	if err := func() error {
		// Open each store and add to peer list.
		peers := make([]string, len(c.Stores))
		for i, s := range c.Stores {
			if err := s.Open(); err != nil {
				return fmt.Errorf("open test store #%d: %s", i, err)
			}
			peers[i] = s.Addr.String()
		}

		// Reset peers on all stores.
		for _, s := range c.Stores {
			if err := s.SetPeers(peers); err != nil {
				return fmt.Errorf("set peers: %s", err)
			}
		}

		return nil
	}(); err != nil {
		c.Close()
		return err
	}
	return nil
}

// Close shuts down all stores.
func (c *Cluster) Close() error {
	for _, s := range c.Stores {
		s.Close()
	}
	return nil
}

// Leader returns the store that is currently leader.
func (c *Cluster) Leader() *Store {
	for _, s := range c.Stores {
		if s.IsLeader() {
			return s
		}
	}
	return nil
}

// MustTempFile returns the path to a non-existent temporary file.
func MustTempFile() string {
	f, _ := ioutil.TempFile("", "influxdb-meta-")
	f.Close()
	os.Remove(f.Name())
	return f.Name()
}

// mockHashPassword is used for most tests to avoid slow calls to bcrypt.
func mockHashPassword(password string) ([]byte, error) {
	return []byte(password), nil
}
