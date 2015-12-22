package meta_test

// import "github.com/davecgh/go-spew/spew"

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/gogo/protobuf/proto"
	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/meta/internal"
)

// Ensure a node can be created.
func TestData_CreateNode(t *testing.T) {
	var data meta.Data
	if err := data.CreateNode("host0"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Nodes, []meta.NodeInfo{{ID: 1, Host: "host0"}}) {
		t.Fatalf("unexpected node: %#v", data.Nodes[0])
	}
}

// Ensure a node can be removed.
func TestData_DeleteNode_Basic(t *testing.T) {
	var data meta.Data
	if err := data.CreateNode("host0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateNode("host1"); err != nil {
		t.Fatal(err)
	} else if err := data.CreateNode("host2"); err != nil {
		t.Fatal(err)
	}

	if err := data.DeleteNode(1, false); err != nil {
		t.Fatal(err)
	} else if len(data.Nodes) != 2 {
		t.Fatalf("unexpected node count: %d", len(data.Nodes))
	} else if data.Nodes[0] != (meta.NodeInfo{ID: 2, Host: "host1"}) {
		t.Fatalf("unexpected node: %#v", data.Nodes[0])
	} else if data.Nodes[1] != (meta.NodeInfo{ID: 3, Host: "host2"}) {
		t.Fatalf("unexpected node: %#v", data.Nodes[1])
	}
}

// Ensure a node can be removed with shard info in play
func TestData_DeleteNode_Shards(t *testing.T) {
	var data meta.Data
	if err := data.CreateNode("host0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateNode("host1"); err != nil {
		t.Fatal(err)
	} else if err := data.CreateNode("host2"); err != nil {
		t.Fatal(err)
	} else if err := data.CreateNode("host3"); err != nil {
		t.Fatal(err)
	}

	if err := data.CreateDatabase("mydb"); err != nil {
		t.Fatal(err)
	}

	rpi := &meta.RetentionPolicyInfo{
		Name:     "myrp",
		ReplicaN: 3,
	}
	if err := data.CreateRetentionPolicy("mydb", rpi); err != nil {
		t.Fatal(err)
	}
	if err := data.CreateShardGroup("mydb", "myrp", time.Now()); err != nil {
		t.Fatal(err)
	}
	if len(data.Databases[0].RetentionPolicies[0].ShardGroups[0].Shards[0].Owners) != 3 {
		t.Fatal("wrong number of shard owners")
	}
	if err := data.DeleteNode(2, false); err != nil {
		t.Fatal(err)
	}
	if got, exp := len(data.Databases[0].RetentionPolicies[0].ShardGroups[0].Shards[0].Owners), 2; exp != got {
		t.Fatalf("wrong number of shard owners, got %d, exp %d", got, exp)
	}
	for _, s := range data.Databases[0].RetentionPolicies[0].ShardGroups[0].Shards {
		if s.OwnedBy(2) {
			t.Fatal("shard still owned by delted node")
		}
	}
}

// Ensure a database can be created.
func TestData_CreateDatabase(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Databases, []meta.DatabaseInfo{{Name: "db0"}}) {
		t.Fatalf("unexpected databases: %#v", data.Databases)
	}
}

// Ensure that creating a database without a name returns an error.
func TestData_CreateDatabase_ErrNameRequired(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase(""); err != meta.ErrDatabaseNameRequired {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that creating an already existing database returns an error.
func TestData_CreateDatabase_ErrDatabaseExists(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}
	if err := data.CreateDatabase("db0"); err != meta.ErrDatabaseExists {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure a database can be removed.
func TestData_DropDatabase(t *testing.T) {
	var data meta.Data
	for i := 0; i < 3; i++ {
		if err := data.CreateDatabase(fmt.Sprintf("db%d", i)); err != nil {
			t.Fatal(err)
		}
	}

	if err := data.DropDatabase("db1"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Databases, []meta.DatabaseInfo{{Name: "db0"}, {Name: "db2"}}) {
		t.Fatalf("unexpected databases: %#v", data.Databases)
	}
}

// Ensure a retention policy can be created.
func TestData_CreateRetentionPolicy(t *testing.T) {
	data := meta.Data{Nodes: []meta.NodeInfo{{ID: 1}, {ID: 2}}}
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}

	// Create policy.
	if err := data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{
		Name:     "rp0",
		ReplicaN: 2,
		Duration: 4 * time.Hour,
	}); err != nil {
		t.Fatal(err)
	}

	// Verify policy exists.
	if !reflect.DeepEqual(data.Databases[0].RetentionPolicies, []meta.RetentionPolicyInfo{
		{
			Name:               "rp0",
			ReplicaN:           2,
			Duration:           4 * time.Hour,
			ShardGroupDuration: 1 * time.Hour,
		},
	}) {
		t.Fatalf("unexpected policies: %#v", data.Databases[0].RetentionPolicies)
	}
}

// Ensure that creating a policy without a name returns an error.
func TestData_CreateRetentionPolicy_ErrNameRequired(t *testing.T) {
	data := meta.Data{Nodes: []meta.NodeInfo{{ID: 1}}}
	if err := data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: ""}); err != meta.ErrRetentionPolicyNameRequired {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that creating a policy with a replication factor less than 1 returns an error.
func TestData_CreateRetentionPolicy_ErrReplicationFactorTooLow(t *testing.T) {
	data := meta.Data{Nodes: []meta.NodeInfo{{ID: 1}}}
	if err := data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 0}); err != meta.ErrReplicationFactorTooLow {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that creating a retention policy on a non-existent database returns an error.
func TestData_CreateRetentionPolicy_ErrDatabaseNotFound(t *testing.T) {
	data := meta.Data{Nodes: []meta.NodeInfo{{ID: 1}}}
	expErr := influxdb.ErrDatabaseNotFound("db0")
	if err := data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err.Error() != expErr.Error() {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that creating an already existing policy returns an error.
func TestData_CreateRetentionPolicy_ErrRetentionPolicyExists(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	}
	if err := data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != meta.ErrRetentionPolicyExists {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure that a retention policy can be updated.
func TestData_UpdateRetentionPolicy(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	}

	// Update the policy.
	var rpu meta.RetentionPolicyUpdate
	rpu.SetName("rp1")
	rpu.SetDuration(10 * time.Hour)
	rpu.SetReplicaN(3)
	if err := data.UpdateRetentionPolicy("db0", "rp0", &rpu); err != nil {
		t.Fatal(err)
	}

	// Verify the policy was changed.
	if rpi, _ := data.RetentionPolicy("db0", "rp1"); !reflect.DeepEqual(rpi, &meta.RetentionPolicyInfo{
		Name:               "rp1",
		Duration:           10 * time.Hour,
		ShardGroupDuration: 3600000000000,
		ReplicaN:           3,
	}) {
		t.Fatalf("unexpected policy: %#v", rpi)
	}
}

// Ensure a retention policy can be removed.
func TestData_DropRetentionPolicy(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	}

	if err := data.DropRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	} else if len(data.Databases[0].RetentionPolicies) != 0 {
		t.Fatalf("unexpected policies: %#v", data.Databases[0].RetentionPolicies)
	}
}

// Ensure an error is returned when deleting a policy from a non-existent database.
func TestData_DropRetentionPolicy_ErrDatabaseNotFound(t *testing.T) {
	var data meta.Data
	expErr := influxdb.ErrDatabaseNotFound("db0")
	if err := data.DropRetentionPolicy("db0", "rp0"); err.Error() != expErr.Error() {
		t.Fatal(err)
	}
}

// Ensure an error is returned when deleting a non-existent policy.
func TestData_DropRetentionPolicy_ErrRetentionPolicyNotFound(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	}
	expErr := influxdb.ErrRetentionPolicyNotFound("rp0")
	if err := data.DropRetentionPolicy("db0", "rp0"); err.Error() != expErr.Error() {
		t.Fatal(err)
	}
}

// Ensure that a retention policy can be retrieved.
func TestData_RetentionPolicy(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp1", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	}

	if rpi, err := data.RetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(rpi, &meta.RetentionPolicyInfo{
		Name:               "rp0",
		ShardGroupDuration: 604800000000000,
		ReplicaN:           1,
	}) {
		t.Fatalf("unexpected value: %#v", rpi)
	}
}

// Ensure that retrieving a policy from a non-existent database returns an error.
func TestData_RetentionPolicy_ErrDatabaseNotFound(t *testing.T) {
	var data meta.Data
	expErr := influxdb.ErrDatabaseNotFound("db0")
	if _, err := data.RetentionPolicy("db0", "rp0"); err.Error() != expErr.Error() {
		t.Fatal(err)
	}
}

// Ensure that a default retention policy can be set.
func TestData_SetDefaultRetentionPolicy(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	}

	// Verify there is no default policy on the database initially.
	if name := data.Database("db0").DefaultRetentionPolicy; name != "" {
		t.Fatalf("unexpected initial default retention policy: %s", name)
	}

	// Set the default policy.
	if err := data.SetDefaultRetentionPolicy("db0", "rp0"); err != nil {
		t.Fatal(err)
	}

	// Verify the default policy is now set.
	if name := data.Database("db0").DefaultRetentionPolicy; name != "rp0" {
		t.Fatalf("unexpected default retention policy: %s", name)
	}
}

// Ensure that a shard group can be created on a database for a given timestamp.
func TestData_CreateShardGroup(t *testing.T) {
	var data meta.Data
	if err := data.CreateNode("node0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateNode("node1"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 2, Duration: 1 * time.Hour}); err != nil {
		t.Fatal(err)
	}

	// Create shard group.
	if err := data.CreateShardGroup("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	}

	// Verify the shard group was created.
	if sgi, _ := data.ShardGroupByTimestamp("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)); !reflect.DeepEqual(sgi, &meta.ShardGroupInfo{
		ID:        1,
		StartTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC),
		EndTime:   time.Date(2000, time.January, 1, 1, 0, 0, 0, time.UTC),
		Shards: []meta.ShardInfo{
			{
				ID: 1,
				Owners: []meta.ShardOwner{
					{NodeID: 1},
					{NodeID: 2},
				},
			},
		},
	}) {
		t.Fatalf("unexpected shard group: %#v", sgi)
	} else if !sgi.Shards[0].OwnedBy(1) || !sgi.Shards[0].OwnedBy(2) || sgi.Shards[0].OwnedBy(3) {
		// Verify shard is correctly owned-by the node.
		t.Fatalf("new shard is not owned by correct node")
	}
}

// Ensure that a shard group is correctly detected as expired.
func TestData_ShardGroupExpiredDeleted(t *testing.T) {
	var data meta.Data
	if err := data.CreateNode("node0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateNode("node1"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 2, Duration: 1 * time.Hour}); err != nil {
		t.Fatal(err)
	}

	// Create shard groups.
	if err := data.CreateShardGroup("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	}
	if err := data.CreateShardGroup("db0", "rp0", time.Date(2001, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	}

	// Check expiration.
	rp, _ := data.RetentionPolicy("db0", "rp0")
	groups := rp.ExpiredShardGroups(time.Date(2001, time.January, 1, 0, 0, 0, 0, time.UTC))
	if len(groups) != 1 {
		t.Fatalf("wrong number of expired shard groups returned, got %d, exp 1", len(groups))
	}
	if groups[0].StartTime != time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC) {
		t.Fatal("wrong shard group marked as expired")
	}

	// Check deletion.
	if err := data.DeleteShardGroup("db0", "rp0", groups[0].ID); err != nil {
		t.Fatal(err)
	}
	groups = rp.DeletedShardGroups()
	if len(groups) != 1 {
		t.Fatalf("wrong number of deleted shard groups returned, got %d, exp 1", len(groups))
	}
	if groups[0].StartTime != time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC) {
		t.Fatal("wrong shard group marked as expired")
	}

}

// Test shard group selection.
func TestShardGroup_Overlaps(t *testing.T) {
	// Make a shard group 1 hour in duration
	startTime, _ := time.Parse(time.RFC3339, "2000-01-01T00:00:00Z")
	endTime := startTime.Add(time.Hour)
	g := &meta.ShardGroupInfo{StartTime: startTime, EndTime: endTime}

	if !g.Overlaps(g.StartTime.Add(-time.Minute), g.EndTime) {
		t.Fatal("shard group not selected when min before start time")
	}

	if !g.Overlaps(g.StartTime.Add(-time.Minute), g.StartTime) {
		t.Fatal("shard group not selected when min before start time and max equals start time")
	}

	if !g.Overlaps(g.StartTime, g.EndTime.Add(time.Minute)) {
		t.Fatal("shard group not selected when max after after end time")
	}

	if !g.Overlaps(g.StartTime.Add(-time.Minute), g.EndTime.Add(time.Minute)) {
		t.Fatal("shard group not selected when min before start time and when max after end time")
	}

	if !g.Overlaps(g.StartTime.Add(time.Minute), g.EndTime.Add(-time.Minute)) {
		t.Fatal("shard group not selected when min after start time and when max before end time")
	}

	if !g.Overlaps(g.StartTime, g.EndTime) {
		t.Fatal("shard group not selected when min at start time and when max at end time")
	}

	if !g.Overlaps(g.StartTime, g.StartTime) {
		t.Fatal("shard group not selected when min and max set to start time")
	}

	if !g.Overlaps(g.StartTime.Add(1*time.Minute), g.EndTime.Add(24*time.Hour)) {
		t.Fatal("shard group selected when both min in range")
	}

	if g.Overlaps(g.EndTime, g.EndTime) {
		t.Fatal("shard group selected when min and max set to end time")
	}

	if g.Overlaps(g.StartTime.Add(-10*time.Hour), g.EndTime.Add(-9*time.Hour)) {
		t.Fatal("shard group selected when both min and max before shard times")
	}

	if g.Overlaps(g.StartTime.Add(24*time.Hour), g.EndTime.Add(25*time.Hour)) {
		t.Fatal("shard group selected when both min and max after shard times")
	}

}

// Ensure a shard group can be removed by ID.
func TestData_DeleteShardGroup(t *testing.T) {
	var data meta.Data
	if err := data.CreateNode("node0"); err != nil {
		t.Fatal(err)
	} else if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateRetentionPolicy("db0", &meta.RetentionPolicyInfo{Name: "rp0", ReplicaN: 1}); err != nil {
		t.Fatal(err)
	} else if err := data.CreateShardGroup("db0", "rp0", time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	}

	if err := data.DeleteShardGroup("db0", "rp0", 1); err != nil {
		t.Fatal(err)
	}
	if sg := data.Databases[0].RetentionPolicies[0].ShardGroups[0]; !sg.Deleted() {
		t.Fatalf("shard group not correctly flagged as deleted")
	}
}

// Ensure a continuous query can be created.
func TestData_CreateContinuousQuery(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err := data.CreateContinuousQuery("db0", "cq0", "SELECT count() FROM foo"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Databases[0].ContinuousQueries, []meta.ContinuousQueryInfo{
		{Name: "cq0", Query: "SELECT count() FROM foo"},
	}) {
		t.Fatalf("unexpected queries: %#v", data.Databases[0].ContinuousQueries)
	}
}

// Ensure a continuous query can be removed.
func TestData_DropContinuousQuery(t *testing.T) {
	var data meta.Data
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err := data.CreateContinuousQuery("db0", "cq0", "SELECT count() FROM foo"); err != nil {
		t.Fatal(err)
	} else if err = data.CreateContinuousQuery("db0", "cq1", "SELECT count() FROM bar"); err != nil {
		t.Fatal(err)
	}

	if err := data.DropContinuousQuery("db0", "cq0"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Databases[0].ContinuousQueries, []meta.ContinuousQueryInfo{
		{Name: "cq1", Query: "SELECT count() FROM bar"},
	}) {
		t.Fatalf("unexpected queries: %#v", data.Databases[0].ContinuousQueries)
	}
}

// Ensure a subscription can be created.
func TestData_CreateSubscription(t *testing.T) {
	var data meta.Data
	rpi := &meta.RetentionPolicyInfo{
		Name:     "rp0",
		ReplicaN: 3,
	}
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err := data.CreateRetentionPolicy("db0", rpi); err != nil {
		t.Fatal(err)
	} else if err := data.CreateSubscription("db0", "rp0", "s0", "ANY", []string{"udp://h0:1234", "udp://h1:1234"}); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Databases[0].RetentionPolicies[0].Subscriptions, []meta.SubscriptionInfo{
		{Name: "s0", Mode: "ANY", Destinations: []string{"udp://h0:1234", "udp://h1:1234"}},
	}) {
		t.Fatalf("unexpected subscriptions: %#v", data.Databases[0].RetentionPolicies[0].Subscriptions)
	}
}

// Ensure a subscription can be removed.
func TestData_DropSubscription(t *testing.T) {
	var data meta.Data
	rpi := &meta.RetentionPolicyInfo{
		Name:     "rp0",
		ReplicaN: 3,
	}
	if err := data.CreateDatabase("db0"); err != nil {
		t.Fatal(err)
	} else if err := data.CreateRetentionPolicy("db0", rpi); err != nil {
		t.Fatal(err)
	} else if err := data.CreateSubscription("db0", "rp0", "s0", "ANY", []string{"udp://h0:1234", "udp://h1:1234"}); err != nil {
		t.Fatal(err)
	} else if err := data.CreateSubscription("db0", "rp0", "s1", "ALL", []string{"udp://h0:1234", "udp://h1:1234"}); err != nil {
		t.Fatal(err)
	}

	if err := data.DropSubscription("db0", "rp0", "s0"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Databases[0].RetentionPolicies[0].Subscriptions, []meta.SubscriptionInfo{
		{Name: "s1", Mode: "ALL", Destinations: []string{"udp://h0:1234", "udp://h1:1234"}},
	}) {
		t.Fatalf("unexpected subscriptions: %#v", data.Databases[0].RetentionPolicies[0].Subscriptions)
	}
}

// Ensure a user can be created.
func TestData_CreateUser(t *testing.T) {
	var data meta.Data
	if err := data.CreateUser("susy", "ABC123", true); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Users, []meta.UserInfo{
		{Name: "susy", Hash: "ABC123", Admin: true},
	}) {
		t.Fatalf("unexpected users: %#v", data.Users)
	}
}

// Ensure that creating a user with no username returns an error.
func TestData_CreateUser_ErrUsernameRequired(t *testing.T) {
	var data meta.Data
	if err := data.CreateUser("", "", false); err != meta.ErrUsernameRequired {
		t.Fatal(err)
	}
}

// Ensure that creating the same user twice returns an error.
func TestData_CreateUser_ErrUserExists(t *testing.T) {
	var data meta.Data
	if err := data.CreateUser("susy", "", false); err != nil {
		t.Fatal(err)
	}
	if err := data.CreateUser("susy", "", false); err != meta.ErrUserExists {
		t.Fatal(err)
	}
}

// Ensure a user can be removed.
func TestData_DropUser(t *testing.T) {
	var data meta.Data
	if err := data.CreateUser("susy", "", false); err != nil {
		t.Fatal(err)
	} else if err := data.CreateUser("bob", "", false); err != nil {
		t.Fatal(err)
	}

	if err := data.DropUser("bob"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.Users, []meta.UserInfo{
		{Name: "susy"},
	}) {
		t.Fatalf("unexpected users: %#v", data.Users)
	}
}

// Ensure that removing a non-existent user returns an error.
func TestData_DropUser_ErrUserNotFound(t *testing.T) {
	var data meta.Data
	if err := data.DropUser("bob"); err != meta.ErrUserNotFound {
		t.Fatal(err)
	}
}

// Ensure a user can be updated.
func TestData_UpdateUser(t *testing.T) {
	var data meta.Data
	if err := data.CreateUser("susy", "", false); err != nil {
		t.Fatal(err)
	} else if err := data.CreateUser("bob", "", false); err != nil {
		t.Fatal(err)
	}

	// Update password hash.
	if err := data.UpdateUser("bob", "XXX"); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(data.User("bob"), &meta.UserInfo{Name: "bob", Hash: "XXX"}) {
		t.Fatalf("unexpected user: %#v", data.User("bob"))
	}
}

// Ensure that updating a non-existent user returns an error.
func TestData_UpdateUser_ErrUserNotFound(t *testing.T) {
	var data meta.Data
	if err := data.UpdateUser("bob", "ZZZ"); err != meta.ErrUserNotFound {
		t.Fatal(err)
	}
}

// Ensure the data can be deeply copied.
func TestData_Clone(t *testing.T) {
	data := meta.Data{
		Term:  10,
		Index: 20,
		Nodes: []meta.NodeInfo{
			{ID: 1, Host: "host0"},
			{ID: 2, Host: "host1"},
		},
		Databases: []meta.DatabaseInfo{
			{
				Name: "db0",
				DefaultRetentionPolicy: "default",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name:               "rp0",
						ReplicaN:           3,
						Duration:           10 * time.Second,
						ShardGroupDuration: 3 * time.Millisecond,
						ShardGroups: []meta.ShardGroupInfo{
							{
								ID:        100,
								StartTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC),
								EndTime:   time.Date(2000, time.February, 1, 0, 0, 0, 0, time.UTC),
								Shards: []meta.ShardInfo{
									{
										ID: 200,
										Owners: []meta.ShardOwner{
											{NodeID: 1},
											{NodeID: 3},
											{NodeID: 4},
										},
									},
								},
							},
						},
					},
				},
				ContinuousQueries: []meta.ContinuousQueryInfo{
					{Query: "SELECT count() FROM foo"},
				},
			},
		},
		Users: []meta.UserInfo{
			{
				Name:       "susy",
				Hash:       "ABC123",
				Admin:      true,
				Privileges: map[string]influxql.Privilege{"db0": influxql.AllPrivileges},
			},
		},
	}

	// Copy the root structure.
	other := data.Clone()

	if !reflect.DeepEqual(data.Nodes, other.Nodes) {
		t.Fatalf("unexpected cloned nodes: %#v", other.Nodes)
	} else if !reflect.DeepEqual(data.Databases, other.Databases) {
		t.Fatalf("unexpected cloned databases: %#v", other.Databases)
	} else if !reflect.DeepEqual(data.Users, other.Users) {
		t.Fatalf("unexpected cloned users: %#v", other.Users)
	}

	// Ensure that changing data in the clone does not affect the original.
	other.Databases[0].RetentionPolicies[0].ShardGroups[0].Shards[0].Owners[1].NodeID = 9
	if v := data.Databases[0].RetentionPolicies[0].ShardGroups[0].Shards[0].Owners[1].NodeID; v != 3 {
		t.Fatalf("editing clone changed original: %v", v)
	}
}

// Ensure the data can be marshaled and unmarshaled.
func TestData_MarshalBinary(t *testing.T) {
	data := meta.Data{
		Term:  10,
		Index: 20,
		Nodes: []meta.NodeInfo{
			{ID: 1, Host: "host0"},
			{ID: 2, Host: "host1"},
		},
		Databases: []meta.DatabaseInfo{
			{
				Name: "db0",
				DefaultRetentionPolicy: "default",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name:               "rp0",
						ReplicaN:           3,
						Duration:           10 * time.Second,
						ShardGroupDuration: 3 * time.Millisecond,
						ShardGroups: []meta.ShardGroupInfo{
							{
								ID:        100,
								StartTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC),
								EndTime:   time.Date(2000, time.February, 1, 0, 0, 0, 0, time.UTC),
								Shards: []meta.ShardInfo{
									{
										ID: 200,
										Owners: []meta.ShardOwner{
											{NodeID: 1},
											{NodeID: 3},
											{NodeID: 4},
										},
									},
								},
							},
						},
					},
				},
				ContinuousQueries: []meta.ContinuousQueryInfo{
					{Query: "SELECT count() FROM foo"},
				},
			},
		},
		Users: []meta.UserInfo{
			{
				Name:       "susy",
				Hash:       "ABC123",
				Admin:      true,
				Privileges: map[string]influxql.Privilege{"db0": influxql.AllPrivileges},
			},
		},
	}

	// Marshal the data struture.
	buf, err := data.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	// Unmarshal into new data.
	var other meta.Data
	if err := other.UnmarshalBinary(buf); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(data.Nodes, other.Nodes) {
		t.Fatalf("unexpected nodes: %#v", other.Nodes)
	} else if !reflect.DeepEqual(data.Databases, other.Databases) {
		spew.Dump(data.Databases)
		spew.Dump(other.Databases)
		t.Fatalf("unexpected databases: %#v", other.Databases)
	} else if !reflect.DeepEqual(data.Users, other.Users) {
		t.Fatalf("unexpected users: %#v", other.Users)
	}
}

// Ensure shards with deprecated "OwnerIDs" can be decoded.
func TestShardInfo_UnmarshalBinary_OwnerIDs(t *testing.T) {
	// Encode deprecated form to bytes.
	buf, err := proto.Marshal(&internal.ShardInfo{
		ID:       proto.Uint64(1),
		OwnerIDs: []uint64{10, 20, 30},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Decode deprecated form.
	var si meta.ShardInfo
	if err := si.UnmarshalBinary(buf); err != nil {
		t.Fatal(err)
	}

	// Verify data is migrated correctly.
	if !reflect.DeepEqual(si, meta.ShardInfo{
		ID: 1,
		Owners: []meta.ShardOwner{
			{NodeID: 10},
			{NodeID: 20},
			{NodeID: 30},
		},
	}) {
		t.Fatalf("unexpected shard info: %s", spew.Sdump(si))
	}
}
