package meta

import (
	"sort"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta/internal"
)

//go:generate protoc --gogo_out=. internal/meta.proto

const (
	// DefaultRetentionPolicyReplicaN is the default value of RetentionPolicyInfo.ReplicaN.
	DefaultRetentionPolicyReplicaN = 1

	// DefaultRetentionPolicyDuration is the default value of RetentionPolicyInfo.Duration.
	DefaultRetentionPolicyDuration = 7 * (24 * time.Hour)

	// MinRetentionPolicyDuration represents the minimum duration for a policy.
	MinRetentionPolicyDuration = time.Hour
)

// Data represents the top level collection of all metadata.
type Data struct {
	Term      uint64 // associated raft term
	Index     uint64 // associated raft index
	ClusterID uint64
	Nodes     []NodeInfo
	Databases []DatabaseInfo
	Users     []UserInfo

	MaxNodeID       uint64
	MaxShardGroupID uint64
	MaxShardID      uint64
}

// Node returns a node by id.
func (data *Data) Node(id uint64) *NodeInfo {
	for i := range data.Nodes {
		if data.Nodes[i].ID == id {
			return &data.Nodes[i]
		}
	}
	return nil
}

// NodeByHost returns a node by hostname.
func (data *Data) NodeByHost(host string) *NodeInfo {
	for i := range data.Nodes {
		if data.Nodes[i].Host == host {
			return &data.Nodes[i]
		}
	}
	return nil
}

// CreateNode adds a node to the metadata.
func (data *Data) CreateNode(host string) error {
	// Ensure a node with the same host doesn't already exist.
	if data.NodeByHost(host) != nil {
		return ErrNodeExists
	}

	// Append new node.
	data.MaxNodeID++
	data.Nodes = append(data.Nodes, NodeInfo{
		ID:   data.MaxNodeID,
		Host: host,
	})

	return nil
}

// DeleteNode removes a node from the metadata.
func (data *Data) DeleteNode(id uint64) error {
	for i := range data.Nodes {
		if data.Nodes[i].ID == id {
			data.Nodes = append(data.Nodes[:i], data.Nodes[i+1:]...)
			return nil
		}
	}
	return ErrNodeNotFound
}

// Database returns a database by name.
func (data *Data) Database(name string) *DatabaseInfo {
	for i := range data.Databases {
		if data.Databases[i].Name == name {
			return &data.Databases[i]
		}
	}
	return nil
}

// CreateDatabase creates a new database.
// Returns an error if name is blank or if a database with the same name already exists.
func (data *Data) CreateDatabase(name string) error {
	if name == "" {
		return ErrDatabaseNameRequired
	} else if data.Database(name) != nil {
		return ErrDatabaseExists
	}

	// Append new node.
	data.Databases = append(data.Databases, DatabaseInfo{Name: name})

	return nil
}

// DropDatabase removes a database by name.
func (data *Data) DropDatabase(name string) error {
	for i := range data.Databases {
		if data.Databases[i].Name == name {
			data.Databases = append(data.Databases[:i], data.Databases[i+1:]...)
			return nil
		}
	}
	return ErrDatabaseNotFound
}

// RetentionPolicy returns a retention policy for a database by name.
func (data *Data) RetentionPolicy(database, name string) (*RetentionPolicyInfo, error) {
	di := data.Database(database)
	if di == nil {
		return nil, ErrDatabaseNotFound
	}

	for i := range di.RetentionPolicies {
		if di.RetentionPolicies[i].Name == name {
			return &di.RetentionPolicies[i], nil
		}
	}
	return nil, ErrRetentionPolicyNotFound
}

// CreateRetentionPolicy creates a new retention policy on a database.
// Returns an error if name is blank or if a database does not exist.
func (data *Data) CreateRetentionPolicy(database string, rpi *RetentionPolicyInfo) error {
	// Validate retention policy.
	if rpi.Name == "" {
		return ErrRetentionPolicyNameRequired
	} else if rpi.ReplicaN != len(data.Nodes) {
		return ErrReplicationFactorMismatch
	}

	// Find database.
	di := data.Database(database)
	if di == nil {
		return ErrDatabaseNotFound
	} else if di.RetentionPolicy(rpi.Name) != nil {
		return ErrRetentionPolicyExists
	}

	// Append new policy.
	di.RetentionPolicies = append(di.RetentionPolicies, RetentionPolicyInfo{
		Name:               rpi.Name,
		Duration:           rpi.Duration,
		ShardGroupDuration: shardGroupDuration(rpi.Duration),
		ReplicaN:           rpi.ReplicaN,
	})

	return nil
}

// DropRetentionPolicy removes a retention policy from a database by name.
func (data *Data) DropRetentionPolicy(database, name string) error {
	// Find database.
	di := data.Database(database)
	if di == nil {
		return ErrDatabaseNotFound
	}

	// Remove from list.
	for i := range di.RetentionPolicies {
		if di.RetentionPolicies[i].Name == name {
			di.RetentionPolicies = append(di.RetentionPolicies[:i], di.RetentionPolicies[i+1:]...)
			return nil
		}
	}
	return ErrRetentionPolicyNotFound
}

// UpdateRetentionPolicy updates an existing retention policy.
func (data *Data) UpdateRetentionPolicy(database, name string, rpu *RetentionPolicyUpdate) error {
	// Find database.
	di := data.Database(database)
	if di == nil {
		return ErrDatabaseNotFound
	}

	// Find policy.
	rpi := di.RetentionPolicy(name)
	if rpi == nil {
		return ErrRetentionPolicyNotFound
	}

	// Ensure new policy doesn't match an existing policy.
	if rpu.Name != nil && *rpu.Name != name && di.RetentionPolicy(*rpu.Name) != nil {
		return ErrRetentionPolicyNameExists
	}

	// Enforce duration of at least MinRetentionPolicyDuration
	if rpu.Duration != nil && *rpu.Duration < MinRetentionPolicyDuration && *rpu.Duration != 0 {
		return ErrRetentionPolicyDurationTooLow
	}

	// Update fields.
	if rpu.Name != nil {
		rpi.Name = *rpu.Name
	}
	if rpu.Duration != nil {
		rpi.Duration = *rpu.Duration
	}
	if rpu.ReplicaN != nil {
		rpi.ReplicaN = *rpu.ReplicaN
	}

	return nil
}

// SetDefaultRetentionPolicy sets the default retention policy for a database.
func (data *Data) SetDefaultRetentionPolicy(database, name string) error {
	// Find database and verify policy exists.
	di := data.Database(database)
	if di == nil {
		return ErrDatabaseNotFound
	} else if di.RetentionPolicy(name) == nil {
		return ErrRetentionPolicyNotFound
	}

	// Set default policy.
	di.DefaultRetentionPolicy = name

	return nil
}

// ShardGroup returns a list of all shard groups on a database and policy.
func (data *Data) ShardGroups(database, policy string) ([]ShardGroupInfo, error) {
	// Find retention policy.
	rpi, err := data.RetentionPolicy(database, policy)
	if err != nil {
		return nil, err
	} else if rpi == nil {
		return nil, ErrRetentionPolicyNotFound
	}
	groups := make([]ShardGroupInfo, 0, len(rpi.ShardGroups))
	for _, g := range rpi.ShardGroups {
		if g.Deleted() {
			continue
		}
		groups = append(groups, g)
	}
	return groups, nil
}

// ShardGroupsByTimeRange returns a list of all shard groups on a database and policy that may contain data
// for the specified time range. Shard groups are sorted by start time.
func (data *Data) ShardGroupsByTimeRange(database, policy string, tmin, tmax time.Time) ([]ShardGroupInfo, error) {
	// Find retention policy.
	rpi, err := data.RetentionPolicy(database, policy)
	if err != nil {
		return nil, err
	} else if rpi == nil {
		return nil, ErrRetentionPolicyNotFound
	}
	groups := make([]ShardGroupInfo, 0, len(rpi.ShardGroups))
	for _, g := range rpi.ShardGroups {
		if g.Deleted() || !g.Overlaps(tmin, tmax) {
			continue
		}
		groups = append(groups, g)
	}
	sort.Sort(ShardGroupInfos(groups))
	return groups, nil
}

// ShardGroupByTimestamp returns the shard group on a database and policy for a given timestamp.
func (data *Data) ShardGroupByTimestamp(database, policy string, timestamp time.Time) (*ShardGroupInfo, error) {
	// Find retention policy.
	rpi, err := data.RetentionPolicy(database, policy)
	if err != nil {
		return nil, err
	} else if rpi == nil {
		return nil, ErrRetentionPolicyNotFound
	}

	return rpi.ShardGroupByTimestamp(timestamp), nil
}

// CreateShardGroup creates a shard group on a database and policy for a given timestamp.
func (data *Data) CreateShardGroup(database, policy string, timestamp time.Time) error {
	// Ensure there are nodes in the metadata.
	if len(data.Nodes) == 0 {
		return ErrNodesRequired
	}

	// Find retention policy.
	rpi, err := data.RetentionPolicy(database, policy)
	if err != nil {
		return err
	} else if rpi == nil {
		return ErrRetentionPolicyNotFound
	}

	// Verify that shard group doesn't already exist for this timestamp.
	if rpi.ShardGroupByTimestamp(timestamp) != nil {
		return ErrShardGroupExists
	}

	// Require at least one replica but no more replicas than nodes.
	replicaN := rpi.ReplicaN
	if replicaN == 0 {
		replicaN = 1
	} else if replicaN > len(data.Nodes) {
		replicaN = len(data.Nodes)
	}

	// Determine shard count by node count divided by replication factor.
	// This will ensure nodes will get distributed across nodes evenly and
	// replicated the correct number of times.
	shardN := len(data.Nodes) / replicaN

	// Create the shard group.
	data.MaxShardGroupID++
	sgi := ShardGroupInfo{}
	sgi.ID = data.MaxShardGroupID
	sgi.StartTime = timestamp.Truncate(rpi.ShardGroupDuration).UTC()
	sgi.EndTime = sgi.StartTime.Add(rpi.ShardGroupDuration).UTC()

	// Create shards on the group.
	sgi.Shards = make([]ShardInfo, shardN)
	for i := range sgi.Shards {
		data.MaxShardID++
		sgi.Shards[i] = ShardInfo{ID: data.MaxShardID}
	}

	// Assign data nodes to shards via round robin.
	// Start from a repeatably "random" place in the node list.
	nodeIndex := int(data.Index % uint64(len(data.Nodes)))
	for i := range sgi.Shards {
		si := &sgi.Shards[i]
		for j := 0; j < replicaN; j++ {
			nodeID := data.Nodes[nodeIndex%len(data.Nodes)].ID
			si.OwnerIDs = append(si.OwnerIDs, nodeID)
			nodeIndex++
		}
	}

	// Retention policy has a new shard group, so update the policy.
	rpi.ShardGroups = append(rpi.ShardGroups, sgi)

	return nil
}

// DeleteShardGroup removes a shard group from a database and retention policy by id.
func (data *Data) DeleteShardGroup(database, policy string, id uint64) error {
	// Find retention policy.
	rpi, err := data.RetentionPolicy(database, policy)
	if err != nil {
		return err
	} else if rpi == nil {
		return ErrRetentionPolicyNotFound
	}

	// Find shard group by ID and set its deletion timestamp.
	for i := range rpi.ShardGroups {
		if rpi.ShardGroups[i].ID == id {
			rpi.ShardGroups[i].DeletedAt = time.Now().UTC()
			return nil
		}
	}

	return ErrShardGroupNotFound
}

// CreateContinuousQuery adds a named continuous query to a database.
func (data *Data) CreateContinuousQuery(database, name, query string) error {
	di := data.Database(database)
	if di == nil {
		return ErrDatabaseNotFound
	}

	// Ensure the name doesn't already exist.
	for i := range di.ContinuousQueries {
		if di.ContinuousQueries[i].Name == name {
			return ErrContinuousQueryExists
		}
	}

	// Append new query.
	di.ContinuousQueries = append(di.ContinuousQueries, ContinuousQueryInfo{
		Name:  name,
		Query: query,
	})

	return nil
}

// DropContinuousQuery removes a continuous query.
func (data *Data) DropContinuousQuery(database, name string) error {
	di := data.Database(database)
	if di == nil {
		return ErrDatabaseNotFound
	}

	for i := range di.ContinuousQueries {
		if di.ContinuousQueries[i].Name == name {
			di.ContinuousQueries = append(di.ContinuousQueries[:i], di.ContinuousQueries[i+1:]...)
			return nil
		}
	}
	return ErrContinuousQueryNotFound
}

// User returns a user by username.
func (data *Data) User(username string) *UserInfo {
	for i := range data.Users {
		if data.Users[i].Name == username {
			return &data.Users[i]
		}
	}
	return nil
}

// CreateUser creates a new user.
func (data *Data) CreateUser(name, hash string, admin bool) error {
	// Ensure the user doesn't already exist.
	if name == "" {
		return ErrUsernameRequired
	} else if data.User(name) != nil {
		return ErrUserExists
	}

	// Append new user.
	data.Users = append(data.Users, UserInfo{
		Name:  name,
		Hash:  hash,
		Admin: admin,
	})

	return nil
}

// DropUser removes an existing user by name.
func (data *Data) DropUser(name string) error {
	for i := range data.Users {
		if data.Users[i].Name == name {
			data.Users = append(data.Users[:i], data.Users[i+1:]...)
			return nil
		}
	}
	return ErrUserNotFound
}

// UpdateUser updates the password hash of an existing user.
func (data *Data) UpdateUser(name, hash string) error {
	for i := range data.Users {
		if data.Users[i].Name == name {
			data.Users[i].Hash = hash
			return nil
		}
	}
	return ErrUserNotFound
}

// SetPrivilege sets a privilege for a user on a database.
func (data *Data) SetPrivilege(name, database string, p influxql.Privilege) error {
	ui := data.User(name)
	if ui == nil {
		return ErrUserNotFound
	}

	if ui.Privileges == nil {
		ui.Privileges = make(map[string]influxql.Privilege)
	}
	ui.Privileges[database] = p

	return nil
}

// SetAdminPrivilege sets the admin privilege for a user.
func (data *Data) SetAdminPrivilege(name string, admin bool) error {
	ui := data.User(name)
	if ui == nil {
		return ErrUserNotFound
	}

	ui.Admin = admin

	return nil
}

// UserPrivileges gets the privileges for a user.
func (data *Data) UserPrivileges(name string) (map[string]influxql.Privilege, error) {
	ui := data.User(name)
	if ui == nil {
		return nil, ErrUserNotFound
	}

	return ui.Privileges, nil
}

// UserPrivilege gets the privilege for a user on a database.
func (data *Data) UserPrivilege(name, database string) (*influxql.Privilege, error) {
	ui := data.User(name)
	if ui == nil {
		return nil, ErrUserNotFound
	}

	for db, p := range ui.Privileges {
		if db == database {
			return &p, nil
		}
	}

	return influxql.NewPrivilege(influxql.NoPrivileges), nil
}

// Clone returns a copy of data with a new version.
func (data *Data) Clone() *Data {
	other := *data

	// Copy nodes.
	if data.Nodes != nil {
		other.Nodes = make([]NodeInfo, len(data.Nodes))
		for i := range data.Nodes {
			other.Nodes[i] = data.Nodes[i].clone()
		}
	}

	// Deep copy databases.
	if data.Databases != nil {
		other.Databases = make([]DatabaseInfo, len(data.Databases))
		for i := range data.Databases {
			other.Databases[i] = data.Databases[i].clone()
		}
	}

	// Copy users.
	if data.Users != nil {
		other.Users = make([]UserInfo, len(data.Users))
		for i := range data.Users {
			other.Users[i] = data.Users[i].clone()
		}
	}

	return &other
}

// marshal serializes to a protobuf representation.
func (data *Data) marshal() *internal.Data {
	pb := &internal.Data{
		Term:      proto.Uint64(data.Term),
		Index:     proto.Uint64(data.Index),
		ClusterID: proto.Uint64(data.ClusterID),

		MaxNodeID:       proto.Uint64(data.MaxNodeID),
		MaxShardGroupID: proto.Uint64(data.MaxShardGroupID),
		MaxShardID:      proto.Uint64(data.MaxShardID),
	}

	pb.Nodes = make([]*internal.NodeInfo, len(data.Nodes))
	for i := range data.Nodes {
		pb.Nodes[i] = data.Nodes[i].marshal()
	}

	pb.Databases = make([]*internal.DatabaseInfo, len(data.Databases))
	for i := range data.Databases {
		pb.Databases[i] = data.Databases[i].marshal()
	}

	pb.Users = make([]*internal.UserInfo, len(data.Users))
	for i := range data.Users {
		pb.Users[i] = data.Users[i].marshal()
	}

	return pb
}

// unmarshal deserializes from a protobuf representation.
func (data *Data) unmarshal(pb *internal.Data) {
	data.Term = pb.GetTerm()
	data.Index = pb.GetIndex()
	data.ClusterID = pb.GetClusterID()

	data.MaxNodeID = pb.GetMaxNodeID()
	data.MaxShardGroupID = pb.GetMaxShardGroupID()
	data.MaxShardID = pb.GetMaxShardID()

	data.Nodes = make([]NodeInfo, len(pb.GetNodes()))
	for i, x := range pb.GetNodes() {
		data.Nodes[i].unmarshal(x)
	}

	data.Databases = make([]DatabaseInfo, len(pb.GetDatabases()))
	for i, x := range pb.GetDatabases() {
		data.Databases[i].unmarshal(x)
	}

	data.Users = make([]UserInfo, len(pb.GetUsers()))
	for i, x := range pb.GetUsers() {
		data.Users[i].unmarshal(x)
	}
}

// MarshalBinary encodes the metadata to a binary format.
func (data *Data) MarshalBinary() ([]byte, error) {
	return proto.Marshal(data.marshal())
}

// UnmarshalBinary decodes the object from a binary format.
func (data *Data) UnmarshalBinary(buf []byte) error {
	var pb internal.Data
	if err := proto.Unmarshal(buf, &pb); err != nil {
		return err
	}
	data.unmarshal(&pb)
	return nil
}

// NodeInfo represents information about a single node in the cluster.
type NodeInfo struct {
	ID   uint64
	Host string
}

// clone returns a deep copy of ni.
func (ni NodeInfo) clone() NodeInfo { return ni }

// marshal serializes to a protobuf representation.
func (ni NodeInfo) marshal() *internal.NodeInfo {
	pb := &internal.NodeInfo{}
	pb.ID = proto.Uint64(ni.ID)
	pb.Host = proto.String(ni.Host)
	return pb
}

// unmarshal deserializes from a protobuf representation.
func (ni *NodeInfo) unmarshal(pb *internal.NodeInfo) {
	ni.ID = pb.GetID()
	ni.Host = pb.GetHost()
}

// DatabaseInfo represents information about a database in the system.
type DatabaseInfo struct {
	Name                   string
	DefaultRetentionPolicy string
	RetentionPolicies      []RetentionPolicyInfo
	ContinuousQueries      []ContinuousQueryInfo
}

// RetentionPolicy returns a retention policy by name.
func (di DatabaseInfo) RetentionPolicy(name string) *RetentionPolicyInfo {
	for i := range di.RetentionPolicies {
		if di.RetentionPolicies[i].Name == name {
			return &di.RetentionPolicies[i]
		}
	}
	return nil
}

// clone returns a deep copy of di.
func (di DatabaseInfo) clone() DatabaseInfo {
	other := di

	if di.RetentionPolicies != nil {
		other.RetentionPolicies = make([]RetentionPolicyInfo, len(di.RetentionPolicies))
		for i := range di.RetentionPolicies {
			other.RetentionPolicies[i] = di.RetentionPolicies[i].clone()
		}
	}

	// Copy continuous queries.
	if di.ContinuousQueries != nil {
		other.ContinuousQueries = make([]ContinuousQueryInfo, len(di.ContinuousQueries))
		for i := range di.ContinuousQueries {
			other.ContinuousQueries[i] = di.ContinuousQueries[i].clone()
		}
	}

	return other
}

// marshal serializes to a protobuf representation.
func (di DatabaseInfo) marshal() *internal.DatabaseInfo {
	pb := &internal.DatabaseInfo{}
	pb.Name = proto.String(di.Name)
	pb.DefaultRetentionPolicy = proto.String(di.DefaultRetentionPolicy)

	pb.RetentionPolicies = make([]*internal.RetentionPolicyInfo, len(di.RetentionPolicies))
	for i := range di.RetentionPolicies {
		pb.RetentionPolicies[i] = di.RetentionPolicies[i].marshal()
	}

	pb.ContinuousQueries = make([]*internal.ContinuousQueryInfo, len(di.ContinuousQueries))
	for i := range di.ContinuousQueries {
		pb.ContinuousQueries[i] = di.ContinuousQueries[i].marshal()
	}
	return pb
}

// unmarshal deserializes from a protobuf representation.
func (di *DatabaseInfo) unmarshal(pb *internal.DatabaseInfo) {
	di.Name = pb.GetName()
	di.DefaultRetentionPolicy = pb.GetDefaultRetentionPolicy()

	di.RetentionPolicies = make([]RetentionPolicyInfo, len(pb.GetRetentionPolicies()))
	for i, x := range pb.GetRetentionPolicies() {
		di.RetentionPolicies[i].unmarshal(x)
	}

	di.ContinuousQueries = make([]ContinuousQueryInfo, len(pb.GetContinuousQueries()))
	for i, x := range pb.GetContinuousQueries() {
		di.ContinuousQueries[i].unmarshal(x)
	}
}

// RetentionPolicyInfo represents metadata about a retention policy.
type RetentionPolicyInfo struct {
	Name               string
	ReplicaN           int
	Duration           time.Duration
	ShardGroupDuration time.Duration
	ShardGroups        []ShardGroupInfo
}

// NewRetentionPolicyInfo returns a new instance of RetentionPolicyInfo with defaults set.
func NewRetentionPolicyInfo(name string) *RetentionPolicyInfo {
	return &RetentionPolicyInfo{
		Name:     name,
		ReplicaN: DefaultRetentionPolicyReplicaN,
		Duration: DefaultRetentionPolicyDuration,
	}
}

// ShardGroupByTimestamp returns the shard group in the policy that contains the timestamp.
func (rpi *RetentionPolicyInfo) ShardGroupByTimestamp(timestamp time.Time) *ShardGroupInfo {
	for i := range rpi.ShardGroups {
		if rpi.ShardGroups[i].Contains(timestamp) && !rpi.ShardGroups[i].Deleted() {
			return &rpi.ShardGroups[i]
		}
	}
	return nil
}

// ExpiredShardGroups returns the Shard Groups which are considered expired, for the given time.
func (rpi *RetentionPolicyInfo) ExpiredShardGroups(t time.Time) []*ShardGroupInfo {
	groups := make([]*ShardGroupInfo, 0)
	for i := range rpi.ShardGroups {
		if rpi.ShardGroups[i].Deleted() {
			continue
		}
		if rpi.Duration != 0 && rpi.ShardGroups[i].EndTime.Add(rpi.Duration).Before(t) {
			groups = append(groups, &rpi.ShardGroups[i])
		}
	}
	return groups
}

// DeletedShardGroups returns the Shard Groups which are marked as deleted.
func (rpi *RetentionPolicyInfo) DeletedShardGroups() []*ShardGroupInfo {
	groups := make([]*ShardGroupInfo, 0)
	for i := range rpi.ShardGroups {
		if rpi.ShardGroups[i].Deleted() {
			groups = append(groups, &rpi.ShardGroups[i])
		}
	}
	return groups
}

// marshal serializes to a protobuf representation.
func (rpi *RetentionPolicyInfo) marshal() *internal.RetentionPolicyInfo {
	pb := &internal.RetentionPolicyInfo{
		Name:               proto.String(rpi.Name),
		ReplicaN:           proto.Uint32(uint32(rpi.ReplicaN)),
		Duration:           proto.Int64(int64(rpi.Duration)),
		ShardGroupDuration: proto.Int64(int64(rpi.ShardGroupDuration)),
	}

	pb.ShardGroups = make([]*internal.ShardGroupInfo, len(rpi.ShardGroups))
	for i, sgi := range rpi.ShardGroups {
		pb.ShardGroups[i] = sgi.marshal()
	}

	return pb
}

// unmarshal deserializes from a protobuf representation.
func (rpi *RetentionPolicyInfo) unmarshal(pb *internal.RetentionPolicyInfo) {
	rpi.Name = pb.GetName()
	rpi.ReplicaN = int(pb.GetReplicaN())
	rpi.Duration = time.Duration(pb.GetDuration())
	rpi.ShardGroupDuration = time.Duration(pb.GetShardGroupDuration())

	rpi.ShardGroups = make([]ShardGroupInfo, len(pb.GetShardGroups()))
	for i, x := range pb.GetShardGroups() {
		rpi.ShardGroups[i].unmarshal(x)
	}
}

// clone returns a deep copy of rpi.
func (rpi RetentionPolicyInfo) clone() RetentionPolicyInfo {
	other := rpi

	if rpi.ShardGroups != nil {
		other.ShardGroups = make([]ShardGroupInfo, len(rpi.ShardGroups))
		for i := range rpi.ShardGroups {
			other.ShardGroups[i] = rpi.ShardGroups[i].clone()
		}
	}

	return other
}

// shardGroupDuration returns the duration for a shard group based on a policy duration.
func shardGroupDuration(d time.Duration) time.Duration {
	if d >= 180*24*time.Hour || d == 0 { // 6 months or 0
		return 7 * 24 * time.Hour
	} else if d >= 2*24*time.Hour { // 2 days
		return 1 * 24 * time.Hour
	}
	return 1 * time.Hour
}

// ShardGroupInfo represents metadata about a shard group. The DeletedAt field is important
// because it makes it clear that a ShardGroup has been marked as deleted, and allow the system
// to be sure that a ShardGroup is not simply missing. If the DeletedAt is set, the system can
// safely delete any associated shards.
type ShardGroupInfo struct {
	ID        uint64
	StartTime time.Time
	EndTime   time.Time
	DeletedAt time.Time
	Shards    []ShardInfo
}

type ShardGroupInfos []ShardGroupInfo

func (a ShardGroupInfos) Len() int           { return len(a) }
func (a ShardGroupInfos) Less(i, j int) bool { return a[i].StartTime.Before(a[j].StartTime) }
func (a ShardGroupInfos) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// Contains return true if the shard group contains data for the timestamp.
func (sgi *ShardGroupInfo) Contains(timestamp time.Time) bool {
	return !sgi.StartTime.After(timestamp) && sgi.EndTime.After(timestamp)
}

// Overlaps return whether the shard group contains data for the time range between min and max
func (sgi *ShardGroupInfo) Overlaps(min, max time.Time) bool {
	return !sgi.StartTime.After(max) && sgi.EndTime.After(min)
}

// Deleted returns whether this ShardGroup has been deleted.
func (sgi *ShardGroupInfo) Deleted() bool {
	return !sgi.DeletedAt.IsZero()
}

// clone returns a deep copy of sgi.
func (sgi ShardGroupInfo) clone() ShardGroupInfo {
	other := sgi

	if sgi.Shards != nil {
		other.Shards = make([]ShardInfo, len(sgi.Shards))
		for i := range sgi.Shards {
			other.Shards[i] = sgi.Shards[i].clone()
		}
	}

	return other
}

// ShardFor returns the ShardInfo for a Point hash
func (s *ShardGroupInfo) ShardFor(hash uint64) ShardInfo {
	return s.Shards[hash%uint64(len(s.Shards))]
}

// marshal serializes to a protobuf representation.
func (sgi *ShardGroupInfo) marshal() *internal.ShardGroupInfo {
	pb := &internal.ShardGroupInfo{
		ID:        proto.Uint64(sgi.ID),
		StartTime: proto.Int64(MarshalTime(sgi.StartTime)),
		EndTime:   proto.Int64(MarshalTime(sgi.EndTime)),
		DeletedAt: proto.Int64(MarshalTime(sgi.DeletedAt)),
	}

	pb.Shards = make([]*internal.ShardInfo, len(sgi.Shards))
	for i := range sgi.Shards {
		pb.Shards[i] = sgi.Shards[i].marshal()
	}

	return pb
}

// unmarshal deserializes from a protobuf representation.
func (sgi *ShardGroupInfo) unmarshal(pb *internal.ShardGroupInfo) {
	sgi.ID = pb.GetID()
	sgi.StartTime = UnmarshalTime(pb.GetStartTime())
	sgi.EndTime = UnmarshalTime(pb.GetEndTime())
	sgi.DeletedAt = UnmarshalTime(pb.GetDeletedAt())

	sgi.Shards = make([]ShardInfo, len(pb.GetShards()))
	for i, x := range pb.GetShards() {
		sgi.Shards[i].unmarshal(x)
	}
}

// ShardInfo represents metadata about a shard.
type ShardInfo struct {
	ID       uint64
	OwnerIDs []uint64
}

// OwnedBy returns whether the shard's owner IDs includes nodeID.
func (si ShardInfo) OwnedBy(nodeID uint64) bool {
	for _, id := range si.OwnerIDs {
		if id == nodeID {
			return true
		}
	}
	return false
}

// clone returns a deep copy of si.
func (si ShardInfo) clone() ShardInfo {
	other := si

	if si.OwnerIDs != nil {
		other.OwnerIDs = make([]uint64, len(si.OwnerIDs))
		copy(other.OwnerIDs, si.OwnerIDs)
	}

	return other
}

// marshal serializes to a protobuf representation.
func (si ShardInfo) marshal() *internal.ShardInfo {
	pb := &internal.ShardInfo{
		ID: proto.Uint64(si.ID),
	}

	pb.OwnerIDs = make([]uint64, len(si.OwnerIDs))
	copy(pb.OwnerIDs, si.OwnerIDs)

	return pb
}

// unmarshal deserializes from a protobuf representation.
func (si *ShardInfo) unmarshal(pb *internal.ShardInfo) {
	si.ID = pb.GetID()
	si.OwnerIDs = make([]uint64, len(pb.GetOwnerIDs()))
	copy(si.OwnerIDs, pb.GetOwnerIDs())
}

// ContinuousQueryInfo represents metadata about a continuous query.
type ContinuousQueryInfo struct {
	Name  string
	Query string
}

// clone returns a deep copy of cqi.
func (cqi ContinuousQueryInfo) clone() ContinuousQueryInfo { return cqi }

// marshal serializes to a protobuf representation.
func (cqi ContinuousQueryInfo) marshal() *internal.ContinuousQueryInfo {
	return &internal.ContinuousQueryInfo{
		Name:  proto.String(cqi.Name),
		Query: proto.String(cqi.Query),
	}
}

// unmarshal deserializes from a protobuf representation.
func (cqi *ContinuousQueryInfo) unmarshal(pb *internal.ContinuousQueryInfo) {
	cqi.Name = pb.GetName()
	cqi.Query = pb.GetQuery()
}

// UserInfo represents metadata about a user in the system.
type UserInfo struct {
	Name       string
	Hash       string
	Admin      bool
	Privileges map[string]influxql.Privilege
}

// Authorize returns true if the user is authorized and false if not.
func (ui *UserInfo) Authorize(privilege influxql.Privilege, database string) bool {
	if ui.Admin {
		return true
	}
	p, ok := ui.Privileges[database]
	return ok && (p == privilege || p == influxql.AllPrivileges)
}

// clone returns a deep copy of si.
func (ui UserInfo) clone() UserInfo {
	other := ui

	if ui.Privileges != nil {
		other.Privileges = make(map[string]influxql.Privilege)
		for k, v := range ui.Privileges {
			other.Privileges[k] = v
		}
	}

	return other
}

// marshal serializes to a protobuf representation.
func (ui UserInfo) marshal() *internal.UserInfo {
	pb := &internal.UserInfo{
		Name:  proto.String(ui.Name),
		Hash:  proto.String(ui.Hash),
		Admin: proto.Bool(ui.Admin),
	}

	for database, privilege := range ui.Privileges {
		pb.Privileges = append(pb.Privileges, &internal.UserPrivilege{
			Database:  proto.String(database),
			Privilege: proto.Int32(int32(privilege)),
		})
	}

	return pb
}

// unmarshal deserializes from a protobuf representation.
func (ui *UserInfo) unmarshal(pb *internal.UserInfo) {
	ui.Name = pb.GetName()
	ui.Hash = pb.GetHash()
	ui.Admin = pb.GetAdmin()

	ui.Privileges = make(map[string]influxql.Privilege)
	for _, p := range pb.GetPrivileges() {
		ui.Privileges[p.GetDatabase()] = influxql.Privilege(p.GetPrivilege())
	}
}

// MarshalTime converts t to nanoseconds since epoch. A zero time returns 0.
func MarshalTime(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UnixNano()
}

// UnmarshalTime converts nanoseconds since epoch to time.
// A zero value returns a zero time.
func UnmarshalTime(v int64) time.Time {
	if v == 0 {
		return time.Time{}
	}
	return time.Unix(0, v).UTC()
}
