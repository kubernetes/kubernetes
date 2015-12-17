package cluster_test

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
)

// Ensures the points writer maps a single point to a single shard.
func TestPointsWriter_MapShards_One(t *testing.T) {
	ms := MetaStore{}
	rp := NewRetentionPolicy("myp", time.Hour, 3)

	ms.NodeIDFn = func() uint64 { return 1 }
	ms.RetentionPolicyFn = func(db, retentionPolicy string) (*meta.RetentionPolicyInfo, error) {
		return rp, nil
	}

	ms.CreateShardGroupIfNotExistsFn = func(database, policy string, timestamp time.Time) (*meta.ShardGroupInfo, error) {
		return &rp.ShardGroups[0], nil
	}

	c := cluster.PointsWriter{MetaStore: ms}
	pr := &cluster.WritePointsRequest{
		Database:         "mydb",
		RetentionPolicy:  "myrp",
		ConsistencyLevel: cluster.ConsistencyLevelOne,
	}
	pr.AddPoint("cpu", 1.0, time.Now(), nil)

	var (
		shardMappings *cluster.ShardMapping
		err           error
	)
	if shardMappings, err = c.MapShards(pr); err != nil {
		t.Fatalf("unexpected an error: %v", err)
	}

	if exp := 1; len(shardMappings.Points) != exp {
		t.Errorf("MapShards() len mismatch. got %v, exp %v", len(shardMappings.Points), exp)
	}
}

// Ensures the points writer maps a multiple points across shard group boundaries.
func TestPointsWriter_MapShards_Multiple(t *testing.T) {
	ms := MetaStore{}
	rp := NewRetentionPolicy("myp", time.Hour, 3)
	AttachShardGroupInfo(rp, []meta.ShardOwner{
		{NodeID: 1},
		{NodeID: 2},
		{NodeID: 3},
	})
	AttachShardGroupInfo(rp, []meta.ShardOwner{
		{NodeID: 1},
		{NodeID: 2},
		{NodeID: 3},
	})

	ms.NodeIDFn = func() uint64 { return 1 }
	ms.RetentionPolicyFn = func(db, retentionPolicy string) (*meta.RetentionPolicyInfo, error) {
		return rp, nil
	}

	ms.CreateShardGroupIfNotExistsFn = func(database, policy string, timestamp time.Time) (*meta.ShardGroupInfo, error) {
		for i, sg := range rp.ShardGroups {
			if timestamp.Equal(sg.StartTime) || timestamp.After(sg.StartTime) && timestamp.Before(sg.EndTime) {
				return &rp.ShardGroups[i], nil
			}
		}
		panic("should not get here")
	}

	c := cluster.PointsWriter{MetaStore: ms}
	pr := &cluster.WritePointsRequest{
		Database:         "mydb",
		RetentionPolicy:  "myrp",
		ConsistencyLevel: cluster.ConsistencyLevelOne,
	}

	// Three points that range over the shardGroup duration (1h) and should map to two
	// distinct shards
	pr.AddPoint("cpu", 1.0, time.Unix(0, 0), nil)
	pr.AddPoint("cpu", 2.0, time.Unix(0, 0).Add(time.Hour), nil)
	pr.AddPoint("cpu", 3.0, time.Unix(0, 0).Add(time.Hour+time.Second), nil)

	var (
		shardMappings *cluster.ShardMapping
		err           error
	)
	if shardMappings, err = c.MapShards(pr); err != nil {
		t.Fatalf("unexpected an error: %v", err)
	}

	if exp := 2; len(shardMappings.Points) != exp {
		t.Errorf("MapShards() len mismatch. got %v, exp %v", len(shardMappings.Points), exp)
	}

	for _, points := range shardMappings.Points {
		// First shard shoud have 1 point w/ first point added
		if len(points) == 1 && points[0].Time() != pr.Points[0].Time() {
			t.Fatalf("MapShards() value mismatch. got %v, exp %v", points[0].Time(), pr.Points[0].Time())
		}

		// Second shard shoud have the last two points added
		if len(points) == 2 && points[0].Time() != pr.Points[1].Time() {
			t.Fatalf("MapShards() value mismatch. got %v, exp %v", points[0].Time(), pr.Points[1].Time())
		}

		if len(points) == 2 && points[1].Time() != pr.Points[2].Time() {
			t.Fatalf("MapShards() value mismatch. got %v, exp %v", points[1].Time(), pr.Points[2].Time())
		}
	}
}

func TestPointsWriter_WritePoints(t *testing.T) {
	tests := []struct {
		name            string
		database        string
		retentionPolicy string
		consistency     cluster.ConsistencyLevel

		// the responses returned by each shard write call.  node ID 1 = pos 0
		err    []error
		expErr error
	}{
		// Consistency one
		{
			name:            "write one success",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelOne,
			err:             []error{nil, nil, nil},
			expErr:          nil,
		},
		{
			name:            "write one error",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelOne,
			err:             []error{fmt.Errorf("a failure"), fmt.Errorf("a failure"), fmt.Errorf("a failure")},
			expErr:          fmt.Errorf("write failed: a failure"),
		},

		// Consistency any
		{
			name:            "write any success",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelAny,
			err:             []error{fmt.Errorf("a failure"), nil, fmt.Errorf("a failure")},
			expErr:          nil,
		},
		// Consistency all
		{
			name:            "write all success",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelAll,
			err:             []error{nil, nil, nil},
			expErr:          nil,
		},
		{
			name:            "write all, 2/3, partial write",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelAll,
			err:             []error{nil, fmt.Errorf("a failure"), nil},
			expErr:          cluster.ErrPartialWrite,
		},
		{
			name:            "write all, 1/3 (failure)",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelAll,
			err:             []error{nil, fmt.Errorf("a failure"), fmt.Errorf("a failure")},
			expErr:          cluster.ErrPartialWrite,
		},

		// Consistency quorum
		{
			name:            "write quorum, 1/3 failure",
			consistency:     cluster.ConsistencyLevelQuorum,
			database:        "mydb",
			retentionPolicy: "myrp",
			err:             []error{fmt.Errorf("a failure"), fmt.Errorf("a failure"), nil},
			expErr:          cluster.ErrPartialWrite,
		},
		{
			name:            "write quorum, 2/3 success",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelQuorum,
			err:             []error{nil, nil, fmt.Errorf("a failure")},
			expErr:          nil,
		},
		{
			name:            "write quorum, 3/3 success",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelQuorum,
			err:             []error{nil, nil, nil},
			expErr:          nil,
		},

		// Error write error
		{
			name:            "no writes succeed",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelOne,
			err:             []error{fmt.Errorf("a failure"), fmt.Errorf("a failure"), fmt.Errorf("a failure")},
			expErr:          fmt.Errorf("write failed: a failure"),
		},

		// Hinted handoff w/ ANY
		{
			name:            "hinted handoff write succeed",
			database:        "mydb",
			retentionPolicy: "myrp",
			consistency:     cluster.ConsistencyLevelAny,
			err:             []error{fmt.Errorf("a failure"), fmt.Errorf("a failure"), fmt.Errorf("a failure")},
			expErr:          nil,
		},

		// Write to non-existent database
		{
			name:            "write to non-existent database",
			database:        "doesnt_exist",
			retentionPolicy: "",
			consistency:     cluster.ConsistencyLevelAny,
			err:             []error{nil, nil, nil},
			expErr:          fmt.Errorf("database not found: doesnt_exist"),
		},
	}

	for _, test := range tests {

		pr := &cluster.WritePointsRequest{
			Database:         test.database,
			RetentionPolicy:  test.retentionPolicy,
			ConsistencyLevel: test.consistency,
		}

		// Three points that range over the shardGroup duration (1h) and should map to two
		// distinct shards
		pr.AddPoint("cpu", 1.0, time.Unix(0, 0), nil)
		pr.AddPoint("cpu", 2.0, time.Unix(0, 0).Add(time.Hour), nil)
		pr.AddPoint("cpu", 3.0, time.Unix(0, 0).Add(time.Hour+time.Second), nil)

		// copy to prevent data race
		theTest := test
		sm := cluster.NewShardMapping()
		sm.MapPoint(
			&meta.ShardInfo{ID: uint64(1), Owners: []meta.ShardOwner{
				{NodeID: 1},
				{NodeID: 2},
				{NodeID: 3},
			}},
			pr.Points[0])
		sm.MapPoint(
			&meta.ShardInfo{ID: uint64(2), Owners: []meta.ShardOwner{
				{NodeID: 1},
				{NodeID: 2},
				{NodeID: 3},
			}},
			pr.Points[1])
		sm.MapPoint(
			&meta.ShardInfo{ID: uint64(2), Owners: []meta.ShardOwner{
				{NodeID: 1},
				{NodeID: 2},
				{NodeID: 3},
			}},
			pr.Points[2])

		// Local cluster.Node ShardWriter
		// lock on the write increment since these functions get called in parallel
		var mu sync.Mutex
		sw := &fakeShardWriter{
			ShardWriteFn: func(shardID, nodeID uint64, points []models.Point) error {
				mu.Lock()
				defer mu.Unlock()
				return theTest.err[int(nodeID)-1]
			},
		}

		store := &fakeStore{
			WriteFn: func(shardID uint64, points []models.Point) error {
				mu.Lock()
				defer mu.Unlock()
				return theTest.err[0]
			},
		}

		hh := &fakeShardWriter{
			ShardWriteFn: func(shardID, nodeID uint64, points []models.Point) error {
				return nil
			},
		}

		ms := NewMetaStore()
		ms.DatabaseFn = func(database string) (*meta.DatabaseInfo, error) {
			return nil, nil
		}
		ms.NodeIDFn = func() uint64 { return 1 }

		subPoints := make(chan *cluster.WritePointsRequest, 1)
		sub := Subscriber{}
		sub.PointsFn = func() chan<- *cluster.WritePointsRequest {
			return subPoints
		}

		c := cluster.NewPointsWriter()
		c.MetaStore = ms
		c.ShardWriter = sw
		c.TSDBStore = store
		c.HintedHandoff = hh
		c.Subscriber = sub

		c.Open()
		defer c.Close()

		err := c.WritePoints(pr)
		if err == nil && test.expErr != nil {
			t.Errorf("PointsWriter.WritePoints(): '%s' error: got %v, exp %v", test.name, err, test.expErr)
		}

		if err != nil && test.expErr == nil {
			t.Errorf("PointsWriter.WritePoints(): '%s' error: got %v, exp %v", test.name, err, test.expErr)
		}
		if err != nil && test.expErr != nil && err.Error() != test.expErr.Error() {
			t.Errorf("PointsWriter.WritePoints(): '%s' error: got %v, exp %v", test.name, err, test.expErr)
		}
		if test.expErr == nil {
			select {
			case p := <-subPoints:
				if p != pr {
					t.Errorf("PointsWriter.WritePoints(): '%s' error: unexpected WritePointsRequest got %v, exp %v", test.name, p, pr)
				}
			default:
				t.Errorf("PointsWriter.WritePoints(): '%s' error: Subscriber.Points not called", test.name)
			}
		}
	}
}

var shardID uint64

type fakeShardWriter struct {
	ShardWriteFn func(shardID, nodeID uint64, points []models.Point) error
}

func (f *fakeShardWriter) WriteShard(shardID, nodeID uint64, points []models.Point) error {
	return f.ShardWriteFn(shardID, nodeID, points)
}

type fakeStore struct {
	WriteFn       func(shardID uint64, points []models.Point) error
	CreateShardfn func(database, retentionPolicy string, shardID uint64) error
}

func (f *fakeStore) WriteToShard(shardID uint64, points []models.Point) error {
	return f.WriteFn(shardID, points)
}

func (f *fakeStore) CreateShard(database, retentionPolicy string, shardID uint64) error {
	return f.CreateShardfn(database, retentionPolicy, shardID)
}

func NewMetaStore() *MetaStore {
	ms := &MetaStore{}
	rp := NewRetentionPolicy("myp", time.Hour, 3)
	AttachShardGroupInfo(rp, []meta.ShardOwner{
		{NodeID: 1},
		{NodeID: 2},
		{NodeID: 3},
	})
	AttachShardGroupInfo(rp, []meta.ShardOwner{
		{NodeID: 1},
		{NodeID: 2},
		{NodeID: 3},
	})

	ms.RetentionPolicyFn = func(db, retentionPolicy string) (*meta.RetentionPolicyInfo, error) {
		return rp, nil
	}

	ms.CreateShardGroupIfNotExistsFn = func(database, policy string, timestamp time.Time) (*meta.ShardGroupInfo, error) {
		for i, sg := range rp.ShardGroups {
			if timestamp.Equal(sg.StartTime) || timestamp.After(sg.StartTime) && timestamp.Before(sg.EndTime) {
				return &rp.ShardGroups[i], nil
			}
		}
		panic("should not get here")
	}
	return ms
}

type MetaStore struct {
	NodeIDFn                      func() uint64
	RetentionPolicyFn             func(database, name string) (*meta.RetentionPolicyInfo, error)
	CreateShardGroupIfNotExistsFn func(database, policy string, timestamp time.Time) (*meta.ShardGroupInfo, error)
	DatabaseFn                    func(database string) (*meta.DatabaseInfo, error)
	ShardOwnerFn                  func(shardID uint64) (string, string, *meta.ShardGroupInfo)
}

func (m MetaStore) NodeID() uint64 { return m.NodeIDFn() }

func (m MetaStore) RetentionPolicy(database, name string) (*meta.RetentionPolicyInfo, error) {
	return m.RetentionPolicyFn(database, name)
}

func (m MetaStore) CreateShardGroupIfNotExists(database, policy string, timestamp time.Time) (*meta.ShardGroupInfo, error) {
	return m.CreateShardGroupIfNotExistsFn(database, policy, timestamp)
}

func (m MetaStore) Database(database string) (*meta.DatabaseInfo, error) {
	return m.DatabaseFn(database)
}

func (m MetaStore) ShardOwner(shardID uint64) (string, string, *meta.ShardGroupInfo) {
	return m.ShardOwnerFn(shardID)
}

type Subscriber struct {
	PointsFn func() chan<- *cluster.WritePointsRequest
}

func (s Subscriber) Points() chan<- *cluster.WritePointsRequest {
	return s.PointsFn()
}

func NewRetentionPolicy(name string, duration time.Duration, nodeCount int) *meta.RetentionPolicyInfo {
	shards := []meta.ShardInfo{}
	owners := []meta.ShardOwner{}
	for i := 1; i <= nodeCount; i++ {
		owners = append(owners, meta.ShardOwner{NodeID: uint64(i)})
	}

	// each node is fully replicated with each other
	shards = append(shards, meta.ShardInfo{
		ID:     nextShardID(),
		Owners: owners,
	})

	rp := &meta.RetentionPolicyInfo{
		Name:               "myrp",
		ReplicaN:           nodeCount,
		Duration:           duration,
		ShardGroupDuration: duration,
		ShardGroups: []meta.ShardGroupInfo{
			meta.ShardGroupInfo{
				ID:        nextShardID(),
				StartTime: time.Unix(0, 0),
				EndTime:   time.Unix(0, 0).Add(duration).Add(-1),
				Shards:    shards,
			},
		},
	}
	return rp
}

func AttachShardGroupInfo(rp *meta.RetentionPolicyInfo, owners []meta.ShardOwner) {
	var startTime, endTime time.Time
	if len(rp.ShardGroups) == 0 {
		startTime = time.Unix(0, 0)
	} else {
		startTime = rp.ShardGroups[len(rp.ShardGroups)-1].StartTime.Add(rp.ShardGroupDuration)
	}
	endTime = startTime.Add(rp.ShardGroupDuration).Add(-1)

	sh := meta.ShardGroupInfo{
		ID:        uint64(len(rp.ShardGroups) + 1),
		StartTime: startTime,
		EndTime:   endTime,
		Shards: []meta.ShardInfo{
			meta.ShardInfo{
				ID:     nextShardID(),
				Owners: owners,
			},
		},
	}
	rp.ShardGroups = append(rp.ShardGroups, sh)
}

func nextShardID() uint64 {
	return atomic.AddUint64(&shardID, 1)
}
