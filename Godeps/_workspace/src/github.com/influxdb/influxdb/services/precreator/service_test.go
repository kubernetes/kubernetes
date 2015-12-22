package precreator

import (
	"sync"
	"testing"
	"time"

	"github.com/influxdb/influxdb/toml"
)

func Test_ShardPrecreation(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC()
	advancePeriod := 5 * time.Minute

	// A test metastaore which returns 2 shard groups, only 1 of which requires a successor.
	var wg sync.WaitGroup
	wg.Add(1)
	ms := metaStore{
		PrecreateShardGroupsFn: func(v, u time.Time) error {
			wg.Done()
			if u != now.Add(advancePeriod) {
				t.Fatalf("precreation called with wrong time, got %s, exp %s", u, now)
			}
			return nil
		},
	}

	srv, err := NewService(Config{
		CheckInterval: toml.Duration(time.Minute),
		AdvancePeriod: toml.Duration(advancePeriod),
	})
	if err != nil {
		t.Fatalf("failed to create shard precreation service: %s", err.Error())
	}
	srv.MetaStore = ms

	err = srv.precreate(now)
	if err != nil {
		t.Fatalf("failed to precreate shards: %s", err.Error())
	}

	wg.Wait() // Ensure metastore test function is called.
	return
}

// PointsWriter represents a mock impl of PointsWriter.
type metaStore struct {
	PrecreateShardGroupsFn func(now, cutoff time.Time) error
}

func (m metaStore) IsLeader() bool {
	return true
}

func (m metaStore) PrecreateShardGroups(now, cutoff time.Time) error {
	return m.PrecreateShardGroupsFn(now, cutoff)
}
