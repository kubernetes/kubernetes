package meta

import (
	"reflect"
	"sort"
	"time"

	"testing"
)

func Test_newShardOwner(t *testing.T) {
	// An error is returned if there are no data nodes available.
	_, err := NewShardOwner(ShardInfo{}, map[int]int{})
	if err == nil {
		t.Error("got no error, but expected one")
	}

	ownerFreqs := map[int]int{1: 15, 2: 11, 3: 12}
	id, err := NewShardOwner(ShardInfo{ID: 4}, ownerFreqs)
	if err != nil {
		t.Fatal(err)
	}

	// The ID that owns the fewest shards is returned.
	if got, exp := id, uint64(2); got != exp {
		t.Errorf("got id %d, expected id %d", got, exp)
	}

	// The ownership frequencies are updated.
	if got, exp := ownerFreqs, map[int]int{1: 15, 2: 12, 3: 12}; !reflect.DeepEqual(got, exp) {
		t.Errorf("got owner frequencies %v, expected %v", got, exp)
	}
}

func TestShardGroupSort(t *testing.T) {
	sg1 := ShardGroupInfo{
		ID:          1,
		StartTime:   time.Unix(1000, 0),
		EndTime:     time.Unix(1100, 0),
		TruncatedAt: time.Unix(1050, 0),
	}

	sg2 := ShardGroupInfo{
		ID:        2,
		StartTime: time.Unix(1000, 0),
		EndTime:   time.Unix(1100, 0),
	}

	sgs := ShardGroupInfos{sg2, sg1}

	sort.Sort(sgs)

	if sgs[len(sgs)-1].ID != 2 {
		t.Fatal("unstable sort for ShardGroupInfos")
	}
}
