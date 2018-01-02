package coordinator

import (
	"testing"
	"time"
)

func TestSgList_ShardGroupAt(t *testing.T) {
	base := time.Date(2016, 10, 19, 0, 0, 0, 0, time.UTC)
	day := func(n int) time.Time {
		return base.Add(time.Duration(24*n) * time.Hour)
	}

	list := sgList{
		{ID: 1, StartTime: day(0), EndTime: day(1)},
		{ID: 2, StartTime: day(1), EndTime: day(2)},
		{ID: 3, StartTime: day(2), EndTime: day(3)},
		// SG day 3 to day 4 missing...
		{ID: 4, StartTime: day(4), EndTime: day(5)},
		{ID: 5, StartTime: day(5), EndTime: day(6)},
	}

	examples := []struct {
		T            time.Time
		ShardGroupID uint64 // 0 will indicate we don't expect a shard group
	}{
		{T: base.Add(-time.Minute), ShardGroupID: 0}, // Before any SG
		{T: day(0), ShardGroupID: 1},
		{T: day(0).Add(time.Minute), ShardGroupID: 1},
		{T: day(1), ShardGroupID: 2},
		{T: day(3).Add(time.Minute), ShardGroupID: 0}, // No matching SG
		{T: day(5).Add(time.Hour), ShardGroupID: 5},
	}

	for i, example := range examples {
		sg := list.ShardGroupAt(example.T)
		var id uint64
		if sg != nil {
			id = sg.ID
		}

		if got, exp := id, example.ShardGroupID; got != exp {
			t.Errorf("[Example %d] got %v, expected %v", i+1, got, exp)
		}
	}
}
