// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

package teststructs

import (
	"time"

	pb "github.com/google/go-cmp/cmp/internal/testprotos"
)

// This is an sanitized example of equality from a real use-case.
// The original equality function was as follows:
/*
func equalBatch(b1, b2 *GermBatch) bool {
	for _, b := range []*GermBatch{b1, b2} {
		for _, l := range b.DirtyGerms {
			sort.Slice(l, func(i, j int) bool { return l[i].String() < l[j].String() })
		}
		for _, l := range b.CleanGerms {
			sort.Slice(l, func(i, j int) bool { return l[i].String() < l[j].String() })
		}
	}
	if !pb.DeepEqual(b1.DirtyGerms, b2.DirtyGerms) ||
		!pb.DeepEqual(b1.CleanGerms, b2.CleanGerms) ||
		!pb.DeepEqual(b1.GermMap, b2.GermMap) {
		return false
	}
	if len(b1.DishMap) != len(b2.DishMap) {
		return false
	}
	for id := range b1.DishMap {
		kpb1, err1 := b1.DishMap[id].Proto()
		kpb2, err2 := b2.DishMap[id].Proto()
		if !pb.Equal(kpb1, kpb2) || !reflect.DeepEqual(err1, err2) {
			return false
		}
	}
	return b1.HasPreviousResult == b2.HasPreviousResult &&
		b1.DirtyID == b2.DirtyID &&
		b1.CleanID == b2.CleanID &&
		b1.GermStrain == b2.GermStrain &&
		b1.TotalDirtyGerms == b2.TotalDirtyGerms &&
		b1.InfectedAt.Equal(b2.InfectedAt)
}
*/

type GermBatch struct {
	DirtyGerms, CleanGerms map[int32][]*pb.Germ
	GermMap                map[int32]*pb.Germ
	DishMap                map[int32]*Dish
	HasPreviousResult      bool
	DirtyID, CleanID       int32
	GermStrain             int32
	TotalDirtyGerms        int
	InfectedAt             time.Time
}

type Dish struct {
	pb  *pb.Dish
	err error
}

func CreateDish(m *pb.Dish, err error) *Dish {
	return &Dish{pb: m, err: err}
}

func (d *Dish) Proto() (*pb.Dish, error) {
	if d.err != nil {
		return nil, d.err
	}
	return d.pb, nil
}
