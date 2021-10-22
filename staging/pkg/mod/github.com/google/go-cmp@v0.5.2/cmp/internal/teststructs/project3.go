// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

package teststructs

import (
	"sync"

	pb "github.com/google/go-cmp/cmp/internal/testprotos"
)

// This is an sanitized example of equality from a real use-case.
// The original equality function was as follows:
/*
func equalDirt(x, y *Dirt) bool {
	if !reflect.DeepEqual(x.table, y.table) ||
		!reflect.DeepEqual(x.ts, y.ts) ||
		x.Discord != y.Discord ||
		!pb.Equal(&x.Proto, &y.Proto) ||
		len(x.wizard) != len(y.wizard) ||
		len(x.sadistic) != len(y.sadistic) ||
		x.lastTime != y.lastTime {
		return false
	}
	for k, vx := range x.wizard {
		vy, ok := y.wizard[k]
		if !ok || !pb.Equal(vx, vy) {
			return false
		}
	}
	for k, vx := range x.sadistic {
		vy, ok := y.sadistic[k]
		if !ok || !pb.Equal(vx, vy) {
			return false
		}
	}
	return true
}
*/

type FakeMutex struct {
	sync.Locker
	x struct{}
}

type Dirt struct {
	table    Table // Always concrete type of MockTable
	ts       Timestamp
	Discord  DiscordState
	Proto    pb.Dirt
	wizard   map[string]*pb.Wizard
	sadistic map[string]*pb.Sadistic
	lastTime int64
	mu       FakeMutex
}

type DiscordState int

type Timestamp int64

func (d *Dirt) SetTable(t Table)                      { d.table = t }
func (d *Dirt) SetTimestamp(t Timestamp)              { d.ts = t }
func (d *Dirt) SetWizard(m map[string]*pb.Wizard)     { d.wizard = m }
func (d *Dirt) SetSadistic(m map[string]*pb.Sadistic) { d.sadistic = m }
func (d *Dirt) SetLastTime(t int64)                   { d.lastTime = t }

type Table interface {
	Operation1() error
	Operation2() error
	Operation3() error
}

type MockTable struct {
	state []string
}

func CreateMockTable(s []string) *MockTable { return &MockTable{s} }
func (mt *MockTable) Operation1() error     { return nil }
func (mt *MockTable) Operation2() error     { return nil }
func (mt *MockTable) Operation3() error     { return nil }
func (mt *MockTable) State() []string       { return mt.state }
