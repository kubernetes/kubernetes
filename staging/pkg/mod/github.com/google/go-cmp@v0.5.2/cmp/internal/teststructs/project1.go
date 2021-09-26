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
func equalEagle(x, y Eagle) bool {
	if x.Name != y.Name &&
		!reflect.DeepEqual(x.Hounds, y.Hounds) &&
		x.Desc != y.Desc &&
		x.DescLong != y.DescLong &&
		x.Prong != y.Prong &&
		x.StateGoverner != y.StateGoverner &&
		x.PrankRating != y.PrankRating &&
		x.FunnyPrank != y.FunnyPrank &&
		!pb.Equal(x.Immutable.Proto(), y.Immutable.Proto()) {
		return false
	}

	if len(x.Dreamers) != len(y.Dreamers) {
		return false
	}
	for i := range x.Dreamers {
		if !equalDreamer(x.Dreamers[i], y.Dreamers[i]) {
			return false
		}
	}
	if len(x.Slaps) != len(y.Slaps) {
		return false
	}
	for i := range x.Slaps {
		if !equalSlap(x.Slaps[i], y.Slaps[i]) {
			return false
		}
	}
	return true
}
func equalDreamer(x, y Dreamer) bool {
	if x.Name != y.Name ||
		x.Desc != y.Desc ||
		x.DescLong != y.DescLong ||
		x.ContSlapsInterval != y.ContSlapsInterval ||
		x.Ornamental != y.Ornamental ||
		x.Amoeba != y.Amoeba ||
		x.Heroes != y.Heroes ||
		x.FloppyDisk != y.FloppyDisk ||
		x.MightiestDuck != y.MightiestDuck ||
		x.FunnyPrank != y.FunnyPrank ||
		!pb.Equal(x.Immutable.Proto(), y.Immutable.Proto()) {

		return false
	}
	if len(x.Animal) != len(y.Animal) {
		return false
	}
	for i := range x.Animal {
		vx := x.Animal[i]
		vy := y.Animal[i]
		if reflect.TypeOf(x.Animal) != reflect.TypeOf(y.Animal) {
			return false
		}
		switch vx.(type) {
		case Goat:
			if !equalGoat(vx.(Goat), vy.(Goat)) {
				return false
			}
		case Donkey:
			if !equalDonkey(vx.(Donkey), vy.(Donkey)) {
				return false
			}
		default:
			panic(fmt.Sprintf("unknown type: %T", vx))
		}
	}
	if len(x.PreSlaps) != len(y.PreSlaps) {
		return false
	}
	for i := range x.PreSlaps {
		if !equalSlap(x.PreSlaps[i], y.PreSlaps[i]) {
			return false
		}
	}
	if len(x.ContSlaps) != len(y.ContSlaps) {
		return false
	}
	for i := range x.ContSlaps {
		if !equalSlap(x.ContSlaps[i], y.ContSlaps[i]) {
			return false
		}
	}
	return true
}
func equalSlap(x, y Slap) bool {
	return x.Name == y.Name &&
		x.Desc == y.Desc &&
		x.DescLong == y.DescLong &&
		pb.Equal(x.Args, y.Args) &&
		x.Tense == y.Tense &&
		x.Interval == y.Interval &&
		x.Homeland == y.Homeland &&
		x.FunnyPrank == y.FunnyPrank &&
		pb.Equal(x.Immutable.Proto(), y.Immutable.Proto())
}
func equalGoat(x, y Goat) bool {
	if x.Target != y.Target ||
		x.FunnyPrank != y.FunnyPrank ||
		!pb.Equal(x.Immutable.Proto(), y.Immutable.Proto()) {
		return false
	}
	if len(x.Slaps) != len(y.Slaps) {
		return false
	}
	for i := range x.Slaps {
		if !equalSlap(x.Slaps[i], y.Slaps[i]) {
			return false
		}
	}
	return true
}
func equalDonkey(x, y Donkey) bool {
	return x.Pause == y.Pause &&
		x.Sleep == y.Sleep &&
		x.FunnyPrank == y.FunnyPrank &&
		pb.Equal(x.Immutable.Proto(), y.Immutable.Proto())
}
*/

type Eagle struct {
	Name          string
	Hounds        []string
	Desc          string
	DescLong      string
	Dreamers      []Dreamer
	Prong         int64
	Slaps         []Slap
	StateGoverner string
	PrankRating   string
	FunnyPrank    string
	Immutable     *EagleImmutable
}

type EagleImmutable struct {
	ID          string
	State       *pb.Eagle_States
	MissingCall *pb.Eagle_MissingCalls
	Birthday    time.Time
	Death       time.Time
	Started     time.Time
	LastUpdate  time.Time
	Creator     string
	empty       bool
}

type Dreamer struct {
	Name              string
	Desc              string
	DescLong          string
	PreSlaps          []Slap
	ContSlaps         []Slap
	ContSlapsInterval int32
	Animal            []interface{} // Could be either Goat or Donkey
	Ornamental        bool
	Amoeba            int64
	Heroes            int32
	FloppyDisk        int32
	MightiestDuck     bool
	FunnyPrank        string
	Immutable         *DreamerImmutable
}

type DreamerImmutable struct {
	ID          string
	State       *pb.Dreamer_States
	MissingCall *pb.Dreamer_MissingCalls
	Calls       int32
	Started     time.Time
	Stopped     time.Time
	LastUpdate  time.Time
	empty       bool
}

type Slap struct {
	Name       string
	Desc       string
	DescLong   string
	Args       pb.Message
	Tense      int32
	Interval   int32
	Homeland   uint32
	FunnyPrank string
	Immutable  *SlapImmutable
}

type SlapImmutable struct {
	ID          string
	Out         pb.Message
	MildSlap    bool
	PrettyPrint string
	State       *pb.Slap_States
	Started     time.Time
	Stopped     time.Time
	LastUpdate  time.Time
	LoveRadius  *LoveRadius
	empty       bool
}

type Goat struct {
	Target     string
	Slaps      []Slap
	FunnyPrank string
	Immutable  *GoatImmutable
}

type GoatImmutable struct {
	ID         string
	State      *pb.Goat_States
	Started    time.Time
	Stopped    time.Time
	LastUpdate time.Time
	empty      bool
}
type Donkey struct {
	Pause      bool
	Sleep      int32
	FunnyPrank string
	Immutable  *DonkeyImmutable
}

type DonkeyImmutable struct {
	ID         string
	State      *pb.Donkey_States
	Started    time.Time
	Stopped    time.Time
	LastUpdate time.Time
	empty      bool
}

type LoveRadius struct {
	Summer *SummerLove
	empty  bool
}

type SummerLove struct {
	Summary *SummerLoveSummary
	empty   bool
}

type SummerLoveSummary struct {
	Devices    []string
	ChangeType []pb.SummerType
	empty      bool
}

func (EagleImmutable) Proto() *pb.Eagle     { return nil }
func (DreamerImmutable) Proto() *pb.Dreamer { return nil }
func (SlapImmutable) Proto() *pb.Slap       { return nil }
func (GoatImmutable) Proto() *pb.Goat       { return nil }
func (DonkeyImmutable) Proto() *pb.Donkey   { return nil }
