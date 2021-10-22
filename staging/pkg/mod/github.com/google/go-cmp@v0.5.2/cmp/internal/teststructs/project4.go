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
func equalCartel(x, y Cartel) bool {
	if !(equalHeadquarter(x.Headquarter, y.Headquarter) &&
		x.Source() == y.Source() &&
		x.CreationDate().Equal(y.CreationDate()) &&
		x.Boss() == y.Boss() &&
		x.LastCrimeDate().Equal(y.LastCrimeDate())) {
		return false
	}
	if len(x.Poisons()) != len(y.Poisons()) {
		return false
	}
	for i := range x.Poisons() {
		if !equalPoison(*x.Poisons()[i], *y.Poisons()[i]) {
			return false
		}
	}
	return true
}
func equalHeadquarter(x, y Headquarter) bool {
	xr, yr := x.Restrictions(), y.Restrictions()
	return x.ID() == y.ID() &&
		x.Location() == y.Location() &&
		reflect.DeepEqual(x.SubDivisions(), y.SubDivisions()) &&
		x.IncorporatedDate().Equal(y.IncorporatedDate()) &&
		pb.Equal(x.MetaData(), y.MetaData()) &&
		bytes.Equal(x.PrivateMessage(), y.PrivateMessage()) &&
		bytes.Equal(x.PublicMessage(), y.PublicMessage()) &&
		x.HorseBack() == y.HorseBack() &&
		x.Rattle() == y.Rattle() &&
		x.Convulsion() == y.Convulsion() &&
		x.Expansion() == y.Expansion() &&
		x.Status() == y.Status() &&
		pb.Equal(&xr, &yr) &&
		x.CreationTime().Equal(y.CreationTime())
}
func equalPoison(x, y Poison) bool {
	return x.PoisonType() == y.PoisonType() &&
		x.Expiration().Equal(y.Expiration()) &&
		x.Manufacturer() == y.Manufacturer() &&
		x.Potency() == y.Potency()
}
*/

type Cartel struct {
	Headquarter
	source        string
	creationDate  time.Time
	boss          string
	lastCrimeDate time.Time
	poisons       []*Poison
}

func (p Cartel) Source() string           { return p.source }
func (p Cartel) CreationDate() time.Time  { return p.creationDate }
func (p Cartel) Boss() string             { return p.boss }
func (p Cartel) LastCrimeDate() time.Time { return p.lastCrimeDate }
func (p Cartel) Poisons() []*Poison       { return p.poisons }

func (p *Cartel) SetSource(x string)           { p.source = x }
func (p *Cartel) SetCreationDate(x time.Time)  { p.creationDate = x }
func (p *Cartel) SetBoss(x string)             { p.boss = x }
func (p *Cartel) SetLastCrimeDate(x time.Time) { p.lastCrimeDate = x }
func (p *Cartel) SetPoisons(x []*Poison)       { p.poisons = x }

type Headquarter struct {
	id               uint64
	location         string
	subDivisions     []string
	incorporatedDate time.Time
	metaData         *pb.MetaData
	privateMessage   []byte
	publicMessage    []byte
	horseBack        string
	rattle           string
	convulsion       bool
	expansion        uint64
	status           pb.HoneyStatus
	restrictions     pb.Restrictions
	creationTime     time.Time
}

func (hq Headquarter) ID() uint64                    { return hq.id }
func (hq Headquarter) Location() string              { return hq.location }
func (hq Headquarter) SubDivisions() []string        { return hq.subDivisions }
func (hq Headquarter) IncorporatedDate() time.Time   { return hq.incorporatedDate }
func (hq Headquarter) MetaData() *pb.MetaData        { return hq.metaData }
func (hq Headquarter) PrivateMessage() []byte        { return hq.privateMessage }
func (hq Headquarter) PublicMessage() []byte         { return hq.publicMessage }
func (hq Headquarter) HorseBack() string             { return hq.horseBack }
func (hq Headquarter) Rattle() string                { return hq.rattle }
func (hq Headquarter) Convulsion() bool              { return hq.convulsion }
func (hq Headquarter) Expansion() uint64             { return hq.expansion }
func (hq Headquarter) Status() pb.HoneyStatus        { return hq.status }
func (hq Headquarter) Restrictions() pb.Restrictions { return hq.restrictions }
func (hq Headquarter) CreationTime() time.Time       { return hq.creationTime }

func (hq *Headquarter) SetID(x uint64)                    { hq.id = x }
func (hq *Headquarter) SetLocation(x string)              { hq.location = x }
func (hq *Headquarter) SetSubDivisions(x []string)        { hq.subDivisions = x }
func (hq *Headquarter) SetIncorporatedDate(x time.Time)   { hq.incorporatedDate = x }
func (hq *Headquarter) SetMetaData(x *pb.MetaData)        { hq.metaData = x }
func (hq *Headquarter) SetPrivateMessage(x []byte)        { hq.privateMessage = x }
func (hq *Headquarter) SetPublicMessage(x []byte)         { hq.publicMessage = x }
func (hq *Headquarter) SetHorseBack(x string)             { hq.horseBack = x }
func (hq *Headquarter) SetRattle(x string)                { hq.rattle = x }
func (hq *Headquarter) SetConvulsion(x bool)              { hq.convulsion = x }
func (hq *Headquarter) SetExpansion(x uint64)             { hq.expansion = x }
func (hq *Headquarter) SetStatus(x pb.HoneyStatus)        { hq.status = x }
func (hq *Headquarter) SetRestrictions(x pb.Restrictions) { hq.restrictions = x }
func (hq *Headquarter) SetCreationTime(x time.Time)       { hq.creationTime = x }

type Poison struct {
	poisonType   pb.PoisonType
	expiration   time.Time
	manufacturer string
	potency      int
}

func (p Poison) PoisonType() pb.PoisonType { return p.poisonType }
func (p Poison) Expiration() time.Time     { return p.expiration }
func (p Poison) Manufacturer() string      { return p.manufacturer }
func (p Poison) Potency() int              { return p.potency }

func (p *Poison) SetPoisonType(x pb.PoisonType) { p.poisonType = x }
func (p *Poison) SetExpiration(x time.Time)     { p.expiration = x }
func (p *Poison) SetManufacturer(x string)      { p.manufacturer = x }
func (p *Poison) SetPotency(x int)              { p.potency = x }
