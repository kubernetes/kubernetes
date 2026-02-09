// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package dbus

import (
	"context"
	"time"
)

// SubscriptionSet returns a subscription set which is like conn.Subscribe but
// can filter to only return events for a set of units.
type SubscriptionSet struct {
	*set
	conn *Conn
}

func (s *SubscriptionSet) filter(unit string) bool {
	return !s.Contains(unit)
}

// SubscribeContext starts listening for dbus events for all of the units in the set.
// Returns channels identical to conn.SubscribeUnits.
func (s *SubscriptionSet) SubscribeContext(ctx context.Context) (<-chan map[string]*UnitStatus, <-chan error) {
	// TODO: Make fully evented by using systemd 209 with properties changed values
	return s.conn.SubscribeUnitsCustomContext(ctx, time.Second, 0,
		mismatchUnitStatus,
		func(unit string) bool { return s.filter(unit) },
	)
}

// Deprecated: use SubscribeContext instead.
func (s *SubscriptionSet) Subscribe() (<-chan map[string]*UnitStatus, <-chan error) {
	return s.SubscribeContext(context.Background())
}

// NewSubscriptionSet returns a new subscription set.
func (c *Conn) NewSubscriptionSet() *SubscriptionSet {
	return &SubscriptionSet{newSet(), c}
}

// mismatchUnitStatus returns true if the provided UnitStatus objects
// are not equivalent. false is returned if the objects are equivalent.
// Only the Name, Description and state-related fields are used in
// the comparison.
func mismatchUnitStatus(u1, u2 *UnitStatus) bool {
	return u1.Name != u2.Name ||
		u1.Description != u2.Description ||
		u1.LoadState != u2.LoadState ||
		u1.ActiveState != u2.ActiveState ||
		u1.SubState != u2.SubState
}
