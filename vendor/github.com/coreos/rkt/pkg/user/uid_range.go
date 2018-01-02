// Copyright 2015 The rkt Authors
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

// For how the uidshift and uidcount are generated please check:
// http://cgit.freedesktop.org/systemd/systemd/commit/?id=03cfe0d51499e86b1573d1

package user

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/hashicorp/errwrap"
)

const DefaultRangeCount = 0x10000

// A UidRange structure used to set uidshift and its range.
type UidRange struct {
	Shift uint32
	Count uint32
}

func generateUidShift() uint32 {
	rand.Seed(time.Now().UnixNano())
	// we force the MSB to 0 because devpts parses the uid,gid options as int
	// instead of as uint.
	// http://lxr.free-electrons.com/source/fs/devpts/inode.c?v=4.1#L189
	// systemd issue: https://github.com/systemd/systemd/issues/956
	n := rand.Intn(0x7FFF) + 1
	uidShift := uint32(n << 16)

	return uidShift
}

func NewBlankUidRange() *UidRange {
	return &UidRange{
		Shift: 0,
		Count: 0}
}

func (r *UidRange) SetRandomUidRange(uidCount uint32) {
	uidShift := generateUidShift()
	r.Shift = uidShift
	r.Count = uidCount
}

func (r *UidRange) ShiftRange(uid uint32, gid uint32) (uint32, uint32, error) {
	if r.Count > 0 && (uid >= r.Count || gid >= r.Count) {
		return 0, 0, fmt.Errorf("uid %d or gid %d are out of range %d", uid, gid, r.Count)
	}
	if math.MaxUint32-r.Shift < uid || math.MaxUint32-r.Shift < gid {
		return 0, 0, fmt.Errorf("uid or gid are out of range %d after shifting", uint32(math.MaxUint32))
	}
	return uid + r.Shift, gid + r.Shift, nil
}

func (r *UidRange) UnshiftRange(uid, gid uint32) (uint32, uint32, error) {
	if uid < r.Shift || gid < r.Shift || (r.Count > 0 && (uid >= r.Shift+r.Count || gid >= r.Shift+r.Count)) {
		return 0, 0, fmt.Errorf("uid %d or gid %d are out of range %d after unshifting", uid, gid, r.Count)
	}
	return uid - r.Shift, gid - r.Shift, nil
}

func (r *UidRange) Serialize() []byte {
	return []byte(fmt.Sprintf("%d:%d", r.Shift, r.Count))
}

func (r *UidRange) Deserialize(uidRange []byte) error {
	if len(uidRange) == 0 {
		return nil
	}
	_, err := fmt.Sscanf(string(uidRange), "%d:%d", &r.Shift, &r.Count)
	if err != nil {
		return errwrap.Wrap(errors.New("error deserializing uid range"), err)
	}

	return nil
}
