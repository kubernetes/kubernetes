// Copyright 2021 The etcd Authors
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

package config

type V2DeprecationEnum string

const (
	// V2Depr0NotYet means v2store isn't deprecated yet.
	// Default in v3.5, and no longer supported in v3.6.
	V2Depr0NotYet = V2DeprecationEnum("not-yet")

	// Deprecated: to be decommissioned in 3.7. Please use V2Depr0NotYet.
	// TODO: remove in 3.7
	//revive:disable-next-line:var-naming
	V2_DEPR_0_NOT_YET = V2Depr0NotYet

	// V2Depr1WriteOnly means only writing v2store is allowed.
	// Default in v3.6.  Meaningful v2 state is not allowed.
	// The V2 files are maintained for v3.5 rollback.
	V2Depr1WriteOnly = V2DeprecationEnum("write-only")

	// Deprecated: to be decommissioned in 3.7. Please use V2Depr1WriteOnly.
	// TODO: remove in 3.7
	//revive:disable-next-line:var-naming
	V2_DEPR_1_WRITE_ONLY = V2Depr1WriteOnly

	// V2Depr1WriteOnlyDrop means v2store is WIPED if found !!!
	// Will be default in 3.7.
	V2Depr1WriteOnlyDrop = V2DeprecationEnum("write-only-drop-data")

	// Deprecated: to be decommissioned in 3.7. Pleae use V2Depr1WriteOnlyDrop.
	// TODO: remove in 3.7
	//revive:disable-next-line:var-naming
	V2_DEPR_1_WRITE_ONLY_DROP = V2Depr1WriteOnlyDrop

	// V2Depr2Gone means v2store is completely gone. The v2store is
	// neither written nor read. Anything related to v2store will be
	// cleaned up in v3.8. Usage of this configuration is blocking
	// ability to rollback to etcd v3.5.
	V2Depr2Gone = V2DeprecationEnum("gone")

	// Deprecated: to be decommissioned in 3.7. Please use V2Depr2Gone.
	// TODO: remove in 3.7
	//revive:disable-next-line:var-naming
	V2_DEPR_2_GONE = V2Depr2Gone

	// V2DeprDefault is the default deprecation level.
	V2DeprDefault = V2Depr1WriteOnly

	// Deprecated: to be decommissioned in 3.7. Please use V2DeprDefault.
	// TODO: remove in 3.7
	//revive:disable-next-line:var-naming
	V2_DEPR_DEFAULT = V2DeprDefault
)

func (e V2DeprecationEnum) IsAtLeast(v2d V2DeprecationEnum) bool {
	return e.level() >= v2d.level()
}

func (e V2DeprecationEnum) level() int {
	switch e {
	case V2Depr0NotYet:
		return 0
	case V2Depr1WriteOnly:
		return 1
	case V2Depr1WriteOnlyDrop:
		return 2
	case V2Depr2Gone:
		return 3
	}
	panic("Unknown V2DeprecationEnum: " + e)
}
