// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package conv

type (
	// these type constraints are redefined after golang.org/x/exp/constraints,
	// because importing that package causes an undesired go upgrade.

	// Signed integer types, cf. [golang.org/x/exp/constraints.Signed]
	Signed interface {
		~int | ~int8 | ~int16 | ~int32 | ~int64
	}

	// Unsigned integer types, cf. [golang.org/x/exp/constraints.Unsigned]
	Unsigned interface {
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
	}

	// Float numerical types, cf. [golang.org/x/exp/constraints.Float]
	Float interface {
		~float32 | ~float64
	}

	// Numerical types
	Numerical interface {
		Signed | Unsigned | Float
	}
)
