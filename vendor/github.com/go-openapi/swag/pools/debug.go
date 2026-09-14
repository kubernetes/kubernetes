// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package pools

// TB is the subset of [testing.TB] used by [AssertNoLeaks].
//
// It is satisfied by *[testing.T] and *[testing.B].
//
// A local interface is used (rather than importing "testing") so that the
// release build does not pull the testing package — and its flags — into
// production binaries.
type TB interface {
	Helper()
	Errorf(format string, args ...any)
	Logf(format string, args ...any)
}
