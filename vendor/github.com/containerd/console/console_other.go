//go:build !darwin && !freebsd && !linux && !netbsd && !openbsd && !solaris && !windows && !zos
// +build !darwin,!freebsd,!linux,!netbsd,!openbsd,!solaris,!windows,!zos

/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package console

// NewPty creates a new pty pair
// The master is returned as the first console and a string
// with the path to the pty slave is returned as the second
func NewPty() (Console, string, error) {
	return nil, "", ErrNotImplemented
}

// checkConsole checks if the provided file is a console
func checkConsole(f File) error {
	return ErrNotAConsole
}

func newMaster(f File) (Console, error) {
	return nil, ErrNotImplemented
}
