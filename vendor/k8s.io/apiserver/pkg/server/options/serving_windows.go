//go:build windows
// +build windows

/*
Copyright 2020 The Kubernetes Authors.

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

package options

import (
	"fmt"
	"syscall"
)

func permitPortReuse(network, address string, c syscall.RawConn) error {
	return fmt.Errorf("port reuse is not supported on Windows")
}

// Windows supports SO_REUSEADDR, but it may cause undefined behavior, as
// there is no protection against port hijacking.
func permitAddressReuse(network, addr string, conn syscall.RawConn) error {
	return fmt.Errorf("address reuse is not supported on Windows")
}
