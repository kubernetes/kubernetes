// Copyright 2015 CoreOS, Inc
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

// +build tpm

package tpm

import (
	"time"

	"github.com/coreos/go-tspi/tpmclient"
)

// we're connecting to localhost, so 10ms is a reasonable timeout value
const timeout = 10 * time.Millisecond

// Extend extends the TPM log with the provided string. Returns any error.
func Extend(description string) error {
	connection := tpmclient.New("localhost:12041", timeout)
	err := connection.Extend(15, 0x1000, nil, description)
	return err
}
