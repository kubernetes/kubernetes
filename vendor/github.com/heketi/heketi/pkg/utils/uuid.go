//
// Copyright (c) 2015 The heketi Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package utils

// From http://www.ashishbanerjee.com/home/go/go-generate-uuid

import (
	"crypto/rand"
	"encoding/hex"
	"github.com/lpabon/godbc"
)

// Return a 16-byte uuid
func GenUUID() string {
	uuid := make([]byte, 16)
	n, err := rand.Read(uuid)
	godbc.Check(n == len(uuid), n, len(uuid))
	godbc.Check(err == nil, err)

	return hex.EncodeToString(uuid)
}
