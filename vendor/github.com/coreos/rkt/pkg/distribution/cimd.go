// Copyright 2016 The rkt Authors
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

package distribution

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

// cimd (container image distribution) is the struct
// that contains the information about a distribution point.
// A cimd URL is encoded as cimd:DistType:Version:Data.
type cimd struct {
	Type    Type
	Version uint32
	Data    string
}

// parseCIMD parses the given url and returns a cimd.
func parseCIMD(u *url.URL) (*cimd, error) {
	if u.Scheme != Scheme {
		return nil, fmt.Errorf("unsupported scheme: %q", u.Scheme)
	}
	parts := strings.SplitN(u.Opaque, ":", 3)
	if len(parts) < 3 {
		return nil, fmt.Errorf("malformed distribution uri: %q", u.String())
	}
	version, err := strconv.ParseUint(strings.TrimPrefix(parts[1], "v="), 10, 32)
	if err != nil {
		return nil, fmt.Errorf("malformed distribution version: %s", parts[1])
	}
	return &cimd{
		Type:    Type(parts[0]),
		Version: uint32(version),
		Data:    parts[2],
	}, nil
}

// NewCIMDString creates a new cimd URL string.
func NewCIMDString(typ Type, version uint32, data string) string {
	return fmt.Sprintf("%s:%s:v=%d:%s", Scheme, typ, version, data)
}
