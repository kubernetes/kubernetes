// Copyright 2014 CoreOS, Inc.
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

package store

import (
	"compress/bzip2"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/appc/spec/aci"
)

// blockTransform creates a path slice from the given string to use as a
// directory prefix. The string must be in hash format:
//    "sha256-abcdefgh"... -> []{"sha256", "ab"}
// Right now it just copies the default of git which is a two byte prefix. We
// will likely want to add re-sharding later.
func blockTransform(s string) []string {
	// TODO(philips): use spec/types.Hash after export typ field
	parts := strings.SplitN(s, "-", 2)
	if len(parts) != 2 {
		panic(fmt.Errorf("blockTransform should never receive non-hash, got %v", s))
	}
	return []string{
		parts[0],
		parts[1][0:2],
	}
}

func parseAlways(s string) *url.URL {
	u, _ := url.Parse(s)
	return u
}

func decompress(rs io.Reader, typ aci.FileType) (io.Reader, error) {
	var (
		dr  io.Reader
		err error
	)
	switch typ {
	case aci.TypeGzip:
		dr, err = gzip.NewReader(rs)
		if err != nil {
			return nil, err
		}
	case aci.TypeBzip2:
		dr = bzip2.NewReader(rs)
	case aci.TypeXz:
		dr = aci.XzReader(rs)
	case aci.TypeTar:
		dr = rs
	case aci.TypeUnknown:
		return nil, errors.New("error: unknown image filetype")
	default:
		return nil, errors.New("no type returned from DetectFileType?")
	}
	return dr, nil
}
