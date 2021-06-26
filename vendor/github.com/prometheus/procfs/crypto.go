// Copyright 2019 The Prometheus Authors
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

package procfs

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// Crypto holds info parsed from /proc/crypto.
type Crypto struct {
	Alignmask   *uint64
	Async       bool
	Blocksize   *uint64
	Chunksize   *uint64
	Ctxsize     *uint64
	Digestsize  *uint64
	Driver      string
	Geniv       string
	Internal    string
	Ivsize      *uint64
	Maxauthsize *uint64
	MaxKeysize  *uint64
	MinKeysize  *uint64
	Module      string
	Name        string
	Priority    *int64
	Refcnt      *int64
	Seedsize    *uint64
	Selftest    string
	Type        string
	Walksize    *uint64
}

// Crypto parses an crypto-file (/proc/crypto) and returns a slice of
// structs containing the relevant info.  More information available here:
// https://kernel.readthedocs.io/en/sphinx-samples/crypto-API.html
func (fs FS) Crypto() ([]Crypto, error) {
	path := fs.proc.Path("crypto")
	b, err := util.ReadFileNoStat(path)
	if err != nil {
		return nil, fmt.Errorf("error reading crypto %q: %w", path, err)
	}

	crypto, err := parseCrypto(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("error parsing crypto %q: %w", path, err)
	}

	return crypto, nil
}

// parseCrypto parses a /proc/crypto stream into Crypto elements.
func parseCrypto(r io.Reader) ([]Crypto, error) {
	var out []Crypto

	s := bufio.NewScanner(r)
	for s.Scan() {
		text := s.Text()
		switch {
		case strings.HasPrefix(text, "name"):
			// Each crypto element begins with its name.
			out = append(out, Crypto{})
		case text == "":
			continue
		}

		kv := strings.Split(text, ":")
		if len(kv) != 2 {
			return nil, fmt.Errorf("malformed crypto line: %q", text)
		}

		k := strings.TrimSpace(kv[0])
		v := strings.TrimSpace(kv[1])

		// Parse the key/value pair into the currently focused element.
		c := &out[len(out)-1]
		if err := c.parseKV(k, v); err != nil {
			return nil, err
		}
	}

	if err := s.Err(); err != nil {
		return nil, err
	}

	return out, nil
}

// parseKV parses a key/value pair into the appropriate field of c.
func (c *Crypto) parseKV(k, v string) error {
	vp := util.NewValueParser(v)

	switch k {
	case "async":
		// Interpret literal yes as true.
		c.Async = v == "yes"
	case "blocksize":
		c.Blocksize = vp.PUInt64()
	case "chunksize":
		c.Chunksize = vp.PUInt64()
	case "digestsize":
		c.Digestsize = vp.PUInt64()
	case "driver":
		c.Driver = v
	case "geniv":
		c.Geniv = v
	case "internal":
		c.Internal = v
	case "ivsize":
		c.Ivsize = vp.PUInt64()
	case "maxauthsize":
		c.Maxauthsize = vp.PUInt64()
	case "max keysize":
		c.MaxKeysize = vp.PUInt64()
	case "min keysize":
		c.MinKeysize = vp.PUInt64()
	case "module":
		c.Module = v
	case "name":
		c.Name = v
	case "priority":
		c.Priority = vp.PInt64()
	case "refcnt":
		c.Refcnt = vp.PInt64()
	case "seedsize":
		c.Seedsize = vp.PUInt64()
	case "selftest":
		c.Selftest = v
	case "type":
		c.Type = v
	case "walksize":
		c.Walksize = vp.PUInt64()
	}

	return vp.Err()
}
