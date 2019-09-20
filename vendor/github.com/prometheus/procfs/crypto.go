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
	"bytes"
	"fmt"
	"io/ioutil"
	"strconv"
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
	data, err := ioutil.ReadFile(fs.proc.Path("crypto"))
	if err != nil {
		return nil, fmt.Errorf("error parsing crypto %s: %s", fs.proc.Path("crypto"), err)
	}
	crypto, err := parseCrypto(data)
	if err != nil {
		return nil, fmt.Errorf("error parsing crypto %s: %s", fs.proc.Path("crypto"), err)
	}
	return crypto, nil
}

func parseCrypto(cryptoData []byte) ([]Crypto, error) {
	crypto := []Crypto{}

	cryptoBlocks := bytes.Split(cryptoData, []byte("\n\n"))

	for _, block := range cryptoBlocks {
		var newCryptoElem Crypto

		lines := strings.Split(string(block), "\n")
		for _, line := range lines {
			if strings.TrimSpace(line) == "" || line[0] == ' ' {
				continue
			}
			fields := strings.Split(line, ":")
			key := strings.TrimSpace(fields[0])
			value := strings.TrimSpace(fields[1])
			vp := util.NewValueParser(value)

			switch strings.TrimSpace(key) {
			case "async":
				b, err := strconv.ParseBool(value)
				if err == nil {
					newCryptoElem.Async = b
				}
			case "blocksize":
				newCryptoElem.Blocksize = vp.PUInt64()
			case "chunksize":
				newCryptoElem.Chunksize = vp.PUInt64()
			case "digestsize":
				newCryptoElem.Digestsize = vp.PUInt64()
			case "driver":
				newCryptoElem.Driver = value
			case "geniv":
				newCryptoElem.Geniv = value
			case "internal":
				newCryptoElem.Internal = value
			case "ivsize":
				newCryptoElem.Ivsize = vp.PUInt64()
			case "maxauthsize":
				newCryptoElem.Maxauthsize = vp.PUInt64()
			case "max keysize":
				newCryptoElem.MaxKeysize = vp.PUInt64()
			case "min keysize":
				newCryptoElem.MinKeysize = vp.PUInt64()
			case "module":
				newCryptoElem.Module = value
			case "name":
				newCryptoElem.Name = value
			case "priority":
				newCryptoElem.Priority = vp.PInt64()
			case "refcnt":
				newCryptoElem.Refcnt = vp.PInt64()
			case "seedsize":
				newCryptoElem.Seedsize = vp.PUInt64()
			case "selftest":
				newCryptoElem.Selftest = value
			case "type":
				newCryptoElem.Type = value
			case "walksize":
				newCryptoElem.Walksize = vp.PUInt64()
			}
		}
		crypto = append(crypto, newCryptoElem)
	}
	return crypto, nil
}
