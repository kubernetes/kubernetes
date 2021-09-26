// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package wycheproof runs a set of the Wycheproof tests
// provided by https://github.com/google/wycheproof.
package wycheproof

import (
	"crypto"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	_ "crypto/sha1"
	_ "crypto/sha256"
	_ "crypto/sha512"
)

const wycheproofModVer = "v0.0.0-20191219022705-2196000605e4"

var wycheproofTestVectorsDir string

func TestMain(m *testing.M) {
	if _, err := exec.LookPath("go"); err != nil {
		log.Printf("skipping test because 'go' command is unavailable: %v", err)
		os.Exit(0)
	}

	// Download the JSON test files from github.com/google/wycheproof
	// using `go mod download -json` so the cached source of the testdata
	// can be used in the following tests.
	path := "github.com/google/wycheproof@" + wycheproofModVer
	cmd := exec.Command("go", "mod", "download", "-json", path)
	// TODO: enable the sumdb once the Trybots proxy supports it.
	cmd.Env = append(os.Environ(), "GONOSUMDB=*")
	output, err := cmd.Output()
	if err != nil {
		log.Fatalf("failed to run `go mod download -json %s`, output: %s", path, output)
	}
	var dm struct {
		Dir string // absolute path to cached source root directory
	}
	if err := json.Unmarshal(output, &dm); err != nil {
		log.Fatal(err)
	}
	// Now that the module has been downloaded, use the absolute path of the
	// cached source as the root directory for all tests going forward.
	wycheproofTestVectorsDir = filepath.Join(dm.Dir, "testvectors")
	os.Exit(m.Run())
}

func readTestVector(t *testing.T, f string, dest interface{}) {
	b, err := ioutil.ReadFile(filepath.Join(wycheproofTestVectorsDir, f))
	if err != nil {
		t.Fatalf("failed to read json file: %v", err)
	}
	if err := json.Unmarshal(b, &dest); err != nil {
		t.Fatalf("failed to unmarshal json file: %v", err)
	}
}

func decodeHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func decodePublicKey(der string) interface{} {
	d := decodeHex(der)
	pub, err := x509.ParsePKIXPublicKey(d)
	if err != nil {
		panic(fmt.Sprintf("failed to parse DER encoded public key: %v", err))
	}
	return pub
}

func parseHash(h string) crypto.Hash {
	switch h {
	case "SHA-1":
		return crypto.SHA1
	case "SHA-256":
		return crypto.SHA256
	case "SHA-224":
		return crypto.SHA224
	case "SHA-384":
		return crypto.SHA384
	case "SHA-512":
		return crypto.SHA512
	case "SHA-512/224":
		return crypto.SHA512_224
	case "SHA-512/256":
		return crypto.SHA512_256
	default:
		panic(fmt.Sprintf("could not identify SHA hash algorithm: %q", h))
	}
}

// shouldPass returns whether or not the test should pass.
// flagsShouldPass is a map associated with whether or not
// a flag for an "acceptable" result should pass.
// Every possible flag value that's associated with an
// "acceptable" result should be explicitly specified,
// otherwise the test will panic.
func shouldPass(result string, flags []string, flagsShouldPass map[string]bool) bool {
	switch result {
	case "valid":
		return true
	case "invalid":
		return false
	case "acceptable":
		for _, flag := range flags {
			pass, ok := flagsShouldPass[flag]
			if !ok {
				panic(fmt.Sprintf("unspecified flag: %q", flag))
			}
			if !pass {
				return false
			}
		}
		return true // There are no flags, or all are meant to pass.
	default:
		panic(fmt.Sprintf("unexpected result: %v", result))
	}
}
