package digest

import (
	"bytes"
	"crypto/rand"
	_ "crypto/sha256"
	_ "crypto/sha512"
	"flag"
	"fmt"
	"strings"
	"testing"
)

func TestFlagInterface(t *testing.T) {
	var (
		alg     Algorithm
		flagSet flag.FlagSet
	)

	flagSet.Var(&alg, "algorithm", "set the digest algorithm")
	for _, testcase := range []struct {
		Name     string
		Args     []string
		Err      error
		Expected Algorithm
	}{
		{
			Name: "Invalid",
			Args: []string{"-algorithm", "bean"},
			Err:  ErrDigestUnsupported,
		},
		{
			Name:     "Default",
			Args:     []string{"unrelated"},
			Expected: "sha256",
		},
		{
			Name:     "Other",
			Args:     []string{"-algorithm", "sha512"},
			Expected: "sha512",
		},
	} {
		t.Run(testcase.Name, func(t *testing.T) {
			alg = Canonical
			if err := flagSet.Parse(testcase.Args); err != testcase.Err {
				if testcase.Err == nil {
					t.Fatal("unexpected error", err)
				}

				// check that flag package returns correct error
				if !strings.Contains(err.Error(), testcase.Err.Error()) {
					t.Fatalf("unexpected error: %v != %v", err, testcase.Err)
				}
				return
			}

			if alg != testcase.Expected {
				t.Fatalf("unexpected algorithm: %v != %v", alg, testcase.Expected)
			}
		})
	}
}

func TestFroms(t *testing.T) {
	p := make([]byte, 1<<20)
	rand.Read(p)

	for alg := range algorithms {
		h := alg.Hash()
		h.Write(p)
		expected := Digest(fmt.Sprintf("%s:%x", alg, h.Sum(nil)))
		readerDgst, err := alg.FromReader(bytes.NewReader(p))
		if err != nil {
			t.Fatalf("error calculating hash from reader: %v", err)
		}

		dgsts := []Digest{
			alg.FromBytes(p),
			alg.FromString(string(p)),
			readerDgst,
		}

		if alg == Canonical {
			readerDgst, err := FromReader(bytes.NewReader(p))
			if err != nil {
				t.Fatalf("error calculating hash from reader: %v", err)
			}

			dgsts = append(dgsts,
				FromBytes(p),
				FromString(string(p)),
				readerDgst)
		}
		for _, dgst := range dgsts {
			if dgst != expected {
				t.Fatalf("unexpected digest %v != %v", dgst, expected)
			}
		}
	}
}
