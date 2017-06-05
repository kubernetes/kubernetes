package digest

import (
	"crypto/sha256"
	"encoding/binary"
	"math/rand"
	"testing"
)

func assertEqualDigests(t *testing.T, d1, d2 Digest) {
	if d1 != d2 {
		t.Fatalf("Digests do not match:\n\tActual: %s\n\tExpected: %s", d1, d2)
	}
}

func TestLookup(t *testing.T) {
	digests := []Digest{
		"sha256:1234511111111111111111111111111111111111111111111111111111111111",
		"sha256:1234111111111111111111111111111111111111111111111111111111111111",
		"sha256:1234611111111111111111111111111111111111111111111111111111111111",
		"sha256:5432111111111111111111111111111111111111111111111111111111111111",
		"sha256:6543111111111111111111111111111111111111111111111111111111111111",
		"sha256:6432111111111111111111111111111111111111111111111111111111111111",
		"sha256:6542111111111111111111111111111111111111111111111111111111111111",
		"sha256:6532111111111111111111111111111111111111111111111111111111111111",
	}

	dset := NewSet()
	for i := range digests {
		if err := dset.Add(digests[i]); err != nil {
			t.Fatal(err)
		}
	}

	dgst, err := dset.Lookup("54")
	if err != nil {
		t.Fatal(err)
	}
	assertEqualDigests(t, dgst, digests[3])

	dgst, err = dset.Lookup("1234")
	if err == nil {
		t.Fatal("Expected ambiguous error looking up: 1234")
	}
	if err != ErrDigestAmbiguous {
		t.Fatal(err)
	}

	dgst, err = dset.Lookup("9876")
	if err == nil {
		t.Fatal("Expected ambiguous error looking up: 9876")
	}
	if err != ErrDigestNotFound {
		t.Fatal(err)
	}

	dgst, err = dset.Lookup("sha256:1234")
	if err == nil {
		t.Fatal("Expected ambiguous error looking up: sha256:1234")
	}
	if err != ErrDigestAmbiguous {
		t.Fatal(err)
	}

	dgst, err = dset.Lookup("sha256:12345")
	if err != nil {
		t.Fatal(err)
	}
	assertEqualDigests(t, dgst, digests[0])

	dgst, err = dset.Lookup("sha256:12346")
	if err != nil {
		t.Fatal(err)
	}
	assertEqualDigests(t, dgst, digests[2])

	dgst, err = dset.Lookup("12346")
	if err != nil {
		t.Fatal(err)
	}
	assertEqualDigests(t, dgst, digests[2])

	dgst, err = dset.Lookup("12345")
	if err != nil {
		t.Fatal(err)
	}
	assertEqualDigests(t, dgst, digests[0])
}

func TestAddDuplication(t *testing.T) {
	digests := []Digest{
		"sha256:1234111111111111111111111111111111111111111111111111111111111111",
		"sha256:1234511111111111111111111111111111111111111111111111111111111111",
		"sha256:1234611111111111111111111111111111111111111111111111111111111111",
		"sha256:5432111111111111111111111111111111111111111111111111111111111111",
		"sha256:6543111111111111111111111111111111111111111111111111111111111111",
		"sha512:65431111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
		"sha512:65421111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
		"sha512:65321111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111",
	}

	dset := NewSet()
	for i := range digests {
		if err := dset.Add(digests[i]); err != nil {
			t.Fatal(err)
		}
	}

	if len(dset.entries) != 8 {
		t.Fatal("Invalid dset size")
	}

	if err := dset.Add(Digest("sha256:1234511111111111111111111111111111111111111111111111111111111111")); err != nil {
		t.Fatal(err)
	}

	if len(dset.entries) != 8 {
		t.Fatal("Duplicate digest insert allowed")
	}

	if err := dset.Add(Digest("sha384:123451111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")); err != nil {
		t.Fatal(err)
	}

	if len(dset.entries) != 9 {
		t.Fatal("Insert with different algorithm not allowed")
	}
}

func TestRemove(t *testing.T) {
	digests, err := createDigests(10)
	if err != nil {
		t.Fatal(err)
	}

	dset := NewSet()
	for i := range digests {
		if err := dset.Add(digests[i]); err != nil {
			t.Fatal(err)
		}
	}

	dgst, err := dset.Lookup(digests[0].String())
	if err != nil {
		t.Fatal(err)
	}
	if dgst != digests[0] {
		t.Fatalf("Unexpected digest value:\n\tExpected: %s\n\tActual: %s", digests[0], dgst)
	}

	if err := dset.Remove(digests[0]); err != nil {
		t.Fatal(err)
	}

	if _, err := dset.Lookup(digests[0].String()); err != ErrDigestNotFound {
		t.Fatalf("Expected error %v when looking up removed digest, got %v", ErrDigestNotFound, err)
	}
}

func TestAll(t *testing.T) {
	digests, err := createDigests(100)
	if err != nil {
		t.Fatal(err)
	}

	dset := NewSet()
	for i := range digests {
		if err := dset.Add(digests[i]); err != nil {
			t.Fatal(err)
		}
	}

	all := map[Digest]struct{}{}
	for _, dgst := range dset.All() {
		all[dgst] = struct{}{}
	}

	if len(all) != len(digests) {
		t.Fatalf("Unexpected number of unique digests found:\n\tExpected: %d\n\tActual: %d", len(digests), len(all))
	}

	for i, dgst := range digests {
		if _, ok := all[dgst]; !ok {
			t.Fatalf("Missing element at position %d: %s", i, dgst)
		}
	}

}

func assertEqualShort(t *testing.T, actual, expected string) {
	if actual != expected {
		t.Fatalf("Unexpected short value:\n\tExpected: %s\n\tActual: %s", expected, actual)
	}
}

func TestShortCodeTable(t *testing.T) {
	digests := []Digest{
		"sha256:1234111111111111111111111111111111111111111111111111111111111111",
		"sha256:1234511111111111111111111111111111111111111111111111111111111111",
		"sha256:1234611111111111111111111111111111111111111111111111111111111111",
		"sha256:5432111111111111111111111111111111111111111111111111111111111111",
		"sha256:6543111111111111111111111111111111111111111111111111111111111111",
		"sha256:6432111111111111111111111111111111111111111111111111111111111111",
		"sha256:6542111111111111111111111111111111111111111111111111111111111111",
		"sha256:6532111111111111111111111111111111111111111111111111111111111111",
	}

	dset := NewSet()
	for i := range digests {
		if err := dset.Add(digests[i]); err != nil {
			t.Fatal(err)
		}
	}

	dump := ShortCodeTable(dset, 2)

	if len(dump) < len(digests) {
		t.Fatalf("Error unexpected size: %d, expecting %d", len(dump), len(digests))
	}
	assertEqualShort(t, dump[digests[0]], "12341")
	assertEqualShort(t, dump[digests[1]], "12345")
	assertEqualShort(t, dump[digests[2]], "12346")
	assertEqualShort(t, dump[digests[3]], "54")
	assertEqualShort(t, dump[digests[4]], "6543")
	assertEqualShort(t, dump[digests[5]], "64")
	assertEqualShort(t, dump[digests[6]], "6542")
	assertEqualShort(t, dump[digests[7]], "653")
}

func createDigests(count int) ([]Digest, error) {
	r := rand.New(rand.NewSource(25823))
	digests := make([]Digest, count)
	for i := range digests {
		h := sha256.New()
		if err := binary.Write(h, binary.BigEndian, r.Int63()); err != nil {
			return nil, err
		}
		digests[i] = NewDigest("sha256", h)
	}
	return digests, nil
}

func benchAddNTable(b *testing.B, n int) {
	digests, err := createDigests(n)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dset := &Set{entries: digestEntries(make([]*digestEntry, 0, n))}
		for j := range digests {
			if err = dset.Add(digests[j]); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func benchLookupNTable(b *testing.B, n int, shortLen int) {
	digests, err := createDigests(n)
	if err != nil {
		b.Fatal(err)
	}
	dset := &Set{entries: digestEntries(make([]*digestEntry, 0, n))}
	for i := range digests {
		if err := dset.Add(digests[i]); err != nil {
			b.Fatal(err)
		}
	}
	shorts := make([]string, 0, n)
	for _, short := range ShortCodeTable(dset, shortLen) {
		shorts = append(shorts, short)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err = dset.Lookup(shorts[i%n]); err != nil {
			b.Fatal(err)
		}
	}
}

func benchRemoveNTable(b *testing.B, n int) {
	digests, err := createDigests(n)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dset := &Set{entries: digestEntries(make([]*digestEntry, 0, n))}
		b.StopTimer()
		for j := range digests {
			if err = dset.Add(digests[j]); err != nil {
				b.Fatal(err)
			}
		}
		b.StartTimer()
		for j := range digests {
			if err = dset.Remove(digests[j]); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func benchShortCodeNTable(b *testing.B, n int, shortLen int) {
	digests, err := createDigests(n)
	if err != nil {
		b.Fatal(err)
	}
	dset := &Set{entries: digestEntries(make([]*digestEntry, 0, n))}
	for i := range digests {
		if err := dset.Add(digests[i]); err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ShortCodeTable(dset, shortLen)
	}
}

func BenchmarkAdd10(b *testing.B) {
	benchAddNTable(b, 10)
}

func BenchmarkAdd100(b *testing.B) {
	benchAddNTable(b, 100)
}

func BenchmarkAdd1000(b *testing.B) {
	benchAddNTable(b, 1000)
}

func BenchmarkRemove10(b *testing.B) {
	benchRemoveNTable(b, 10)
}

func BenchmarkRemove100(b *testing.B) {
	benchRemoveNTable(b, 100)
}

func BenchmarkRemove1000(b *testing.B) {
	benchRemoveNTable(b, 1000)
}

func BenchmarkLookup10(b *testing.B) {
	benchLookupNTable(b, 10, 12)
}

func BenchmarkLookup100(b *testing.B) {
	benchLookupNTable(b, 100, 12)
}

func BenchmarkLookup1000(b *testing.B) {
	benchLookupNTable(b, 1000, 12)
}

func BenchmarkShortCode10(b *testing.B) {
	benchShortCodeNTable(b, 10, 12)
}
func BenchmarkShortCode100(b *testing.B) {
	benchShortCodeNTable(b, 100, 12)
}
func BenchmarkShortCode1000(b *testing.B) {
	benchShortCodeNTable(b, 1000, 12)
}
