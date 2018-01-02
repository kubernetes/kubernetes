package libtrust

import (
	"testing"
)

func compareKeySlices(t *testing.T, sliceA, sliceB []PublicKey) {
	if len(sliceA) != len(sliceB) {
		t.Fatalf("slice size %d, expected %d", len(sliceA), len(sliceB))
	}

	for i, itemA := range sliceA {
		itemB := sliceB[i]
		if itemA != itemB {
			t.Fatalf("slice index %d not equal: %#v != %#v", i, itemA, itemB)
		}
	}
}

func TestFilter(t *testing.T) {
	keys := make([]PublicKey, 0, 8)

	// Create 8 keys and add host entries.
	for i := 0; i < cap(keys); i++ {
		key, err := GenerateECP256PrivateKey()
		if err != nil {
			t.Fatal(err)
		}

		// we use both []interface{} and []string here because jwt uses
		// []interface{} format, while PEM uses []string
		switch {
		case i == 0:
			// Don't add entries for this key, key 0.
			break
		case i%2 == 0:
			// Should catch keys 2, 4, and 6.
			key.AddExtendedField("hosts", []interface{}{"*.even.example.com"})
		case i == 7:
			// Should catch only the last key, and make it match any hostname.
			key.AddExtendedField("hosts", []string{"*"})
		default:
			// should catch keys 1, 3, 5.
			key.AddExtendedField("hosts", []string{"*.example.com"})
		}

		keys = append(keys, key)
	}

	// Should match 2 keys, the empty one, and the one that matches all hosts.
	matchedKeys, err := FilterByHosts(keys, "foo.bar.com", true)
	if err != nil {
		t.Fatal(err)
	}
	expectedMatch := []PublicKey{keys[0], keys[7]}
	compareKeySlices(t, expectedMatch, matchedKeys)

	// Should match 1 key, the one that matches any host.
	matchedKeys, err = FilterByHosts(keys, "foo.bar.com", false)
	if err != nil {
		t.Fatal(err)
	}
	expectedMatch = []PublicKey{keys[7]}
	compareKeySlices(t, expectedMatch, matchedKeys)

	// Should match keys that end in "example.com", and the key that matches anything.
	matchedKeys, err = FilterByHosts(keys, "foo.example.com", false)
	if err != nil {
		t.Fatal(err)
	}
	expectedMatch = []PublicKey{keys[1], keys[3], keys[5], keys[7]}
	compareKeySlices(t, expectedMatch, matchedKeys)

	// Should match all of the keys except the empty key.
	matchedKeys, err = FilterByHosts(keys, "foo.even.example.com", false)
	if err != nil {
		t.Fatal(err)
	}
	expectedMatch = keys[1:]
	compareKeySlices(t, expectedMatch, matchedKeys)
}
