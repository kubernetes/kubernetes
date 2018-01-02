package reference

import (
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/docker/distribution/reference"
	"github.com/opencontainers/go-digest"
)

var (
	saveLoadTestCases = map[string]digest.Digest{
		"registry:5000/foobar:HEAD":                                                        "sha256:470022b8af682154f57a2163d030eb369549549cba00edc69e1b99b46bb924d6",
		"registry:5000/foobar:alternate":                                                   "sha256:ae300ebc4a4f00693702cfb0a5e0b7bc527b353828dc86ad09fb95c8a681b793",
		"registry:5000/foobar:latest":                                                      "sha256:6153498b9ac00968d71b66cca4eac37e990b5f9eb50c26877eb8799c8847451b",
		"registry:5000/foobar:master":                                                      "sha256:6c9917af4c4e05001b346421959d7ea81b6dc9d25718466a37a6add865dfd7fc",
		"jess/hollywood:latest":                                                            "sha256:ae7a5519a0a55a2d4ef20ddcbd5d0ca0888a1f7ab806acc8e2a27baf46f529fe",
		"registry@sha256:367eb40fd0330a7e464777121e39d2f5b3e8e23a1e159342e53ab05c9e4d94e6": "sha256:24126a56805beb9711be5f4590cc2eb55ab8d4a85ebd618eed72bb19fc50631c",
		"busybox:latest": "sha256:91e54dfb11794fad694460162bf0cb0a4fa710cfa3f60979c177d920813e267c",
	}

	marshalledSaveLoadTestCases = []byte(`{"Repositories":{"busybox":{"busybox:latest":"sha256:91e54dfb11794fad694460162bf0cb0a4fa710cfa3f60979c177d920813e267c"},"jess/hollywood":{"jess/hollywood:latest":"sha256:ae7a5519a0a55a2d4ef20ddcbd5d0ca0888a1f7ab806acc8e2a27baf46f529fe"},"registry":{"registry@sha256:367eb40fd0330a7e464777121e39d2f5b3e8e23a1e159342e53ab05c9e4d94e6":"sha256:24126a56805beb9711be5f4590cc2eb55ab8d4a85ebd618eed72bb19fc50631c"},"registry:5000/foobar":{"registry:5000/foobar:HEAD":"sha256:470022b8af682154f57a2163d030eb369549549cba00edc69e1b99b46bb924d6","registry:5000/foobar:alternate":"sha256:ae300ebc4a4f00693702cfb0a5e0b7bc527b353828dc86ad09fb95c8a681b793","registry:5000/foobar:latest":"sha256:6153498b9ac00968d71b66cca4eac37e990b5f9eb50c26877eb8799c8847451b","registry:5000/foobar:master":"sha256:6c9917af4c4e05001b346421959d7ea81b6dc9d25718466a37a6add865dfd7fc"}}}`)
)

func TestLoad(t *testing.T) {
	jsonFile, err := ioutil.TempFile("", "tag-store-test")
	if err != nil {
		t.Fatalf("error creating temp file: %v", err)
	}
	defer os.RemoveAll(jsonFile.Name())

	// Write canned json to the temp file
	_, err = jsonFile.Write(marshalledSaveLoadTestCases)
	if err != nil {
		t.Fatalf("error writing to temp file: %v", err)
	}
	jsonFile.Close()

	store, err := NewReferenceStore(jsonFile.Name(), runtime.GOOS)
	if err != nil {
		t.Fatalf("error creating tag store: %v", err)
	}

	for refStr, expectedID := range saveLoadTestCases {
		ref, err := reference.ParseNormalizedNamed(refStr)
		if err != nil {
			t.Fatalf("failed to parse reference: %v", err)
		}
		id, err := store.Get(ref)
		if err != nil {
			t.Fatalf("could not find reference %s: %v", refStr, err)
		}
		if id != expectedID {
			t.Fatalf("expected %s - got %s", expectedID, id)
		}
	}
}

func TestSave(t *testing.T) {
	jsonFile, err := ioutil.TempFile("", "tag-store-test")
	if err != nil {
		t.Fatalf("error creating temp file: %v", err)
	}
	_, err = jsonFile.Write([]byte(`{}`))
	jsonFile.Close()
	defer os.RemoveAll(jsonFile.Name())

	store, err := NewReferenceStore(jsonFile.Name(), runtime.GOOS)
	if err != nil {
		t.Fatalf("error creating tag store: %v", err)
	}

	for refStr, id := range saveLoadTestCases {
		ref, err := reference.ParseNormalizedNamed(refStr)
		if err != nil {
			t.Fatalf("failed to parse reference: %v", err)
		}
		if canonical, ok := ref.(reference.Canonical); ok {
			err = store.AddDigest(canonical, id, false)
			if err != nil {
				t.Fatalf("could not add digest reference %s: %v", refStr, err)
			}
		} else {
			err = store.AddTag(ref, id, false)
			if err != nil {
				t.Fatalf("could not add reference %s: %v", refStr, err)
			}
		}
	}

	jsonBytes, err := ioutil.ReadFile(jsonFile.Name())
	if err != nil {
		t.Fatalf("could not read json file: %v", err)
	}

	if !bytes.Equal(jsonBytes, marshalledSaveLoadTestCases) {
		t.Fatalf("save output did not match expectations\nexpected:\n%s\ngot:\n%s", marshalledSaveLoadTestCases, jsonBytes)
	}
}

func TestAddDeleteGet(t *testing.T) {
	jsonFile, err := ioutil.TempFile("", "tag-store-test")
	if err != nil {
		t.Fatalf("error creating temp file: %v", err)
	}
	_, err = jsonFile.Write([]byte(`{}`))
	jsonFile.Close()
	defer os.RemoveAll(jsonFile.Name())

	store, err := NewReferenceStore(jsonFile.Name(), runtime.GOOS)
	if err != nil {
		t.Fatalf("error creating tag store: %v", err)
	}

	testImageID1 := digest.Digest("sha256:9655aef5fd742a1b4e1b7b163aa9f1c76c186304bf39102283d80927c916ca9c")
	testImageID2 := digest.Digest("sha256:9655aef5fd742a1b4e1b7b163aa9f1c76c186304bf39102283d80927c916ca9d")
	testImageID3 := digest.Digest("sha256:9655aef5fd742a1b4e1b7b163aa9f1c76c186304bf39102283d80927c916ca9e")

	// Try adding a reference with no tag or digest
	nameOnly, err := reference.ParseNormalizedNamed("username/repo")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	if err = store.AddTag(nameOnly, testImageID1, false); err != nil {
		t.Fatalf("error adding to store: %v", err)
	}

	// Add a few references
	ref1, err := reference.ParseNormalizedNamed("username/repo1:latest")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	if err = store.AddTag(ref1, testImageID1, false); err != nil {
		t.Fatalf("error adding to store: %v", err)
	}

	ref2, err := reference.ParseNormalizedNamed("username/repo1:old")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	if err = store.AddTag(ref2, testImageID2, false); err != nil {
		t.Fatalf("error adding to store: %v", err)
	}

	ref3, err := reference.ParseNormalizedNamed("username/repo1:alias")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	if err = store.AddTag(ref3, testImageID1, false); err != nil {
		t.Fatalf("error adding to store: %v", err)
	}

	ref4, err := reference.ParseNormalizedNamed("username/repo2:latest")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	if err = store.AddTag(ref4, testImageID2, false); err != nil {
		t.Fatalf("error adding to store: %v", err)
	}

	ref5, err := reference.ParseNormalizedNamed("username/repo3@sha256:58153dfb11794fad694460162bf0cb0a4fa710cfa3f60979c177d920813e267c")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	if err = store.AddDigest(ref5.(reference.Canonical), testImageID2, false); err != nil {
		t.Fatalf("error adding to store: %v", err)
	}

	// Attempt to overwrite with force == false
	if err = store.AddTag(ref4, testImageID3, false); err == nil || !strings.HasPrefix(err.Error(), "Conflict:") {
		t.Fatalf("did not get expected error on overwrite attempt - got %v", err)
	}
	// Repeat to overwrite with force == true
	if err = store.AddTag(ref4, testImageID3, true); err != nil {
		t.Fatalf("failed to force tag overwrite: %v", err)
	}

	// Check references so far
	id, err := store.Get(nameOnly)
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if id != testImageID1 {
		t.Fatalf("id mismatch: got %s instead of %s", id.String(), testImageID1.String())
	}

	id, err = store.Get(ref1)
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if id != testImageID1 {
		t.Fatalf("id mismatch: got %s instead of %s", id.String(), testImageID1.String())
	}

	id, err = store.Get(ref2)
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if id != testImageID2 {
		t.Fatalf("id mismatch: got %s instead of %s", id.String(), testImageID2.String())
	}

	id, err = store.Get(ref3)
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if id != testImageID1 {
		t.Fatalf("id mismatch: got %s instead of %s", id.String(), testImageID1.String())
	}

	id, err = store.Get(ref4)
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if id != testImageID3 {
		t.Fatalf("id mismatch: got %s instead of %s", id.String(), testImageID3.String())
	}

	id, err = store.Get(ref5)
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if id != testImageID2 {
		t.Fatalf("id mismatch: got %s instead of %s", id.String(), testImageID3.String())
	}

	// Get should return ErrDoesNotExist for a nonexistent repo
	nonExistRepo, err := reference.ParseNormalizedNamed("username/nonexistrepo:latest")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	if _, err = store.Get(nonExistRepo); err != ErrDoesNotExist {
		t.Fatal("Expected ErrDoesNotExist from Get")
	}

	// Get should return ErrDoesNotExist for a nonexistent tag
	nonExistTag, err := reference.ParseNormalizedNamed("username/repo1:nonexist")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	if _, err = store.Get(nonExistTag); err != ErrDoesNotExist {
		t.Fatal("Expected ErrDoesNotExist from Get")
	}

	// Check References
	refs := store.References(testImageID1)
	if len(refs) != 3 {
		t.Fatal("unexpected number of references")
	}
	// Looking for the references in this order verifies that they are
	// returned lexically sorted.
	if refs[0].String() != ref3.String() {
		t.Fatalf("unexpected reference: %v", refs[0].String())
	}
	if refs[1].String() != ref1.String() {
		t.Fatalf("unexpected reference: %v", refs[1].String())
	}
	if refs[2].String() != nameOnly.String()+":latest" {
		t.Fatalf("unexpected reference: %v", refs[2].String())
	}

	// Check ReferencesByName
	repoName, err := reference.ParseNormalizedNamed("username/repo1")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	associations := store.ReferencesByName(repoName)
	if len(associations) != 3 {
		t.Fatal("unexpected number of associations")
	}
	// Looking for the associations in this order verifies that they are
	// returned lexically sorted.
	if associations[0].Ref.String() != ref3.String() {
		t.Fatalf("unexpected reference: %v", associations[0].Ref.String())
	}
	if associations[0].ID != testImageID1 {
		t.Fatalf("unexpected reference: %v", associations[0].Ref.String())
	}
	if associations[1].Ref.String() != ref1.String() {
		t.Fatalf("unexpected reference: %v", associations[1].Ref.String())
	}
	if associations[1].ID != testImageID1 {
		t.Fatalf("unexpected reference: %v", associations[1].Ref.String())
	}
	if associations[2].Ref.String() != ref2.String() {
		t.Fatalf("unexpected reference: %v", associations[2].Ref.String())
	}
	if associations[2].ID != testImageID2 {
		t.Fatalf("unexpected reference: %v", associations[2].Ref.String())
	}

	// Delete should return ErrDoesNotExist for a nonexistent repo
	if _, err = store.Delete(nonExistRepo); err != ErrDoesNotExist {
		t.Fatal("Expected ErrDoesNotExist from Delete")
	}

	// Delete should return ErrDoesNotExist for a nonexistent tag
	if _, err = store.Delete(nonExistTag); err != ErrDoesNotExist {
		t.Fatal("Expected ErrDoesNotExist from Delete")
	}

	// Delete a few references
	if deleted, err := store.Delete(ref1); err != nil || deleted != true {
		t.Fatal("Delete failed")
	}
	if _, err := store.Get(ref1); err != ErrDoesNotExist {
		t.Fatal("Expected ErrDoesNotExist from Get")
	}
	if deleted, err := store.Delete(ref5); err != nil || deleted != true {
		t.Fatal("Delete failed")
	}
	if _, err := store.Get(ref5); err != ErrDoesNotExist {
		t.Fatal("Expected ErrDoesNotExist from Get")
	}
	if deleted, err := store.Delete(nameOnly); err != nil || deleted != true {
		t.Fatal("Delete failed")
	}
	if _, err := store.Get(nameOnly); err != ErrDoesNotExist {
		t.Fatal("Expected ErrDoesNotExist from Get")
	}
}

func TestInvalidTags(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "tag-store-test")
	defer os.RemoveAll(tmpDir)

	store, err := NewReferenceStore(filepath.Join(tmpDir, "repositories.json"), runtime.GOOS)
	if err != nil {
		t.Fatalf("error creating tag store: %v", err)
	}
	id := digest.Digest("sha256:470022b8af682154f57a2163d030eb369549549cba00edc69e1b99b46bb924d6")

	// sha256 as repo name
	ref, err := reference.ParseNormalizedNamed("sha256:abc")
	if err != nil {
		t.Fatal(err)
	}
	err = store.AddTag(ref, id, true)
	if err == nil {
		t.Fatalf("expected setting tag %q to fail", ref)
	}

	// setting digest as a tag
	ref, err = reference.ParseNormalizedNamed("registry@sha256:367eb40fd0330a7e464777121e39d2f5b3e8e23a1e159342e53ab05c9e4d94e6")
	if err != nil {
		t.Fatal(err)
	}
	err = store.AddTag(ref, id, true)
	if err == nil {
		t.Fatalf("expected setting tag %q to fail", ref)
	}

}
