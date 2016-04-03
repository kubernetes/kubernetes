package graph

import (
	"archive/tar"
	"bytes"
	"io"
	"os"
	"path"
	"testing"

	"github.com/docker/docker/daemon/events"
	"github.com/docker/docker/daemon/graphdriver"
	_ "github.com/docker/docker/daemon/graphdriver/vfs" // import the vfs driver so it is used in the tests
	"github.com/docker/docker/image"
	"github.com/docker/docker/trust"
	"github.com/docker/docker/utils"
)

const (
	testOfficialImageName    = "myapp"
	testOfficialImageID      = "1a2d3c4d4e5fa2d2a21acea242a5e2345d3aefc3e7dfa2a2a2a21a2a2ad2d234"
	testOfficialImageIDShort = "1a2d3c4d4e5f"
	testPrivateImageName     = "127.0.0.1:8000/privateapp"
	testPrivateImageID       = "5bc255f8699e4ee89ac4469266c3d11515da88fdcbde45d7b069b636ff4efd81"
	testPrivateImageIDShort  = "5bc255f8699e"
	testPrivateImageDigest   = "sha256:bc8813ea7b3603864987522f02a76101c17ad122e1c46d790efc0fca78ca7bfb"
	testPrivateImageTag      = "sometag"
)

func fakeTar() (io.Reader, error) {
	uid := os.Getuid()
	gid := os.Getgid()

	content := []byte("Hello world!\n")
	buf := new(bytes.Buffer)
	tw := tar.NewWriter(buf)
	for _, name := range []string{"/etc/postgres/postgres.conf", "/etc/passwd", "/var/log/postgres/postgres.conf"} {
		hdr := new(tar.Header)

		// Leaving these fields blank requires root privileges
		hdr.Uid = uid
		hdr.Gid = gid

		hdr.Size = int64(len(content))
		hdr.Name = name
		if err := tw.WriteHeader(hdr); err != nil {
			return nil, err
		}
		tw.Write([]byte(content))
	}
	tw.Close()
	return buf, nil
}

func mkTestTagStore(root string, t *testing.T) *TagStore {
	driver, err := graphdriver.New(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := NewGraph(root, driver)
	if err != nil {
		t.Fatal(err)
	}

	trust, err := trust.NewTrustStore(root + "/trust")
	if err != nil {
		t.Fatal(err)
	}

	tagCfg := &TagStoreConfig{
		Graph:  graph,
		Events: events.New(),
		Trust:  trust,
	}
	store, err := NewTagStore(path.Join(root, "tags"), tagCfg)
	if err != nil {
		t.Fatal(err)
	}
	officialArchive, err := fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	img := &image.Image{ID: testOfficialImageID}
	if err := graph.Register(img, officialArchive); err != nil {
		t.Fatal(err)
	}
	if err := store.Tag(testOfficialImageName, "", testOfficialImageID, false); err != nil {
		t.Fatal(err)
	}
	privateArchive, err := fakeTar()
	if err != nil {
		t.Fatal(err)
	}
	img = &image.Image{ID: testPrivateImageID}
	if err := graph.Register(img, privateArchive); err != nil {
		t.Fatal(err)
	}
	if err := store.Tag(testPrivateImageName, "", testPrivateImageID, false); err != nil {
		t.Fatal(err)
	}
	if err := store.SetDigest(testPrivateImageName, testPrivateImageDigest, testPrivateImageID); err != nil {
		t.Fatal(err)
	}
	return store
}

func TestLookupImage(t *testing.T) {
	tmp, err := utils.TestDirectory("")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	store := mkTestTagStore(tmp, t)
	defer store.graph.driver.Cleanup()

	officialLookups := []string{
		testOfficialImageID,
		testOfficialImageIDShort,
		testOfficialImageName + ":" + testOfficialImageID,
		testOfficialImageName + ":" + testOfficialImageIDShort,
		testOfficialImageName,
		testOfficialImageName + ":" + DEFAULTTAG,
		"docker.io/" + testOfficialImageName,
		"docker.io/" + testOfficialImageName + ":" + DEFAULTTAG,
		"index.docker.io/" + testOfficialImageName,
		"index.docker.io/" + testOfficialImageName + ":" + DEFAULTTAG,
		"library/" + testOfficialImageName,
		"library/" + testOfficialImageName + ":" + DEFAULTTAG,
		"docker.io/library/" + testOfficialImageName,
		"docker.io/library/" + testOfficialImageName + ":" + DEFAULTTAG,
		"index.docker.io/library/" + testOfficialImageName,
		"index.docker.io/library/" + testOfficialImageName + ":" + DEFAULTTAG,
	}

	privateLookups := []string{
		testPrivateImageID,
		testPrivateImageIDShort,
		testPrivateImageName + ":" + testPrivateImageID,
		testPrivateImageName + ":" + testPrivateImageIDShort,
		testPrivateImageName,
		testPrivateImageName + ":" + DEFAULTTAG,
	}

	invalidLookups := []string{
		testOfficialImageName + ":" + "fail",
		"fail:fail",
	}

	digestLookups := []string{
		testPrivateImageName + "@" + testPrivateImageDigest,
	}

	for _, name := range officialLookups {
		if img, err := store.LookupImage(name); err != nil {
			t.Errorf("Error looking up %s: %s", name, err)
		} else if img == nil {
			t.Errorf("Expected 1 image, none found: %s", name)
		} else if img.ID != testOfficialImageID {
			t.Errorf("Expected ID '%s' found '%s'", testOfficialImageID, img.ID)
		}
	}

	for _, name := range privateLookups {
		if img, err := store.LookupImage(name); err != nil {
			t.Errorf("Error looking up %s: %s", name, err)
		} else if img == nil {
			t.Errorf("Expected 1 image, none found: %s", name)
		} else if img.ID != testPrivateImageID {
			t.Errorf("Expected ID '%s' found '%s'", testPrivateImageID, img.ID)
		}
	}

	for _, name := range invalidLookups {
		if img, err := store.LookupImage(name); err == nil {
			t.Errorf("Expected error, none found: %s", name)
		} else if img != nil {
			t.Errorf("Expected 0 image, 1 found: %s", name)
		}
	}

	for _, name := range digestLookups {
		if img, err := store.LookupImage(name); err != nil {
			t.Errorf("Error looking up %s: %s", name, err)
		} else if img == nil {
			t.Errorf("Expected 1 image, none found: %s", name)
		} else if img.ID != testPrivateImageID {
			t.Errorf("Expected ID '%s' found '%s'", testPrivateImageID, img.ID)
		}
	}
}

func TestValidateDigest(t *testing.T) {
	tests := []struct {
		input       string
		expectError bool
	}{
		{"", true},
		{"latest", true},
		{"sha256:b", false},
		{"tarsum+v1+sha256:bY852-_.+=", false},
		{"#$%#$^:$%^#$%", true},
	}

	for i, test := range tests {
		err := validateDigest(test.input)
		gotError := err != nil
		if e, a := test.expectError, gotError; e != a {
			t.Errorf("%d: with input %s, expected error=%t, got %t: %s", i, test.input, test.expectError, gotError, err)
		}
	}
}
