package archive

import (
	"archive/tar"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"runtime"
	"sort"
	"testing"
)

func TestHardLinkOrder(t *testing.T) {
	//TODO Should run for Solaris
	if runtime.GOOS == "solaris" {
		t.Skip("gcp failures on Solaris")
	}
	names := []string{"file1.txt", "file2.txt", "file3.txt"}
	msg := []byte("Hey y'all")

	// Create dir
	src, err := ioutil.TempDir("", "docker-hardlink-test-src-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(src)
	for _, name := range names {
		func() {
			fh, err := os.Create(path.Join(src, name))
			if err != nil {
				t.Fatal(err)
			}
			defer fh.Close()
			if _, err = fh.Write(msg); err != nil {
				t.Fatal(err)
			}
		}()
	}
	// Create dest, with changes that includes hardlinks
	dest, err := ioutil.TempDir("", "docker-hardlink-test-dest-")
	if err != nil {
		t.Fatal(err)
	}
	os.RemoveAll(dest) // we just want the name, at first
	if err := copyDir(src, dest); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dest)
	for _, name := range names {
		for i := 0; i < 5; i++ {
			if err := os.Link(path.Join(dest, name), path.Join(dest, fmt.Sprintf("%s.link%d", name, i))); err != nil {
				t.Fatal(err)
			}
		}
	}

	// get changes
	changes, err := ChangesDirs(dest, src)
	if err != nil {
		t.Fatal(err)
	}

	// sort
	sort.Sort(changesByPath(changes))

	// ExportChanges
	ar, err := ExportChanges(dest, changes, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	hdrs, err := walkHeaders(ar)
	if err != nil {
		t.Fatal(err)
	}

	// reverse sort
	sort.Sort(sort.Reverse(changesByPath(changes)))
	// ExportChanges
	arRev, err := ExportChanges(dest, changes, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	hdrsRev, err := walkHeaders(arRev)
	if err != nil {
		t.Fatal(err)
	}

	// line up the two sets
	sort.Sort(tarHeaders(hdrs))
	sort.Sort(tarHeaders(hdrsRev))

	// compare Size and LinkName
	for i := range hdrs {
		if hdrs[i].Name != hdrsRev[i].Name {
			t.Errorf("headers - expected name %q; but got %q", hdrs[i].Name, hdrsRev[i].Name)
		}
		if hdrs[i].Size != hdrsRev[i].Size {
			t.Errorf("headers - %q expected size %d; but got %d", hdrs[i].Name, hdrs[i].Size, hdrsRev[i].Size)
		}
		if hdrs[i].Typeflag != hdrsRev[i].Typeflag {
			t.Errorf("headers - %q expected type %d; but got %d", hdrs[i].Name, hdrs[i].Typeflag, hdrsRev[i].Typeflag)
		}
		if hdrs[i].Linkname != hdrsRev[i].Linkname {
			t.Errorf("headers - %q expected linkname %q; but got %q", hdrs[i].Name, hdrs[i].Linkname, hdrsRev[i].Linkname)
		}
	}

}

type tarHeaders []tar.Header

func (th tarHeaders) Len() int           { return len(th) }
func (th tarHeaders) Swap(i, j int)      { th[j], th[i] = th[i], th[j] }
func (th tarHeaders) Less(i, j int) bool { return th[i].Name < th[j].Name }

func walkHeaders(r io.Reader) ([]tar.Header, error) {
	t := tar.NewReader(r)
	headers := []tar.Header{}
	for {
		hdr, err := t.Next()
		if err != nil {
			if err == io.EOF {
				break
			}
			return headers, err
		}
		headers = append(headers, *hdr)
	}
	return headers, nil
}
