package backuptar

import (
	"bytes"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/go-winio/archive/tar"
)

func ensurePresent(t *testing.T, m map[string]string, keys ...string) {
	for _, k := range keys {
		if _, ok := m[k]; !ok {
			t.Error(k, "not present in tar header")
		}
	}
}

func TestRoundTrip(t *testing.T) {
	f, err := ioutil.TempFile("", "tst")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	defer os.Remove(f.Name())

	if _, err = f.Write([]byte("testing 1 2 3\n")); err != nil {
		t.Fatal(err)
	}

	if _, err = f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}

	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}

	bi, err := winio.GetFileBasicInfo(f)
	if err != nil {
		t.Fatal(err)
	}

	br := winio.NewBackupFileReader(f, true)
	defer br.Close()

	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)

	err = WriteTarFileFromBackupStream(tw, br, f.Name(), fi.Size(), bi)
	if err != nil {
		t.Fatal(err)
	}

	tr := tar.NewReader(&buf)
	hdr, err := tr.Next()
	if err != nil {
		t.Fatal(err)
	}

	name, size, bi2, err := FileInfoFromHeader(hdr)
	if err != nil {
		t.Fatal(err)
	}

	if name != filepath.ToSlash(f.Name()) {
		t.Errorf("got name %s, expected %s", name, filepath.ToSlash(f.Name()))
	}

	if size != fi.Size() {
		t.Errorf("got size %d, expected %d", size, fi.Size())
	}

	if !reflect.DeepEqual(*bi, *bi2) {
		t.Errorf("got %#v, expected %#v", *bi, *bi2)
	}

	ensurePresent(t, hdr.Winheaders, "fileattr", "rawsd")
}
