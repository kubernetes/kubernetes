package afero

import (
	"regexp"
	"testing"
)

func TestFilterReadOnly(t *testing.T) {
	fs := &ReadOnlyFs{source: &MemMapFs{}}
	_, err := fs.Create("/file.txt")
	if err == nil {
		t.Errorf("Did not fail to create file")
	}
	// t.Logf("ERR=%s", err)
}

func TestFilterReadonlyRemoveAndRead(t *testing.T) {
	mfs := &MemMapFs{}
	fh, err := mfs.Create("/file.txt")
	fh.Write([]byte("content here"))
	fh.Close()

	fs := NewReadOnlyFs(mfs)
	err = fs.Remove("/file.txt")
	if err == nil {
		t.Errorf("Did not fail to remove file")
	}

	fh, err = fs.Open("/file.txt")
	if err != nil {
		t.Errorf("Failed to open file: %s", err)
	}

	buf := make([]byte, len("content here"))
	_, err = fh.Read(buf)
	fh.Close()
	if string(buf) != "content here" {
		t.Errorf("Failed to read file: %s", err)
	}

	err = mfs.Remove("/file.txt")
	if err != nil {
		t.Errorf("Failed to remove file")
	}

	fh, err = fs.Open("/file.txt")
	if err == nil {
		fh.Close()
		t.Errorf("File still present")
	}
}

func TestFilterRegexp(t *testing.T) {
	fs := NewRegexpFs(&MemMapFs{}, regexp.MustCompile(`\.txt$`))
	_, err := fs.Create("/file.html")
	if err == nil {

		t.Errorf("Did not fail to create file")
	}
	// t.Logf("ERR=%s", err)
}

func TestFilterRORegexpChain(t *testing.T) {
	rofs := &ReadOnlyFs{source: &MemMapFs{}}
	fs := &RegexpFs{re: regexp.MustCompile(`\.txt$`), source: rofs}
	_, err := fs.Create("/file.txt")
	if err == nil {
		t.Errorf("Did not fail to create file")
	}
	// t.Logf("ERR=%s", err)
}

func TestFilterRegexReadDir(t *testing.T) {
	mfs := &MemMapFs{}
	fs1 := &RegexpFs{re: regexp.MustCompile(`\.txt$`), source: mfs}
	fs := &RegexpFs{re: regexp.MustCompile(`^a`), source: fs1}

	mfs.MkdirAll("/dir/sub", 0777)
	for _, name := range []string{"afile.txt", "afile.html", "bfile.txt"} {
		for _, dir := range []string{"/dir/", "/dir/sub/"} {
			fh, _ := mfs.Create(dir + name)
			fh.Close()
		}
	}

	files, _ := ReadDir(fs, "/dir")
	if len(files) != 2 { // afile.txt, sub
		t.Errorf("Got wrong number of files: %#v", files)
	}

	f, _ := fs.Open("/dir/sub")
	names, _ := f.Readdirnames(-1)
	if len(names) != 1 {
		t.Errorf("Got wrong number of names: %v", names)
	}
}
