package afero

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"
	"time"
)

var tempDirs []string

func NewTempOsBaseFs(t *testing.T) Fs {
	name, err := TempDir(NewOsFs(), "", "")
	if err != nil {
		t.Error("error creating tempDir", err)
	}

	tempDirs = append(tempDirs, name)

	return NewBasePathFs(NewOsFs(), name)
}

func CleanupTempDirs(t *testing.T) {
	osfs := NewOsFs()
	type ev struct{
		path string
		e error
	}

	errs := []ev{}

	for _, x := range tempDirs {
		err := osfs.RemoveAll(x)
		if err != nil {
			errs = append(errs, ev{path:x,e: err})
		}
	}

	for _, e := range errs {
		fmt.Println("error removing tempDir", e.path, e.e)
	}

	if len(errs) > 0 {
		t.Error("error cleaning up tempDirs")
	}
	tempDirs = []string{}
}

func TestUnionCreateExisting(t *testing.T) {
	base := &MemMapFs{}
	roBase := &ReadOnlyFs{source: base}

	ufs := NewCopyOnWriteFs(roBase, &MemMapFs{})

	base.MkdirAll("/home/test", 0777)
	fh, _ := base.Create("/home/test/file.txt")
	fh.WriteString("This is a test")
	fh.Close()

	fh, err := ufs.OpenFile("/home/test/file.txt", os.O_RDWR, 0666)
	if err != nil {
		t.Errorf("Failed to open file r/w: %s", err)
	}

	_, err = fh.Write([]byte("####"))
	if err != nil {
		t.Errorf("Failed to write file: %s", err)
	}
	fh.Seek(0, 0)
	data, err := ioutil.ReadAll(fh)
	if err != nil {
		t.Errorf("Failed to read file: %s", err)
	}
	if string(data) != "#### is a test" {
		t.Errorf("Got wrong data")
	}
	fh.Close()

	fh, _ = base.Open("/home/test/file.txt")
	data, err = ioutil.ReadAll(fh)
	if string(data) != "This is a test" {
		t.Errorf("Got wrong data in base file")
	}
	fh.Close()

	fh, err = ufs.Create("/home/test/file.txt")
	switch err {
	case nil:
		if fi, _ := fh.Stat(); fi.Size() != 0 {
			t.Errorf("Create did not truncate file")
		}
		fh.Close()
	default:
		t.Errorf("Create failed on existing file")
	}

}

func TestUnionMergeReaddir(t *testing.T) {
	base := &MemMapFs{}
	roBase := &ReadOnlyFs{source: base}

	ufs := &CopyOnWriteFs{base: roBase, layer: &MemMapFs{}}

	base.MkdirAll("/home/test", 0777)
	fh, _ := base.Create("/home/test/file.txt")
	fh.WriteString("This is a test")
	fh.Close()

	fh, _ = ufs.Create("/home/test/file2.txt")
	fh.WriteString("This is a test")
	fh.Close()

	fh, _ = ufs.Open("/home/test")
	files, err := fh.Readdirnames(-1)
	if err != nil {
		t.Errorf("Readdirnames failed")
	}
	if len(files) != 2 {
		t.Errorf("Got wrong number of files: %v", files)
	}
}

func TestExistingDirectoryCollisionReaddir(t *testing.T) {
	base := &MemMapFs{}
	roBase := &ReadOnlyFs{source: base}
	overlay := &MemMapFs{}

	ufs := &CopyOnWriteFs{base: roBase, layer: overlay}

	base.MkdirAll("/home/test", 0777)
	fh, _ := base.Create("/home/test/file.txt")
	fh.WriteString("This is a test")
	fh.Close()

	overlay.MkdirAll("home/test", 0777)
	fh, _ = overlay.Create("/home/test/file2.txt")
	fh.WriteString("This is a test")
	fh.Close()

	fh, _ = ufs.Create("/home/test/file3.txt")
	fh.WriteString("This is a test")
	fh.Close()

	fh, _ = ufs.Open("/home/test")
	files, err := fh.Readdirnames(-1)
	if err != nil {
		t.Errorf("Readdirnames failed")
	}
	if len(files) != 3 {
		t.Errorf("Got wrong number of files in union: %v", files)
	}

	fh, _ = overlay.Open("/home/test")
	files, err = fh.Readdirnames(-1)
	if err != nil {
		t.Errorf("Readdirnames failed")
	}
	if len(files) != 2 {
		t.Errorf("Got wrong number of files in overlay: %v", files)
	}
}

func TestNestedDirBaseReaddir(t *testing.T) {
	base := &MemMapFs{}
	roBase := &ReadOnlyFs{source: base}
	overlay := &MemMapFs{}

	ufs := &CopyOnWriteFs{base: roBase, layer: overlay}

	base.MkdirAll("/home/test/foo/bar", 0777)
	fh, _ := base.Create("/home/test/file.txt")
	fh.WriteString("This is a test")
	fh.Close()

	fh, _ = base.Create("/home/test/foo/file2.txt")
	fh.WriteString("This is a test")
	fh.Close()
	fh, _ = base.Create("/home/test/foo/bar/file3.txt")
	fh.WriteString("This is a test")
	fh.Close()

	overlay.MkdirAll("/", 0777)

	// Opening something only in the base
	fh, _ = ufs.Open("/home/test/foo")
	list, err := fh.Readdir(-1)
	if err != nil {
		t.Errorf("Readdir failed", err)
	}
	if len(list) != 2 {
		for _, x := range list {
			fmt.Println(x.Name())
		}
		t.Errorf("Got wrong number of files in union: %v", len(list))
	}
}

func TestNestedDirOverlayReaddir(t *testing.T) {
	base := &MemMapFs{}
	roBase := &ReadOnlyFs{source: base}
	overlay := &MemMapFs{}

	ufs := &CopyOnWriteFs{base: roBase, layer: overlay}

	base.MkdirAll("/", 0777)
	overlay.MkdirAll("/home/test/foo/bar", 0777)
	fh, _ := overlay.Create("/home/test/file.txt")
	fh.WriteString("This is a test")
	fh.Close()
	fh, _ = overlay.Create("/home/test/foo/file2.txt")
	fh.WriteString("This is a test")
	fh.Close()
	fh, _ = overlay.Create("/home/test/foo/bar/file3.txt")
	fh.WriteString("This is a test")
	fh.Close()

	// Opening nested dir only in the overlay
	fh, _ = ufs.Open("/home/test/foo")
	list, err := fh.Readdir(-1)
	if err != nil {
		t.Errorf("Readdir failed", err)
	}
	if len(list) != 2 {
		for _, x := range list {
			fmt.Println(x.Name())
		}
		t.Errorf("Got wrong number of files in union: %v", len(list))
	}
}

func TestNestedDirOverlayOsFsReaddir(t *testing.T) {
	defer CleanupTempDirs(t)
	base := NewTempOsBaseFs(t)
	roBase := &ReadOnlyFs{source: base}
	overlay := NewTempOsBaseFs(t)

	ufs := &CopyOnWriteFs{base: roBase, layer: overlay}

	base.MkdirAll("/", 0777)
	overlay.MkdirAll("/home/test/foo/bar", 0777)
	fh, _ := overlay.Create("/home/test/file.txt")
	fh.WriteString("This is a test")
	fh.Close()
	fh, _ = overlay.Create("/home/test/foo/file2.txt")
	fh.WriteString("This is a test")
	fh.Close()
	fh, _ = overlay.Create("/home/test/foo/bar/file3.txt")
	fh.WriteString("This is a test")
	fh.Close()

	// Opening nested dir only in the overlay
	fh, _ = ufs.Open("/home/test/foo")
	list, err := fh.Readdir(-1)
	fh.Close()
	if err != nil {
		t.Errorf("Readdir failed", err)
	}
	if len(list) != 2 {
		for _, x := range list {
			fmt.Println(x.Name())
		}
		t.Errorf("Got wrong number of files in union: %v", len(list))
	}
}

func TestCopyOnWriteFsWithOsFs(t *testing.T) {
	defer CleanupTempDirs(t)
	base := NewTempOsBaseFs(t)
	roBase := &ReadOnlyFs{source: base}
	overlay := NewTempOsBaseFs(t)

	ufs := &CopyOnWriteFs{base: roBase, layer: overlay}

	base.MkdirAll("/home/test", 0777)
	fh, _ := base.Create("/home/test/file.txt")
	fh.WriteString("This is a test")
	fh.Close()

	overlay.MkdirAll("home/test", 0777)
	fh, _ = overlay.Create("/home/test/file2.txt")
	fh.WriteString("This is a test")
	fh.Close()

	fh, _ = ufs.Create("/home/test/file3.txt")
	fh.WriteString("This is a test")
	fh.Close()

	fh, _ = ufs.Open("/home/test")
	files, err := fh.Readdirnames(-1)
	fh.Close()
	if err != nil {
		t.Errorf("Readdirnames failed")
	}
	if len(files) != 3 {
		t.Errorf("Got wrong number of files in union: %v", files)
	}

	fh, _ = overlay.Open("/home/test")
	files, err = fh.Readdirnames(-1)
	fh.Close()
	if err != nil {
		t.Errorf("Readdirnames failed")
	}
	if len(files) != 2 {
		t.Errorf("Got wrong number of files in overlay: %v", files)
	}
}

func TestUnionCacheWrite(t *testing.T) {
	base := &MemMapFs{}
	layer := &MemMapFs{}

	ufs := NewCacheOnReadFs(base, layer, 0)

	base.Mkdir("/data", 0777)

	fh, err := ufs.Create("/data/file.txt")
	if err != nil {
		t.Errorf("Failed to create file")
	}
	_, err = fh.Write([]byte("This is a test"))
	if err != nil {
		t.Errorf("Failed to write file")
	}

	fh.Seek(0, os.SEEK_SET)
	buf := make([]byte, 4)
	_, err = fh.Read(buf)
	fh.Write([]byte(" IS A"))
	fh.Close()

	baseData, _ := ReadFile(base, "/data/file.txt")
	layerData, _ := ReadFile(layer, "/data/file.txt")
	if string(baseData) != string(layerData) {
		t.Errorf("Different data: %s <=> %s", baseData, layerData)
	}
}

func TestUnionCacheExpire(t *testing.T) {
	base := &MemMapFs{}
	layer := &MemMapFs{}
	ufs := &CacheOnReadFs{base: base, layer: layer, cacheTime: 1 * time.Second}

	base.Mkdir("/data", 0777)

	fh, err := ufs.Create("/data/file.txt")
	if err != nil {
		t.Errorf("Failed to create file")
	}
	_, err = fh.Write([]byte("This is a test"))
	if err != nil {
		t.Errorf("Failed to write file")
	}
	fh.Close()

	fh, _ = base.Create("/data/file.txt")
	// sleep some time, so we really get a different time.Now() on write...
	time.Sleep(2 * time.Second)
	fh.WriteString("Another test")
	fh.Close()

	data, _ := ReadFile(ufs, "/data/file.txt")
	if string(data) != "Another test" {
		t.Errorf("cache time failed: <%s>", data)
	}
}
