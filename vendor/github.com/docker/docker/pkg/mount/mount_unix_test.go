// +build !windows,!solaris

package mount

import (
	"os"
	"path"
	"testing"
)

func TestMountOptionsParsing(t *testing.T) {
	options := "noatime,ro,size=10k"

	flag, data := parseOptions(options)

	if data != "size=10k" {
		t.Fatalf("Expected size=10 got %s", data)
	}

	expectedFlag := NOATIME | RDONLY

	if flag != expectedFlag {
		t.Fatalf("Expected %d got %d", expectedFlag, flag)
	}
}

func TestMounted(t *testing.T) {
	tmp := path.Join(os.TempDir(), "mount-tests")
	if err := os.MkdirAll(tmp, 0777); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	var (
		sourceDir  = path.Join(tmp, "source")
		targetDir  = path.Join(tmp, "target")
		sourcePath = path.Join(sourceDir, "file.txt")
		targetPath = path.Join(targetDir, "file.txt")
	)

	os.Mkdir(sourceDir, 0777)
	os.Mkdir(targetDir, 0777)

	f, err := os.Create(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("hello")
	f.Close()

	f, err = os.Create(targetPath)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	if err := Mount(sourceDir, targetDir, "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(targetDir); err != nil {
			t.Fatal(err)
		}
	}()

	mounted, err := Mounted(targetDir)
	if err != nil {
		t.Fatal(err)
	}
	if !mounted {
		t.Fatalf("Expected %s to be mounted", targetDir)
	}
	if _, err := os.Stat(targetDir); err != nil {
		t.Fatal(err)
	}
}

func TestMountReadonly(t *testing.T) {
	tmp := path.Join(os.TempDir(), "mount-tests")
	if err := os.MkdirAll(tmp, 0777); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	var (
		sourceDir  = path.Join(tmp, "source")
		targetDir  = path.Join(tmp, "target")
		sourcePath = path.Join(sourceDir, "file.txt")
		targetPath = path.Join(targetDir, "file.txt")
	)

	os.Mkdir(sourceDir, 0777)
	os.Mkdir(targetDir, 0777)

	f, err := os.Create(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("hello")
	f.Close()

	f, err = os.Create(targetPath)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	if err := Mount(sourceDir, targetDir, "none", "bind,ro"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(targetDir); err != nil {
			t.Fatal(err)
		}
	}()

	f, err = os.OpenFile(targetPath, os.O_RDWR, 0777)
	if err == nil {
		t.Fatal("Should not be able to open a ro file as rw")
	}
}

func TestGetMounts(t *testing.T) {
	mounts, err := GetMounts()
	if err != nil {
		t.Fatal(err)
	}

	root := false
	for _, entry := range mounts {
		if entry.Mountpoint == "/" {
			root = true
		}
	}

	if !root {
		t.Fatal("/ should be mounted at least")
	}
}

func TestMergeTmpfsOptions(t *testing.T) {
	options := []string{"noatime", "ro", "size=10k", "defaults", "atime", "defaults", "rw", "rprivate", "size=1024k", "slave"}
	expected := []string{"atime", "rw", "size=1024k", "slave"}
	merged, err := MergeTmpfsOptions(options)
	if err != nil {
		t.Fatal(err)
	}
	if len(expected) != len(merged) {
		t.Fatalf("Expected %s got %s", expected, merged)
	}
	for index := range merged {
		if merged[index] != expected[index] {
			t.Fatalf("Expected %s for the %dth option, got %s", expected, index, merged)
		}
	}

	options = []string{"noatime", "ro", "size=10k", "atime", "rw", "rprivate", "size=1024k", "slave", "size"}
	_, err = MergeTmpfsOptions(options)
	if err == nil {
		t.Fatal("Expected error got nil")
	}
}
