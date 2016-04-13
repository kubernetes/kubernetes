package zfs_test

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/mistifyio/go-zfs"
)

func sleep(delay int) {
	time.Sleep(time.Duration(delay) * time.Second)
}

func pow2(x int) int64 {
	return int64(math.Pow(2, float64(x)))
}

//https://github.com/benbjohnson/testing
// assert fails the test if the condition is false.
func assert(tb testing.TB, condition bool, msg string, v ...interface{}) {
	if !condition {
		_, file, line, _ := runtime.Caller(1)
		fmt.Printf("\033[31m%s:%d: "+msg+"\033[39m\n\n", append([]interface{}{filepath.Base(file), line}, v...)...)
		tb.FailNow()
	}
}

// ok fails the test if an err is not nil.
func ok(tb testing.TB, err error) {
	if err != nil {
		_, file, line, _ := runtime.Caller(1)
		fmt.Printf("\033[31m%s:%d: unexpected error: %s\033[39m\n\n", filepath.Base(file), line, err.Error())
		tb.FailNow()
	}
}

// equals fails the test if exp is not equal to act.
func equals(tb testing.TB, exp, act interface{}) {
	if !reflect.DeepEqual(exp, act) {
		_, file, line, _ := runtime.Caller(1)
		fmt.Printf("\033[31m%s:%d:\n\n\texp: %#v\n\n\tgot: %#v\033[39m\n\n", filepath.Base(file), line, exp, act)
		tb.FailNow()
	}
}

func zpoolTest(t *testing.T, fn func()) {
	tempfiles := make([]string, 3)
	for i := range tempfiles {
		f, _ := ioutil.TempFile("/tmp/", "zfs-")
		defer f.Close()
		err := f.Truncate(pow2(30))
		ok(t, err)
		tempfiles[i] = f.Name()
		defer os.Remove(f.Name())
	}

	pool, err := zfs.CreateZpool("test", nil, tempfiles...)
	ok(t, err)
	defer pool.Destroy()
	ok(t, err)
	fn()

}

func TestDatasets(t *testing.T) {
	zpoolTest(t, func() {
		_, err := zfs.Datasets("")
		ok(t, err)

		ds, err := zfs.GetDataset("test")
		ok(t, err)
		equals(t, zfs.DatasetFilesystem, ds.Type)
		equals(t, "", ds.Origin)
		assert(t, ds.Logicalused > 0, "Logicalused is not greater than 0")
	})
}

func TestSnapshots(t *testing.T) {

	zpoolTest(t, func() {
		snapshots, err := zfs.Snapshots("")
		ok(t, err)

		for _, snapshot := range snapshots {
			equals(t, zfs.DatasetSnapshot, snapshot.Type)
		}
	})
}

func TestFilesystems(t *testing.T) {
	zpoolTest(t, func() {
		f, err := zfs.CreateFilesystem("test/filesystem-test", nil)
		ok(t, err)

		filesystems, err := zfs.Filesystems("")
		ok(t, err)

		for _, filesystem := range filesystems {
			equals(t, zfs.DatasetFilesystem, filesystem.Type)
		}

		ok(t, f.Destroy(zfs.DestroyDefault))
	})
}

func TestCreateFilesystemWithProperties(t *testing.T) {
	zpoolTest(t, func() {
		props := map[string]string{
			"compression": "lz4",
		}

		f, err := zfs.CreateFilesystem("test/filesystem-test", props)
		ok(t, err)

		equals(t, "lz4", f.Compression)

		filesystems, err := zfs.Filesystems("")
		ok(t, err)

		for _, filesystem := range filesystems {
			equals(t, zfs.DatasetFilesystem, filesystem.Type)
		}

		ok(t, f.Destroy(zfs.DestroyDefault))
	})
}

func TestVolumes(t *testing.T) {
	zpoolTest(t, func() {
		v, err := zfs.CreateVolume("test/volume-test", uint64(pow2(23)), nil)
		ok(t, err)

		// volumes are sometimes "busy" if you try to manipulate them right away
		sleep(1)

		equals(t, zfs.DatasetVolume, v.Type)
		volumes, err := zfs.Volumes("")
		ok(t, err)

		for _, volume := range volumes {
			equals(t, zfs.DatasetVolume, volume.Type)
		}

		ok(t, v.Destroy(zfs.DestroyDefault))
	})
}

func TestSnapshot(t *testing.T) {
	zpoolTest(t, func() {
		f, err := zfs.CreateFilesystem("test/snapshot-test", nil)
		ok(t, err)

		filesystems, err := zfs.Filesystems("")
		ok(t, err)

		for _, filesystem := range filesystems {
			equals(t, zfs.DatasetFilesystem, filesystem.Type)
		}

		s, err := f.Snapshot("test", false)
		ok(t, err)

		equals(t, zfs.DatasetSnapshot, s.Type)

		equals(t, "test/snapshot-test@test", s.Name)

		ok(t, s.Destroy(zfs.DestroyDefault))

		ok(t, f.Destroy(zfs.DestroyDefault))
	})
}

func TestClone(t *testing.T) {
	zpoolTest(t, func() {
		f, err := zfs.CreateFilesystem("test/snapshot-test", nil)
		ok(t, err)

		filesystems, err := zfs.Filesystems("")
		ok(t, err)

		for _, filesystem := range filesystems {
			equals(t, zfs.DatasetFilesystem, filesystem.Type)
		}

		s, err := f.Snapshot("test", false)
		ok(t, err)

		equals(t, zfs.DatasetSnapshot, s.Type)
		equals(t, "test/snapshot-test@test", s.Name)

		c, err := s.Clone("test/clone-test", nil)
		ok(t, err)

		equals(t, zfs.DatasetFilesystem, c.Type)

		ok(t, c.Destroy(zfs.DestroyDefault))

		ok(t, s.Destroy(zfs.DestroyDefault))

		ok(t, f.Destroy(zfs.DestroyDefault))
	})
}

func TestSendSnapshot(t *testing.T) {
	zpoolTest(t, func() {
		f, err := zfs.CreateFilesystem("test/snapshot-test", nil)
		ok(t, err)

		filesystems, err := zfs.Filesystems("")
		ok(t, err)

		for _, filesystem := range filesystems {
			equals(t, zfs.DatasetFilesystem, filesystem.Type)
		}

		s, err := f.Snapshot("test", false)
		ok(t, err)

		file, _ := ioutil.TempFile("/tmp/", "zfs-")
		defer file.Close()
		err = file.Truncate(pow2(30))
		ok(t, err)
		defer os.Remove(file.Name())

		err = s.SendSnapshot(file)
		ok(t, err)

		ok(t, s.Destroy(zfs.DestroyDefault))

		ok(t, f.Destroy(zfs.DestroyDefault))
	})
}

func TestChildren(t *testing.T) {
	zpoolTest(t, func() {
		f, err := zfs.CreateFilesystem("test/snapshot-test", nil)
		ok(t, err)

		s, err := f.Snapshot("test", false)
		ok(t, err)

		equals(t, zfs.DatasetSnapshot, s.Type)
		equals(t, "test/snapshot-test@test", s.Name)

		children, err := f.Children(0)
		ok(t, err)

		equals(t, 1, len(children))
		equals(t, "test/snapshot-test@test", children[0].Name)

		ok(t, s.Destroy(zfs.DestroyDefault))
		ok(t, f.Destroy(zfs.DestroyDefault))
	})
}

func TestListZpool(t *testing.T) {
	zpoolTest(t, func() {
		pools, err := zfs.ListZpools()
		ok(t, err)
		equals(t, "test", pools[0].Name)

	})
}

func TestRollback(t *testing.T) {
	zpoolTest(t, func() {
		f, err := zfs.CreateFilesystem("test/snapshot-test", nil)
		ok(t, err)

		filesystems, err := zfs.Filesystems("")
		ok(t, err)

		for _, filesystem := range filesystems {
			equals(t, zfs.DatasetFilesystem, filesystem.Type)
		}

		s1, err := f.Snapshot("test", false)
		ok(t, err)

		_, err = f.Snapshot("test2", false)
		ok(t, err)

		s3, err := f.Snapshot("test3", false)
		ok(t, err)

		err = s3.Rollback(false)
		ok(t, err)

		err = s1.Rollback(false)
		assert(t, err != nil, "should error when rolling back beyond most recent without destroyMoreRecent = true")

		err = s1.Rollback(true)
		ok(t, err)

		ok(t, s1.Destroy(zfs.DestroyDefault))

		ok(t, f.Destroy(zfs.DestroyDefault))
	})
}

func TestDiff(t *testing.T) {
	zpoolTest(t, func() {
		fs, err := zfs.CreateFilesystem("test/origin", nil)
		ok(t, err)

		linkedFile, err := os.Create(filepath.Join(fs.Mountpoint, "linked"))
		ok(t, err)

		movedFile, err := os.Create(filepath.Join(fs.Mountpoint, "file"))
		ok(t, err)

		snapshot, err := fs.Snapshot("snapshot", false)
		ok(t, err)

		unicodeFile, err := os.Create(filepath.Join(fs.Mountpoint, "i ❤ unicode"))
		ok(t, err)

		err = os.Rename(movedFile.Name(), movedFile.Name()+"-new")
		ok(t, err)

		err = os.Link(linkedFile.Name(), linkedFile.Name()+"_hard")
		ok(t, err)

		inodeChanges, err := fs.Diff(snapshot.Name)
		ok(t, err)
		equals(t, 4, len(inodeChanges))

		equals(t, "/test/origin/", inodeChanges[0].Path)
		equals(t, zfs.Directory, inodeChanges[0].Type)
		equals(t, zfs.Modified, inodeChanges[0].Change)

		equals(t, "/test/origin/linked", inodeChanges[1].Path)
		equals(t, zfs.File, inodeChanges[1].Type)
		equals(t, zfs.Modified, inodeChanges[1].Change)
		equals(t, 1, inodeChanges[1].ReferenceCountChange)

		equals(t, "/test/origin/file", inodeChanges[2].Path)
		equals(t, "/test/origin/file-new", inodeChanges[2].NewPath)
		equals(t, zfs.File, inodeChanges[2].Type)
		equals(t, zfs.Renamed, inodeChanges[2].Change)

		equals(t, "/test/origin/i ❤ unicode", inodeChanges[3].Path)
		equals(t, zfs.File, inodeChanges[3].Type)
		equals(t, zfs.Created, inodeChanges[3].Change)

		ok(t, movedFile.Close())
		ok(t, unicodeFile.Close())
		ok(t, linkedFile.Close())
		ok(t, snapshot.Destroy(zfs.DestroyForceUmount))
		ok(t, fs.Destroy(zfs.DestroyForceUmount))
	})
}
