package chrootarchive

import (
	"bytes"
	"fmt"
	"hash/crc32"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/docker/pkg/system"
)

func init() {
	reexec.Init()
}

var chrootArchiver = NewArchiver(nil)

func TarUntar(src, dst string) error {
	return chrootArchiver.TarUntar(src, dst)
}

func CopyFileWithTar(src, dst string) (err error) {
	return chrootArchiver.CopyFileWithTar(src, dst)
}

func UntarPath(src, dst string) error {
	return chrootArchiver.UntarPath(src, dst)
}

func CopyWithTar(src, dst string) error {
	return chrootArchiver.CopyWithTar(src, dst)
}

func TestChrootTarUntar(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootTarUntar")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	src := filepath.Join(tmpdir, "src")
	if err := system.MkdirAll(src, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(src, "toto"), []byte("hello toto"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(src, "lolo"), []byte("hello lolo"), 0644); err != nil {
		t.Fatal(err)
	}
	stream, err := archive.Tar(src, archive.Uncompressed)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(tmpdir, "src")
	if err := system.MkdirAll(dest, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if err := Untar(stream, dest, &archive.TarOptions{ExcludePatterns: []string{"lolo"}}); err != nil {
		t.Fatal(err)
	}
}

// gh#10426: Verify the fix for having a huge excludes list (like on `docker load` with large # of
// local images)
func TestChrootUntarWithHugeExcludesList(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootUntarHugeExcludes")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	src := filepath.Join(tmpdir, "src")
	if err := system.MkdirAll(src, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(src, "toto"), []byte("hello toto"), 0644); err != nil {
		t.Fatal(err)
	}
	stream, err := archive.Tar(src, archive.Uncompressed)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(tmpdir, "dest")
	if err := system.MkdirAll(dest, 0700, ""); err != nil {
		t.Fatal(err)
	}
	options := &archive.TarOptions{}
	//65534 entries of 64-byte strings ~= 4MB of environment space which should overflow
	//on most systems when passed via environment or command line arguments
	excludes := make([]string, 65534)
	for i := 0; i < 65534; i++ {
		excludes[i] = strings.Repeat(string(i), 64)
	}
	options.ExcludePatterns = excludes
	if err := Untar(stream, dest, options); err != nil {
		t.Fatal(err)
	}
}

func TestChrootUntarEmptyArchive(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootUntarEmptyArchive")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	if err := Untar(nil, tmpdir, nil); err == nil {
		t.Fatal("expected error on empty archive")
	}
}

func prepareSourceDirectory(numberOfFiles int, targetPath string, makeSymLinks bool) (int, error) {
	fileData := []byte("fooo")
	for n := 0; n < numberOfFiles; n++ {
		fileName := fmt.Sprintf("file-%d", n)
		if err := ioutil.WriteFile(filepath.Join(targetPath, fileName), fileData, 0700); err != nil {
			return 0, err
		}
		if makeSymLinks {
			if err := os.Symlink(filepath.Join(targetPath, fileName), filepath.Join(targetPath, fileName+"-link")); err != nil {
				return 0, err
			}
		}
	}
	totalSize := numberOfFiles * len(fileData)
	return totalSize, nil
}

func getHash(filename string) (uint32, error) {
	stream, err := ioutil.ReadFile(filename)
	if err != nil {
		return 0, err
	}
	hash := crc32.NewIEEE()
	hash.Write(stream)
	return hash.Sum32(), nil
}

func compareDirectories(src string, dest string) error {
	changes, err := archive.ChangesDirs(dest, src)
	if err != nil {
		return err
	}
	if len(changes) > 0 {
		return fmt.Errorf("Unexpected differences after untar: %v", changes)
	}
	return nil
}

func compareFiles(src string, dest string) error {
	srcHash, err := getHash(src)
	if err != nil {
		return err
	}
	destHash, err := getHash(dest)
	if err != nil {
		return err
	}
	if srcHash != destHash {
		return fmt.Errorf("%s is different from %s", src, dest)
	}
	return nil
}

func TestChrootTarUntarWithSymlink(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootTarUntarWithSymlink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	src := filepath.Join(tmpdir, "src")
	if err := system.MkdirAll(src, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareSourceDirectory(10, src, false); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(tmpdir, "dest")
	if err := TarUntar(src, dest); err != nil {
		t.Fatal(err)
	}
	if err := compareDirectories(src, dest); err != nil {
		t.Fatal(err)
	}
}

func TestChrootCopyWithTar(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" || runtime.GOOS == "solaris" {
		t.Skip("Failing on Windows and Solaris")
	}
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootCopyWithTar")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	src := filepath.Join(tmpdir, "src")
	if err := system.MkdirAll(src, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareSourceDirectory(10, src, true); err != nil {
		t.Fatal(err)
	}

	// Copy directory
	dest := filepath.Join(tmpdir, "dest")
	if err := CopyWithTar(src, dest); err != nil {
		t.Fatal(err)
	}
	if err := compareDirectories(src, dest); err != nil {
		t.Fatal(err)
	}

	// Copy file
	srcfile := filepath.Join(src, "file-1")
	dest = filepath.Join(tmpdir, "destFile")
	destfile := filepath.Join(dest, "file-1")
	if err := CopyWithTar(srcfile, destfile); err != nil {
		t.Fatal(err)
	}
	if err := compareFiles(srcfile, destfile); err != nil {
		t.Fatal(err)
	}

	// Copy symbolic link
	srcLinkfile := filepath.Join(src, "file-1-link")
	dest = filepath.Join(tmpdir, "destSymlink")
	destLinkfile := filepath.Join(dest, "file-1-link")
	if err := CopyWithTar(srcLinkfile, destLinkfile); err != nil {
		t.Fatal(err)
	}
	if err := compareFiles(srcLinkfile, destLinkfile); err != nil {
		t.Fatal(err)
	}
}

func TestChrootCopyFileWithTar(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootCopyFileWithTar")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	src := filepath.Join(tmpdir, "src")
	if err := system.MkdirAll(src, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareSourceDirectory(10, src, true); err != nil {
		t.Fatal(err)
	}

	// Copy directory
	dest := filepath.Join(tmpdir, "dest")
	if err := CopyFileWithTar(src, dest); err == nil {
		t.Fatal("Expected error on copying directory")
	}

	// Copy file
	srcfile := filepath.Join(src, "file-1")
	dest = filepath.Join(tmpdir, "destFile")
	destfile := filepath.Join(dest, "file-1")
	if err := CopyFileWithTar(srcfile, destfile); err != nil {
		t.Fatal(err)
	}
	if err := compareFiles(srcfile, destfile); err != nil {
		t.Fatal(err)
	}

	// Copy symbolic link
	srcLinkfile := filepath.Join(src, "file-1-link")
	dest = filepath.Join(tmpdir, "destSymlink")
	destLinkfile := filepath.Join(dest, "file-1-link")
	if err := CopyFileWithTar(srcLinkfile, destLinkfile); err != nil {
		t.Fatal(err)
	}
	if err := compareFiles(srcLinkfile, destLinkfile); err != nil {
		t.Fatal(err)
	}
}

func TestChrootUntarPath(t *testing.T) {
	// TODO Windows: Figure out why this is failing
	if runtime.GOOS == "windows" {
		t.Skip("Failing on Windows")
	}
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootUntarPath")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	src := filepath.Join(tmpdir, "src")
	if err := system.MkdirAll(src, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareSourceDirectory(10, src, false); err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(tmpdir, "dest")
	// Untar a directory
	if err := UntarPath(src, dest); err == nil {
		t.Fatal("Expected error on untaring a directory")
	}

	// Untar a tar file
	stream, err := archive.Tar(src, archive.Uncompressed)
	if err != nil {
		t.Fatal(err)
	}
	buf := new(bytes.Buffer)
	buf.ReadFrom(stream)
	tarfile := filepath.Join(tmpdir, "src.tar")
	if err := ioutil.WriteFile(tarfile, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	if err := UntarPath(tarfile, dest); err != nil {
		t.Fatal(err)
	}
	if err := compareDirectories(src, dest); err != nil {
		t.Fatal(err)
	}
}

type slowEmptyTarReader struct {
	size      int
	offset    int
	chunkSize int
}

// Read is a slow reader of an empty tar (like the output of "tar c --files-from /dev/null")
func (s *slowEmptyTarReader) Read(p []byte) (int, error) {
	time.Sleep(100 * time.Millisecond)
	count := s.chunkSize
	if len(p) < s.chunkSize {
		count = len(p)
	}
	for i := 0; i < count; i++ {
		p[i] = 0
	}
	s.offset += count
	if s.offset > s.size {
		return count, io.EOF
	}
	return count, nil
}

func TestChrootUntarEmptyArchiveFromSlowReader(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootUntarEmptyArchiveFromSlowReader")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	dest := filepath.Join(tmpdir, "dest")
	if err := system.MkdirAll(dest, 0700, ""); err != nil {
		t.Fatal(err)
	}
	stream := &slowEmptyTarReader{size: 10240, chunkSize: 1024}
	if err := Untar(stream, dest, nil); err != nil {
		t.Fatal(err)
	}
}

func TestChrootApplyEmptyArchiveFromSlowReader(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootApplyEmptyArchiveFromSlowReader")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	dest := filepath.Join(tmpdir, "dest")
	if err := system.MkdirAll(dest, 0700, ""); err != nil {
		t.Fatal(err)
	}
	stream := &slowEmptyTarReader{size: 10240, chunkSize: 1024}
	if _, err := ApplyLayer(dest, stream); err != nil {
		t.Fatal(err)
	}
}

func TestChrootApplyDotDotFile(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "docker-TestChrootApplyDotDotFile")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	src := filepath.Join(tmpdir, "src")
	if err := system.MkdirAll(src, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(src, "..gitme"), []byte(""), 0644); err != nil {
		t.Fatal(err)
	}
	stream, err := archive.Tar(src, archive.Uncompressed)
	if err != nil {
		t.Fatal(err)
	}
	dest := filepath.Join(tmpdir, "dest")
	if err := system.MkdirAll(dest, 0700, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyLayer(dest, stream); err != nil {
		t.Fatal(err)
	}
}
