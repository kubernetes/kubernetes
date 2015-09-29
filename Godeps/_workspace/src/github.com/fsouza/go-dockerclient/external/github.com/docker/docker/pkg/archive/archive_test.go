package archive

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/system"
)

func TestIsArchiveNilHeader(t *testing.T) {
	out := IsArchive(nil)
	if out {
		t.Fatalf("isArchive should return false as nil is not a valid archive header")
	}
}

func TestIsArchiveInvalidHeader(t *testing.T) {
	header := []byte{0x00, 0x01, 0x02}
	out := IsArchive(header)
	if out {
		t.Fatalf("isArchive should return false as %s is not a valid archive header", header)
	}
}

func TestIsArchiveBzip2(t *testing.T) {
	header := []byte{0x42, 0x5A, 0x68}
	out := IsArchive(header)
	if !out {
		t.Fatalf("isArchive should return true as %s is a bz2 header", header)
	}
}

func TestIsArchive7zip(t *testing.T) {
	header := []byte{0x50, 0x4b, 0x03, 0x04}
	out := IsArchive(header)
	if out {
		t.Fatalf("isArchive should return false as %s is a 7z header and it is not supported", header)
	}
}

func TestDecompressStreamGzip(t *testing.T) {
	cmd := exec.Command("/bin/sh", "-c", "touch /tmp/archive && gzip -f /tmp/archive")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Fail to create an archive file for test : %s.", output)
	}
	archive, err := os.Open("/tmp/archive.gz")
	_, err = DecompressStream(archive)
	if err != nil {
		t.Fatalf("Failed to decompress a gzip file.")
	}
}

func TestDecompressStreamBzip2(t *testing.T) {
	cmd := exec.Command("/bin/sh", "-c", "touch /tmp/archive && bzip2 -f /tmp/archive")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Fail to create an archive file for test : %s.", output)
	}
	archive, err := os.Open("/tmp/archive.bz2")
	_, err = DecompressStream(archive)
	if err != nil {
		t.Fatalf("Failed to decompress a bzip2 file.")
	}
}

func TestDecompressStreamXz(t *testing.T) {
	cmd := exec.Command("/bin/sh", "-c", "touch /tmp/archive && xz -f /tmp/archive")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Fail to create an archive file for test : %s.", output)
	}
	archive, err := os.Open("/tmp/archive.xz")
	_, err = DecompressStream(archive)
	if err != nil {
		t.Fatalf("Failed to decompress a xz file.")
	}
}

func TestCompressStreamXzUnsuported(t *testing.T) {
	dest, err := os.Create("/tmp/dest")
	if err != nil {
		t.Fatalf("Fail to create the destination file")
	}
	_, err = CompressStream(dest, Xz)
	if err == nil {
		t.Fatalf("Should fail as xz is unsupported for compression format.")
	}
}

func TestCompressStreamBzip2Unsupported(t *testing.T) {
	dest, err := os.Create("/tmp/dest")
	if err != nil {
		t.Fatalf("Fail to create the destination file")
	}
	_, err = CompressStream(dest, Xz)
	if err == nil {
		t.Fatalf("Should fail as xz is unsupported for compression format.")
	}
}

func TestCompressStreamInvalid(t *testing.T) {
	dest, err := os.Create("/tmp/dest")
	if err != nil {
		t.Fatalf("Fail to create the destination file")
	}
	_, err = CompressStream(dest, -1)
	if err == nil {
		t.Fatalf("Should fail as xz is unsupported for compression format.")
	}
}

func TestExtensionInvalid(t *testing.T) {
	compression := Compression(-1)
	output := compression.Extension()
	if output != "" {
		t.Fatalf("The extension of an invalid compression should be an empty string.")
	}
}

func TestExtensionUncompressed(t *testing.T) {
	compression := Uncompressed
	output := compression.Extension()
	if output != "tar" {
		t.Fatalf("The extension of a uncompressed archive should be 'tar'.")
	}
}
func TestExtensionBzip2(t *testing.T) {
	compression := Bzip2
	output := compression.Extension()
	if output != "tar.bz2" {
		t.Fatalf("The extension of a bzip2 archive should be 'tar.bz2'")
	}
}
func TestExtensionGzip(t *testing.T) {
	compression := Gzip
	output := compression.Extension()
	if output != "tar.gz" {
		t.Fatalf("The extension of a bzip2 archive should be 'tar.gz'")
	}
}
func TestExtensionXz(t *testing.T) {
	compression := Xz
	output := compression.Extension()
	if output != "tar.xz" {
		t.Fatalf("The extension of a bzip2 archive should be 'tar.xz'")
	}
}

func TestCmdStreamLargeStderr(t *testing.T) {
	cmd := exec.Command("/bin/sh", "-c", "dd if=/dev/zero bs=1k count=1000 of=/dev/stderr; echo hello")
	out, err := CmdStream(cmd, nil)
	if err != nil {
		t.Fatalf("Failed to start command: %s", err)
	}
	errCh := make(chan error)
	go func() {
		_, err := io.Copy(ioutil.Discard, out)
		errCh <- err
	}()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("Command should not have failed (err=%.100s...)", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("Command did not complete in 5 seconds; probable deadlock")
	}
}

func TestCmdStreamBad(t *testing.T) {
	badCmd := exec.Command("/bin/sh", "-c", "echo hello; echo >&2 error couldn\\'t reverse the phase pulser; exit 1")
	out, err := CmdStream(badCmd, nil)
	if err != nil {
		t.Fatalf("Failed to start command: %s", err)
	}
	if output, err := ioutil.ReadAll(out); err == nil {
		t.Fatalf("Command should have failed")
	} else if err.Error() != "exit status 1: error couldn't reverse the phase pulser\n" {
		t.Fatalf("Wrong error value (%s)", err)
	} else if s := string(output); s != "hello\n" {
		t.Fatalf("Command output should be '%s', not '%s'", "hello\\n", output)
	}
}

func TestCmdStreamGood(t *testing.T) {
	cmd := exec.Command("/bin/sh", "-c", "echo hello; exit 0")
	out, err := CmdStream(cmd, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output, err := ioutil.ReadAll(out); err != nil {
		t.Fatalf("Command should not have failed (err=%s)", err)
	} else if s := string(output); s != "hello\n" {
		t.Fatalf("Command output should be '%s', not '%s'", "hello\\n", output)
	}
}

func TestUntarPathWithInvalidDest(t *testing.T) {
	tempFolder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempFolder)
	invalidDestFolder := path.Join(tempFolder, "invalidDest")
	// Create a src file
	srcFile := path.Join(tempFolder, "src")
	_, err = os.Create(srcFile)
	if err != nil {
		t.Fatalf("Fail to create the source file")
	}
	err = UntarPath(srcFile, invalidDestFolder)
	if err == nil {
		t.Fatalf("UntarPath with invalid destination path should throw an error.")
	}
}

func TestUntarPathWithInvalidSrc(t *testing.T) {
	dest, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatalf("Fail to create the destination file")
	}
	defer os.RemoveAll(dest)
	err = UntarPath("/invalid/path", dest)
	if err == nil {
		t.Fatalf("UntarPath with invalid src path should throw an error.")
	}
}

func TestUntarPath(t *testing.T) {
	tmpFolder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpFolder)
	srcFile := path.Join(tmpFolder, "src")
	tarFile := path.Join(tmpFolder, "src.tar")
	os.Create(path.Join(tmpFolder, "src"))
	cmd := exec.Command("/bin/sh", "-c", "tar cf "+tarFile+" "+srcFile)
	_, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	destFolder := path.Join(tmpFolder, "dest")
	err = os.MkdirAll(destFolder, 0740)
	if err != nil {
		t.Fatalf("Fail to create the destination file")
	}
	err = UntarPath(tarFile, destFolder)
	if err != nil {
		t.Fatalf("UntarPath shouldn't throw an error, %s.", err)
	}
	expectedFile := path.Join(destFolder, srcFile)
	_, err = os.Stat(expectedFile)
	if err != nil {
		t.Fatalf("Destination folder should contain the source file but did not.")
	}
}

// Do the same test as above but with the destination as file, it should fail
func TestUntarPathWithDestinationFile(t *testing.T) {
	tmpFolder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpFolder)
	srcFile := path.Join(tmpFolder, "src")
	tarFile := path.Join(tmpFolder, "src.tar")
	os.Create(path.Join(tmpFolder, "src"))
	cmd := exec.Command("/bin/sh", "-c", "tar cf "+tarFile+" "+srcFile)
	_, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	destFile := path.Join(tmpFolder, "dest")
	_, err = os.Create(destFile)
	if err != nil {
		t.Fatalf("Fail to create the destination file")
	}
	err = UntarPath(tarFile, destFile)
	if err == nil {
		t.Fatalf("UntarPath should throw an error if the destination if a file")
	}
}

// Do the same test as above but with the destination folder already exists
// and the destination file is a directory
// It's working, see https://github.com/docker/docker/issues/10040
func TestUntarPathWithDestinationSrcFileAsFolder(t *testing.T) {
	tmpFolder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpFolder)
	srcFile := path.Join(tmpFolder, "src")
	tarFile := path.Join(tmpFolder, "src.tar")
	os.Create(srcFile)
	cmd := exec.Command("/bin/sh", "-c", "tar cf "+tarFile+" "+srcFile)
	_, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	destFolder := path.Join(tmpFolder, "dest")
	err = os.MkdirAll(destFolder, 0740)
	if err != nil {
		t.Fatalf("Fail to create the destination folder")
	}
	// Let's create a folder that will has the same path as the extracted file (from tar)
	destSrcFileAsFolder := path.Join(destFolder, srcFile)
	err = os.MkdirAll(destSrcFileAsFolder, 0740)
	if err != nil {
		t.Fatal(err)
	}
	err = UntarPath(tarFile, destFolder)
	if err != nil {
		t.Fatalf("UntarPath should throw not throw an error if the extracted file already exists and is a folder")
	}
}

func TestCopyWithTarInvalidSrc(t *testing.T) {
	tempFolder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(nil)
	}
	destFolder := path.Join(tempFolder, "dest")
	invalidSrc := path.Join(tempFolder, "doesnotexists")
	err = os.MkdirAll(destFolder, 0740)
	if err != nil {
		t.Fatal(err)
	}
	err = CopyWithTar(invalidSrc, destFolder)
	if err == nil {
		t.Fatalf("archiver.CopyWithTar with invalid src path should throw an error.")
	}
}

func TestCopyWithTarInexistentDestWillCreateIt(t *testing.T) {
	tempFolder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(nil)
	}
	srcFolder := path.Join(tempFolder, "src")
	inexistentDestFolder := path.Join(tempFolder, "doesnotexists")
	err = os.MkdirAll(srcFolder, 0740)
	if err != nil {
		t.Fatal(err)
	}
	err = CopyWithTar(srcFolder, inexistentDestFolder)
	if err != nil {
		t.Fatalf("CopyWithTar with an inexistent folder shouldn't fail.")
	}
	_, err = os.Stat(inexistentDestFolder)
	if err != nil {
		t.Fatalf("CopyWithTar with an inexistent folder should create it.")
	}
}

// Test CopyWithTar with a file as src
func TestCopyWithTarSrcFile(t *testing.T) {
	folder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(folder)
	dest := path.Join(folder, "dest")
	srcFolder := path.Join(folder, "src")
	src := path.Join(folder, path.Join("src", "src"))
	err = os.MkdirAll(srcFolder, 0740)
	if err != nil {
		t.Fatal(err)
	}
	err = os.MkdirAll(dest, 0740)
	if err != nil {
		t.Fatal(err)
	}
	ioutil.WriteFile(src, []byte("content"), 0777)
	err = CopyWithTar(src, dest)
	if err != nil {
		t.Fatalf("archiver.CopyWithTar shouldn't throw an error, %s.", err)
	}
	_, err = os.Stat(dest)
	// FIXME Check the content
	if err != nil {
		t.Fatalf("Destination file should be the same as the source.")
	}
}

// Test CopyWithTar with a folder as src
func TestCopyWithTarSrcFolder(t *testing.T) {
	folder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(folder)
	dest := path.Join(folder, "dest")
	src := path.Join(folder, path.Join("src", "folder"))
	err = os.MkdirAll(src, 0740)
	if err != nil {
		t.Fatal(err)
	}
	err = os.MkdirAll(dest, 0740)
	if err != nil {
		t.Fatal(err)
	}
	ioutil.WriteFile(path.Join(src, "file"), []byte("content"), 0777)
	err = CopyWithTar(src, dest)
	if err != nil {
		t.Fatalf("archiver.CopyWithTar shouldn't throw an error, %s.", err)
	}
	_, err = os.Stat(dest)
	// FIXME Check the content (the file inside)
	if err != nil {
		t.Fatalf("Destination folder should contain the source file but did not.")
	}
}

func TestCopyFileWithTarInvalidSrc(t *testing.T) {
	tempFolder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempFolder)
	destFolder := path.Join(tempFolder, "dest")
	err = os.MkdirAll(destFolder, 0740)
	if err != nil {
		t.Fatal(err)
	}
	invalidFile := path.Join(tempFolder, "doesnotexists")
	err = CopyFileWithTar(invalidFile, destFolder)
	if err == nil {
		t.Fatalf("archiver.CopyWithTar with invalid src path should throw an error.")
	}
}

func TestCopyFileWithTarInexistentDestWillCreateIt(t *testing.T) {
	tempFolder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(nil)
	}
	defer os.RemoveAll(tempFolder)
	srcFile := path.Join(tempFolder, "src")
	inexistentDestFolder := path.Join(tempFolder, "doesnotexists")
	_, err = os.Create(srcFile)
	if err != nil {
		t.Fatal(err)
	}
	err = CopyFileWithTar(srcFile, inexistentDestFolder)
	if err != nil {
		t.Fatalf("CopyWithTar with an inexistent folder shouldn't fail.")
	}
	_, err = os.Stat(inexistentDestFolder)
	if err != nil {
		t.Fatalf("CopyWithTar with an inexistent folder should create it.")
	}
	// FIXME Test the src file and content
}

func TestCopyFileWithTarSrcFolder(t *testing.T) {
	folder, err := ioutil.TempDir("", "docker-archive-copyfilewithtar-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(folder)
	dest := path.Join(folder, "dest")
	src := path.Join(folder, "srcfolder")
	err = os.MkdirAll(src, 0740)
	if err != nil {
		t.Fatal(err)
	}
	err = os.MkdirAll(dest, 0740)
	if err != nil {
		t.Fatal(err)
	}
	err = CopyFileWithTar(src, dest)
	if err == nil {
		t.Fatalf("CopyFileWithTar should throw an error with a folder.")
	}
}

func TestCopyFileWithTarSrcFile(t *testing.T) {
	folder, err := ioutil.TempDir("", "docker-archive-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(folder)
	dest := path.Join(folder, "dest")
	srcFolder := path.Join(folder, "src")
	src := path.Join(folder, path.Join("src", "src"))
	err = os.MkdirAll(srcFolder, 0740)
	if err != nil {
		t.Fatal(err)
	}
	err = os.MkdirAll(dest, 0740)
	if err != nil {
		t.Fatal(err)
	}
	ioutil.WriteFile(src, []byte("content"), 0777)
	err = CopyWithTar(src, dest+"/")
	if err != nil {
		t.Fatalf("archiver.CopyFileWithTar shouldn't throw an error, %s.", err)
	}
	_, err = os.Stat(dest)
	if err != nil {
		t.Fatalf("Destination folder should contain the source file but did not.")
	}
}

func TestTarFiles(t *testing.T) {
	// try without hardlinks
	if err := checkNoChanges(1000, false); err != nil {
		t.Fatal(err)
	}
	// try with hardlinks
	if err := checkNoChanges(1000, true); err != nil {
		t.Fatal(err)
	}
}

func checkNoChanges(fileNum int, hardlinks bool) error {
	srcDir, err := ioutil.TempDir("", "docker-test-srcDir")
	if err != nil {
		return err
	}
	defer os.RemoveAll(srcDir)

	destDir, err := ioutil.TempDir("", "docker-test-destDir")
	if err != nil {
		return err
	}
	defer os.RemoveAll(destDir)

	_, err = prepareUntarSourceDirectory(fileNum, srcDir, hardlinks)
	if err != nil {
		return err
	}

	err = TarUntar(srcDir, destDir)
	if err != nil {
		return err
	}

	changes, err := ChangesDirs(destDir, srcDir)
	if err != nil {
		return err
	}
	if len(changes) > 0 {
		return fmt.Errorf("with %d files and %v hardlinks: expected 0 changes, got %d", fileNum, hardlinks, len(changes))
	}
	return nil
}

func tarUntar(t *testing.T, origin string, options *TarOptions) ([]Change, error) {
	archive, err := TarWithOptions(origin, options)
	if err != nil {
		t.Fatal(err)
	}
	defer archive.Close()

	buf := make([]byte, 10)
	if _, err := archive.Read(buf); err != nil {
		return nil, err
	}
	wrap := io.MultiReader(bytes.NewReader(buf), archive)

	detectedCompression := DetectCompression(buf)
	compression := options.Compression
	if detectedCompression.Extension() != compression.Extension() {
		return nil, fmt.Errorf("Wrong compression detected. Actual compression: %s, found %s", compression.Extension(), detectedCompression.Extension())
	}

	tmp, err := ioutil.TempDir("", "docker-test-untar")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	if err := Untar(wrap, tmp, nil); err != nil {
		return nil, err
	}
	if _, err := os.Stat(tmp); err != nil {
		return nil, err
	}

	return ChangesDirs(origin, tmp)
}

func TestTarUntar(t *testing.T) {
	origin, err := ioutil.TempDir("", "docker-test-untar-origin")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(origin)
	if err := ioutil.WriteFile(path.Join(origin, "1"), []byte("hello world"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(origin, "2"), []byte("welcome!"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(origin, "3"), []byte("will be ignored"), 0700); err != nil {
		t.Fatal(err)
	}

	for _, c := range []Compression{
		Uncompressed,
		Gzip,
	} {
		changes, err := tarUntar(t, origin, &TarOptions{
			Compression:     c,
			ExcludePatterns: []string{"3"},
		})

		if err != nil {
			t.Fatalf("Error tar/untar for compression %s: %s", c.Extension(), err)
		}

		if len(changes) != 1 || changes[0].Path != "/3" {
			t.Fatalf("Unexpected differences after tarUntar: %v", changes)
		}
	}
}

func TestTarUntarWithXattr(t *testing.T) {
	origin, err := ioutil.TempDir("", "docker-test-untar-origin")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(origin)
	if err := ioutil.WriteFile(path.Join(origin, "1"), []byte("hello world"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(origin, "2"), []byte("welcome!"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(origin, "3"), []byte("will be ignored"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := system.Lsetxattr(path.Join(origin, "2"), "security.capability", []byte{0x00}, 0); err != nil {
		t.Fatal(err)
	}

	for _, c := range []Compression{
		Uncompressed,
		Gzip,
	} {
		changes, err := tarUntar(t, origin, &TarOptions{
			Compression:     c,
			ExcludePatterns: []string{"3"},
		})

		if err != nil {
			t.Fatalf("Error tar/untar for compression %s: %s", c.Extension(), err)
		}

		if len(changes) != 1 || changes[0].Path != "/3" {
			t.Fatalf("Unexpected differences after tarUntar: %v", changes)
		}
		capability, _ := system.Lgetxattr(path.Join(origin, "2"), "security.capability")
		if capability == nil && capability[0] != 0x00 {
			t.Fatalf("Untar should have kept the 'security.capability' xattr.")
		}
	}
}

func TestTarWithOptions(t *testing.T) {
	origin, err := ioutil.TempDir("", "docker-test-untar-origin")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ioutil.TempDir(origin, "folder"); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(origin)
	if err := ioutil.WriteFile(path.Join(origin, "1"), []byte("hello world"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(origin, "2"), []byte("welcome!"), 0700); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		opts       *TarOptions
		numChanges int
	}{
		{&TarOptions{IncludeFiles: []string{"1"}}, 2},
		{&TarOptions{ExcludePatterns: []string{"2"}}, 1},
		{&TarOptions{ExcludePatterns: []string{"1", "folder*"}}, 2},
		{&TarOptions{IncludeFiles: []string{"1", "1"}}, 2},
		{&TarOptions{Name: "test", IncludeFiles: []string{"1"}}, 4},
	}
	for _, testCase := range cases {
		changes, err := tarUntar(t, origin, testCase.opts)
		if err != nil {
			t.Fatalf("Error tar/untar when testing inclusion/exclusion: %s", err)
		}
		if len(changes) != testCase.numChanges {
			t.Errorf("Expected %d changes, got %d for %+v:",
				testCase.numChanges, len(changes), testCase.opts)
		}
	}
}

// Some tar archives such as http://haproxy.1wt.eu/download/1.5/src/devel/haproxy-1.5-dev21.tar.gz
// use PAX Global Extended Headers.
// Failing prevents the archives from being uncompressed during ADD
func TestTypeXGlobalHeaderDoesNotFail(t *testing.T) {
	hdr := tar.Header{Typeflag: tar.TypeXGlobalHeader}
	tmpDir, err := ioutil.TempDir("", "docker-test-archive-pax-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)
	err = createTarFile(filepath.Join(tmpDir, "pax_global_header"), tmpDir, &hdr, nil, true, nil)
	if err != nil {
		t.Fatal(err)
	}
}

// Some tar have both GNU specific (huge uid) and Ustar specific (long name) things.
// Not supposed to happen (should use PAX instead of Ustar for long name) but it does and it should still work.
func TestUntarUstarGnuConflict(t *testing.T) {
	f, err := os.Open("testdata/broken.tar")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	tr := tar.NewReader(f)
	// Iterate through the files in the archive.
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			// end of tar archive
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if hdr.Name == "root/.cpanm/work/1395823785.24209/Plack-1.0030/blib/man3/Plack::Middleware::LighttpdScriptNameFix.3pm" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("%s not found in the archive", "root/.cpanm/work/1395823785.24209/Plack-1.0030/blib/man3/Plack::Middleware::LighttpdScriptNameFix.3pm")
	}
}

func TestTarWithBlockCharFifo(t *testing.T) {
	origin, err := ioutil.TempDir("", "docker-test-tar-hardlink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(origin)
	if err := ioutil.WriteFile(path.Join(origin, "1"), []byte("hello world"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := system.Mknod(path.Join(origin, "2"), syscall.S_IFBLK, int(system.Mkdev(int64(12), int64(5)))); err != nil {
		t.Fatal(err)
	}
	if err := system.Mknod(path.Join(origin, "3"), syscall.S_IFCHR, int(system.Mkdev(int64(12), int64(5)))); err != nil {
		t.Fatal(err)
	}
	if err := system.Mknod(path.Join(origin, "4"), syscall.S_IFIFO, int(system.Mkdev(int64(12), int64(5)))); err != nil {
		t.Fatal(err)
	}

	dest, err := ioutil.TempDir("", "docker-test-tar-hardlink-dest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dest)

	// we'll do this in two steps to separate failure
	fh, err := Tar(origin, Uncompressed)
	if err != nil {
		t.Fatal(err)
	}

	// ensure we can read the whole thing with no error, before writing back out
	buf, err := ioutil.ReadAll(fh)
	if err != nil {
		t.Fatal(err)
	}

	bRdr := bytes.NewReader(buf)
	err = Untar(bRdr, dest, &TarOptions{Compression: Uncompressed})
	if err != nil {
		t.Fatal(err)
	}

	changes, err := ChangesDirs(origin, dest)
	if err != nil {
		t.Fatal(err)
	}
	if len(changes) > 0 {
		t.Fatalf("Tar with special device (block, char, fifo) should keep them (recreate them when untar) : %v", changes)
	}
}

func TestTarWithHardLink(t *testing.T) {
	origin, err := ioutil.TempDir("", "docker-test-tar-hardlink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(origin)
	if err := ioutil.WriteFile(path.Join(origin, "1"), []byte("hello world"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(path.Join(origin, "1"), path.Join(origin, "2")); err != nil {
		t.Fatal(err)
	}

	var i1, i2 uint64
	if i1, err = getNlink(path.Join(origin, "1")); err != nil {
		t.Fatal(err)
	}
	// sanity check that we can hardlink
	if i1 != 2 {
		t.Skipf("skipping since hardlinks don't work here; expected 2 links, got %d", i1)
	}

	dest, err := ioutil.TempDir("", "docker-test-tar-hardlink-dest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dest)

	// we'll do this in two steps to separate failure
	fh, err := Tar(origin, Uncompressed)
	if err != nil {
		t.Fatal(err)
	}

	// ensure we can read the whole thing with no error, before writing back out
	buf, err := ioutil.ReadAll(fh)
	if err != nil {
		t.Fatal(err)
	}

	bRdr := bytes.NewReader(buf)
	err = Untar(bRdr, dest, &TarOptions{Compression: Uncompressed})
	if err != nil {
		t.Fatal(err)
	}

	if i1, err = getInode(path.Join(dest, "1")); err != nil {
		t.Fatal(err)
	}
	if i2, err = getInode(path.Join(dest, "2")); err != nil {
		t.Fatal(err)
	}

	if i1 != i2 {
		t.Errorf("expected matching inodes, but got %d and %d", i1, i2)
	}
}

func getNlink(path string) (uint64, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	statT, ok := stat.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, fmt.Errorf("expected type *syscall.Stat_t, got %t", stat.Sys())
	}
	// We need this conversion on ARM64
	return uint64(statT.Nlink), nil
}

func getInode(path string) (uint64, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	statT, ok := stat.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, fmt.Errorf("expected type *syscall.Stat_t, got %t", stat.Sys())
	}
	return statT.Ino, nil
}

func prepareUntarSourceDirectory(numberOfFiles int, targetPath string, makeLinks bool) (int, error) {
	fileData := []byte("fooo")
	for n := 0; n < numberOfFiles; n++ {
		fileName := fmt.Sprintf("file-%d", n)
		if err := ioutil.WriteFile(path.Join(targetPath, fileName), fileData, 0700); err != nil {
			return 0, err
		}
		if makeLinks {
			if err := os.Link(path.Join(targetPath, fileName), path.Join(targetPath, fileName+"-link")); err != nil {
				return 0, err
			}
		}
	}
	totalSize := numberOfFiles * len(fileData)
	return totalSize, nil
}

func BenchmarkTarUntar(b *testing.B) {
	origin, err := ioutil.TempDir("", "docker-test-untar-origin")
	if err != nil {
		b.Fatal(err)
	}
	tempDir, err := ioutil.TempDir("", "docker-test-untar-destination")
	if err != nil {
		b.Fatal(err)
	}
	target := path.Join(tempDir, "dest")
	n, err := prepareUntarSourceDirectory(100, origin, false)
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(origin)
	defer os.RemoveAll(tempDir)

	b.ResetTimer()
	b.SetBytes(int64(n))
	for n := 0; n < b.N; n++ {
		err := TarUntar(origin, target)
		if err != nil {
			b.Fatal(err)
		}
		os.RemoveAll(target)
	}
}

func BenchmarkTarUntarWithLinks(b *testing.B) {
	origin, err := ioutil.TempDir("", "docker-test-untar-origin")
	if err != nil {
		b.Fatal(err)
	}
	tempDir, err := ioutil.TempDir("", "docker-test-untar-destination")
	if err != nil {
		b.Fatal(err)
	}
	target := path.Join(tempDir, "dest")
	n, err := prepareUntarSourceDirectory(100, origin, true)
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(origin)
	defer os.RemoveAll(tempDir)

	b.ResetTimer()
	b.SetBytes(int64(n))
	for n := 0; n < b.N; n++ {
		err := TarUntar(origin, target)
		if err != nil {
			b.Fatal(err)
		}
		os.RemoveAll(target)
	}
}

func TestUntarInvalidFilenames(t *testing.T) {
	for i, headers := range [][]*tar.Header{
		{
			{
				Name:     "../victim/dotdot",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
		{
			{
				// Note the leading slash
				Name:     "/../victim/slash-dotdot",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
	} {
		if err := testBreakout("untar", "docker-TestUntarInvalidFilenames", headers); err != nil {
			t.Fatalf("i=%d. %v", i, err)
		}
	}
}

func TestUntarHardlinkToSymlink(t *testing.T) {
	for i, headers := range [][]*tar.Header{
		{
			{
				Name:     "symlink1",
				Typeflag: tar.TypeSymlink,
				Linkname: "regfile",
				Mode:     0644,
			},
			{
				Name:     "symlink2",
				Typeflag: tar.TypeLink,
				Linkname: "symlink1",
				Mode:     0644,
			},
			{
				Name:     "regfile",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
	} {
		if err := testBreakout("untar", "docker-TestUntarHardlinkToSymlink", headers); err != nil {
			t.Fatalf("i=%d. %v", i, err)
		}
	}
}

func TestUntarInvalidHardlink(t *testing.T) {
	for i, headers := range [][]*tar.Header{
		{ // try reading victim/hello (../)
			{
				Name:     "dotdot",
				Typeflag: tar.TypeLink,
				Linkname: "../victim/hello",
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (/../)
			{
				Name:     "slash-dotdot",
				Typeflag: tar.TypeLink,
				// Note the leading slash
				Linkname: "/../victim/hello",
				Mode:     0644,
			},
		},
		{ // try writing victim/file
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeLink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "loophole-victim/file",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (hardlink, symlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeLink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "symlink",
				Typeflag: tar.TypeSymlink,
				Linkname: "loophole-victim/hello",
				Mode:     0644,
			},
		},
		{ // Try reading victim/hello (hardlink, hardlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeLink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "hardlink",
				Typeflag: tar.TypeLink,
				Linkname: "loophole-victim/hello",
				Mode:     0644,
			},
		},
		{ // Try removing victim directory (hardlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeLink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
	} {
		if err := testBreakout("untar", "docker-TestUntarInvalidHardlink", headers); err != nil {
			t.Fatalf("i=%d. %v", i, err)
		}
	}
}

func TestUntarInvalidSymlink(t *testing.T) {
	for i, headers := range [][]*tar.Header{
		{ // try reading victim/hello (../)
			{
				Name:     "dotdot",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim/hello",
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (/../)
			{
				Name:     "slash-dotdot",
				Typeflag: tar.TypeSymlink,
				// Note the leading slash
				Linkname: "/../victim/hello",
				Mode:     0644,
			},
		},
		{ // try writing victim/file
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "loophole-victim/file",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (symlink, symlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "symlink",
				Typeflag: tar.TypeSymlink,
				Linkname: "loophole-victim/hello",
				Mode:     0644,
			},
		},
		{ // try reading victim/hello (symlink, hardlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "hardlink",
				Typeflag: tar.TypeLink,
				Linkname: "loophole-victim/hello",
				Mode:     0644,
			},
		},
		{ // try removing victim directory (symlink)
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeSymlink,
				Linkname: "../victim",
				Mode:     0755,
			},
			{
				Name:     "loophole-victim",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
		{ // try writing to victim/newdir/newfile with a symlink in the path
			{
				// this header needs to be before the next one, or else there is an error
				Name:     "dir/loophole",
				Typeflag: tar.TypeSymlink,
				Linkname: "../../victim",
				Mode:     0755,
			},
			{
				Name:     "dir/loophole/newdir/newfile",
				Typeflag: tar.TypeReg,
				Mode:     0644,
			},
		},
	} {
		if err := testBreakout("untar", "docker-TestUntarInvalidSymlink", headers); err != nil {
			t.Fatalf("i=%d. %v", i, err)
		}
	}
}

func TestTempArchiveCloseMultipleTimes(t *testing.T) {
	reader := ioutil.NopCloser(strings.NewReader("hello"))
	tempArchive, err := NewTempArchive(reader, "")
	buf := make([]byte, 10)
	n, err := tempArchive.Read(buf)
	if n != 5 {
		t.Fatalf("Expected to read 5 bytes. Read %d instead", n)
	}
	for i := 0; i < 3; i++ {
		if err = tempArchive.Close(); err != nil {
			t.Fatalf("i=%d. Unexpected error closing temp archive: %v", i, err)
		}
	}
}
