package main

import (
	"os"

	"github.com/go-check/check"
)

// docker cp LOCALPATH CONTAINER:PATH

// Try all of the test cases from the archive package which implements the
// internals of `docker cp` and ensure that the behavior matches when actually
// copying to and from containers.

// Basic assumptions about SRC and DST:
// 1. SRC must exist.
// 2. If SRC ends with a trailing separator, it must be a directory.
// 3. DST parent directory must exist.
// 4. If DST exists as a file, it must not end with a trailing separator.

// First get these easy error cases out of the way.

// Test for error when SRC does not exist.
func (s *DockerSuite) TestCpToErrSrcNotExists(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-err-src-not-exists")
	defer os.RemoveAll(tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(cID, "file1")

	err := runDockerCp(c, srcPath, dstPath)
	if err == nil {
		c.Fatal("expected IsNotExist error, but got nil instead")
	}

	if !isCpNotExist(err) {
		c.Fatalf("expected IsNotExist error, but got %T: %s", err, err)
	}
}

// Test for error when SRC ends in a trailing
// path separator but it exists as a file.
func (s *DockerSuite) TestCpToErrSrcNotDir(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-err-src-not-dir")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPathTrailingSep(tmpDir, "file1")
	dstPath := containerCpPath(cID, "testDir")

	err := runDockerCp(c, srcPath, dstPath)
	if err == nil {
		c.Fatal("expected IsNotDir error, but got nil instead")
	}

	if !isCpNotDir(err) {
		c.Fatalf("expected IsNotDir error, but got %T: %s", err, err)
	}
}

// Test for error when SRC is a valid file or directory,
// bu the DST parent directory does not exist.
func (s *DockerSuite) TestCpToErrDstParentNotExists(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{addContent: true})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-err-dst-parent-not-exists")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	// Try with a file source.
	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(cID, "/notExists", "file1")

	err := runDockerCp(c, srcPath, dstPath)
	if err == nil {
		c.Fatal("expected IsNotExist error, but got nil instead")
	}

	if !isCpNotExist(err) {
		c.Fatalf("expected IsNotExist error, but got %T: %s", err, err)
	}

	// Try with a directory source.
	srcPath = cpPath(tmpDir, "dir1")

	if err := runDockerCp(c, srcPath, dstPath); err == nil {
		c.Fatal("expected IsNotExist error, but got nil instead")
	}

	if !isCpNotExist(err) {
		c.Fatalf("expected IsNotExist error, but got %T: %s", err, err)
	}
}

// Test for error when DST ends in a trailing path separator but exists as a
// file. Also test that we cannot overwirite an existing directory with a
// non-directory and cannot overwrite an existing
func (s *DockerSuite) TestCpToErrDstNotDir(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{addContent: true})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-err-dst-not-dir")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	// Try with a file source.
	srcPath := cpPath(tmpDir, "dir1/file1-1")
	dstPath := containerCpPathTrailingSep(cID, "file1")

	// The client should encounter an error trying to stat the destination
	// and then be unable to copy since the destination is asserted to be a
	// directory but does not exist.
	err := runDockerCp(c, srcPath, dstPath)
	if err == nil {
		c.Fatal("expected DirNotExist error, but got nil instead")
	}

	if !isCpDirNotExist(err) {
		c.Fatalf("expected DirNotExist error, but got %T: %s", err, err)
	}

	// Try with a directory source.
	srcPath = cpPath(tmpDir, "dir1")

	// The client should encounter an error trying to stat the destination and
	// then decide to extract to the parent directory instead with a rebased
	// name in the source archive, but this directory would overwrite the
	// existing file with the same name.
	err = runDockerCp(c, srcPath, dstPath)
	if err == nil {
		c.Fatal("expected CannotOverwriteNonDirWithDir error, but got nil instead")
	}

	if !isCannotOverwriteNonDirWithDir(err) {
		c.Fatalf("expected CannotOverwriteNonDirWithDir error, but got %T: %s", err, err)
	}
}

// Possibilities are reduced to the remaining 10 cases:
//
//  case | srcIsDir | onlyDirContents | dstExists | dstIsDir | dstTrSep | action
// ===================================================================================================
//   A   |  no      |  -              |  no       |  -       |  no      |  create file
//   B   |  no      |  -              |  no       |  -       |  yes     |  error
//   C   |  no      |  -              |  yes      |  no      |  -       |  overwrite file
//   D   |  no      |  -              |  yes      |  yes     |  -       |  create file in dst dir
//   E   |  yes     |  no             |  no       |  -       |  -       |  create dir, copy contents
//   F   |  yes     |  no             |  yes      |  no      |  -       |  error
//   G   |  yes     |  no             |  yes      |  yes     |  -       |  copy dir and contents
//   H   |  yes     |  yes            |  no       |  -       |  -       |  create dir, copy contents
//   I   |  yes     |  yes            |  yes      |  no      |  -       |  error
//   J   |  yes     |  yes            |  yes      |  yes     |  -       |  copy dir contents
//

// A. SRC specifies a file and DST (no trailing path separator) doesn't
//    exist. This should create a file with the name DST and copy the
//    contents of the source file into it.
func (s *DockerSuite) TestCpToCaseA(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		workDir: "/root", command: makeCatFileCommand("itWorks.txt"),
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-a")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(cID, "/root/itWorks.txt")

	if err := runDockerCp(c, srcPath, dstPath); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	if err := containerStartOutputEquals(c, cID, "file1\n"); err != nil {
		c.Fatal(err)
	}
}

// B. SRC specifies a file and DST (with trailing path separator) doesn't
//    exist. This should cause an error because the copy operation cannot
//    create a directory when copying a single file.
func (s *DockerSuite) TestCpToCaseB(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("testDir/file1"),
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-b")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstDir := containerCpPathTrailingSep(cID, "testDir")

	err := runDockerCp(c, srcPath, dstDir)
	if err == nil {
		c.Fatal("expected DirNotExists error, but got nil instead")
	}

	if !isCpDirNotExist(err) {
		c.Fatalf("expected DirNotExists error, but got %T: %s", err, err)
	}
}

// C. SRC specifies a file and DST exists as a file. This should overwrite
//    the file at DST with the contents of the source file.
func (s *DockerSuite) TestCpToCaseC(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
		command: makeCatFileCommand("file2"),
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-c")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(cID, "/root/file2")

	// Ensure the container's file starts with the original content.
	if err := containerStartOutputEquals(c, cID, "file2\n"); err != nil {
		c.Fatal(err)
	}

	if err := runDockerCp(c, srcPath, dstPath); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1's contents.
	if err := containerStartOutputEquals(c, cID, "file1\n"); err != nil {
		c.Fatal(err)
	}
}

// D. SRC specifies a file and DST exists as a directory. This should place
//    a copy of the source file inside it using the basename from SRC. Ensure
//    this works whether DST has a trailing path separator or not.
func (s *DockerSuite) TestCpToCaseD(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		addContent: true,
		command:    makeCatFileCommand("/dir1/file1"),
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-d")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstDir := containerCpPath(cID, "dir1")

	// Ensure that dstPath doesn't exist.
	if err := containerStartOutputEquals(c, cID, ""); err != nil {
		c.Fatal(err)
	}

	if err := runDockerCp(c, srcPath, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1's contents.
	if err := containerStartOutputEquals(c, cID, "file1\n"); err != nil {
		c.Fatal(err)
	}

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	cID = makeTestContainer(c, testContainerOptions{
		addContent: true,
		command:    makeCatFileCommand("/dir1/file1"),
	})
	defer deleteContainer(cID)

	dstDir = containerCpPathTrailingSep(cID, "dir1")

	// Ensure that dstPath doesn't exist.
	if err := containerStartOutputEquals(c, cID, ""); err != nil {
		c.Fatal(err)
	}

	if err := runDockerCp(c, srcPath, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1's contents.
	if err := containerStartOutputEquals(c, cID, "file1\n"); err != nil {
		c.Fatal(err)
	}
}

// E. SRC specifies a directory and DST does not exist. This should create a
//    directory at DST and copy the contents of the SRC directory into the DST
//    directory. Ensure this works whether DST has a trailing path separator or
//    not.
func (s *DockerSuite) TestCpToCaseE(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/testDir/file1-1"),
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-e")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPath(tmpDir, "dir1")
	dstDir := containerCpPath(cID, "testDir")

	if err := runDockerCp(c, srcDir, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1-1's contents.
	if err := containerStartOutputEquals(c, cID, "file1-1\n"); err != nil {
		c.Fatal(err)
	}

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	cID = makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/testDir/file1-1"),
	})
	defer deleteContainer(cID)

	dstDir = containerCpPathTrailingSep(cID, "testDir")

	err := runDockerCp(c, srcDir, dstDir)
	if err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1-1's contents.
	if err := containerStartOutputEquals(c, cID, "file1-1\n"); err != nil {
		c.Fatal(err)
	}
}

// F. SRC specifies a directory and DST exists as a file. This should cause an
//    error as it is not possible to overwrite a file with a directory.
func (s *DockerSuite) TestCpToCaseF(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-f")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPath(tmpDir, "dir1")
	dstFile := containerCpPath(cID, "/root/file1")

	err := runDockerCp(c, srcDir, dstFile)
	if err == nil {
		c.Fatal("expected ErrCannotCopyDir error, but got nil instead")
	}

	if !isCpCannotCopyDir(err) {
		c.Fatalf("expected ErrCannotCopyDir error, but got %T: %s", err, err)
	}
}

// G. SRC specifies a directory and DST exists as a directory. This should copy
//    the SRC directory and all its contents to the DST directory. Ensure this
//    works whether DST has a trailing path separator or not.
func (s *DockerSuite) TestCpToCaseG(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
		command: makeCatFileCommand("dir2/dir1/file1-1"),
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-g")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPath(tmpDir, "dir1")
	dstDir := containerCpPath(cID, "/root/dir2")

	// Ensure that dstPath doesn't exist.
	if err := containerStartOutputEquals(c, cID, ""); err != nil {
		c.Fatal(err)
	}

	if err := runDockerCp(c, srcDir, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1-1's contents.
	if err := containerStartOutputEquals(c, cID, "file1-1\n"); err != nil {
		c.Fatal(err)
	}

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	cID = makeTestContainer(c, testContainerOptions{
		addContent: true,
		command:    makeCatFileCommand("/dir2/dir1/file1-1"),
	})
	defer deleteContainer(cID)

	dstDir = containerCpPathTrailingSep(cID, "/dir2")

	// Ensure that dstPath doesn't exist.
	if err := containerStartOutputEquals(c, cID, ""); err != nil {
		c.Fatal(err)
	}

	if err := runDockerCp(c, srcDir, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1-1's contents.
	if err := containerStartOutputEquals(c, cID, "file1-1\n"); err != nil {
		c.Fatal(err)
	}
}

// H. SRC specifies a directory's contents only and DST does not exist. This
//    should create a directory at DST and copy the contents of the SRC
//    directory (but not the directory itself) into the DST directory. Ensure
//    this works whether DST has a trailing path separator or not.
func (s *DockerSuite) TestCpToCaseH(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/testDir/file1-1"),
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-h")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPathTrailingSep(tmpDir, "dir1") + "."
	dstDir := containerCpPath(cID, "testDir")

	if err := runDockerCp(c, srcDir, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1-1's contents.
	if err := containerStartOutputEquals(c, cID, "file1-1\n"); err != nil {
		c.Fatal(err)
	}

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	cID = makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/testDir/file1-1"),
	})
	defer deleteContainer(cID)

	dstDir = containerCpPathTrailingSep(cID, "testDir")

	if err := runDockerCp(c, srcDir, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1-1's contents.
	if err := containerStartOutputEquals(c, cID, "file1-1\n"); err != nil {
		c.Fatal(err)
	}
}

// I. SRC specifies a direcotry's contents only and DST exists as a file. This
//    should cause an error as it is not possible to overwrite a file with a
//    directory.
func (s *DockerSuite) TestCpToCaseI(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-i")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPathTrailingSep(tmpDir, "dir1") + "."
	dstFile := containerCpPath(cID, "/root/file1")

	err := runDockerCp(c, srcDir, dstFile)
	if err == nil {
		c.Fatal("expected ErrCannotCopyDir error, but got nil instead")
	}

	if !isCpCannotCopyDir(err) {
		c.Fatalf("expected ErrCannotCopyDir error, but got %T: %s", err, err)
	}
}

// J. SRC specifies a directory's contents only and DST exists as a directory.
//    This should copy the contents of the SRC directory (but not the directory
//    itself) into the DST directory. Ensure this works whether DST has a
//    trailing path separator or not.
func (s *DockerSuite) TestCpToCaseJ(c *check.C) {
	cID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
		command: makeCatFileCommand("/dir2/file1-1"),
	})
	defer deleteContainer(cID)

	tmpDir := getTestDir(c, "test-cp-to-case-j")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPathTrailingSep(tmpDir, "dir1") + "."
	dstDir := containerCpPath(cID, "/dir2")

	// Ensure that dstPath doesn't exist.
	if err := containerStartOutputEquals(c, cID, ""); err != nil {
		c.Fatal(err)
	}

	if err := runDockerCp(c, srcDir, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1-1's contents.
	if err := containerStartOutputEquals(c, cID, "file1-1\n"); err != nil {
		c.Fatal(err)
	}

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	cID = makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/dir2/file1-1"),
	})
	defer deleteContainer(cID)

	dstDir = containerCpPathTrailingSep(cID, "/dir2")

	// Ensure that dstPath doesn't exist.
	if err := containerStartOutputEquals(c, cID, ""); err != nil {
		c.Fatal(err)
	}

	if err := runDockerCp(c, srcDir, dstDir); err != nil {
		c.Fatalf("unexpected error %T: %s", err, err)
	}

	// Should now contain file1-1's contents.
	if err := containerStartOutputEquals(c, cID, "file1-1\n"); err != nil {
		c.Fatal(err)
	}
}

// The `docker cp` command should also ensure that you cannot
// write to a container rootfs that is marked as read-only.
func (s *DockerSuite) TestCpToErrReadOnlyRootfs(c *check.C) {
	tmpDir := getTestDir(c, "test-cp-to-err-read-only-rootfs")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	cID := makeTestContainer(c, testContainerOptions{
		readOnly: true, workDir: "/root",
		command: makeCatFileCommand("shouldNotExist"),
	})
	defer deleteContainer(cID)

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(cID, "/root/shouldNotExist")

	err := runDockerCp(c, srcPath, dstPath)
	if err == nil {
		c.Fatal("expected ErrContainerRootfsReadonly error, but got nil instead")
	}

	if !isCpCannotCopyReadOnly(err) {
		c.Fatalf("expected ErrContainerRootfsReadonly error, but got %T: %s", err, err)
	}

	// Ensure that dstPath doesn't exist.
	if err := containerStartOutputEquals(c, cID, ""); err != nil {
		c.Fatal(err)
	}
}

// The `docker cp` command should also ensure that you
// cannot write to a volume that is mounted as read-only.
func (s *DockerSuite) TestCpToErrReadOnlyVolume(c *check.C) {
	tmpDir := getTestDir(c, "test-cp-to-err-read-only-volume")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	cID := makeTestContainer(c, testContainerOptions{
		volumes: defaultVolumes(tmpDir), workDir: "/root",
		command: makeCatFileCommand("/vol_ro/shouldNotExist"),
	})
	defer deleteContainer(cID)

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(cID, "/vol_ro/shouldNotExist")

	err := runDockerCp(c, srcPath, dstPath)
	if err == nil {
		c.Fatal("expected ErrVolumeReadonly error, but got nil instead")
	}

	if !isCpCannotCopyReadOnly(err) {
		c.Fatalf("expected ErrVolumeReadonly error, but got %T: %s", err, err)
	}

	// Ensure that dstPath doesn't exist.
	if err := containerStartOutputEquals(c, cID, ""); err != nil {
		c.Fatal(err)
	}
}
