package main

import (
	"os"

	"github.com/docker/docker/integration-cli/checker"
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
	containerID := makeTestContainer(c, testContainerOptions{})

	tmpDir := getTestDir(c, "test-cp-to-err-src-not-exists")
	defer os.RemoveAll(tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(containerID, "file1")

	err := runDockerCp(c, srcPath, dstPath, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpNotExist(err), checker.True, check.Commentf("expected IsNotExist error, but got %T: %s", err, err))
}

// Test for error when SRC ends in a trailing
// path separator but it exists as a file.
func (s *DockerSuite) TestCpToErrSrcNotDir(c *check.C) {
	containerID := makeTestContainer(c, testContainerOptions{})

	tmpDir := getTestDir(c, "test-cp-to-err-src-not-dir")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPathTrailingSep(tmpDir, "file1")
	dstPath := containerCpPath(containerID, "testDir")

	err := runDockerCp(c, srcPath, dstPath, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpNotDir(err), checker.True, check.Commentf("expected IsNotDir error, but got %T: %s", err, err))
}

// Test for error when SRC is a valid file or directory,
// but the DST parent directory does not exist.
func (s *DockerSuite) TestCpToErrDstParentNotExists(c *check.C) {
	testRequires(c, DaemonIsLinux)
	containerID := makeTestContainer(c, testContainerOptions{addContent: true})

	tmpDir := getTestDir(c, "test-cp-to-err-dst-parent-not-exists")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	// Try with a file source.
	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(containerID, "/notExists", "file1")

	err := runDockerCp(c, srcPath, dstPath, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpNotExist(err), checker.True, check.Commentf("expected IsNotExist error, but got %T: %s", err, err))

	// Try with a directory source.
	srcPath = cpPath(tmpDir, "dir1")

	err = runDockerCp(c, srcPath, dstPath, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpNotExist(err), checker.True, check.Commentf("expected IsNotExist error, but got %T: %s", err, err))
}

// Test for error when DST ends in a trailing path separator but exists as a
// file. Also test that we cannot overwrite an existing directory with a
// non-directory and cannot overwrite an existing
func (s *DockerSuite) TestCpToErrDstNotDir(c *check.C) {
	testRequires(c, DaemonIsLinux)
	containerID := makeTestContainer(c, testContainerOptions{addContent: true})

	tmpDir := getTestDir(c, "test-cp-to-err-dst-not-dir")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	// Try with a file source.
	srcPath := cpPath(tmpDir, "dir1/file1-1")
	dstPath := containerCpPathTrailingSep(containerID, "file1")

	// The client should encounter an error trying to stat the destination
	// and then be unable to copy since the destination is asserted to be a
	// directory but does not exist.
	err := runDockerCp(c, srcPath, dstPath, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpDirNotExist(err), checker.True, check.Commentf("expected DirNotExist error, but got %T: %s", err, err))

	// Try with a directory source.
	srcPath = cpPath(tmpDir, "dir1")

	// The client should encounter an error trying to stat the destination and
	// then decide to extract to the parent directory instead with a rebased
	// name in the source archive, but this directory would overwrite the
	// existing file with the same name.
	err = runDockerCp(c, srcPath, dstPath, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCannotOverwriteNonDirWithDir(err), checker.True, check.Commentf("expected CannotOverwriteNonDirWithDir error, but got %T: %s", err, err))
}

// Check that copying from a local path to a symlink in a container copies to
// the symlink target and does not overwrite the container symlink itself.
func (s *DockerSuite) TestCpToSymlinkDestination(c *check.C) {
	//  stat /tmp/test-cp-to-symlink-destination-262430901/vol3 gets permission denied for the user
	testRequires(c, NotUserNamespace)
	testRequires(c, DaemonIsLinux)
	testRequires(c, SameHostDaemon) // Requires local volume mount bind.

	testVol := getTestDir(c, "test-cp-to-symlink-destination-")
	defer os.RemoveAll(testVol)

	makeTestContentInDir(c, testVol)

	containerID := makeTestContainer(c, testContainerOptions{
		volumes: defaultVolumes(testVol), // Our bind mount is at /vol2
	})

	// First, copy a local file to a symlink to a file in the container. This
	// should overwrite the symlink target contents with the source contents.
	srcPath := cpPath(testVol, "file2")
	dstPath := containerCpPath(containerID, "/vol2/symlinkToFile1")

	c.Assert(runDockerCp(c, srcPath, dstPath, nil), checker.IsNil)

	// The symlink should not have been modified.
	c.Assert(symlinkTargetEquals(c, cpPath(testVol, "symlinkToFile1"), "file1"), checker.IsNil)

	// The file should have the contents of "file2" now.
	c.Assert(fileContentEquals(c, cpPath(testVol, "file1"), "file2\n"), checker.IsNil)

	// Next, copy a local file to a symlink to a directory in the container.
	// This should copy the file into the symlink target directory.
	dstPath = containerCpPath(containerID, "/vol2/symlinkToDir1")

	c.Assert(runDockerCp(c, srcPath, dstPath, nil), checker.IsNil)

	// The symlink should not have been modified.
	c.Assert(symlinkTargetEquals(c, cpPath(testVol, "symlinkToDir1"), "dir1"), checker.IsNil)

	// The file should have the contents of "file2" now.
	c.Assert(fileContentEquals(c, cpPath(testVol, "file2"), "file2\n"), checker.IsNil)

	// Next, copy a file to a symlink to a file that does not exist (a broken
	// symlink) in the container. This should create the target file with the
	// contents of the source file.
	dstPath = containerCpPath(containerID, "/vol2/brokenSymlinkToFileX")

	c.Assert(runDockerCp(c, srcPath, dstPath, nil), checker.IsNil)

	// The symlink should not have been modified.
	c.Assert(symlinkTargetEquals(c, cpPath(testVol, "brokenSymlinkToFileX"), "fileX"), checker.IsNil)

	// The file should have the contents of "file2" now.
	c.Assert(fileContentEquals(c, cpPath(testVol, "fileX"), "file2\n"), checker.IsNil)

	// Next, copy a local directory to a symlink to a directory in the
	// container. This should copy the directory into the symlink target
	// directory and not modify the symlink.
	srcPath = cpPath(testVol, "/dir2")
	dstPath = containerCpPath(containerID, "/vol2/symlinkToDir1")

	c.Assert(runDockerCp(c, srcPath, dstPath, nil), checker.IsNil)

	// The symlink should not have been modified.
	c.Assert(symlinkTargetEquals(c, cpPath(testVol, "symlinkToDir1"), "dir1"), checker.IsNil)

	// The directory should now contain a copy of "dir2".
	c.Assert(fileContentEquals(c, cpPath(testVol, "dir1/dir2/file2-1"), "file2-1\n"), checker.IsNil)

	// Next, copy a local directory to a symlink to a local directory that does
	// not exist (a broken symlink) in the container. This should create the
	// target as a directory with the contents of the source directory. It
	// should not modify the symlink.
	dstPath = containerCpPath(containerID, "/vol2/brokenSymlinkToDirX")

	c.Assert(runDockerCp(c, srcPath, dstPath, nil), checker.IsNil)

	// The symlink should not have been modified.
	c.Assert(symlinkTargetEquals(c, cpPath(testVol, "brokenSymlinkToDirX"), "dirX"), checker.IsNil)

	// The "dirX" directory should now be a copy of "dir2".
	c.Assert(fileContentEquals(c, cpPath(testVol, "dirX/file2-1"), "file2-1\n"), checker.IsNil)
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
	containerID := makeTestContainer(c, testContainerOptions{
		workDir: "/root", command: makeCatFileCommand("itWorks.txt"),
	})

	tmpDir := getTestDir(c, "test-cp-to-case-a")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(containerID, "/root/itWorks.txt")

	c.Assert(runDockerCp(c, srcPath, dstPath, nil), checker.IsNil)

	c.Assert(containerStartOutputEquals(c, containerID, "file1\n"), checker.IsNil)
}

// B. SRC specifies a file and DST (with trailing path separator) doesn't
//    exist. This should cause an error because the copy operation cannot
//    create a directory when copying a single file.
func (s *DockerSuite) TestCpToCaseB(c *check.C) {
	containerID := makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("testDir/file1"),
	})

	tmpDir := getTestDir(c, "test-cp-to-case-b")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstDir := containerCpPathTrailingSep(containerID, "testDir")

	err := runDockerCp(c, srcPath, dstDir, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpDirNotExist(err), checker.True, check.Commentf("expected DirNotExists error, but got %T: %s", err, err))
}

// C. SRC specifies a file and DST exists as a file. This should overwrite
//    the file at DST with the contents of the source file.
func (s *DockerSuite) TestCpToCaseC(c *check.C) {
	testRequires(c, DaemonIsLinux)
	containerID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
		command: makeCatFileCommand("file2"),
	})

	tmpDir := getTestDir(c, "test-cp-to-case-c")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(containerID, "/root/file2")

	// Ensure the container's file starts with the original content.
	c.Assert(containerStartOutputEquals(c, containerID, "file2\n"), checker.IsNil)

	c.Assert(runDockerCp(c, srcPath, dstPath, nil), checker.IsNil)

	// Should now contain file1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1\n"), checker.IsNil)
}

// D. SRC specifies a file and DST exists as a directory. This should place
//    a copy of the source file inside it using the basename from SRC. Ensure
//    this works whether DST has a trailing path separator or not.
func (s *DockerSuite) TestCpToCaseD(c *check.C) {
	testRequires(c, DaemonIsLinux)
	containerID := makeTestContainer(c, testContainerOptions{
		addContent: true,
		command:    makeCatFileCommand("/dir1/file1"),
	})

	tmpDir := getTestDir(c, "test-cp-to-case-d")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcPath := cpPath(tmpDir, "file1")
	dstDir := containerCpPath(containerID, "dir1")

	// Ensure that dstPath doesn't exist.
	c.Assert(containerStartOutputEquals(c, containerID, ""), checker.IsNil)

	c.Assert(runDockerCp(c, srcPath, dstDir, nil), checker.IsNil)

	// Should now contain file1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1\n"), checker.IsNil)

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	containerID = makeTestContainer(c, testContainerOptions{
		addContent: true,
		command:    makeCatFileCommand("/dir1/file1"),
	})

	dstDir = containerCpPathTrailingSep(containerID, "dir1")

	// Ensure that dstPath doesn't exist.
	c.Assert(containerStartOutputEquals(c, containerID, ""), checker.IsNil)

	c.Assert(runDockerCp(c, srcPath, dstDir, nil), checker.IsNil)

	// Should now contain file1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1\n"), checker.IsNil)
}

// E. SRC specifies a directory and DST does not exist. This should create a
//    directory at DST and copy the contents of the SRC directory into the DST
//    directory. Ensure this works whether DST has a trailing path separator or
//    not.
func (s *DockerSuite) TestCpToCaseE(c *check.C) {
	containerID := makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/testDir/file1-1"),
	})

	tmpDir := getTestDir(c, "test-cp-to-case-e")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPath(tmpDir, "dir1")
	dstDir := containerCpPath(containerID, "testDir")

	c.Assert(runDockerCp(c, srcDir, dstDir, nil), checker.IsNil)

	// Should now contain file1-1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1-1\n"), checker.IsNil)

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	containerID = makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/testDir/file1-1"),
	})

	dstDir = containerCpPathTrailingSep(containerID, "testDir")

	c.Assert(runDockerCp(c, srcDir, dstDir, nil), checker.IsNil)

	// Should now contain file1-1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1-1\n"), checker.IsNil)
}

// F. SRC specifies a directory and DST exists as a file. This should cause an
//    error as it is not possible to overwrite a file with a directory.
func (s *DockerSuite) TestCpToCaseF(c *check.C) {
	testRequires(c, DaemonIsLinux)
	containerID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
	})

	tmpDir := getTestDir(c, "test-cp-to-case-f")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPath(tmpDir, "dir1")
	dstFile := containerCpPath(containerID, "/root/file1")

	err := runDockerCp(c, srcDir, dstFile, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpCannotCopyDir(err), checker.True, check.Commentf("expected ErrCannotCopyDir error, but got %T: %s", err, err))
}

// G. SRC specifies a directory and DST exists as a directory. This should copy
//    the SRC directory and all its contents to the DST directory. Ensure this
//    works whether DST has a trailing path separator or not.
func (s *DockerSuite) TestCpToCaseG(c *check.C) {
	testRequires(c, DaemonIsLinux)
	containerID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
		command: makeCatFileCommand("dir2/dir1/file1-1"),
	})

	tmpDir := getTestDir(c, "test-cp-to-case-g")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPath(tmpDir, "dir1")
	dstDir := containerCpPath(containerID, "/root/dir2")

	// Ensure that dstPath doesn't exist.
	c.Assert(containerStartOutputEquals(c, containerID, ""), checker.IsNil)

	c.Assert(runDockerCp(c, srcDir, dstDir, nil), checker.IsNil)

	// Should now contain file1-1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1-1\n"), checker.IsNil)

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	containerID = makeTestContainer(c, testContainerOptions{
		addContent: true,
		command:    makeCatFileCommand("/dir2/dir1/file1-1"),
	})

	dstDir = containerCpPathTrailingSep(containerID, "/dir2")

	// Ensure that dstPath doesn't exist.
	c.Assert(containerStartOutputEquals(c, containerID, ""), checker.IsNil)

	c.Assert(runDockerCp(c, srcDir, dstDir, nil), checker.IsNil)

	// Should now contain file1-1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1-1\n"), checker.IsNil)
}

// H. SRC specifies a directory's contents only and DST does not exist. This
//    should create a directory at DST and copy the contents of the SRC
//    directory (but not the directory itself) into the DST directory. Ensure
//    this works whether DST has a trailing path separator or not.
func (s *DockerSuite) TestCpToCaseH(c *check.C) {
	containerID := makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/testDir/file1-1"),
	})

	tmpDir := getTestDir(c, "test-cp-to-case-h")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPathTrailingSep(tmpDir, "dir1") + "."
	dstDir := containerCpPath(containerID, "testDir")

	c.Assert(runDockerCp(c, srcDir, dstDir, nil), checker.IsNil)

	// Should now contain file1-1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1-1\n"), checker.IsNil)

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	containerID = makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/testDir/file1-1"),
	})

	dstDir = containerCpPathTrailingSep(containerID, "testDir")

	c.Assert(runDockerCp(c, srcDir, dstDir, nil), checker.IsNil)

	// Should now contain file1-1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1-1\n"), checker.IsNil)
}

// I. SRC specifies a directory's contents only and DST exists as a file. This
//    should cause an error as it is not possible to overwrite a file with a
//    directory.
func (s *DockerSuite) TestCpToCaseI(c *check.C) {
	testRequires(c, DaemonIsLinux)
	containerID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
	})

	tmpDir := getTestDir(c, "test-cp-to-case-i")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPathTrailingSep(tmpDir, "dir1") + "."
	dstFile := containerCpPath(containerID, "/root/file1")

	err := runDockerCp(c, srcDir, dstFile, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpCannotCopyDir(err), checker.True, check.Commentf("expected ErrCannotCopyDir error, but got %T: %s", err, err))
}

// J. SRC specifies a directory's contents only and DST exists as a directory.
//    This should copy the contents of the SRC directory (but not the directory
//    itself) into the DST directory. Ensure this works whether DST has a
//    trailing path separator or not.
func (s *DockerSuite) TestCpToCaseJ(c *check.C) {
	testRequires(c, DaemonIsLinux)
	containerID := makeTestContainer(c, testContainerOptions{
		addContent: true, workDir: "/root",
		command: makeCatFileCommand("/dir2/file1-1"),
	})

	tmpDir := getTestDir(c, "test-cp-to-case-j")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	srcDir := cpPathTrailingSep(tmpDir, "dir1") + "."
	dstDir := containerCpPath(containerID, "/dir2")

	// Ensure that dstPath doesn't exist.
	c.Assert(containerStartOutputEquals(c, containerID, ""), checker.IsNil)

	c.Assert(runDockerCp(c, srcDir, dstDir, nil), checker.IsNil)

	// Should now contain file1-1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1-1\n"), checker.IsNil)

	// Now try again but using a trailing path separator for dstDir.

	// Make new destination container.
	containerID = makeTestContainer(c, testContainerOptions{
		command: makeCatFileCommand("/dir2/file1-1"),
	})

	dstDir = containerCpPathTrailingSep(containerID, "/dir2")

	// Ensure that dstPath doesn't exist.
	c.Assert(containerStartOutputEquals(c, containerID, ""), checker.IsNil)

	c.Assert(runDockerCp(c, srcDir, dstDir, nil), checker.IsNil)

	// Should now contain file1-1's contents.
	c.Assert(containerStartOutputEquals(c, containerID, "file1-1\n"), checker.IsNil)
}

// The `docker cp` command should also ensure that you cannot
// write to a container rootfs that is marked as read-only.
func (s *DockerSuite) TestCpToErrReadOnlyRootfs(c *check.C) {
	// --read-only + userns has remount issues
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	tmpDir := getTestDir(c, "test-cp-to-err-read-only-rootfs")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	containerID := makeTestContainer(c, testContainerOptions{
		readOnly: true, workDir: "/root",
		command: makeCatFileCommand("shouldNotExist"),
	})

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(containerID, "/root/shouldNotExist")

	err := runDockerCp(c, srcPath, dstPath, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpCannotCopyReadOnly(err), checker.True, check.Commentf("expected ErrContainerRootfsReadonly error, but got %T: %s", err, err))

	// Ensure that dstPath doesn't exist.
	c.Assert(containerStartOutputEquals(c, containerID, ""), checker.IsNil)
}

// The `docker cp` command should also ensure that you
// cannot write to a volume that is mounted as read-only.
func (s *DockerSuite) TestCpToErrReadOnlyVolume(c *check.C) {
	// --read-only + userns has remount issues
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	tmpDir := getTestDir(c, "test-cp-to-err-read-only-volume")
	defer os.RemoveAll(tmpDir)

	makeTestContentInDir(c, tmpDir)

	containerID := makeTestContainer(c, testContainerOptions{
		volumes: defaultVolumes(tmpDir), workDir: "/root",
		command: makeCatFileCommand("/vol_ro/shouldNotExist"),
	})

	srcPath := cpPath(tmpDir, "file1")
	dstPath := containerCpPath(containerID, "/vol_ro/shouldNotExist")

	err := runDockerCp(c, srcPath, dstPath, nil)
	c.Assert(err, checker.NotNil)

	c.Assert(isCpCannotCopyReadOnly(err), checker.True, check.Commentf("expected ErrVolumeReadonly error, but got %T: %s", err, err))

	// Ensure that dstPath doesn't exist.
	c.Assert(containerStartOutputEquals(c, containerID, ""), checker.IsNil)
}
