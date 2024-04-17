package dmz

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/system"
)

type SealFunc func(**os.File) error

var (
	_ SealFunc = sealMemfd
	_ SealFunc = sealFile
)

func isExecutable(f *os.File) bool {
	if err := unix.Faccessat(int(f.Fd()), "", unix.X_OK, unix.AT_EACCESS|unix.AT_EMPTY_PATH); err == nil {
		return true
	} else if err == unix.EACCES {
		return false
	}
	path := "/proc/self/fd/" + strconv.Itoa(int(f.Fd()))
	if err := unix.Access(path, unix.X_OK); err == nil {
		return true
	} else if err == unix.EACCES {
		return false
	}
	// Cannot check -- assume it's executable (if not, exec will fail).
	logrus.Debugf("cannot do X_OK check on binary %s -- assuming it's executable", f.Name())
	return true
}

const baseMemfdSeals = unix.F_SEAL_SEAL | unix.F_SEAL_SHRINK | unix.F_SEAL_GROW | unix.F_SEAL_WRITE

func sealMemfd(f **os.File) error {
	if err := (*f).Chmod(0o511); err != nil {
		return err
	}
	// Try to set the newer memfd sealing flags, but we ignore
	// errors because they are not needed and we want to continue
	// to work on older kernels.
	fd := (*f).Fd()
	// F_SEAL_FUTURE_WRITE -- Linux 5.1
	_, _ = unix.FcntlInt(fd, unix.F_ADD_SEALS, unix.F_SEAL_FUTURE_WRITE)
	// F_SEAL_EXEC -- Linux 6.3
	const F_SEAL_EXEC = 0x20 //nolint:revive // this matches the unix.* name
	_, _ = unix.FcntlInt(fd, unix.F_ADD_SEALS, F_SEAL_EXEC)
	// Apply all original memfd seals.
	_, err := unix.FcntlInt(fd, unix.F_ADD_SEALS, baseMemfdSeals)
	return os.NewSyscallError("fcntl(F_ADD_SEALS)", err)
}

// Memfd creates a sealable executable memfd (supported since Linux 3.17).
func Memfd(comment string) (*os.File, SealFunc, error) {
	file, err := system.ExecutableMemfd("runc_cloned:"+comment, unix.MFD_ALLOW_SEALING|unix.MFD_CLOEXEC)
	return file, sealMemfd, err
}

func sealFile(f **os.File) error {
	if err := (*f).Chmod(0o511); err != nil {
		return err
	}
	// When sealing an O_TMPFILE-style descriptor we need to
	// re-open the path as O_PATH to clear the existing write
	// handle we have.
	opath, err := os.OpenFile(fmt.Sprintf("/proc/self/fd/%d", (*f).Fd()), unix.O_PATH|unix.O_CLOEXEC, 0)
	if err != nil {
		return fmt.Errorf("reopen tmpfile: %w", err)
	}
	_ = (*f).Close()
	*f = opath
	return nil
}

// otmpfile creates an open(O_TMPFILE) file in the given directory (supported
// since Linux 3.11).
func otmpfile(dir string) (*os.File, SealFunc, error) {
	file, err := os.OpenFile(dir, unix.O_TMPFILE|unix.O_RDWR|unix.O_EXCL|unix.O_CLOEXEC, 0o700)
	if err != nil {
		return nil, nil, fmt.Errorf("O_TMPFILE creation failed: %w", err)
	}
	// Make sure we actually got an unlinked O_TMPFILE descriptor.
	var stat unix.Stat_t
	if err := unix.Fstat(int(file.Fd()), &stat); err != nil {
		file.Close()
		return nil, nil, fmt.Errorf("cannot fstat O_TMPFILE fd: %w", err)
	} else if stat.Nlink != 0 {
		file.Close()
		return nil, nil, errors.New("O_TMPFILE has non-zero nlink")
	}
	return file, sealFile, err
}

// mktemp creates a classic unlinked file in the given directory.
func mktemp(dir string) (*os.File, SealFunc, error) {
	file, err := os.CreateTemp(dir, "runc.")
	if err != nil {
		return nil, nil, err
	}
	// Unlink the file and verify it was unlinked.
	if err := os.Remove(file.Name()); err != nil {
		return nil, nil, fmt.Errorf("unlinking classic tmpfile: %w", err)
	}
	var stat unix.Stat_t
	if err := unix.Fstat(int(file.Fd()), &stat); err != nil {
		return nil, nil, fmt.Errorf("cannot fstat classic tmpfile: %w", err)
	} else if stat.Nlink != 0 {
		return nil, nil, fmt.Errorf("classic tmpfile %s has non-zero nlink after unlink", file.Name())
	}
	return file, sealFile, err
}

func getSealableFile(comment, tmpDir string) (file *os.File, sealFn SealFunc, err error) {
	// First, try an executable memfd (supported since Linux 3.17).
	file, sealFn, err = Memfd(comment)
	if err == nil {
		return
	}
	logrus.Debugf("memfd cloned binary failed, falling back to O_TMPFILE: %v", err)

	// The tmpDir here (c.root) might be mounted noexec, so we need a couple of
	// fallbacks to try. It's possible that none of these are writable and
	// executable, in which case there's nothing we can practically do (other
	// than mounting our own executable tmpfs, which would have its own
	// issues).
	tmpDirs := []string{
		tmpDir,
		os.TempDir(),
		"/tmp",
		".",
		"/bin",
		"/",
	}

	// Try to fallback to O_TMPFILE (supported since Linux 3.11).
	for _, dir := range tmpDirs {
		file, sealFn, err = otmpfile(dir)
		if err != nil {
			continue
		}
		if !isExecutable(file) {
			logrus.Debugf("tmpdir %s is noexec -- trying a different tmpdir", dir)
			file.Close()
			continue
		}
		return
	}
	logrus.Debugf("O_TMPFILE cloned binary failed, falling back to mktemp(): %v", err)
	// Finally, try a classic unlinked temporary file.
	for _, dir := range tmpDirs {
		file, sealFn, err = mktemp(dir)
		if err != nil {
			continue
		}
		if !isExecutable(file) {
			logrus.Debugf("tmpdir %s is noexec -- trying a different tmpdir", dir)
			file.Close()
			continue
		}
		return
	}
	return nil, nil, fmt.Errorf("could not create sealable file for cloned binary: %w", err)
}

// CloneBinary creates a "sealed" clone of a given binary, which can be used to
// thwart attempts by the container process to gain access to host binaries
// through procfs magic-link shenanigans. For more details on why this is
// necessary, see CVE-2019-5736.
func CloneBinary(src io.Reader, size int64, name, tmpDir string) (*os.File, error) {
	logrus.Debugf("cloning %s binary (%d bytes)", name, size)
	file, sealFn, err := getSealableFile(name, tmpDir)
	if err != nil {
		return nil, err
	}
	copied, err := system.Copy(file, src)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("copy binary: %w", err)
	} else if copied != size {
		file.Close()
		return nil, fmt.Errorf("copied binary size mismatch: %d != %d", copied, size)
	}
	if err := sealFn(&file); err != nil {
		file.Close()
		return nil, fmt.Errorf("could not seal fd: %w", err)
	}
	return file, nil
}

// IsCloned returns whether the given file can be guaranteed to be a safe exe.
func IsCloned(exe *os.File) bool {
	seals, err := unix.FcntlInt(exe.Fd(), unix.F_GET_SEALS, 0)
	if err != nil {
		// /proc/self/exe is probably not a memfd
		logrus.Debugf("F_GET_SEALS on %s failed: %v", exe.Name(), err)
		return false
	}
	// The memfd must have all of the base seals applied.
	logrus.Debugf("checking %s memfd seals: 0x%x", exe.Name(), seals)
	return seals&baseMemfdSeals == baseMemfdSeals
}

// CloneSelfExe makes a clone of the current process's binary (through
// /proc/self/exe). This binary can then be used for "runc init" in order to
// make sure the container process can never resolve the original runc binary.
// For more details on why this is necessary, see CVE-2019-5736.
func CloneSelfExe(tmpDir string) (*os.File, error) {
	selfExe, err := os.Open("/proc/self/exe")
	if err != nil {
		return nil, fmt.Errorf("opening current binary: %w", err)
	}
	defer selfExe.Close()

	stat, err := selfExe.Stat()
	if err != nil {
		return nil, fmt.Errorf("checking /proc/self/exe size: %w", err)
	}
	size := stat.Size()

	return CloneBinary(selfExe, size, "/proc/self/exe", tmpDir)
}

// IsSelfExeCloned returns whether /proc/self/exe is a cloned binary that can
// be guaranteed to be safe. This means that it must be a sealed memfd. Other
// types of clones cannot be completely verified as safe.
func IsSelfExeCloned() bool {
	selfExe, err := os.Open("/proc/self/exe")
	if err != nil {
		logrus.Debugf("open /proc/self/exe failed: %v", err)
		return false
	}
	defer selfExe.Close()
	return IsCloned(selfExe)
}
