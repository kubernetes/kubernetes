// +build windows

package lcow

import (
	"bytes"
	"fmt"
	"io"
	"runtime"
	"strings"
	"sync"

	"github.com/Microsoft/hcsshim"
	"github.com/Microsoft/opengcs/service/gcsutils/remotefs"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/containerfs"
	"github.com/sirupsen/logrus"
)

type lcowfs struct {
	root        string
	d           *Driver
	mappedDisks []hcsshim.MappedVirtualDisk
	vmID        string
	currentSVM  *serviceVM
	sync.Mutex
}

var _ containerfs.ContainerFS = &lcowfs{}

// ErrNotSupported is an error for unsupported operations in the remotefs
var ErrNotSupported = fmt.Errorf("not supported")

// Functions to implement the ContainerFS interface
func (l *lcowfs) Path() string {
	return l.root
}

func (l *lcowfs) ResolveScopedPath(path string, rawPath bool) (string, error) {
	logrus.Debugf("remotefs.resolvescopedpath inputs: %s %s ", path, l.root)

	arg1 := l.Join(l.root, path)
	if !rawPath {
		// The l.Join("/", path) will make path an absolute path and then clean it
		// so if path = ../../X, it will become /X.
		arg1 = l.Join(l.root, l.Join("/", path))
	}
	arg2 := l.root

	output := &bytes.Buffer{}
	if err := l.runRemoteFSProcess(nil, output, remotefs.ResolvePathCmd, arg1, arg2); err != nil {
		return "", err
	}

	logrus.Debugf("remotefs.resolvescopedpath success. Output: %s\n", output.String())
	return output.String(), nil
}

func (l *lcowfs) OS() string {
	return "linux"
}

func (l *lcowfs) Architecture() string {
	return runtime.GOARCH
}

// Other functions that are used by docker like the daemon Archiver/Extractor
func (l *lcowfs) ExtractArchive(src io.Reader, dst string, opts *archive.TarOptions) error {
	logrus.Debugf("remotefs.ExtractArchve inputs: %s %+v", dst, opts)

	tarBuf := &bytes.Buffer{}
	if err := remotefs.WriteTarOptions(tarBuf, opts); err != nil {
		return fmt.Errorf("failed to marshall tar opts: %s", err)
	}

	input := io.MultiReader(tarBuf, src)
	if err := l.runRemoteFSProcess(input, nil, remotefs.ExtractArchiveCmd, dst); err != nil {
		return fmt.Errorf("failed to extract archive to %s: %s", dst, err)
	}
	return nil
}

func (l *lcowfs) ArchivePath(src string, opts *archive.TarOptions) (io.ReadCloser, error) {
	logrus.Debugf("remotefs.ArchivePath: %s %+v", src, opts)

	tarBuf := &bytes.Buffer{}
	if err := remotefs.WriteTarOptions(tarBuf, opts); err != nil {
		return nil, fmt.Errorf("failed to marshall tar opts: %s", err)
	}

	r, w := io.Pipe()
	go func() {
		defer w.Close()
		if err := l.runRemoteFSProcess(tarBuf, w, remotefs.ArchivePathCmd, src); err != nil {
			logrus.Debugf("REMOTEFS: Failed to extract archive: %s %+v %s", src, opts, err)
		}
	}()
	return r, nil
}

// Helper functions
func (l *lcowfs) startVM() error {
	l.Lock()
	defer l.Unlock()
	if l.currentSVM != nil {
		return nil
	}

	svm, err := l.d.startServiceVMIfNotRunning(l.vmID, l.mappedDisks, fmt.Sprintf("lcowfs.startVM"))
	if err != nil {
		return err
	}

	if err = svm.createUnionMount(l.root, l.mappedDisks...); err != nil {
		return err
	}
	l.currentSVM = svm
	return nil
}

func (l *lcowfs) runRemoteFSProcess(stdin io.Reader, stdout io.Writer, args ...string) error {
	if err := l.startVM(); err != nil {
		return err
	}

	// Append remotefs prefix and setup as a command line string
	cmd := fmt.Sprintf("%s %s", remotefs.RemotefsCmd, strings.Join(args, " "))
	stderr := &bytes.Buffer{}
	if err := l.currentSVM.runProcess(cmd, stdin, stdout, stderr); err != nil {
		return err
	}

	eerr, err := remotefs.ReadError(stderr)
	if eerr != nil {
		// Process returned an error so return that.
		return remotefs.ExportedToError(eerr)
	}
	return err
}
