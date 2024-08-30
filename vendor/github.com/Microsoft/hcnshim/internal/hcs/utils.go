//go:build windows

package hcs

import (
	"context"
	"io"
	"syscall"

	"github.com/Microsoft/go-winio"
	diskutil "github.com/Microsoft/go-winio/vhd"
	"github.com/Microsoft/hcnshim/computestorage"
	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
)

// makeOpenFiles calls winio.NewOpenFile for each handle in a slice but closes all the handles
// if there is an error.
func makeOpenFiles(hs []syscall.Handle) (_ []io.ReadWriteCloser, err error) {
	fs := make([]io.ReadWriteCloser, len(hs))
	for i, h := range hs {
		if h != syscall.Handle(0) {
			if err == nil {
				fs[i], err = winio.NewOpenFile(windows.Handle(h))
			}
			if err != nil {
				syscall.Close(h)
			}
		}
	}
	if err != nil {
		for _, f := range fs {
			if f != nil {
				f.Close()
			}
		}
		return nil, err
	}
	return fs, nil
}

// CreateNTFSVHD creates a VHD formatted with NTFS of size `sizeGB` at the given `vhdPath`.
func CreateNTFSVHD(ctx context.Context, vhdPath string, sizeGB uint32) (err error) {
	if err := diskutil.CreateVhdx(vhdPath, sizeGB, 1); err != nil {
		return errors.Wrap(err, "failed to create VHD")
	}

	vhd, err := diskutil.OpenVirtualDisk(vhdPath, diskutil.VirtualDiskAccessNone, diskutil.OpenVirtualDiskFlagNone)
	if err != nil {
		return errors.Wrap(err, "failed to open VHD")
	}
	defer func() {
		err2 := windows.CloseHandle(windows.Handle(vhd))
		if err == nil {
			err = errors.Wrap(err2, "failed to close VHD")
		}
	}()

	if err := computestorage.FormatWritableLayerVhd(ctx, windows.Handle(vhd)); err != nil {
		return errors.Wrap(err, "failed to format VHD")
	}

	return nil
}
