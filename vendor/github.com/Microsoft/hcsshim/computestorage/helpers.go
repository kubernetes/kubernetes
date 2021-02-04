package computestorage

import (
	"context"
	"os"
	"path/filepath"
	"syscall"

	"github.com/Microsoft/go-winio/pkg/security"
	"github.com/Microsoft/go-winio/vhd"
	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
)

const defaultVHDXBlockSizeInMB = 1

// SetupContainerBaseLayer is a helper to setup a containers scratch. It
// will create and format the vhdx's inside and the size is configurable with the sizeInGB
// parameter.
//
// `layerPath` is the path to the base container layer on disk.
//
// `baseVhdPath` is the path to where the base vhdx for the base layer should be created.
//
// `diffVhdPath` is the path where the differencing disk for the base layer should be created.
//
// `sizeInGB` is the size in gigabytes to make the base vhdx.
func SetupContainerBaseLayer(ctx context.Context, layerPath, baseVhdPath, diffVhdPath string, sizeInGB uint64) (err error) {
	var (
		hivesPath  = filepath.Join(layerPath, "Hives")
		layoutPath = filepath.Join(layerPath, "Layout")
	)

	// We need to remove the hives directory and layout file as `SetupBaseOSLayer` fails if these files
	// already exist. `SetupBaseOSLayer` will create these files internally. We also remove the base and
	// differencing disks if they exist in case we're asking for a different size.
	if _, err := os.Stat(hivesPath); err == nil {
		if err := os.RemoveAll(hivesPath); err != nil {
			return errors.Wrap(err, "failed to remove prexisting hives directory")
		}
	}
	if _, err := os.Stat(layoutPath); err == nil {
		if err := os.RemoveAll(layoutPath); err != nil {
			return errors.Wrap(err, "failed to remove prexisting layout file")
		}
	}

	if _, err := os.Stat(baseVhdPath); err == nil {
		if err := os.RemoveAll(baseVhdPath); err != nil {
			return errors.Wrap(err, "failed to remove base vhdx path")
		}
	}
	if _, err := os.Stat(diffVhdPath); err == nil {
		if err := os.RemoveAll(diffVhdPath); err != nil {
			return errors.Wrap(err, "failed to remove differencing vhdx")
		}
	}

	createParams := &vhd.CreateVirtualDiskParameters{
		Version: 2,
		Version2: vhd.CreateVersion2{
			MaximumSize:      sizeInGB * 1024 * 1024 * 1024,
			BlockSizeInBytes: defaultVHDXBlockSizeInMB * 1024 * 1024,
		},
	}
	handle, err := vhd.CreateVirtualDisk(baseVhdPath, vhd.VirtualDiskAccessNone, vhd.CreateVirtualDiskFlagNone, createParams)
	if err != nil {
		return errors.Wrap(err, "failed to create vhdx")
	}

	defer func() {
		if err != nil {
			syscall.CloseHandle(handle)
			os.RemoveAll(baseVhdPath)
			if os.Stat(diffVhdPath); err == nil {
				os.RemoveAll(diffVhdPath)
			}
		}
	}()

	if err = FormatWritableLayerVhd(ctx, windows.Handle(handle)); err != nil {
		return err
	}
	// Base vhd handle must be closed before calling SetupBaseLayer in case of Container layer
	if err = syscall.CloseHandle(handle); err != nil {
		return errors.Wrap(err, "failed to close vhdx handle")
	}

	options := OsLayerOptions{
		Type: OsLayerTypeContainer,
	}

	// SetupBaseOSLayer expects an empty vhd handle for a container layer and will
	// error out otherwise.
	if err = SetupBaseOSLayer(ctx, layerPath, 0, options); err != nil {
		return err
	}
	// Create the differencing disk that will be what's copied for the final rw layer
	// for a container.
	if err = vhd.CreateDiffVhd(diffVhdPath, baseVhdPath, defaultVHDXBlockSizeInMB); err != nil {
		return errors.Wrap(err, "failed to create differencing disk")
	}

	if err = security.GrantVmGroupAccess(baseVhdPath); err != nil {
		return errors.Wrapf(err, "failed to grant vm group access to %s", baseVhdPath)
	}
	if err = security.GrantVmGroupAccess(diffVhdPath); err != nil {
		return errors.Wrapf(err, "failed to grant vm group access to %s", diffVhdPath)
	}
	return nil
}

// SetupUtilityVMBaseLayer is a helper to setup a UVMs scratch space. It will create and format
// the vhdx inside and the size is configurable by the sizeInGB parameter.
//
// `uvmPath` is the path to the UtilityVM filesystem.
//
// `baseVhdPath` is the path to where the base vhdx for the UVM should be created.
//
// `diffVhdPath` is the path where the differencing disk for the UVM should be created.
//
// `sizeInGB` specifies the size in gigabytes to make the base vhdx.
func SetupUtilityVMBaseLayer(ctx context.Context, uvmPath, baseVhdPath, diffVhdPath string, sizeInGB uint64) (err error) {
	// Remove the base and differencing disks if they exist in case we're asking for a different size.
	if _, err := os.Stat(baseVhdPath); err == nil {
		if err := os.RemoveAll(baseVhdPath); err != nil {
			return errors.Wrap(err, "failed to remove base vhdx")
		}
	}
	if _, err := os.Stat(diffVhdPath); err == nil {
		if err := os.RemoveAll(diffVhdPath); err != nil {
			return errors.Wrap(err, "failed to remove differencing vhdx")
		}
	}

	// Just create the vhdx for utilityVM layer, no need to format it.
	createParams := &vhd.CreateVirtualDiskParameters{
		Version: 2,
		Version2: vhd.CreateVersion2{
			MaximumSize:      sizeInGB * 1024 * 1024 * 1024,
			BlockSizeInBytes: defaultVHDXBlockSizeInMB * 1024 * 1024,
		},
	}
	handle, err := vhd.CreateVirtualDisk(baseVhdPath, vhd.VirtualDiskAccessNone, vhd.CreateVirtualDiskFlagNone, createParams)
	if err != nil {
		return errors.Wrap(err, "failed to create vhdx")
	}

	defer func() {
		if err != nil {
			syscall.CloseHandle(handle)
			os.RemoveAll(baseVhdPath)
			if os.Stat(diffVhdPath); err == nil {
				os.RemoveAll(diffVhdPath)
			}
		}
	}()

	// If it is a UtilityVM layer then the base vhdx must be attached when calling
	// `SetupBaseOSLayer`
	attachParams := &vhd.AttachVirtualDiskParameters{
		Version: 2,
	}
	if err := vhd.AttachVirtualDisk(handle, vhd.AttachVirtualDiskFlagNone, attachParams); err != nil {
		return errors.Wrapf(err, "failed to attach virtual disk")
	}

	options := OsLayerOptions{
		Type: OsLayerTypeVM,
	}
	if err := SetupBaseOSLayer(ctx, uvmPath, windows.Handle(handle), options); err != nil {
		return err
	}

	// Detach and close the handle after setting up the layer as we don't need the handle
	// for anything else and we no longer need to be attached either.
	if err = vhd.DetachVirtualDisk(handle); err != nil {
		return errors.Wrap(err, "failed to detach vhdx")
	}
	if err = syscall.CloseHandle(handle); err != nil {
		return errors.Wrap(err, "failed to close vhdx handle")
	}

	// Create the differencing disk that will be what's copied for the final rw layer
	// for a container.
	if err = vhd.CreateDiffVhd(diffVhdPath, baseVhdPath, defaultVHDXBlockSizeInMB); err != nil {
		return errors.Wrap(err, "failed to create differencing disk")
	}

	if err := security.GrantVmGroupAccess(baseVhdPath); err != nil {
		return errors.Wrapf(err, "failed to grant vm group access to %s", baseVhdPath)
	}
	if err := security.GrantVmGroupAccess(diffVhdPath); err != nil {
		return errors.Wrapf(err, "failed to grant vm group access to %s", diffVhdPath)
	}
	return nil
}
