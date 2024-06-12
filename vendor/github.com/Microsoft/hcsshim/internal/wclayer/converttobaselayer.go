//go:build windows

package wclayer

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/longpath"
	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/Microsoft/hcsshim/internal/safefile"
	"github.com/Microsoft/hcsshim/internal/winapi"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"
	"golang.org/x/sys/windows"
)

var hiveNames = []string{"DEFAULT", "SAM", "SECURITY", "SOFTWARE", "SYSTEM"}

// Ensure the given file exists as an ordinary file, and create a minimal hive file if not.
func ensureHive(path string, root *os.File) (err error) {
	_, err = safefile.LstatRelative(path, root)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("accessing %s: %w", path, err)
	}

	version := windows.RtlGetVersion()
	if version == nil {
		return fmt.Errorf("failed to get OS version")
	}

	var fullPath string
	fullPath, err = longpath.LongAbs(filepath.Join(root.Name(), path))
	if err != nil {
		return fmt.Errorf("getting path: %w", err)
	}

	var key winapi.ORHKey
	err = winapi.ORCreateHive(&key)
	if err != nil {
		return fmt.Errorf("creating hive: %w", err)
	}

	defer func() {
		closeErr := winapi.ORCloseHive(key)
		if closeErr != nil && err == nil {
			err = fmt.Errorf("closing hive key: %w", closeErr)
		}
	}()

	err = winapi.ORSaveHive(key, fullPath, version.MajorVersion, version.MinorVersion)
	if err != nil {
		return fmt.Errorf("saving hive: %w", err)
	}

	return nil
}

func ensureBaseLayer(root *os.File) (hasUtilityVM bool, err error) {
	// The base layer registry hives will be copied from here
	const hiveSourcePath = "Files\\Windows\\System32\\config"
	if err = safefile.MkdirAllRelative(hiveSourcePath, root); err != nil {
		return
	}

	for _, hiveName := range hiveNames {
		hivePath := filepath.Join(hiveSourcePath, hiveName)
		if err = ensureHive(hivePath, root); err != nil {
			return
		}
	}

	stat, err := safefile.LstatRelative(UtilityVMFilesPath, root)

	if os.IsNotExist(err) {
		return false, nil
	}

	if err != nil {
		return
	}

	if !stat.Mode().IsDir() {
		fullPath := filepath.Join(root.Name(), UtilityVMFilesPath)
		return false, errors.Errorf("%s has unexpected file mode %s", fullPath, stat.Mode().String())
	}

	const bcdRelativePath = "EFI\\Microsoft\\Boot\\BCD"

	// Just check that this exists as a regular file. If it exists but is not a valid registry hive,
	// ProcessUtilityVMImage will complain:
	// "The registry could not read in, or write out, or flush, one of the files that contain the system's image of the registry."
	bcdPath := filepath.Join(UtilityVMFilesPath, bcdRelativePath)

	stat, err = safefile.LstatRelative(bcdPath, root)
	if err != nil {
		return false, errors.Wrapf(err, "UtilityVM must contain '%s'", bcdRelativePath)
	}

	if !stat.Mode().IsRegular() {
		fullPath := filepath.Join(root.Name(), bcdPath)
		return false, errors.Errorf("%s has unexpected file mode %s", fullPath, stat.Mode().String())
	}

	return true, nil
}

func convertToBaseLayer(ctx context.Context, root *os.File) error {
	hasUtilityVM, err := ensureBaseLayer(root)

	if err != nil {
		return err
	}

	if err := ProcessBaseLayer(ctx, root.Name()); err != nil {
		return err
	}

	if !hasUtilityVM {
		return nil
	}

	err = safefile.EnsureNotReparsePointRelative(UtilityVMPath, root)
	if err != nil {
		return err
	}

	utilityVMPath := filepath.Join(root.Name(), UtilityVMPath)
	return ProcessUtilityVMImage(ctx, utilityVMPath)
}

// ConvertToBaseLayer processes a candidate base layer, i.e. a directory
// containing the desired file content under Files/, and optionally the
// desired file content for a UtilityVM under UtilityVM/Files/
func ConvertToBaseLayer(ctx context.Context, path string) (err error) {
	title := "hcsshim::ConvertToBaseLayer"
	ctx, span := trace.StartSpan(ctx, title)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("path", path))

	root, err := safefile.OpenRoot(path)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	defer func() {
		if err2 := root.Close(); err == nil && err2 != nil {
			err = hcserror.New(err2, title+" - failed", "")
		}
	}()

	if err = convertToBaseLayer(ctx, root); err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
