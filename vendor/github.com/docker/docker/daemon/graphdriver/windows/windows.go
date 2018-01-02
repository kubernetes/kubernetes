//+build windows

package windows

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/go-winio/archive/tar"
	"github.com/Microsoft/go-winio/backuptar"
	"github.com/Microsoft/hcsshim"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/longpath"
	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/docker/pkg/system"
	units "github.com/docker/go-units"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/windows"
)

// filterDriver is an HCSShim driver type for the Windows Filter driver.
const filterDriver = 1

var (
	// mutatedFiles is a list of files that are mutated by the import process
	// and must be backed up and restored.
	mutatedFiles = map[string]string{
		"UtilityVM/Files/EFI/Microsoft/Boot/BCD":      "bcd.bak",
		"UtilityVM/Files/EFI/Microsoft/Boot/BCD.LOG":  "bcd.log.bak",
		"UtilityVM/Files/EFI/Microsoft/Boot/BCD.LOG1": "bcd.log1.bak",
		"UtilityVM/Files/EFI/Microsoft/Boot/BCD.LOG2": "bcd.log2.bak",
	}
	noreexec = false
)

// init registers the windows graph drivers to the register.
func init() {
	graphdriver.Register("windowsfilter", InitFilter)
	// DOCKER_WINDOWSFILTER_NOREEXEC allows for inline processing which makes
	// debugging issues in the re-exec codepath significantly easier.
	if os.Getenv("DOCKER_WINDOWSFILTER_NOREEXEC") != "" {
		logrus.Warnf("WindowsGraphDriver is set to not re-exec. This is intended for debugging purposes only.")
		noreexec = true
	} else {
		reexec.Register("docker-windows-write-layer", writeLayerReexec)
	}
}

type checker struct {
}

func (c *checker) IsMounted(path string) bool {
	return false
}

// Driver represents a windows graph driver.
type Driver struct {
	// info stores the shim driver information
	info hcsshim.DriverInfo
	ctr  *graphdriver.RefCounter
	// it is safe for windows to use a cache here because it does not support
	// restoring containers when the daemon dies.
	cacheMu sync.Mutex
	cache   map[string]string
}

// InitFilter returns a new Windows storage filter driver.
func InitFilter(home string, options []string, uidMaps, gidMaps []idtools.IDMap) (graphdriver.Driver, error) {
	logrus.Debugf("WindowsGraphDriver InitFilter at %s", home)

	fsType, err := getFileSystemType(string(home[0]))
	if err != nil {
		return nil, err
	}
	if strings.ToLower(fsType) == "refs" {
		return nil, fmt.Errorf("%s is on an ReFS volume - ReFS volumes are not supported", home)
	}

	if err := idtools.MkdirAllAs(home, 0700, 0, 0); err != nil {
		return nil, fmt.Errorf("windowsfilter failed to create '%s': %v", home, err)
	}

	d := &Driver{
		info: hcsshim.DriverInfo{
			HomeDir: home,
			Flavour: filterDriver,
		},
		cache: make(map[string]string),
		ctr:   graphdriver.NewRefCounter(&checker{}),
	}
	return d, nil
}

// win32FromHresult is a helper function to get the win32 error code from an HRESULT
func win32FromHresult(hr uintptr) uintptr {
	if hr&0x1fff0000 == 0x00070000 {
		return hr & 0xffff
	}
	return hr
}

// getFileSystemType obtains the type of a file system through GetVolumeInformation
// https://msdn.microsoft.com/en-us/library/windows/desktop/aa364993(v=vs.85).aspx
func getFileSystemType(drive string) (fsType string, hr error) {
	var (
		modkernel32              = windows.NewLazySystemDLL("kernel32.dll")
		procGetVolumeInformation = modkernel32.NewProc("GetVolumeInformationW")
		buf                      = make([]uint16, 255)
		size                     = windows.MAX_PATH + 1
	)
	if len(drive) != 1 {
		hr = errors.New("getFileSystemType must be called with a drive letter")
		return
	}
	drive += `:\`
	n := uintptr(unsafe.Pointer(nil))
	r0, _, _ := syscall.Syscall9(procGetVolumeInformation.Addr(), 8, uintptr(unsafe.Pointer(windows.StringToUTF16Ptr(drive))), n, n, n, n, n, uintptr(unsafe.Pointer(&buf[0])), uintptr(size), 0)
	if int32(r0) < 0 {
		hr = syscall.Errno(win32FromHresult(r0))
	}
	fsType = windows.UTF16ToString(buf)
	return
}

// String returns the string representation of a driver. This should match
// the name the graph driver has been registered with.
func (d *Driver) String() string {
	return "windowsfilter"
}

// Status returns the status of the driver.
func (d *Driver) Status() [][2]string {
	return [][2]string{
		{"Windows", ""},
	}
}

// panicIfUsedByLcow does exactly what it says.
// TODO @jhowardmsft - this is a temporary measure for the bring-up of
// Linux containers on Windows. It is a failsafe to ensure that the right
// graphdriver is used.
func panicIfUsedByLcow() {
	if system.LCOWSupported() {
		panic("inconsistency - windowsfilter graphdriver should not be used when in LCOW mode")
	}
}

// Exists returns true if the given id is registered with this driver.
func (d *Driver) Exists(id string) bool {
	panicIfUsedByLcow()
	rID, err := d.resolveID(id)
	if err != nil {
		return false
	}
	result, err := hcsshim.LayerExists(d.info, rID)
	if err != nil {
		return false
	}
	return result
}

// CreateReadWrite creates a layer that is writable for use as a container
// file system.
func (d *Driver) CreateReadWrite(id, parent string, opts *graphdriver.CreateOpts) error {
	panicIfUsedByLcow()
	if opts != nil {
		return d.create(id, parent, opts.MountLabel, false, opts.StorageOpt)
	}
	return d.create(id, parent, "", false, nil)
}

// Create creates a new read-only layer with the given id.
func (d *Driver) Create(id, parent string, opts *graphdriver.CreateOpts) error {
	panicIfUsedByLcow()
	if opts != nil {
		return d.create(id, parent, opts.MountLabel, true, opts.StorageOpt)
	}
	return d.create(id, parent, "", true, nil)
}

func (d *Driver) create(id, parent, mountLabel string, readOnly bool, storageOpt map[string]string) error {
	rPId, err := d.resolveID(parent)
	if err != nil {
		return err
	}

	parentChain, err := d.getLayerChain(rPId)
	if err != nil {
		return err
	}

	var layerChain []string

	if rPId != "" {
		parentPath, err := hcsshim.GetLayerMountPath(d.info, rPId)
		if err != nil {
			return err
		}
		if _, err := os.Stat(filepath.Join(parentPath, "Files")); err == nil {
			// This is a legitimate parent layer (not the empty "-init" layer),
			// so include it in the layer chain.
			layerChain = []string{parentPath}
		}
	}

	layerChain = append(layerChain, parentChain...)

	if readOnly {
		if err := hcsshim.CreateLayer(d.info, id, rPId); err != nil {
			return err
		}
	} else {
		var parentPath string
		if len(layerChain) != 0 {
			parentPath = layerChain[0]
		}

		if err := hcsshim.CreateSandboxLayer(d.info, id, parentPath, layerChain); err != nil {
			return err
		}

		storageOptions, err := parseStorageOpt(storageOpt)
		if err != nil {
			return fmt.Errorf("Failed to parse storage options - %s", err)
		}

		if storageOptions.size != 0 {
			if err := hcsshim.ExpandSandboxSize(d.info, id, storageOptions.size); err != nil {
				return err
			}
		}
	}

	if _, err := os.Lstat(d.dir(parent)); err != nil {
		if err2 := hcsshim.DestroyLayer(d.info, id); err2 != nil {
			logrus.Warnf("Failed to DestroyLayer %s: %s", id, err2)
		}
		return fmt.Errorf("Cannot create layer with missing parent %s: %s", parent, err)
	}

	if err := d.setLayerChain(id, layerChain); err != nil {
		if err2 := hcsshim.DestroyLayer(d.info, id); err2 != nil {
			logrus.Warnf("Failed to DestroyLayer %s: %s", id, err2)
		}
		return err
	}

	return nil
}

// dir returns the absolute path to the layer.
func (d *Driver) dir(id string) string {
	return filepath.Join(d.info.HomeDir, filepath.Base(id))
}

// Remove unmounts and removes the dir information.
func (d *Driver) Remove(id string) error {
	panicIfUsedByLcow()
	rID, err := d.resolveID(id)
	if err != nil {
		return err
	}

	// This retry loop is due to a bug in Windows (Internal bug #9432268)
	// if GetContainers fails with ErrVmcomputeOperationInvalidState
	// it is a transient error. Retry until it succeeds.
	var computeSystems []hcsshim.ContainerProperties
	retryCount := 0
	osv := system.GetOSVersion()
	for {
		// Get and terminate any template VMs that are currently using the layer.
		// Note: It is unfortunate that we end up in the graphdrivers Remove() call
		// for both containers and images, but the logic for template VMs is only
		// needed for images - specifically we are looking to see if a base layer
		// is in use by a template VM as a result of having started a Hyper-V
		// container at some point.
		//
		// We have a retry loop for ErrVmcomputeOperationInvalidState and
		// ErrVmcomputeOperationAccessIsDenied as there is a race condition
		// in RS1 and RS2 building during enumeration when a silo is going away
		// for example under it, in HCS. AccessIsDenied added to fix 30278.
		//
		// TODO @jhowardmsft - For RS3, we can remove the retries. Also consider
		// using platform APIs (if available) to get this more succinctly. Also
		// consider enhancing the Remove() interface to have context of why
		// the remove is being called - that could improve efficiency by not
		// enumerating compute systems during a remove of a container as it's
		// not required.
		computeSystems, err = hcsshim.GetContainers(hcsshim.ComputeSystemQuery{})
		if err != nil {
			if (osv.Build < 15139) &&
				((err == hcsshim.ErrVmcomputeOperationInvalidState) || (err == hcsshim.ErrVmcomputeOperationAccessIsDenied)) {
				if retryCount >= 500 {
					break
				}
				retryCount++
				time.Sleep(10 * time.Millisecond)
				continue
			}
			return err
		}
		break
	}

	for _, computeSystem := range computeSystems {
		if strings.Contains(computeSystem.RuntimeImagePath, id) && computeSystem.IsRuntimeTemplate {
			container, err := hcsshim.OpenContainer(computeSystem.ID)
			if err != nil {
				return err
			}
			defer container.Close()
			err = container.Terminate()
			if hcsshim.IsPending(err) {
				err = container.Wait()
			} else if hcsshim.IsAlreadyStopped(err) {
				err = nil
			}

			if err != nil {
				return err
			}
		}
	}

	layerPath := filepath.Join(d.info.HomeDir, rID)
	tmpID := fmt.Sprintf("%s-removing", rID)
	tmpLayerPath := filepath.Join(d.info.HomeDir, tmpID)
	if err := os.Rename(layerPath, tmpLayerPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	if err := hcsshim.DestroyLayer(d.info, tmpID); err != nil {
		logrus.Errorf("Failed to DestroyLayer %s: %s", id, err)
	}

	return nil
}

// Get returns the rootfs path for the id. This will mount the dir at its given path.
func (d *Driver) Get(id, mountLabel string) (string, error) {
	panicIfUsedByLcow()
	logrus.Debugf("WindowsGraphDriver Get() id %s mountLabel %s", id, mountLabel)
	var dir string

	rID, err := d.resolveID(id)
	if err != nil {
		return "", err
	}
	if count := d.ctr.Increment(rID); count > 1 {
		return d.cache[rID], nil
	}

	// Getting the layer paths must be done outside of the lock.
	layerChain, err := d.getLayerChain(rID)
	if err != nil {
		d.ctr.Decrement(rID)
		return "", err
	}

	if err := hcsshim.ActivateLayer(d.info, rID); err != nil {
		d.ctr.Decrement(rID)
		return "", err
	}
	if err := hcsshim.PrepareLayer(d.info, rID, layerChain); err != nil {
		d.ctr.Decrement(rID)
		if err2 := hcsshim.DeactivateLayer(d.info, rID); err2 != nil {
			logrus.Warnf("Failed to Deactivate %s: %s", id, err)
		}
		return "", err
	}

	mountPath, err := hcsshim.GetLayerMountPath(d.info, rID)
	if err != nil {
		d.ctr.Decrement(rID)
		if err := hcsshim.UnprepareLayer(d.info, rID); err != nil {
			logrus.Warnf("Failed to Unprepare %s: %s", id, err)
		}
		if err2 := hcsshim.DeactivateLayer(d.info, rID); err2 != nil {
			logrus.Warnf("Failed to Deactivate %s: %s", id, err)
		}
		return "", err
	}
	d.cacheMu.Lock()
	d.cache[rID] = mountPath
	d.cacheMu.Unlock()

	// If the layer has a mount path, use that. Otherwise, use the
	// folder path.
	if mountPath != "" {
		dir = mountPath
	} else {
		dir = d.dir(id)
	}

	return dir, nil
}

// Put adds a new layer to the driver.
func (d *Driver) Put(id string) error {
	panicIfUsedByLcow()
	logrus.Debugf("WindowsGraphDriver Put() id %s", id)

	rID, err := d.resolveID(id)
	if err != nil {
		return err
	}
	if count := d.ctr.Decrement(rID); count > 0 {
		return nil
	}
	d.cacheMu.Lock()
	_, exists := d.cache[rID]
	delete(d.cache, rID)
	d.cacheMu.Unlock()

	// If the cache was not populated, then the layer was left unprepared and deactivated
	if !exists {
		return nil
	}

	if err := hcsshim.UnprepareLayer(d.info, rID); err != nil {
		return err
	}
	return hcsshim.DeactivateLayer(d.info, rID)
}

// Cleanup ensures the information the driver stores is properly removed.
// We use this opportunity to cleanup any -removing folders which may be
// still left if the daemon was killed while it was removing a layer.
func (d *Driver) Cleanup() error {
	items, err := ioutil.ReadDir(d.info.HomeDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	// Note we don't return an error below - it's possible the files
	// are locked. However, next time around after the daemon exits,
	// we likely will be able to to cleanup successfully. Instead we log
	// warnings if there are errors.
	for _, item := range items {
		if item.IsDir() && strings.HasSuffix(item.Name(), "-removing") {
			if err := hcsshim.DestroyLayer(d.info, item.Name()); err != nil {
				logrus.Warnf("Failed to cleanup %s: %s", item.Name(), err)
			} else {
				logrus.Infof("Cleaned up %s", item.Name())
			}
		}
	}

	return nil
}

// Diff produces an archive of the changes between the specified
// layer and its parent layer which may be "".
// The layer should be mounted when calling this function
func (d *Driver) Diff(id, parent string) (_ io.ReadCloser, err error) {
	panicIfUsedByLcow()
	rID, err := d.resolveID(id)
	if err != nil {
		return
	}

	layerChain, err := d.getLayerChain(rID)
	if err != nil {
		return
	}

	// this is assuming that the layer is unmounted
	if err := hcsshim.UnprepareLayer(d.info, rID); err != nil {
		return nil, err
	}
	prepare := func() {
		if err := hcsshim.PrepareLayer(d.info, rID, layerChain); err != nil {
			logrus.Warnf("Failed to Deactivate %s: %s", rID, err)
		}
	}

	arch, err := d.exportLayer(rID, layerChain)
	if err != nil {
		prepare()
		return
	}
	return ioutils.NewReadCloserWrapper(arch, func() error {
		err := arch.Close()
		prepare()
		return err
	}), nil
}

// Changes produces a list of changes between the specified layer
// and its parent layer. If parent is "", then all changes will be ADD changes.
// The layer should not be mounted when calling this function.
func (d *Driver) Changes(id, parent string) ([]archive.Change, error) {
	panicIfUsedByLcow()
	rID, err := d.resolveID(id)
	if err != nil {
		return nil, err
	}
	parentChain, err := d.getLayerChain(rID)
	if err != nil {
		return nil, err
	}

	if err := hcsshim.ActivateLayer(d.info, rID); err != nil {
		return nil, err
	}
	defer func() {
		if err2 := hcsshim.DeactivateLayer(d.info, rID); err2 != nil {
			logrus.Errorf("changes() failed to DeactivateLayer %s %s: %s", id, rID, err2)
		}
	}()

	var changes []archive.Change
	err = winio.RunWithPrivilege(winio.SeBackupPrivilege, func() error {
		r, err := hcsshim.NewLayerReader(d.info, id, parentChain)
		if err != nil {
			return err
		}
		defer r.Close()

		for {
			name, _, fileInfo, err := r.Next()
			if err == io.EOF {
				return nil
			}
			if err != nil {
				return err
			}
			name = filepath.ToSlash(name)
			if fileInfo == nil {
				changes = append(changes, archive.Change{Path: name, Kind: archive.ChangeDelete})
			} else {
				// Currently there is no way to tell between an add and a modify.
				changes = append(changes, archive.Change{Path: name, Kind: archive.ChangeModify})
			}
		}
	})
	if err != nil {
		return nil, err
	}

	return changes, nil
}

// ApplyDiff extracts the changeset from the given diff into the
// layer with the specified id and parent, returning the size of the
// new layer in bytes.
// The layer should not be mounted when calling this function
func (d *Driver) ApplyDiff(id, parent string, diff io.Reader) (int64, error) {
	panicIfUsedByLcow()
	var layerChain []string
	if parent != "" {
		rPId, err := d.resolveID(parent)
		if err != nil {
			return 0, err
		}
		parentChain, err := d.getLayerChain(rPId)
		if err != nil {
			return 0, err
		}
		parentPath, err := hcsshim.GetLayerMountPath(d.info, rPId)
		if err != nil {
			return 0, err
		}
		layerChain = append(layerChain, parentPath)
		layerChain = append(layerChain, parentChain...)
	}

	size, err := d.importLayer(id, diff, layerChain)
	if err != nil {
		return 0, err
	}

	if err = d.setLayerChain(id, layerChain); err != nil {
		return 0, err
	}

	return size, nil
}

// DiffSize calculates the changes between the specified layer
// and its parent and returns the size in bytes of the changes
// relative to its base filesystem directory.
func (d *Driver) DiffSize(id, parent string) (size int64, err error) {
	panicIfUsedByLcow()
	rPId, err := d.resolveID(parent)
	if err != nil {
		return
	}

	changes, err := d.Changes(id, rPId)
	if err != nil {
		return
	}

	layerFs, err := d.Get(id, "")
	if err != nil {
		return
	}
	defer d.Put(id)

	return archive.ChangesSize(layerFs, changes), nil
}

// GetMetadata returns custom driver information.
func (d *Driver) GetMetadata(id string) (map[string]string, error) {
	panicIfUsedByLcow()
	m := make(map[string]string)
	m["dir"] = d.dir(id)
	return m, nil
}

func writeTarFromLayer(r hcsshim.LayerReader, w io.Writer) error {
	t := tar.NewWriter(w)
	for {
		name, size, fileInfo, err := r.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if fileInfo == nil {
			// Write a whiteout file.
			hdr := &tar.Header{
				Name: filepath.ToSlash(filepath.Join(filepath.Dir(name), archive.WhiteoutPrefix+filepath.Base(name))),
			}
			err := t.WriteHeader(hdr)
			if err != nil {
				return err
			}
		} else {
			err = backuptar.WriteTarFileFromBackupStream(t, r, name, size, fileInfo)
			if err != nil {
				return err
			}
		}
	}
	return t.Close()
}

// exportLayer generates an archive from a layer based on the given ID.
func (d *Driver) exportLayer(id string, parentLayerPaths []string) (io.ReadCloser, error) {
	archive, w := io.Pipe()
	go func() {
		err := winio.RunWithPrivilege(winio.SeBackupPrivilege, func() error {
			r, err := hcsshim.NewLayerReader(d.info, id, parentLayerPaths)
			if err != nil {
				return err
			}

			err = writeTarFromLayer(r, w)
			cerr := r.Close()
			if err == nil {
				err = cerr
			}
			return err
		})
		w.CloseWithError(err)
	}()

	return archive, nil
}

// writeBackupStreamFromTarAndSaveMutatedFiles reads data from a tar stream and
// writes it to a backup stream, and also saves any files that will be mutated
// by the import layer process to a backup location.
func writeBackupStreamFromTarAndSaveMutatedFiles(buf *bufio.Writer, w io.Writer, t *tar.Reader, hdr *tar.Header, root string) (nextHdr *tar.Header, err error) {
	var bcdBackup *os.File
	var bcdBackupWriter *winio.BackupFileWriter
	if backupPath, ok := mutatedFiles[hdr.Name]; ok {
		bcdBackup, err = os.Create(filepath.Join(root, backupPath))
		if err != nil {
			return nil, err
		}
		defer func() {
			cerr := bcdBackup.Close()
			if err == nil {
				err = cerr
			}
		}()

		bcdBackupWriter = winio.NewBackupFileWriter(bcdBackup, false)
		defer func() {
			cerr := bcdBackupWriter.Close()
			if err == nil {
				err = cerr
			}
		}()

		buf.Reset(io.MultiWriter(w, bcdBackupWriter))
	} else {
		buf.Reset(w)
	}

	defer func() {
		ferr := buf.Flush()
		if err == nil {
			err = ferr
		}
	}()

	return backuptar.WriteBackupStreamFromTarFile(buf, t, hdr)
}

func writeLayerFromTar(r io.Reader, w hcsshim.LayerWriter, root string) (int64, error) {
	t := tar.NewReader(r)
	hdr, err := t.Next()
	totalSize := int64(0)
	buf := bufio.NewWriter(nil)
	for err == nil {
		base := path.Base(hdr.Name)
		if strings.HasPrefix(base, archive.WhiteoutPrefix) {
			name := path.Join(path.Dir(hdr.Name), base[len(archive.WhiteoutPrefix):])
			err = w.Remove(filepath.FromSlash(name))
			if err != nil {
				return 0, err
			}
			hdr, err = t.Next()
		} else if hdr.Typeflag == tar.TypeLink {
			err = w.AddLink(filepath.FromSlash(hdr.Name), filepath.FromSlash(hdr.Linkname))
			if err != nil {
				return 0, err
			}
			hdr, err = t.Next()
		} else {
			var (
				name     string
				size     int64
				fileInfo *winio.FileBasicInfo
			)
			name, size, fileInfo, err = backuptar.FileInfoFromHeader(hdr)
			if err != nil {
				return 0, err
			}
			err = w.Add(filepath.FromSlash(name), fileInfo)
			if err != nil {
				return 0, err
			}
			hdr, err = writeBackupStreamFromTarAndSaveMutatedFiles(buf, w, t, hdr, root)
			totalSize += size
		}
	}
	if err != io.EOF {
		return 0, err
	}
	return totalSize, nil
}

// importLayer adds a new layer to the tag and graph store based on the given data.
func (d *Driver) importLayer(id string, layerData io.Reader, parentLayerPaths []string) (size int64, err error) {
	if !noreexec {
		cmd := reexec.Command(append([]string{"docker-windows-write-layer", d.info.HomeDir, id}, parentLayerPaths...)...)
		output := bytes.NewBuffer(nil)
		cmd.Stdin = layerData
		cmd.Stdout = output
		cmd.Stderr = output

		if err = cmd.Start(); err != nil {
			return
		}

		if err = cmd.Wait(); err != nil {
			return 0, fmt.Errorf("re-exec error: %v: output: %s", err, output)
		}

		return strconv.ParseInt(output.String(), 10, 64)
	}
	return writeLayer(layerData, d.info.HomeDir, id, parentLayerPaths...)
}

// writeLayerReexec is the re-exec entry point for writing a layer from a tar file
func writeLayerReexec() {
	size, err := writeLayer(os.Stdin, os.Args[1], os.Args[2], os.Args[3:]...)
	if err != nil {
		fmt.Fprint(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprint(os.Stdout, size)
}

// writeLayer writes a layer from a tar file.
func writeLayer(layerData io.Reader, home string, id string, parentLayerPaths ...string) (int64, error) {
	err := winio.EnableProcessPrivileges([]string{winio.SeBackupPrivilege, winio.SeRestorePrivilege})
	if err != nil {
		return 0, err
	}
	if noreexec {
		defer func() {
			if err := winio.DisableProcessPrivileges([]string{winio.SeBackupPrivilege, winio.SeRestorePrivilege}); err != nil {
				// This should never happen, but just in case when in debugging mode.
				// See https://github.com/docker/docker/pull/28002#discussion_r86259241 for rationale.
				panic("Failed to disabled process privileges while in non re-exec mode")
			}
		}()
	}

	info := hcsshim.DriverInfo{
		Flavour: filterDriver,
		HomeDir: home,
	}

	w, err := hcsshim.NewLayerWriter(info, id, parentLayerPaths)
	if err != nil {
		return 0, err
	}

	size, err := writeLayerFromTar(layerData, w, filepath.Join(home, id))
	if err != nil {
		return 0, err
	}

	err = w.Close()
	if err != nil {
		return 0, err
	}

	return size, nil
}

// resolveID computes the layerID information based on the given id.
func (d *Driver) resolveID(id string) (string, error) {
	content, err := ioutil.ReadFile(filepath.Join(d.dir(id), "layerID"))
	if os.IsNotExist(err) {
		return id, nil
	} else if err != nil {
		return "", err
	}
	return string(content), nil
}

// setID stores the layerId in disk.
func (d *Driver) setID(id, altID string) error {
	return ioutil.WriteFile(filepath.Join(d.dir(id), "layerId"), []byte(altID), 0600)
}

// getLayerChain returns the layer chain information.
func (d *Driver) getLayerChain(id string) ([]string, error) {
	jPath := filepath.Join(d.dir(id), "layerchain.json")
	content, err := ioutil.ReadFile(jPath)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, fmt.Errorf("Unable to read layerchain file - %s", err)
	}

	var layerChain []string
	err = json.Unmarshal(content, &layerChain)
	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshall layerchain json - %s", err)
	}

	return layerChain, nil
}

// setLayerChain stores the layer chain information in disk.
func (d *Driver) setLayerChain(id string, chain []string) error {
	content, err := json.Marshal(&chain)
	if err != nil {
		return fmt.Errorf("Failed to marshall layerchain json - %s", err)
	}

	jPath := filepath.Join(d.dir(id), "layerchain.json")
	err = ioutil.WriteFile(jPath, content, 0600)
	if err != nil {
		return fmt.Errorf("Unable to write layerchain file - %s", err)
	}

	return nil
}

type fileGetCloserWithBackupPrivileges struct {
	path string
}

func (fg *fileGetCloserWithBackupPrivileges) Get(filename string) (io.ReadCloser, error) {
	if backupPath, ok := mutatedFiles[filename]; ok {
		return os.Open(filepath.Join(fg.path, backupPath))
	}

	var f *os.File
	// Open the file while holding the Windows backup privilege. This ensures that the
	// file can be opened even if the caller does not actually have access to it according
	// to the security descriptor. Also use sequential file access to avoid depleting the
	// standby list - Microsoft VSO Bug Tracker #9900466
	err := winio.RunWithPrivilege(winio.SeBackupPrivilege, func() error {
		path := longpath.AddPrefix(filepath.Join(fg.path, filename))
		p, err := windows.UTF16FromString(path)
		if err != nil {
			return err
		}
		const fileFlagSequentialScan = 0x08000000 // FILE_FLAG_SEQUENTIAL_SCAN
		h, err := windows.CreateFile(&p[0], windows.GENERIC_READ, windows.FILE_SHARE_READ, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS|fileFlagSequentialScan, 0)
		if err != nil {
			return &os.PathError{Op: "open", Path: path, Err: err}
		}
		f = os.NewFile(uintptr(h), path)
		return nil
	})
	return f, err
}

func (fg *fileGetCloserWithBackupPrivileges) Close() error {
	return nil
}

// DiffGetter returns a FileGetCloser that can read files from the directory that
// contains files for the layer differences. Used for direct access for tar-split.
func (d *Driver) DiffGetter(id string) (graphdriver.FileGetCloser, error) {
	panicIfUsedByLcow()
	id, err := d.resolveID(id)
	if err != nil {
		return nil, err
	}

	return &fileGetCloserWithBackupPrivileges{d.dir(id)}, nil
}

type storageOptions struct {
	size uint64
}

func parseStorageOpt(storageOpt map[string]string) (*storageOptions, error) {
	options := storageOptions{}

	// Read size to change the block device size per container.
	for key, val := range storageOpt {
		key := strings.ToLower(key)
		switch key {
		case "size":
			size, err := units.RAMInBytes(val)
			if err != nil {
				return nil, err
			}
			options.size = uint64(size)
		default:
			return nil, fmt.Errorf("Unknown storage option: %s", key)
		}
	}
	return &options, nil
}
