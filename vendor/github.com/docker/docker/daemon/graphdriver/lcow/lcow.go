// +build windows

// Maintainer:  jhowardmsft
// Locale:      en-gb
// About:       Graph-driver for Linux Containers On Windows (LCOW)
//
// This graphdriver runs in two modes. Yet to be determined which one will
// be the shipping mode. The global mode is where a single utility VM
// is used for all service VM tool operations. This isn't safe security-wise
// as it's attaching a sandbox of multiple containers to it, containing
// untrusted data. This may be fine for client devops scenarios. In
// safe mode, a unique utility VM is instantiated for all service VM tool
// operations. The downside of safe-mode is that operations are slower as
// a new service utility VM has to be started and torn-down when needed.
//
// Options:
//
// The following options are read by the graphdriver itself:
//
//   * lcow.globalmode - Enables global service VM Mode
//        -- Possible values:     true/false
//        -- Default if omitted:  false
//
//   * lcow.sandboxsize - Specifies a custom sandbox size in GB for starting a container
//        -- Possible values:      >= default sandbox size (opengcs defined, currently 20)
//        -- Default if omitted:  20
//
// The following options are read by opengcs:
//
//   * lcow.kirdpath - Specifies a custom path to a kernel/initrd pair
//        -- Possible values:      Any local path that is not a mapped drive
//        -- Default if omitted:  %ProgramFiles%\Linux Containers
//
//   * lcow.kernel - Specifies a custom kernel file located in the `lcow.kirdpath` path
//        -- Possible values:      Any valid filename
//        -- Default if omitted:  bootx64.efi
//
//   * lcow.initrd - Specifies a custom initrd file located in the `lcow.kirdpath` path
//        -- Possible values:      Any valid filename
//        -- Default if omitted:  initrd.img
//
//   * lcow.bootparameters - Specifies additional boot parameters for booting in kernel+initrd mode
//        -- Possible values:      Any valid linux kernel boot options
//        -- Default if omitted:  <nil>
//
//   * lcow.vhdx - Specifies a custom vhdx file to boot (instead of a kernel+initrd)
//        -- Possible values:      Any valid filename
//        -- Default if omitted:  uvm.vhdx under `lcow.kirdpath`
//
//   * lcow.timeout - Specifies a timeout for utility VM operations in seconds
//        -- Possible values:      >=0
//        -- Default if omitted:  300

// TODO: Grab logs from SVM at terminate or errors

package lcow

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Microsoft/hcsshim"
	"github.com/Microsoft/opengcs/client"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/containerfs"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/system"
	"github.com/sirupsen/logrus"
)

// init registers this driver to the register. It gets initialised by the
// function passed in the second parameter, implemented in this file.
func init() {
	graphdriver.Register("lcow", InitDriver)
}

const (
	// sandboxFilename is the name of the file containing a layer's sandbox (read-write layer).
	sandboxFilename = "sandbox.vhdx"

	// scratchFilename is the name of the scratch-space used by an SVM to avoid running out of memory.
	scratchFilename = "scratch.vhdx"

	// layerFilename is the name of the file containing a layer's read-only contents.
	// Note this really is VHD format, not VHDX.
	layerFilename = "layer.vhd"

	// toolsScratchPath is a location in a service utility VM that the tools can use as a
	// scratch space to avoid running out of memory.
	toolsScratchPath = "/tmp/scratch"

	// svmGlobalID is the ID used in the serviceVMs map for the global service VM when running in "global" mode.
	svmGlobalID = "_lcow_global_svm_"

	// cacheDirectory is the sub-folder under the driver's data-root used to cache blank sandbox and scratch VHDs.
	cacheDirectory = "cache"

	// scratchDirectory is the sub-folder under the driver's data-root used for scratch VHDs in service VMs
	scratchDirectory = "scratch"

	// errOperationPending is the HRESULT returned by the HCS when the VM termination operation is still pending.
	errOperationPending syscall.Errno = 0xc0370103
)

// Driver represents an LCOW graph driver.
type Driver struct {
	dataRoot           string     // Root path on the host where we are storing everything.
	cachedSandboxFile  string     // Location of the local default-sized cached sandbox.
	cachedSandboxMutex sync.Mutex // Protects race conditions from multiple threads creating the cached sandbox.
	cachedScratchFile  string     // Location of the local cached empty scratch space.
	cachedScratchMutex sync.Mutex // Protects race conditions from multiple threads creating the cached scratch.
	options            []string   // Graphdriver options we are initialised with.
	globalMode         bool       // Indicates if running in an unsafe/global service VM mode.

	// NOTE: It is OK to use a cache here because Windows does not support
	// restoring containers when the daemon dies.
	serviceVms *serviceVMMap // Map of the configs representing the service VM(s) we are running.
}

// layerDetails is the structure returned by a helper function `getLayerDetails`
// for getting information about a layer folder
type layerDetails struct {
	filename  string // \path\to\sandbox.vhdx or \path\to\layer.vhd
	size      int64  // size of the above file
	isSandbox bool   // true if sandbox.vhdx
}

// deletefiles is a helper function for initialisation where we delete any
// left-over scratch files in case we were previously forcibly terminated.
func deletefiles(path string, f os.FileInfo, err error) error {
	if strings.HasSuffix(f.Name(), ".vhdx") {
		logrus.Warnf("lcowdriver: init: deleting stale scratch file %s", path)
		return os.Remove(path)
	}
	return nil
}

// InitDriver returns a new LCOW storage driver.
func InitDriver(dataRoot string, options []string, _, _ []idtools.IDMap) (graphdriver.Driver, error) {
	title := "lcowdriver: init:"

	cd := filepath.Join(dataRoot, cacheDirectory)
	sd := filepath.Join(dataRoot, scratchDirectory)

	d := &Driver{
		dataRoot:          dataRoot,
		options:           options,
		cachedSandboxFile: filepath.Join(cd, sandboxFilename),
		cachedScratchFile: filepath.Join(cd, scratchFilename),
		serviceVms: &serviceVMMap{
			svms: make(map[string]*serviceVMMapItem),
		},
		globalMode: false,
	}

	// Looks for relevant options
	for _, v := range options {
		opt := strings.SplitN(v, "=", 2)
		if len(opt) == 2 {
			switch strings.ToLower(opt[0]) {
			case "lcow.globalmode":
				var err error
				d.globalMode, err = strconv.ParseBool(opt[1])
				if err != nil {
					return nil, fmt.Errorf("%s failed to parse value for 'lcow.globalmode' - must be 'true' or 'false'", title)
				}
				break
			}
		}
	}

	// Make sure the dataRoot directory is created
	if err := idtools.MkdirAllAndChown(dataRoot, 0700, idtools.IDPair{UID: 0, GID: 0}); err != nil {
		return nil, fmt.Errorf("%s failed to create '%s': %v", title, dataRoot, err)
	}

	// Make sure the cache directory is created under dataRoot
	if err := idtools.MkdirAllAndChown(cd, 0700, idtools.IDPair{UID: 0, GID: 0}); err != nil {
		return nil, fmt.Errorf("%s failed to create '%s': %v", title, cd, err)
	}

	// Make sure the scratch directory is created under dataRoot
	if err := idtools.MkdirAllAndChown(sd, 0700, idtools.IDPair{UID: 0, GID: 0}); err != nil {
		return nil, fmt.Errorf("%s failed to create '%s': %v", title, sd, err)
	}

	// Delete any items in the scratch directory
	filepath.Walk(sd, deletefiles)

	logrus.Infof("%s dataRoot: %s globalMode: %t", title, dataRoot, d.globalMode)

	return d, nil
}

func (d *Driver) getVMID(id string) string {
	if d.globalMode {
		return svmGlobalID
	}
	return id
}

// startServiceVMIfNotRunning starts a service utility VM if it is not currently running.
// It can optionally be started with a mapped virtual disk. Returns a opengcs config structure
// representing the VM.
func (d *Driver) startServiceVMIfNotRunning(id string, mvdToAdd []hcsshim.MappedVirtualDisk, context string) (_ *serviceVM, err error) {
	// Use the global ID if in global mode
	id = d.getVMID(id)

	title := fmt.Sprintf("lcowdriver: startservicevmifnotrunning %s:", id)

	// Attempt to add ID to the service vm map
	logrus.Debugf("%s: Adding entry to service vm map", title)
	svm, exists, err := d.serviceVms.add(id)
	if err != nil && err == errVMisTerminating {
		// VM is in the process of terminating. Wait until it's done and and then try again
		logrus.Debugf("%s: VM with current ID still in the process of terminating: %s", title, id)
		if err := svm.getStopError(); err != nil {
			logrus.Debugf("%s: VM %s did not stop successfully: %s", title, id, err)
			return nil, err
		}
		return d.startServiceVMIfNotRunning(id, mvdToAdd, context)
	} else if err != nil {
		logrus.Debugf("%s: failed to add service vm to map: %s", err)
		return nil, fmt.Errorf("%s: failed to add to service vm map: %s", title, err)
	}

	if exists {
		// Service VM is already up and running. In this case, just hot add the vhds.
		logrus.Debugf("%s: service vm already exists. Just hot adding: %+v", title, mvdToAdd)
		if err := svm.hotAddVHDs(mvdToAdd...); err != nil {
			logrus.Debugf("%s: failed to hot add vhds on service vm creation: %s", title, err)
			return nil, fmt.Errorf("%s: failed to hot add vhds on service vm: %s", title, err)
		}
		return svm, nil
	}

	// We are the first service for this id, so we need to start it
	logrus.Debugf("%s: service vm doesn't exist. Now starting it up: %s", title, id)

	defer func() {
		// Signal that start has finished, passing in the error if any.
		svm.signalStartFinished(err)
		if err != nil {
			// We added a ref to the VM, since we failed, we should delete the ref.
			d.terminateServiceVM(id, "error path on startServiceVMIfNotRunning", false)
		}
	}()

	// Generate a default configuration
	if err := svm.config.GenerateDefault(d.options); err != nil {
		return nil, fmt.Errorf("%s failed to generate default gogcs configuration for global svm (%s): %s", title, context, err)
	}

	// For the name, we deliberately suffix if safe-mode to ensure that it doesn't
	// clash with another utility VM which may be running for the container itself.
	// This also makes it easier to correlate through Get-ComputeProcess.
	if id == svmGlobalID {
		svm.config.Name = svmGlobalID
	} else {
		svm.config.Name = fmt.Sprintf("%s_svm", id)
	}

	// Ensure we take the cached scratch mutex around the check to ensure the file is complete
	// and not in the process of being created by another thread.
	scratchTargetFile := filepath.Join(d.dataRoot, scratchDirectory, fmt.Sprintf("%s.vhdx", id))

	logrus.Debugf("%s locking cachedScratchMutex", title)
	d.cachedScratchMutex.Lock()
	if _, err := os.Stat(d.cachedScratchFile); err == nil {
		// Make a copy of cached scratch to the scratch directory
		logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) cloning cached scratch for mvd", context)
		if err := client.CopyFile(d.cachedScratchFile, scratchTargetFile, true); err != nil {
			logrus.Debugf("%s releasing cachedScratchMutex on err: %s", title, err)
			d.cachedScratchMutex.Unlock()
			return nil, err
		}

		// Add the cached clone as a mapped virtual disk
		logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) adding cloned scratch as mvd", context)
		mvd := hcsshim.MappedVirtualDisk{
			HostPath:          scratchTargetFile,
			ContainerPath:     toolsScratchPath,
			CreateInUtilityVM: true,
		}
		svm.config.MappedVirtualDisks = append(svm.config.MappedVirtualDisks, mvd)
		svm.scratchAttached = true
	}

	logrus.Debugf("%s releasing cachedScratchMutex", title)
	d.cachedScratchMutex.Unlock()

	// If requested to start it with a mapped virtual disk, add it now.
	svm.config.MappedVirtualDisks = append(svm.config.MappedVirtualDisks, mvdToAdd...)
	for _, mvd := range svm.config.MappedVirtualDisks {
		svm.attachedVHDs[mvd.HostPath] = 1
	}

	// Start it.
	logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) starting %s", context, svm.config.Name)
	if err := svm.config.StartUtilityVM(); err != nil {
		return nil, fmt.Errorf("failed to start service utility VM (%s): %s", context, err)
	}

	// defer function to terminate the VM if the next steps fail
	defer func() {
		if err != nil {
			waitTerminate(svm, fmt.Sprintf("startServiceVmIfNotRunning: %s (%s)", id, context))
		}
	}()

	// Now we have a running service VM, we can create the cached scratch file if it doesn't exist.
	logrus.Debugf("%s locking cachedScratchMutex", title)
	d.cachedScratchMutex.Lock()
	if _, err := os.Stat(d.cachedScratchFile); err != nil {
		logrus.Debugf("%s (%s): creating an SVM scratch", title, context)

		// Don't use svm.CreateExt4Vhdx since that only works when the service vm is setup,
		// but we're still in that process right now.
		if err := svm.config.CreateExt4Vhdx(scratchTargetFile, client.DefaultVhdxSizeGB, d.cachedScratchFile); err != nil {
			logrus.Debugf("%s (%s): releasing cachedScratchMutex on error path", title, context)
			d.cachedScratchMutex.Unlock()
			logrus.Debugf("%s: failed to create vm scratch %s: %s", title, scratchTargetFile, err)
			return nil, fmt.Errorf("failed to create SVM scratch VHDX (%s): %s", context, err)
		}
	}
	logrus.Debugf("%s (%s): releasing cachedScratchMutex", title, context)
	d.cachedScratchMutex.Unlock()

	// Hot-add the scratch-space if not already attached
	if !svm.scratchAttached {
		logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) hot-adding scratch %s", context, scratchTargetFile)
		if err := svm.hotAddVHDsAtStart(hcsshim.MappedVirtualDisk{
			HostPath:          scratchTargetFile,
			ContainerPath:     toolsScratchPath,
			CreateInUtilityVM: true,
		}); err != nil {
			logrus.Debugf("%s: failed to hot-add scratch %s: %s", title, scratchTargetFile, err)
			return nil, fmt.Errorf("failed to hot-add %s failed: %s", scratchTargetFile, err)
		}
		svm.scratchAttached = true
	}

	logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) success", context)
	return svm, nil
}

// terminateServiceVM terminates a service utility VM if its running if it's,
// not being used by any goroutine, but does nothing when in global mode as it's
// lifetime is limited to that of the daemon. If the force flag is set, then
// the VM will be killed regardless of the ref count or if it's global.
func (d *Driver) terminateServiceVM(id, context string, force bool) (err error) {
	// We don't do anything in safe mode unless the force flag has been passed, which
	// is only the case for cleanup at driver termination.
	if d.globalMode && !force {
		logrus.Debugf("lcowdriver: terminateservicevm: %s (%s) - doing nothing as in global mode", id, context)
		return nil
	}

	id = d.getVMID(id)

	var svm *serviceVM
	var lastRef bool
	if !force {
		// In the not force case, we ref count
		svm, lastRef, err = d.serviceVms.decrementRefCount(id)
	} else {
		// In the force case, we ignore the ref count and just set it to 0
		svm, err = d.serviceVms.setRefCountZero(id)
		lastRef = true
	}

	if err == errVMUnknown {
		return nil
	} else if err == errVMisTerminating {
		return svm.getStopError()
	} else if !lastRef {
		return nil
	}

	// We run the deletion of the scratch as a deferred function to at least attempt
	// clean-up in case of errors.
	defer func() {
		if svm.scratchAttached {
			scratchTargetFile := filepath.Join(d.dataRoot, scratchDirectory, fmt.Sprintf("%s.vhdx", id))
			logrus.Debugf("lcowdriver: terminateservicevm: %s (%s) - deleting scratch %s", id, context, scratchTargetFile)
			if errRemove := os.Remove(scratchTargetFile); errRemove != nil {
				logrus.Warnf("failed to remove scratch file %s (%s): %s", scratchTargetFile, context, errRemove)
				err = errRemove
			}
		}

		// This function shouldn't actually return error unless there is a bug
		if errDelete := d.serviceVms.deleteID(id); errDelete != nil {
			logrus.Warnf("failed to service vm from svm map %s (%s): %s", id, context, errDelete)
		}

		// Signal that this VM has stopped
		svm.signalStopFinished(err)
	}()

	// Now it's possible that the serivce VM failed to start and now we are trying to termiante it.
	// In this case, we will relay the error to the goroutines waiting for this vm to stop.
	if err := svm.getStartError(); err != nil {
		logrus.Debugf("lcowdriver: terminateservicevm: %s had failed to start up: %s", id, err)
		return err
	}

	if err := waitTerminate(svm, fmt.Sprintf("terminateservicevm: %s (%s)", id, context)); err != nil {
		return err
	}

	logrus.Debugf("lcowdriver: terminateservicevm: %s (%s) - success", id, context)
	return nil
}

func waitTerminate(svm *serviceVM, context string) error {
	if svm.config == nil {
		return fmt.Errorf("lcowdriver: waitTermiante: Nil utility VM. %s", context)
	}

	logrus.Debugf("lcowdriver: waitTerminate: Calling terminate: %s", context)
	if err := svm.config.Uvm.Terminate(); err != nil {
		// We might get operation still pending from the HCS. In that case, we shouldn't return
		// an error since we call wait right after.
		underlyingError := err
		if conterr, ok := err.(*hcsshim.ContainerError); ok {
			underlyingError = conterr.Err
		}

		if syscallErr, ok := underlyingError.(syscall.Errno); ok {
			underlyingError = syscallErr
		}

		if underlyingError != errOperationPending {
			return fmt.Errorf("failed to terminate utility VM (%s): %s", context, err)
		}
		logrus.Debugf("lcowdriver: waitTerminate: uvm.Terminate() returned operation pending (%s)", context)
	}

	logrus.Debugf("lcowdriver: waitTerminate: (%s) - waiting for utility VM to terminate", context)
	if err := svm.config.Uvm.WaitTimeout(time.Duration(svm.config.UvmTimeoutSeconds) * time.Second); err != nil {
		return fmt.Errorf("failed waiting for utility VM to terminate (%s): %s", context, err)
	}
	return nil
}

// String returns the string representation of a driver. This should match
// the name the graph driver has been registered with.
func (d *Driver) String() string {
	return "lcow"
}

// Status returns the status of the driver.
func (d *Driver) Status() [][2]string {
	return [][2]string{
		{"LCOW", ""},
		// TODO: Add some more info here - mode, home, ....
	}
}

// Exists returns true if the given id is registered with this driver.
func (d *Driver) Exists(id string) bool {
	_, err := os.Lstat(d.dir(id))
	logrus.Debugf("lcowdriver: exists: id %s %t", id, err == nil)
	return err == nil
}

// CreateReadWrite creates a layer that is writable for use as a container
// file system. That equates to creating a sandbox.
func (d *Driver) CreateReadWrite(id, parent string, opts *graphdriver.CreateOpts) error {
	title := fmt.Sprintf("lcowdriver: createreadwrite: id %s", id)
	logrus.Debugf(title)

	// First we need to create the folder
	if err := d.Create(id, parent, opts); err != nil {
		return err
	}

	// Look for an explicit sandbox size option.
	sandboxSize := uint64(client.DefaultVhdxSizeGB)
	for k, v := range opts.StorageOpt {
		switch strings.ToLower(k) {
		case "lcow.sandboxsize":
			var err error
			sandboxSize, err = strconv.ParseUint(v, 10, 32)
			if err != nil {
				return fmt.Errorf("%s failed to parse value '%s' for 'lcow.sandboxsize'", title, v)
			}
			if sandboxSize < client.DefaultVhdxSizeGB {
				return fmt.Errorf("%s 'lcow.sandboxsize' option cannot be less than %d", title, client.DefaultVhdxSizeGB)
			}
			break
		}
	}

	// Massive perf optimisation here. If we know that the RW layer is the default size,
	// and that the cached sandbox already exists, and we are running in safe mode, we
	// can just do a simple copy into the layers sandbox file without needing to start a
	// unique service VM. For a global service VM, it doesn't really matter. Of course,
	// this is only the case where the sandbox is the default size.
	//
	// Make sure we have the sandbox mutex taken while we are examining it.
	if sandboxSize == client.DefaultVhdxSizeGB {
		logrus.Debugf("%s: locking cachedSandboxMutex", title)
		d.cachedSandboxMutex.Lock()
		_, err := os.Stat(d.cachedSandboxFile)
		logrus.Debugf("%s: releasing cachedSandboxMutex", title)
		d.cachedSandboxMutex.Unlock()
		if err == nil {
			logrus.Debugf("%s: using cached sandbox to populate", title)
			if err := client.CopyFile(d.cachedSandboxFile, filepath.Join(d.dir(id), sandboxFilename), true); err != nil {
				return err
			}
			return nil
		}
	}

	logrus.Debugf("%s: creating SVM to create sandbox", title)
	svm, err := d.startServiceVMIfNotRunning(id, nil, "createreadwrite")
	if err != nil {
		return err
	}
	defer d.terminateServiceVM(id, "createreadwrite", false)

	// So the sandbox needs creating. If default size ensure we are the only thread populating the cache.
	// Non-default size we don't store, just create them one-off so no need to lock the cachedSandboxMutex.
	if sandboxSize == client.DefaultVhdxSizeGB {
		logrus.Debugf("%s: locking cachedSandboxMutex for creation", title)
		d.cachedSandboxMutex.Lock()
		defer func() {
			logrus.Debugf("%s: releasing cachedSandboxMutex for creation", title)
			d.cachedSandboxMutex.Unlock()
		}()
	}

	// Make sure we don't write to our local cached copy if this is for a non-default size request.
	targetCacheFile := d.cachedSandboxFile
	if sandboxSize != client.DefaultVhdxSizeGB {
		targetCacheFile = ""
	}

	// Create the ext4 vhdx
	logrus.Debugf("%s: creating sandbox ext4 vhdx", title)
	if err := svm.createExt4VHDX(filepath.Join(d.dir(id), sandboxFilename), uint32(sandboxSize), targetCacheFile); err != nil {
		logrus.Debugf("%s: failed to create sandbox vhdx for %s: %s", title, id, err)
		return err
	}
	return nil
}

// Create creates the folder for the layer with the given id, and
// adds it to the layer chain.
func (d *Driver) Create(id, parent string, opts *graphdriver.CreateOpts) error {
	logrus.Debugf("lcowdriver: create: id %s parent: %s", id, parent)

	parentChain, err := d.getLayerChain(parent)
	if err != nil {
		return err
	}

	var layerChain []string
	if parent != "" {
		if !d.Exists(parent) {
			return fmt.Errorf("lcowdriver: cannot create layer folder with missing parent %s", parent)
		}
		layerChain = []string{d.dir(parent)}
	}
	layerChain = append(layerChain, parentChain...)

	// Make sure layers are created with the correct ACL so that VMs can access them.
	layerPath := d.dir(id)
	logrus.Debugf("lcowdriver: create: id %s: creating %s", id, layerPath)
	if err := system.MkdirAllWithACL(layerPath, 755, system.SddlNtvmAdministratorsLocalSystem); err != nil {
		return err
	}

	if err := d.setLayerChain(id, layerChain); err != nil {
		if err2 := os.RemoveAll(layerPath); err2 != nil {
			logrus.Warnf("failed to remove layer %s: %s", layerPath, err2)
		}
		return err
	}
	logrus.Debugf("lcowdriver: create: id %s: success", id)

	return nil
}

// Remove unmounts and removes the dir information.
func (d *Driver) Remove(id string) error {
	logrus.Debugf("lcowdriver: remove: id %s", id)
	tmpID := fmt.Sprintf("%s-removing", id)
	tmpLayerPath := d.dir(tmpID)
	layerPath := d.dir(id)

	logrus.Debugf("lcowdriver: remove: id %s: layerPath %s", id, layerPath)

	// Unmount all the layers
	err := d.Put(id)
	if err != nil {
		logrus.Debugf("lcowdriver: remove id %s: failed to unmount: %s", id, err)
		return err
	}

	// for non-global case just kill the vm
	if !d.globalMode {
		if err := d.terminateServiceVM(id, fmt.Sprintf("Remove %s", id), true); err != nil {
			return err
		}
	}

	if err := os.Rename(layerPath, tmpLayerPath); err != nil && !os.IsNotExist(err) {
		return err
	}

	if err := os.RemoveAll(tmpLayerPath); err != nil {
		return err
	}

	logrus.Debugf("lcowdriver: remove: id %s: layerPath %s succeeded", id, layerPath)
	return nil
}

// Get returns the rootfs path for the id. It is reference counted and
// effectively can be thought of as a "mount the layer into the utility
// vm if it isn't already". The contract from the caller of this is that
// all Gets and Puts are matched. It -should- be the case that on cleanup,
// nothing is mounted.
//
// For optimisation, we don't actually mount the filesystem (which in our
// case means [hot-]adding it to a service VM. But we track that and defer
// the actual adding to the point we need to access it.
func (d *Driver) Get(id, mountLabel string) (containerfs.ContainerFS, error) {
	title := fmt.Sprintf("lcowdriver: get: %s", id)
	logrus.Debugf(title)

	// Generate the mounts needed for the defered operation.
	disks, err := d.getAllMounts(id)
	if err != nil {
		logrus.Debugf("%s failed to get all layer details for %s: %s", title, d.dir(id), err)
		return nil, fmt.Errorf("%s failed to get layer details for %s: %s", title, d.dir(id), err)
	}

	logrus.Debugf("%s: got layer mounts: %+v", title, disks)
	return &lcowfs{
		root:        unionMountName(disks),
		d:           d,
		mappedDisks: disks,
		vmID:        d.getVMID(id),
	}, nil
}

// Put does the reverse of get. If there are no more references to
// the layer, it unmounts it from the utility VM.
func (d *Driver) Put(id string) error {
	title := fmt.Sprintf("lcowdriver: put: %s", id)

	// Get the service VM that we need to remove from
	svm, err := d.serviceVms.get(d.getVMID(id))
	if err == errVMUnknown {
		return nil
	} else if err == errVMisTerminating {
		return svm.getStopError()
	}

	// Generate the mounts that Get() might have mounted
	disks, err := d.getAllMounts(id)
	if err != nil {
		logrus.Debugf("%s failed to get all layer details for %s: %s", title, d.dir(id), err)
		return fmt.Errorf("%s failed to get layer details for %s: %s", title, d.dir(id), err)
	}

	// Now, we want to perform the unmounts, hot-remove and stop the service vm.
	// We want to go though all the steps even if we have an error to clean up properly
	err = svm.deleteUnionMount(unionMountName(disks), disks...)
	if err != nil {
		logrus.Debugf("%s failed to delete union mount %s: %s", title, id, err)
	}

	err1 := svm.hotRemoveVHDs(disks...)
	if err1 != nil {
		logrus.Debugf("%s failed to hot remove vhds %s: %s", title, id, err)
		if err == nil {
			err = err1
		}
	}

	err1 = d.terminateServiceVM(id, fmt.Sprintf("Put %s", id), false)
	if err1 != nil {
		logrus.Debugf("%s failed to terminate service vm %s: %s", title, id, err1)
		if err == nil {
			err = err1
		}
	}
	logrus.Debugf("Put succeeded on id %s", id)
	return err
}

// Cleanup ensures the information the driver stores is properly removed.
// We use this opportunity to cleanup any -removing folders which may be
// still left if the daemon was killed while it was removing a layer.
func (d *Driver) Cleanup() error {
	title := "lcowdriver: cleanup"

	items, err := ioutil.ReadDir(d.dataRoot)
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
			if err := os.RemoveAll(filepath.Join(d.dataRoot, item.Name())); err != nil {
				logrus.Warnf("%s failed to cleanup %s: %s", title, item.Name(), err)
			} else {
				logrus.Infof("%s cleaned up %s", title, item.Name())
			}
		}
	}

	// Cleanup any service VMs we have running, along with their scratch spaces.
	// We don't take the lock for this as it's taken in terminateServiceVm.
	for k, v := range d.serviceVms.svms {
		logrus.Debugf("%s svm entry: %s: %+v", title, k, v)
		d.terminateServiceVM(k, "cleanup", true)
	}

	return nil
}

// Diff takes a layer (and it's parent layer which may be null, but
// is ignored by this implementation below) and returns a reader for
// a tarstream representing the layers contents. The id could be
// a read-only "layer.vhd" or a read-write "sandbox.vhdx". The semantics
// of this function dictate that the layer is already mounted.
// However, as we do lazy mounting as a performance optimisation,
// this will likely not be the case.
func (d *Driver) Diff(id, parent string) (io.ReadCloser, error) {
	title := fmt.Sprintf("lcowdriver: diff: %s", id)

	// Get VHDX info
	ld, err := getLayerDetails(d.dir(id))
	if err != nil {
		logrus.Debugf("%s: failed to get vhdx information of %s: %s", title, d.dir(id), err)
		return nil, err
	}

	// Start the SVM with a mapped virtual disk. Note that if the SVM is
	// already running and we are in global mode, this will be
	// hot-added.
	mvd := hcsshim.MappedVirtualDisk{
		HostPath:          ld.filename,
		ContainerPath:     hostToGuest(ld.filename),
		CreateInUtilityVM: true,
		ReadOnly:          true,
	}

	logrus.Debugf("%s: starting service VM", title)
	svm, err := d.startServiceVMIfNotRunning(id, []hcsshim.MappedVirtualDisk{mvd}, fmt.Sprintf("diff %s", id))
	if err != nil {
		return nil, err
	}

	logrus.Debugf("lcowdriver: diff: waiting for svm to finish booting")
	err = svm.getStartError()
	if err != nil {
		d.terminateServiceVM(id, fmt.Sprintf("diff %s", id), false)
		return nil, fmt.Errorf("lcowdriver: diff: svm failed to boot: %s", err)
	}

	// Obtain the tar stream for it
	logrus.Debugf("%s: %s %s, size %d, ReadOnly %t", title, ld.filename, mvd.ContainerPath, ld.size, ld.isSandbox)
	tarReadCloser, err := svm.config.VhdToTar(mvd.HostPath, mvd.ContainerPath, ld.isSandbox, ld.size)
	if err != nil {
		svm.hotRemoveVHDs(mvd)
		d.terminateServiceVM(id, fmt.Sprintf("diff %s", id), false)
		return nil, fmt.Errorf("%s failed to export layer to tar stream for id: %s, parent: %s : %s", title, id, parent, err)
	}

	logrus.Debugf("%s id %s parent %s completed successfully", title, id, parent)

	// In safe/non-global mode, we can't tear down the service VM until things have been read.
	return ioutils.NewReadCloserWrapper(tarReadCloser, func() error {
		tarReadCloser.Close()
		svm.hotRemoveVHDs(mvd)
		d.terminateServiceVM(id, fmt.Sprintf("diff %s", id), false)
		return nil
	}), nil
}

// ApplyDiff extracts the changeset from the given diff into the
// layer with the specified id and parent, returning the size of the
// new layer in bytes. The layer should not be mounted when calling
// this function. Another way of describing this is that ApplyDiff writes
// to a new layer (a VHD in LCOW) the contents of a tarstream it's given.
func (d *Driver) ApplyDiff(id, parent string, diff io.Reader) (int64, error) {
	logrus.Debugf("lcowdriver: applydiff: id %s", id)

	svm, err := d.startServiceVMIfNotRunning(id, nil, fmt.Sprintf("applydiff %s", id))
	if err != nil {
		return 0, err
	}
	defer d.terminateServiceVM(id, fmt.Sprintf("applydiff %s", id), false)

	logrus.Debugf("lcowdriver: applydiff: waiting for svm to finish booting")
	err = svm.getStartError()
	if err != nil {
		return 0, fmt.Errorf("lcowdriver: applydiff: svm failed to boot: %s", err)
	}

	// TODO @jhowardmsft - the retries are temporary to overcome platform reliability issues.
	// Obviously this will be removed as platform bugs are fixed.
	retries := 0
	for {
		retries++
		size, err := svm.config.TarToVhd(filepath.Join(d.dataRoot, id, layerFilename), diff)
		if err != nil {
			if retries <= 10 {
				continue
			}
			return 0, err
		}
		return size, err
	}
}

// Changes produces a list of changes between the specified layer
// and its parent layer. If parent is "", then all changes will be ADD changes.
// The layer should not be mounted when calling this function.
func (d *Driver) Changes(id, parent string) ([]archive.Change, error) {
	logrus.Debugf("lcowdriver: changes: id %s parent %s", id, parent)
	// TODO @gupta-ak. Needs implementation with assistance from service VM
	return nil, nil
}

// DiffSize calculates the changes between the specified layer
// and its parent and returns the size in bytes of the changes
// relative to its base filesystem directory.
func (d *Driver) DiffSize(id, parent string) (size int64, err error) {
	logrus.Debugf("lcowdriver: diffsize: id %s", id)
	// TODO @gupta-ak. Needs implementation with assistance from service VM
	return 0, nil
}

// GetMetadata returns custom driver information.
func (d *Driver) GetMetadata(id string) (map[string]string, error) {
	logrus.Debugf("lcowdriver: getmetadata: id %s", id)
	m := make(map[string]string)
	m["dir"] = d.dir(id)
	return m, nil
}

// GetLayerPath gets the layer path on host (path to VHD/VHDX)
func (d *Driver) GetLayerPath(id string) (string, error) {
	return d.dir(id), nil
}

// dir returns the absolute path to the layer.
func (d *Driver) dir(id string) string {
	return filepath.Join(d.dataRoot, filepath.Base(id))
}

// getLayerChain returns the layer chain information.
func (d *Driver) getLayerChain(id string) ([]string, error) {
	jPath := filepath.Join(d.dir(id), "layerchain.json")
	logrus.Debugf("lcowdriver: getlayerchain: id %s json %s", id, jPath)
	content, err := ioutil.ReadFile(jPath)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, fmt.Errorf("lcowdriver: getlayerchain: %s unable to read layerchain file %s: %s", id, jPath, err)
	}

	var layerChain []string
	err = json.Unmarshal(content, &layerChain)
	if err != nil {
		return nil, fmt.Errorf("lcowdriver: getlayerchain: %s failed to unmarshall layerchain file %s: %s", id, jPath, err)
	}
	return layerChain, nil
}

// setLayerChain stores the layer chain information on disk.
func (d *Driver) setLayerChain(id string, chain []string) error {
	content, err := json.Marshal(&chain)
	if err != nil {
		return fmt.Errorf("lcowdriver: setlayerchain: %s failed to marshall layerchain json: %s", id, err)
	}

	jPath := filepath.Join(d.dir(id), "layerchain.json")
	logrus.Debugf("lcowdriver: setlayerchain: id %s json %s", id, jPath)
	err = ioutil.WriteFile(jPath, content, 0600)
	if err != nil {
		return fmt.Errorf("lcowdriver: setlayerchain: %s failed to write layerchain file: %s", id, err)
	}
	return nil
}

// getLayerDetails is a utility for getting a file name, size and indication of
// sandbox for a VHD(x) in a folder. A read-only layer will be layer.vhd. A
// read-write layer will be sandbox.vhdx.
func getLayerDetails(folder string) (*layerDetails, error) {
	var fileInfo os.FileInfo
	ld := &layerDetails{
		isSandbox: false,
		filename:  filepath.Join(folder, layerFilename),
	}

	fileInfo, err := os.Stat(ld.filename)
	if err != nil {
		ld.filename = filepath.Join(folder, sandboxFilename)
		if fileInfo, err = os.Stat(ld.filename); err != nil {
			return nil, fmt.Errorf("failed to locate layer or sandbox in %s", folder)
		}
		ld.isSandbox = true
	}
	ld.size = fileInfo.Size()

	return ld, nil
}

func (d *Driver) getAllMounts(id string) ([]hcsshim.MappedVirtualDisk, error) {
	layerChain, err := d.getLayerChain(id)
	if err != nil {
		return nil, err
	}
	layerChain = append([]string{d.dir(id)}, layerChain...)

	logrus.Debugf("getting all  layers: %v", layerChain)
	disks := make([]hcsshim.MappedVirtualDisk, len(layerChain), len(layerChain))
	for i := range layerChain {
		ld, err := getLayerDetails(layerChain[i])
		if err != nil {
			logrus.Debugf("Failed to get LayerVhdDetails from %s: %s", layerChain[i], err)
			return nil, err
		}
		disks[i].HostPath = ld.filename
		disks[i].ContainerPath = hostToGuest(ld.filename)
		disks[i].CreateInUtilityVM = true
		disks[i].ReadOnly = !ld.isSandbox
	}
	return disks, nil
}

func hostToGuest(hostpath string) string {
	return fmt.Sprintf("/tmp/%s", filepath.Base(filepath.Dir(hostpath)))
}

func unionMountName(disks []hcsshim.MappedVirtualDisk) string {
	return fmt.Sprintf("%s-mount", disks[0].ContainerPath)
}
