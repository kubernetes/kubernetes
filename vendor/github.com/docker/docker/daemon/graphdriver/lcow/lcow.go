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
// To enable global mode, run with --storage-opt lcow.globalmode=true

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
	"time"

	"github.com/Microsoft/hcsshim"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/system"
	"github.com/jhowardmsft/opengcs/gogcs/client"
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
	// TODO @jhowardmsft. I really dislike this path! But needs a platform change or passing parameters to the tools.
	toolsScratchPath = "/mnt/gcs/LinuxServiceVM/scratch"

	// svmGlobalID is the ID used in the serviceVMs map for the global service VM when running in "global" mode.
	svmGlobalID = "_lcow_global_svm_"

	// cacheDirectory is the sub-folder under the driver's data-root used to cache blank sandbox and scratch VHDs.
	cacheDirectory = "cache"

	// scratchDirectory is the sub-folder under the driver's data-root used for scratch VHDs in service VMs
	scratchDirectory = "scratch"
)

// cacheItem is our internal structure representing an item in our local cache
// of things that have been mounted.
type cacheItem struct {
	sync.Mutex        // Protects operations performed on this item
	uvmPath    string // Path in utility VM
	hostPath   string // Path on host
	refCount   int    // How many times its been mounted
	isSandbox  bool   // True if a sandbox
	isMounted  bool   // True when mounted in a service VM
}

// serviceVMItem is our internal structure representing an item in our
// map of service VMs we are maintaining.
type serviceVMItem struct {
	sync.Mutex                     // Serialises operations being performed in this service VM.
	scratchAttached bool           // Has a scratch been attached?
	config          *client.Config // Represents the service VM item.
}

// Driver represents an LCOW graph driver.
type Driver struct {
	dataRoot           string                    // Root path on the host where we are storing everything.
	cachedSandboxFile  string                    // Location of the local default-sized cached sandbox.
	cachedSandboxMutex sync.Mutex                // Protects race conditions from multiple threads creating the cached sandbox.
	cachedScratchFile  string                    // Location of the local cached empty scratch space.
	cachedScratchMutex sync.Mutex                // Protects race conditions from multiple threads creating the cached scratch.
	options            []string                  // Graphdriver options we are initialised with.
	serviceVmsMutex    sync.Mutex                // Protects add/updates/delete to the serviceVMs map.
	serviceVms         map[string]*serviceVMItem // Map of the configs representing the service VM(s) we are running.
	globalMode         bool                      // Indicates if running in an unsafe/global service VM mode.

	// NOTE: It is OK to use a cache here because Windows does not support
	// restoring containers when the daemon dies.

	cacheMutex sync.Mutex            // Protects add/update/deletes to cache.
	cache      map[string]*cacheItem // Map holding a cache of all the IDs we've mounted/unmounted.
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
		cache:             make(map[string]*cacheItem),
		serviceVms:        make(map[string]*serviceVMItem),
		globalMode:        false,
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
	if err := idtools.MkdirAllAs(dataRoot, 0700, 0, 0); err != nil {
		return nil, fmt.Errorf("%s failed to create '%s': %v", title, dataRoot, err)
	}

	// Make sure the cache directory is created under dataRoot
	if err := idtools.MkdirAllAs(cd, 0700, 0, 0); err != nil {
		return nil, fmt.Errorf("%s failed to create '%s': %v", title, cd, err)
	}

	// Make sure the scratch directory is created under dataRoot
	if err := idtools.MkdirAllAs(sd, 0700, 0, 0); err != nil {
		return nil, fmt.Errorf("%s failed to create '%s': %v", title, sd, err)
	}

	// Delete any items in the scratch directory
	filepath.Walk(sd, deletefiles)

	logrus.Infof("%s dataRoot: %s globalMode: %t", title, dataRoot, d.globalMode)

	return d, nil
}

// startServiceVMIfNotRunning starts a service utility VM if it is not currently running.
// It can optionally be started with a mapped virtual disk. Returns a opengcs config structure
// representing the VM.
func (d *Driver) startServiceVMIfNotRunning(id string, mvdToAdd *hcsshim.MappedVirtualDisk, context string) (*serviceVMItem, error) {
	// Use the global ID if in global mode
	if d.globalMode {
		id = svmGlobalID
	}

	title := fmt.Sprintf("lcowdriver: startservicevmifnotrunning %s:", id)

	// Make sure thread-safe when interrogating the map
	logrus.Debugf("%s taking serviceVmsMutex", title)
	d.serviceVmsMutex.Lock()

	// Nothing to do if it's already running except add the mapped drive if supplied.
	if svm, ok := d.serviceVms[id]; ok {
		logrus.Debugf("%s exists, releasing serviceVmsMutex", title)
		d.serviceVmsMutex.Unlock()

		if mvdToAdd != nil {
			logrus.Debugf("hot-adding %s to %s", mvdToAdd.HostPath, mvdToAdd.ContainerPath)

			// Ensure the item is locked while doing this
			logrus.Debugf("%s locking serviceVmItem %s", title, svm.config.Name)
			svm.Lock()

			if err := svm.config.HotAddVhd(mvdToAdd.HostPath, mvdToAdd.ContainerPath); err != nil {
				logrus.Debugf("%s releasing serviceVmItem %s on hot-add failure %s", title, svm.config.Name, err)
				svm.Unlock()
				return nil, fmt.Errorf("%s hot add %s to %s failed: %s", title, mvdToAdd.HostPath, mvdToAdd.ContainerPath, err)
			}

			logrus.Debugf("%s releasing serviceVmItem %s", title, svm.config.Name)
			svm.Unlock()
		}
		return svm, nil
	}

	// Release the lock early
	logrus.Debugf("%s releasing serviceVmsMutex", title)
	d.serviceVmsMutex.Unlock()

	// So we are starting one. First need an enpty structure.
	svm := &serviceVMItem{
		config: &client.Config{},
	}

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
	if mvdToAdd != nil {
		svm.config.MappedVirtualDisks = append(svm.config.MappedVirtualDisks, *mvdToAdd)
	}

	// Start it.
	logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) starting %s", context, svm.config.Name)
	if err := svm.config.Create(); err != nil {
		return nil, fmt.Errorf("failed to start service utility VM (%s): %s", context, err)
	}

	// As it's now running, add it to the map, checking for a race where another
	// thread has simultaneously tried to start it.
	logrus.Debugf("%s locking serviceVmsMutex for insertion", title)
	d.serviceVmsMutex.Lock()
	if svm, ok := d.serviceVms[id]; ok {
		logrus.Debugf("%s releasing serviceVmsMutex after insertion but exists", title)
		d.serviceVmsMutex.Unlock()
		return svm, nil
	}
	d.serviceVms[id] = svm
	logrus.Debugf("%s releasing serviceVmsMutex after insertion", title)
	d.serviceVmsMutex.Unlock()

	// Now we have a running service VM, we can create the cached scratch file if it doesn't exist.
	logrus.Debugf("%s locking cachedScratchMutex", title)
	d.cachedScratchMutex.Lock()
	if _, err := os.Stat(d.cachedScratchFile); err != nil {
		// TODO: Not a typo, but needs fixing when the platform sandbox stuff has been sorted out.
		logrus.Debugf("%s (%s): creating an SVM scratch - locking serviceVM", title, context)
		svm.Lock()
		if err := svm.config.CreateSandbox(d.cachedScratchFile, client.DefaultSandboxSizeMB, d.cachedSandboxFile); err != nil {
			logrus.Debugf("%s (%s): releasing serviceVM on error path", title, context)
			svm.Unlock()
			logrus.Debugf("%s (%s): releasing cachedScratchMutex on error path", title, context)
			d.cachedScratchMutex.Unlock()
			// TODO: NEED TO REMOVE FROM MAP HERE AND STOP IT
			return nil, fmt.Errorf("failed to create SVM scratch VHDX (%s): %s", context, err)
		}
		logrus.Debugf("%s (%s): releasing serviceVM on error path", title, context)
		svm.Unlock()
	}
	logrus.Debugf("%s (%s): releasing cachedScratchMutex", title, context)
	d.cachedScratchMutex.Unlock()

	// Hot-add the scratch-space if not already attached
	if !svm.scratchAttached {
		// Make a copy of it to the layer directory
		logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) cloning cached scratch for hot-add", context)
		if err := client.CopyFile(d.cachedScratchFile, scratchTargetFile, true); err != nil {
			// TODO: NEED TO REMOVE FROM MAP HERE AND STOP IT
			return nil, err
		}

		logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) hot-adding scratch %s - locking serviceVM", context, scratchTargetFile)
		svm.Lock()
		if err := svm.config.HotAddVhd(scratchTargetFile, toolsScratchPath); err != nil {
			logrus.Debugf("%s (%s): releasing serviceVM on error path", title, context)
			svm.Unlock()
			// TODOL NEED TO REMOVE FROM MAP HERE AND STOP IT
			return nil, fmt.Errorf("failed to hot-add %s failed: %s", scratchTargetFile, err)
		}
		logrus.Debugf("%s (%s): releasing serviceVM", title, context)
		svm.Unlock()
	}

	logrus.Debugf("lcowdriver: startServiceVmIfNotRunning: (%s) success", context)
	return svm, nil
}

// getServiceVM returns the appropriate service utility VM instance, optionally
// deleting it from the map (but not the global one)
func (d *Driver) getServiceVM(id string, deleteFromMap bool) (*serviceVMItem, error) {
	logrus.Debugf("lcowdriver: getservicevm:locking serviceVmsMutex")
	d.serviceVmsMutex.Lock()
	defer func() {
		logrus.Debugf("lcowdriver: getservicevm:releasing serviceVmsMutex")
		d.serviceVmsMutex.Unlock()
	}()
	if d.globalMode {
		id = svmGlobalID
	}
	if _, ok := d.serviceVms[id]; !ok {
		return nil, fmt.Errorf("getservicevm for %s failed as not found", id)
	}
	svm := d.serviceVms[id]
	if deleteFromMap && id != svmGlobalID {
		logrus.Debugf("lcowdriver: getservicevm: removing %s from map", id)
		delete(d.serviceVms, id)
	}
	return svm, nil
}

// terminateServiceVM terminates a service utility VM if its running, but does nothing
// when in global mode as it's lifetime is limited to that of the daemon.
func (d *Driver) terminateServiceVM(id, context string, force bool) error {

	// We don't do anything in safe mode unless the force flag has been passed, which
	// is only the case for cleanup at driver termination.
	if d.globalMode {
		if !force {
			logrus.Debugf("lcowdriver: terminateservicevm: %s (%s) - doing nothing as in global mode", id, context)
			return nil
		}
		id = svmGlobalID
	}

	// Get the service VM and delete it from the map
	svm, err := d.getServiceVM(id, true)
	if err != nil {
		return err
	}

	// We run the deletion of the scratch as a deferred function to at least attempt
	// clean-up in case of errors.
	defer func() {
		if svm.scratchAttached {
			scratchTargetFile := filepath.Join(d.dataRoot, scratchDirectory, fmt.Sprintf("%s.vhdx", id))
			logrus.Debugf("lcowdriver: terminateservicevm: %s (%s) - deleting scratch %s", id, context, scratchTargetFile)
			if err := os.Remove(scratchTargetFile); err != nil {
				logrus.Warnf("failed to remove scratch file %s (%s): %s", scratchTargetFile, context, err)
			}
		}
	}()

	// Nothing to do if it's not running
	if svm.config.Uvm != nil {
		logrus.Debugf("lcowdriver: terminateservicevm: %s (%s) - calling terminate", id, context)
		if err := svm.config.Uvm.Terminate(); err != nil {
			return fmt.Errorf("failed to terminate utility VM (%s): %s", context, err)
		}

		logrus.Debugf("lcowdriver: terminateservicevm: %s (%s) - waiting for utility VM to terminate", id, context)
		if err := svm.config.Uvm.WaitTimeout(time.Duration(svm.config.UvmTimeoutSeconds) * time.Second); err != nil {
			return fmt.Errorf("failed waiting for utility VM to terminate (%s): %s", context, err)
		}
	}

	logrus.Debugf("lcowdriver: terminateservicevm: %s (%s) - success", id, context)
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

	// Massive perf optimisation here. If we know that the RW layer is the default size,
	// and that the cached sandbox already exists, and we are running in safe mode, we
	// can just do a simple copy into the layers sandbox file without needing to start a
	// unique service VM. For a global service VM, it doesn't really matter.
	//
	// TODO: @jhowardmsft Where are we going to get the required size from?
	// We need to look at the CreateOpts for that, I think....

	// Make sure we have the sandbox mutex taken while we are examining it.
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

	logrus.Debugf("%s: creating SVM to create sandbox", title)
	svm, err := d.startServiceVMIfNotRunning(id, nil, "createreadwrite")
	if err != nil {
		return err
	}
	defer d.terminateServiceVM(id, "createreadwrite", false)

	// So the cached sandbox needs creating. Ensure we are the only thread creating it.
	logrus.Debugf("%s: locking cachedSandboxMutex for creation", title)
	d.cachedSandboxMutex.Lock()
	defer func() {
		logrus.Debugf("%s: releasing cachedSandboxMutex for creation", title)
		d.cachedSandboxMutex.Unlock()
	}()

	// Synchronise the operation in the service VM.
	logrus.Debugf("%s: locking svm for sandbox creation", title)
	svm.Lock()
	defer func() {
		logrus.Debugf("%s: releasing svm for sandbox creation", title)
		svm.Unlock()
	}()
	if err := svm.config.CreateSandbox(filepath.Join(d.dir(id), sandboxFilename), client.DefaultSandboxSizeMB, d.cachedSandboxFile); err != nil {
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
func (d *Driver) Get(id, mountLabel string) (string, error) {
	title := fmt.Sprintf("lcowdriver: get: %s", id)
	logrus.Debugf(title)

	// Work out what we are working on
	vhdFilename, vhdSize, isSandbox, err := getLayerDetails(d.dir(id))
	if err != nil {
		logrus.Debugf("%s failed to get layer details from %s: %s", title, d.dir(id), err)
		return "", fmt.Errorf("%s failed to open layer or sandbox VHD to open in %s: %s", title, d.dir(id), err)
	}
	logrus.Debugf("%s %s, size %d, isSandbox %t", title, vhdFilename, vhdSize, isSandbox)

	// Add item to cache, or update existing item, but ensure we have the
	// lock while updating items.
	logrus.Debugf("%s: locking cacheMutex", title)
	d.cacheMutex.Lock()
	var cacheEntry *cacheItem
	if entry, ok := d.cache[id]; !ok {
		// The item is not currently in the cache.
		cacheEntry = &cacheItem{
			refCount:  1,
			isSandbox: isSandbox,
			hostPath:  vhdFilename,
			uvmPath:   fmt.Sprintf("/mnt/%s", id),
			isMounted: false, // we defer this as an optimisation
		}
		d.cache[id] = cacheEntry
		logrus.Debugf("%s: added cache entry %+v", title, cacheEntry)
	} else {
		// Increment the reference counter in the cache.
		logrus.Debugf("%s: locking cache item for increment", title)
		entry.Lock()
		entry.refCount++
		logrus.Debugf("%s: releasing cache item for increment", title)
		entry.Unlock()
		logrus.Debugf("%s: incremented refcount on cache entry %+v", title, cacheEntry)
	}
	logrus.Debugf("%s: releasing cacheMutex", title)
	d.cacheMutex.Unlock()

	logrus.Debugf("%s %s success. %s: %+v: size %d", title, id, d.dir(id), cacheEntry, vhdSize)
	return d.dir(id), nil
}

// Put does the reverse of get. If there are no more references to
// the layer, it unmounts it from the utility VM.
func (d *Driver) Put(id string) error {
	title := fmt.Sprintf("lcowdriver: put: %s", id)

	logrus.Debugf("%s: locking cacheMutex", title)
	d.cacheMutex.Lock()
	entry, ok := d.cache[id]
	if !ok {
		logrus.Debugf("%s: releasing cacheMutex on error path", title)
		d.cacheMutex.Unlock()
		return fmt.Errorf("%s possible ref-count error, or invalid id was passed to the graphdriver. Cannot handle id %s as it's not in the cache", title, id)
	}

	// Are we just decrementing the reference count?
	logrus.Debugf("%s: locking cache item for possible decrement", title)
	entry.Lock()
	if entry.refCount > 1 {
		entry.refCount--
		logrus.Debugf("%s: releasing cache item for decrement and early get-out as refCount is now %d", title, entry.refCount)
		entry.Unlock()
		logrus.Debugf("%s: refCount decremented to %d. Releasing cacheMutex", title, entry.refCount)
		d.cacheMutex.Unlock()
		return nil
	}
	logrus.Debugf("%s: releasing cache item", title)
	entry.Unlock()
	logrus.Debugf("%s: releasing cacheMutex. Ref count has dropped to zero", title)
	d.cacheMutex.Unlock()

	// To reach this point, the reference count has dropped to zero. If we have
	// done a mount and we are in global mode, then remove it. We don't
	// need to remove in safe mode as the service VM is going to be torn down
	// anyway.

	if d.globalMode {
		logrus.Debugf("%s: locking cache item at zero ref-count", title)
		entry.Lock()
		defer func() {
			logrus.Debugf("%s: releasing cache item at zero ref-count", title)
			entry.Unlock()
		}()
		if entry.isMounted {
			svm, err := d.getServiceVM(id, false)
			if err != nil {
				return err
			}

			logrus.Debugf("%s: Hot-Removing %s. Locking svm", title, entry.hostPath)
			svm.Lock()
			if err := svm.config.HotRemoveVhd(entry.hostPath); err != nil {
				logrus.Debugf("%s: releasing svm on error path", title)
				svm.Unlock()
				return fmt.Errorf("%s failed to hot-remove %s from global service utility VM: %s", title, entry.hostPath, err)
			}
			logrus.Debugf("%s: releasing svm", title)
			svm.Unlock()
		}
	}

	// Remove from the cache map.
	logrus.Debugf("%s: Locking cacheMutex to delete item from cache", title)
	d.cacheMutex.Lock()
	delete(d.cache, id)
	logrus.Debugf("%s: releasing cacheMutex after item deleted from cache", title)
	d.cacheMutex.Unlock()

	logrus.Debugf("%s %s: refCount 0. %s (%s) completed successfully", title, id, entry.hostPath, entry.uvmPath)
	return nil
}

// Cleanup ensures the information the driver stores is properly removed.
// We use this opportunity to cleanup any -removing folders which may be
// still left if the daemon was killed while it was removing a layer.
func (d *Driver) Cleanup() error {
	title := "lcowdriver: cleanup"

	d.cacheMutex.Lock()
	for k, v := range d.cache {
		logrus.Debugf("%s cache entry: %s: %+v", title, k, v)
		if v.refCount > 0 {
			logrus.Warnf("%s leaked %s: %+v", title, k, v)
		}
	}
	d.cacheMutex.Unlock()

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
	for k, v := range d.serviceVms {
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

	logrus.Debugf("%s: locking cacheMutex", title)
	d.cacheMutex.Lock()
	if _, ok := d.cache[id]; !ok {
		logrus.Debugf("%s: releasing cacheMutex on error path", title)
		d.cacheMutex.Unlock()
		return nil, fmt.Errorf("%s fail as %s is not in the cache", title, id)
	}
	cacheEntry := d.cache[id]
	logrus.Debugf("%s: releasing cacheMutex", title)
	d.cacheMutex.Unlock()

	// Stat to get size
	logrus.Debugf("%s: locking cacheEntry", title)
	cacheEntry.Lock()
	fileInfo, err := os.Stat(cacheEntry.hostPath)
	if err != nil {
		logrus.Debugf("%s: releasing cacheEntry on error path", title)
		cacheEntry.Unlock()
		return nil, fmt.Errorf("%s failed to stat %s: %s", title, cacheEntry.hostPath, err)
	}
	logrus.Debugf("%s: releasing cacheEntry", title)
	cacheEntry.Unlock()

	// Start the SVM with a mapped virtual disk. Note that if the SVM is
	// already runing and we are in global mode, this will be
	// hot-added.
	mvd := &hcsshim.MappedVirtualDisk{
		HostPath:          cacheEntry.hostPath,
		ContainerPath:     cacheEntry.uvmPath,
		CreateInUtilityVM: true,
		ReadOnly:          true,
	}

	logrus.Debugf("%s: starting service VM", title)
	svm, err := d.startServiceVMIfNotRunning(id, mvd, fmt.Sprintf("diff %s", id))
	if err != nil {
		return nil, err
	}

	// Set `isMounted` for the cache entry. Note that we re-scan the cache
	// at this point as it's possible the cacheEntry changed during the long-
	// running operation above when we weren't holding the cacheMutex lock.
	logrus.Debugf("%s: locking cacheMutex for updating isMounted", title)
	d.cacheMutex.Lock()
	if _, ok := d.cache[id]; !ok {
		logrus.Debugf("%s: releasing cacheMutex on error path of isMounted", title)
		d.cacheMutex.Unlock()
		d.terminateServiceVM(id, fmt.Sprintf("diff %s", id), false)
		return nil, fmt.Errorf("%s fail as %s is not in the cache when updating isMounted", title, id)
	}
	cacheEntry = d.cache[id]
	logrus.Debugf("%s: locking cacheEntry for updating isMounted", title)
	cacheEntry.Lock()
	cacheEntry.isMounted = true
	logrus.Debugf("%s: releasing cacheEntry for updating isMounted", title)
	cacheEntry.Unlock()
	logrus.Debugf("%s: releasing cacheMutex for updating isMounted", title)
	d.cacheMutex.Unlock()

	// Obtain the tar stream for it
	logrus.Debugf("%s %s, size %d, isSandbox %t", title, cacheEntry.hostPath, fileInfo.Size(), cacheEntry.isSandbox)
	tarReadCloser, err := svm.config.VhdToTar(cacheEntry.hostPath, cacheEntry.uvmPath, cacheEntry.isSandbox, fileInfo.Size())
	if err != nil {
		d.terminateServiceVM(id, fmt.Sprintf("diff %s", id), false)
		return nil, fmt.Errorf("%s failed to export layer to tar stream for id: %s, parent: %s : %s", title, id, parent, err)
	}

	logrus.Debugf("%s id %s parent %s completed successfully", title, id, parent)

	// In safe/non-global mode, we can't tear down the service VM until things have been read.
	if !d.globalMode {
		return ioutils.NewReadCloserWrapper(tarReadCloser, func() error {
			tarReadCloser.Close()
			d.terminateServiceVM(id, fmt.Sprintf("diff %s", id), false)
			return nil
		}), nil
	}
	return tarReadCloser, nil
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

	// TODO @jhowardmsft - the retries are temporary to overcome platform reliablity issues.
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
func getLayerDetails(folder string) (string, int64, bool, error) {
	var fileInfo os.FileInfo
	isSandbox := false
	filename := filepath.Join(folder, layerFilename)
	var err error

	if fileInfo, err = os.Stat(filename); err != nil {
		filename = filepath.Join(folder, sandboxFilename)
		if fileInfo, err = os.Stat(filename); err != nil {
			if os.IsNotExist(err) {
				return "", 0, isSandbox, fmt.Errorf("could not find layer or sandbox in %s", folder)
			}
			return "", 0, isSandbox, fmt.Errorf("error locating layer or sandbox in %s: %s", folder, err)
		}
		isSandbox = true
	}
	return filename, fileInfo.Size(), isSandbox, nil
}
