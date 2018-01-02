package libcontainerd

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"golang.org/x/net/context"

	"github.com/Microsoft/hcsshim"
	"github.com/docker/docker/pkg/sysinfo"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
)

type client struct {
	clientCommon

	// Platform specific properties below here (none presently on Windows)
}

// Win32 error codes that are used for various workarounds
// These really should be ALL_CAPS to match golangs syscall library and standard
// Win32 error conventions, but golint insists on CamelCase.
const (
	CoEClassstring     = syscall.Errno(0x800401F3) // Invalid class string
	ErrorNoNetwork     = syscall.Errno(1222)       // The network is not present or not started
	ErrorBadPathname   = syscall.Errno(161)        // The specified path is invalid
	ErrorInvalidObject = syscall.Errno(0x800710D8) // The object identifier does not represent a valid object
)

// defaultOwner is a tag passed to HCS to allow it to differentiate between
// container creator management stacks. We hard code "docker" in the case
// of docker.
const defaultOwner = "docker"

// Create is the entrypoint to create a container from a spec, and if successfully
// created, start it too. Table below shows the fields required for HCS JSON calling parameters,
// where if not populated, is omitted.
// +-----------------+--------------------------------------------+---------------------------------------------------+
// |                 | Isolation=Process                          | Isolation=Hyper-V                                 |
// +-----------------+--------------------------------------------+---------------------------------------------------+
// | VolumePath      | \\?\\Volume{GUIDa}                         |                                                   |
// | LayerFolderPath | %root%\windowsfilter\containerID           | %root%\windowsfilter\containerID (servicing only) |
// | Layers[]        | ID=GUIDb;Path=%root%\windowsfilter\layerID | ID=GUIDb;Path=%root%\windowsfilter\layerID        |
// | HvRuntime       |                                            | ImagePath=%root%\BaseLayerID\UtilityVM            |
// +-----------------+--------------------------------------------+---------------------------------------------------+
//
// Isolation=Process example:
//
// {
//	"SystemType": "Container",
//	"Name": "5e0055c814a6005b8e57ac59f9a522066e0af12b48b3c26a9416e23907698776",
//	"Owner": "docker",
//	"VolumePath": "\\\\\\\\?\\\\Volume{66d1ef4c-7a00-11e6-8948-00155ddbef9d}",
//	"IgnoreFlushesDuringBoot": true,
//	"LayerFolderPath": "C:\\\\control\\\\windowsfilter\\\\5e0055c814a6005b8e57ac59f9a522066e0af12b48b3c26a9416e23907698776",
//	"Layers": [{
//		"ID": "18955d65-d45a-557b-bf1c-49d6dfefc526",
//		"Path": "C:\\\\control\\\\windowsfilter\\\\65bf96e5760a09edf1790cb229e2dfb2dbd0fcdc0bf7451bae099106bfbfea0c"
//	}],
//	"HostName": "5e0055c814a6",
//	"MappedDirectories": [],
//	"HvPartition": false,
//	"EndpointList": ["eef2649d-bb17-4d53-9937-295a8efe6f2c"],
//	"Servicing": false
//}
//
// Isolation=Hyper-V example:
//
//{
//	"SystemType": "Container",
//	"Name": "475c2c58933b72687a88a441e7e0ca4bd72d76413c5f9d5031fee83b98f6045d",
//	"Owner": "docker",
//	"IgnoreFlushesDuringBoot": true,
//	"Layers": [{
//		"ID": "18955d65-d45a-557b-bf1c-49d6dfefc526",
//		"Path": "C:\\\\control\\\\windowsfilter\\\\65bf96e5760a09edf1790cb229e2dfb2dbd0fcdc0bf7451bae099106bfbfea0c"
//	}],
//	"HostName": "475c2c58933b",
//	"MappedDirectories": [],
//	"HvPartition": true,
//	"EndpointList": ["e1bb1e61-d56f-405e-b75d-fd520cefa0cb"],
//	"DNSSearchList": "a.com,b.com,c.com",
//	"HvRuntime": {
//		"ImagePath": "C:\\\\control\\\\windowsfilter\\\\65bf96e5760a09edf1790cb229e2dfb2dbd0fcdc0bf7451bae099106bfbfea0c\\\\UtilityVM"
//	},
//	"Servicing": false
//}
func (clnt *client) Create(containerID string, checkpoint string, checkpointDir string, spec specs.Spec, attachStdio StdioCallback, options ...CreateOption) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	if b, err := json.Marshal(spec); err == nil {
		logrus.Debugln("libcontainerd: client.Create() with spec", string(b))
	}
	osName := spec.Platform.OS
	if osName == "windows" {
		return clnt.createWindows(containerID, checkpoint, checkpointDir, spec, attachStdio, options...)
	}
	return clnt.createLinux(containerID, checkpoint, checkpointDir, spec, attachStdio, options...)
}

func (clnt *client) createWindows(containerID string, checkpoint string, checkpointDir string, spec specs.Spec, attachStdio StdioCallback, options ...CreateOption) error {
	configuration := &hcsshim.ContainerConfig{
		SystemType: "Container",
		Name:       containerID,
		Owner:      defaultOwner,
		IgnoreFlushesDuringBoot: false,
		HostName:                spec.Hostname,
		HvPartition:             false,
	}

	if spec.Windows.Resources != nil {
		if spec.Windows.Resources.CPU != nil {
			if spec.Windows.Resources.CPU.Count != nil {
				// This check is being done here rather than in adaptContainerSettings
				// because we don't want to update the HostConfig in case this container
				// is moved to a host with more CPUs than this one.
				cpuCount := *spec.Windows.Resources.CPU.Count
				hostCPUCount := uint64(sysinfo.NumCPU())
				if cpuCount > hostCPUCount {
					logrus.Warnf("Changing requested CPUCount of %d to current number of processors, %d", cpuCount, hostCPUCount)
					cpuCount = hostCPUCount
				}
				configuration.ProcessorCount = uint32(cpuCount)
			}
			if spec.Windows.Resources.CPU.Shares != nil {
				configuration.ProcessorWeight = uint64(*spec.Windows.Resources.CPU.Shares)
			}
			if spec.Windows.Resources.CPU.Maximum != nil {
				configuration.ProcessorMaximum = int64(*spec.Windows.Resources.CPU.Maximum)
			}
		}
		if spec.Windows.Resources.Memory != nil {
			if spec.Windows.Resources.Memory.Limit != nil {
				configuration.MemoryMaximumInMB = int64(*spec.Windows.Resources.Memory.Limit) / 1024 / 1024
			}
		}
		if spec.Windows.Resources.Storage != nil {
			if spec.Windows.Resources.Storage.Bps != nil {
				configuration.StorageBandwidthMaximum = *spec.Windows.Resources.Storage.Bps
			}
			if spec.Windows.Resources.Storage.Iops != nil {
				configuration.StorageIOPSMaximum = *spec.Windows.Resources.Storage.Iops
			}
		}
	}

	var layerOpt *LayerOption
	for _, option := range options {
		if s, ok := option.(*ServicingOption); ok {
			configuration.Servicing = s.IsServicing
			continue
		}
		if f, ok := option.(*FlushOption); ok {
			configuration.IgnoreFlushesDuringBoot = f.IgnoreFlushesDuringBoot
			continue
		}
		if h, ok := option.(*HyperVIsolationOption); ok {
			configuration.HvPartition = h.IsHyperV
			continue
		}
		if l, ok := option.(*LayerOption); ok {
			layerOpt = l
		}
		if n, ok := option.(*NetworkEndpointsOption); ok {
			configuration.EndpointList = n.Endpoints
			configuration.AllowUnqualifiedDNSQuery = n.AllowUnqualifiedDNSQuery
			if n.DNSSearchList != nil {
				configuration.DNSSearchList = strings.Join(n.DNSSearchList, ",")
			}
			configuration.NetworkSharedContainerName = n.NetworkSharedContainerID
			continue
		}
		if c, ok := option.(*CredentialsOption); ok {
			configuration.Credentials = c.Credentials
			continue
		}
	}

	// We must have a layer option with at least one path
	if layerOpt == nil || layerOpt.LayerPaths == nil {
		return fmt.Errorf("no layer option or paths were supplied to the runtime")
	}

	if configuration.HvPartition {
		// Find the upper-most utility VM image, since the utility VM does not
		// use layering in RS1.
		// TODO @swernli/jhowardmsft at some point post RS1 this may be re-locatable.
		var uvmImagePath string
		for _, path := range layerOpt.LayerPaths {
			fullPath := filepath.Join(path, "UtilityVM")
			_, err := os.Stat(fullPath)
			if err == nil {
				uvmImagePath = fullPath
				break
			}
			if !os.IsNotExist(err) {
				return err
			}
		}
		if uvmImagePath == "" {
			return errors.New("utility VM image could not be found")
		}
		configuration.HvRuntime = &hcsshim.HvRuntime{ImagePath: uvmImagePath}
	} else {
		configuration.VolumePath = spec.Root.Path
	}

	configuration.LayerFolderPath = layerOpt.LayerFolderPath

	for _, layerPath := range layerOpt.LayerPaths {
		_, filename := filepath.Split(layerPath)
		g, err := hcsshim.NameToGuid(filename)
		if err != nil {
			return err
		}
		configuration.Layers = append(configuration.Layers, hcsshim.Layer{
			ID:   g.ToString(),
			Path: layerPath,
		})
	}

	// Add the mounts (volumes, bind mounts etc) to the structure
	mds := make([]hcsshim.MappedDir, len(spec.Mounts))
	for i, mount := range spec.Mounts {
		mds[i] = hcsshim.MappedDir{
			HostPath:      mount.Source,
			ContainerPath: mount.Destination,
			ReadOnly:      false,
		}
		for _, o := range mount.Options {
			if strings.ToLower(o) == "ro" {
				mds[i].ReadOnly = true
			}
		}
	}
	configuration.MappedDirectories = mds

	hcsContainer, err := hcsshim.CreateContainer(containerID, configuration)
	if err != nil {
		return err
	}

	// Construct a container object for calling start on it.
	container := &container{
		containerCommon: containerCommon{
			process: process{
				processCommon: processCommon{
					containerID:  containerID,
					client:       clnt,
					friendlyName: InitFriendlyName,
				},
			},
			processes: make(map[string]*process),
		},
		ociSpec:      spec,
		hcsContainer: hcsContainer,
	}

	container.options = options
	for _, option := range options {
		if err := option.Apply(container); err != nil {
			logrus.Errorf("libcontainerd: %v", err)
		}
	}

	// Call start, and if it fails, delete the container from our
	// internal structure, start will keep HCS in sync by deleting the
	// container there.
	logrus.Debugf("libcontainerd: createWindows() id=%s, Calling start()", containerID)
	if err := container.start(attachStdio); err != nil {
		clnt.deleteContainer(containerID)
		return err
	}

	logrus.Debugf("libcontainerd: createWindows() id=%s completed successfully", containerID)
	return nil

}

func (clnt *client) createLinux(containerID string, checkpoint string, checkpointDir string, spec specs.Spec, attachStdio StdioCallback, options ...CreateOption) error {
	logrus.Debugf("libcontainerd: createLinux(): containerId %s ", containerID)

	// TODO @jhowardmsft LCOW Support: This needs to be configurable, not hard-coded.
	// However, good-enough for the LCOW bring-up.
	configuration := &hcsshim.ContainerConfig{
		HvPartition:   true,
		Name:          containerID,
		SystemType:    "container",
		ContainerType: "linux",
		Owner:         defaultOwner,
		TerminateOnLastHandleClosed: true,
		HvRuntime: &hcsshim.HvRuntime{
			ImagePath:       `c:\Program Files\Linux Containers`,
			LinuxKernelFile: `bootx64.efi`,
			LinuxInitrdFile: `initrd.img`,
		},
	}

	var layerOpt *LayerOption
	for _, option := range options {
		if l, ok := option.(*LayerOption); ok {
			layerOpt = l
		}
	}

	// We must have a layer option with at least one path
	if layerOpt == nil || layerOpt.LayerPaths == nil {
		return fmt.Errorf("no layer option or paths were supplied to the runtime")
	}

	// LayerFolderPath (writeable layer) + Layers (Guid + path)
	configuration.LayerFolderPath = layerOpt.LayerFolderPath
	for _, layerPath := range layerOpt.LayerPaths {
		_, filename := filepath.Split(layerPath)
		g, err := hcsshim.NameToGuid(filename)
		if err != nil {
			return err
		}
		configuration.Layers = append(configuration.Layers, hcsshim.Layer{
			ID:   g.ToString(),
			Path: filepath.Join(layerPath, "layer.vhd"),
		})
	}

	for _, option := range options {
		if n, ok := option.(*NetworkEndpointsOption); ok {
			configuration.EndpointList = n.Endpoints
			configuration.AllowUnqualifiedDNSQuery = n.AllowUnqualifiedDNSQuery
			if n.DNSSearchList != nil {
				configuration.DNSSearchList = strings.Join(n.DNSSearchList, ",")
			}
			configuration.NetworkSharedContainerName = n.NetworkSharedContainerID
			break
		}
	}

	hcsContainer, err := hcsshim.CreateContainer(containerID, configuration)
	if err != nil {
		return err
	}

	// Construct a container object for calling start on it.
	container := &container{
		containerCommon: containerCommon{
			process: process{
				processCommon: processCommon{
					containerID:  containerID,
					client:       clnt,
					friendlyName: InitFriendlyName,
				},
			},
			processes: make(map[string]*process),
		},
		ociSpec:      spec,
		hcsContainer: hcsContainer,
	}

	container.options = options
	for _, option := range options {
		if err := option.Apply(container); err != nil {
			logrus.Errorf("libcontainerd: createLinux() %v", err)
		}
	}

	// Call start, and if it fails, delete the container from our
	// internal structure, start will keep HCS in sync by deleting the
	// container there.
	logrus.Debugf("libcontainerd: createLinux() id=%s, Calling start()", containerID)
	if err := container.start(attachStdio); err != nil {
		clnt.deleteContainer(containerID)
		return err
	}

	logrus.Debugf("libcontainerd: createLinux() id=%s completed successfully", containerID)
	return nil
}

// AddProcess is the handler for adding a process to an already running
// container. It's called through docker exec. It returns the system pid of the
// exec'd process.
func (clnt *client) AddProcess(ctx context.Context, containerID, processFriendlyName string, procToAdd Process, attachStdio StdioCallback) (int, error) {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	container, err := clnt.getContainer(containerID)
	if err != nil {
		return -1, err
	}
	// Note we always tell HCS to
	// create stdout as it's required regardless of '-i' or '-t' options, so that
	// docker can always grab the output through logs. We also tell HCS to always
	// create stdin, even if it's not used - it will be closed shortly. Stderr
	// is only created if it we're not -t.
	createProcessParms := hcsshim.ProcessConfig{
		CreateStdInPipe:  true,
		CreateStdOutPipe: true,
		CreateStdErrPipe: !procToAdd.Terminal,
	}
	if procToAdd.Terminal {
		createProcessParms.EmulateConsole = true
		createProcessParms.ConsoleSize[0] = uint(procToAdd.ConsoleSize.Height)
		createProcessParms.ConsoleSize[1] = uint(procToAdd.ConsoleSize.Width)
	}

	// Take working directory from the process to add if it is defined,
	// otherwise take from the first process.
	if procToAdd.Cwd != "" {
		createProcessParms.WorkingDirectory = procToAdd.Cwd
	} else {
		createProcessParms.WorkingDirectory = container.ociSpec.Process.Cwd
	}

	// Configure the environment for the process
	createProcessParms.Environment = setupEnvironmentVariables(procToAdd.Env)
	if container.ociSpec.Platform.OS == "windows" {
		createProcessParms.CommandLine = strings.Join(procToAdd.Args, " ")
	} else {
		createProcessParms.CommandArgs = procToAdd.Args
	}
	createProcessParms.User = procToAdd.User.Username

	logrus.Debugf("libcontainerd: commandLine: %s", createProcessParms.CommandLine)

	// Start the command running in the container.
	var stdout, stderr io.ReadCloser
	var stdin io.WriteCloser
	newProcess, err := container.hcsContainer.CreateProcess(&createProcessParms)
	if err != nil {
		logrus.Errorf("libcontainerd: AddProcess(%s) CreateProcess() failed %s", containerID, err)
		return -1, err
	}

	pid := newProcess.Pid()

	stdin, stdout, stderr, err = newProcess.Stdio()
	if err != nil {
		logrus.Errorf("libcontainerd: %s getting std pipes failed %s", containerID, err)
		return -1, err
	}

	iopipe := &IOPipe{Terminal: procToAdd.Terminal}
	iopipe.Stdin = createStdInCloser(stdin, newProcess)

	// Convert io.ReadClosers to io.Readers
	if stdout != nil {
		iopipe.Stdout = ioutil.NopCloser(&autoClosingReader{ReadCloser: stdout})
	}
	if stderr != nil {
		iopipe.Stderr = ioutil.NopCloser(&autoClosingReader{ReadCloser: stderr})
	}

	proc := &process{
		processCommon: processCommon{
			containerID:  containerID,
			friendlyName: processFriendlyName,
			client:       clnt,
			systemPid:    uint32(pid),
		},
		hcsProcess: newProcess,
	}

	// Add the process to the container's list of processes
	container.processes[processFriendlyName] = proc

	// Tell the engine to attach streams back to the client
	if err := attachStdio(*iopipe); err != nil {
		return -1, err
	}

	// Spin up a go routine waiting for exit to handle cleanup
	go container.waitExit(proc, false)

	return pid, nil
}

// Signal handles `docker stop` on Windows. While Linux has support for
// the full range of signals, signals aren't really implemented on Windows.
// We fake supporting regular stop and -9 to force kill.
func (clnt *client) Signal(containerID string, sig int) error {
	var (
		cont *container
		err  error
	)

	// Get the container as we need it to get the container handle.
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	if cont, err = clnt.getContainer(containerID); err != nil {
		return err
	}

	cont.manualStopRequested = true

	logrus.Debugf("libcontainerd: Signal() containerID=%s sig=%d pid=%d", containerID, sig, cont.systemPid)

	if syscall.Signal(sig) == syscall.SIGKILL {
		// Terminate the compute system
		if err := cont.hcsContainer.Terminate(); err != nil {
			if !hcsshim.IsPending(err) {
				logrus.Errorf("libcontainerd: failed to terminate %s - %q", containerID, err)
			}
		}
	} else {
		// Shut down the container
		if err := cont.hcsContainer.Shutdown(); err != nil {
			if !hcsshim.IsPending(err) && !hcsshim.IsAlreadyStopped(err) {
				// ignore errors
				logrus.Warnf("libcontainerd: failed to shutdown container %s: %q", containerID, err)
			}
		}
	}

	return nil
}

// While Linux has support for the full range of signals, signals aren't really implemented on Windows.
// We try to terminate the specified process whatever signal is requested.
func (clnt *client) SignalProcess(containerID string, processFriendlyName string, sig int) error {
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	cont, err := clnt.getContainer(containerID)
	if err != nil {
		return err
	}

	for _, p := range cont.processes {
		if p.friendlyName == processFriendlyName {
			return p.hcsProcess.Kill()
		}
	}

	return fmt.Errorf("SignalProcess could not find process %s in %s", processFriendlyName, containerID)
}

// Resize handles a CLI event to resize an interactive docker run or docker exec
// window.
func (clnt *client) Resize(containerID, processFriendlyName string, width, height int) error {
	// Get the libcontainerd container object
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	cont, err := clnt.getContainer(containerID)
	if err != nil {
		return err
	}

	h, w := uint16(height), uint16(width)

	if processFriendlyName == InitFriendlyName {
		logrus.Debugln("libcontainerd: resizing systemPID in", containerID, cont.process.systemPid)
		return cont.process.hcsProcess.ResizeConsole(w, h)
	}

	for _, p := range cont.processes {
		if p.friendlyName == processFriendlyName {
			logrus.Debugln("libcontainerd: resizing exec'd process", containerID, p.systemPid)
			return p.hcsProcess.ResizeConsole(w, h)
		}
	}

	return fmt.Errorf("Resize could not find containerID %s to resize", containerID)

}

// Pause handles pause requests for containers
func (clnt *client) Pause(containerID string) error {
	unlockContainer := true
	// Get the libcontainerd container object
	clnt.lock(containerID)
	defer func() {
		if unlockContainer {
			clnt.unlock(containerID)
		}
	}()
	container, err := clnt.getContainer(containerID)
	if err != nil {
		return err
	}

	for _, option := range container.options {
		if h, ok := option.(*HyperVIsolationOption); ok {
			if !h.IsHyperV {
				return errors.New("cannot pause Windows Server Containers")
			}
			break
		}
	}

	err = container.hcsContainer.Pause()
	if err != nil {
		return err
	}

	// Unlock container before calling back into the daemon
	unlockContainer = false
	clnt.unlock(containerID)

	return clnt.backend.StateChanged(containerID, StateInfo{
		CommonStateInfo: CommonStateInfo{
			State: StatePause,
		}})
}

// Resume handles resume requests for containers
func (clnt *client) Resume(containerID string) error {
	unlockContainer := true
	// Get the libcontainerd container object
	clnt.lock(containerID)
	defer func() {
		if unlockContainer {
			clnt.unlock(containerID)
		}
	}()
	container, err := clnt.getContainer(containerID)
	if err != nil {
		return err
	}

	// This should never happen, since Windows Server Containers cannot be paused
	for _, option := range container.options {
		if h, ok := option.(*HyperVIsolationOption); ok {
			if !h.IsHyperV {
				return errors.New("cannot resume Windows Server Containers")
			}
			break
		}
	}

	err = container.hcsContainer.Resume()
	if err != nil {
		return err
	}

	// Unlock container before calling back into the daemon
	unlockContainer = false
	clnt.unlock(containerID)

	return clnt.backend.StateChanged(containerID, StateInfo{
		CommonStateInfo: CommonStateInfo{
			State: StateResume,
		}})
}

// Stats handles stats requests for containers
func (clnt *client) Stats(containerID string) (*Stats, error) {
	// Get the libcontainerd container object
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	container, err := clnt.getContainer(containerID)
	if err != nil {
		return nil, err
	}
	s, err := container.hcsContainer.Statistics()
	if err != nil {
		return nil, err
	}
	st := Stats(s)
	return &st, nil
}

// Restore is the handler for restoring a container
func (clnt *client) Restore(containerID string, _ StdioCallback, unusedOnWindows ...CreateOption) error {
	logrus.Debugf("libcontainerd: Restore(%s)", containerID)

	// TODO Windows: On RS1, a re-attach isn't possible.
	// However, there is a scenario in which there is an issue.
	// Consider a background container. The daemon dies unexpectedly.
	// HCS will still have the compute service alive and running.
	// For consistence, we call in to shoot it regardless if HCS knows about it
	// We explicitly just log a warning if the terminate fails.
	// Then we tell the backend the container exited.
	if hc, err := hcsshim.OpenContainer(containerID); err == nil {
		const terminateTimeout = time.Minute * 2
		err := hc.Terminate()

		if hcsshim.IsPending(err) {
			err = hc.WaitTimeout(terminateTimeout)
		} else if hcsshim.IsAlreadyStopped(err) {
			err = nil
		}

		if err != nil {
			logrus.Warnf("libcontainerd: failed to terminate %s on restore - %q", containerID, err)
			return err
		}
	}
	return clnt.backend.StateChanged(containerID, StateInfo{
		CommonStateInfo: CommonStateInfo{
			State:    StateExit,
			ExitCode: 1 << 31,
		}})
}

// GetPidsForContainer returns a list of process IDs running in a container.
// Not used on Windows.
func (clnt *client) GetPidsForContainer(containerID string) ([]int, error) {
	return nil, errors.New("not implemented on Windows")
}

// Summary returns a summary of the processes running in a container.
// This is present in Windows to support docker top. In linux, the
// engine shells out to ps to get process information. On Windows, as
// the containers could be Hyper-V containers, they would not be
// visible on the container host. However, libcontainerd does have
// that information.
func (clnt *client) Summary(containerID string) ([]Summary, error) {

	// Get the libcontainerd container object
	clnt.lock(containerID)
	defer clnt.unlock(containerID)
	container, err := clnt.getContainer(containerID)
	if err != nil {
		return nil, err
	}
	p, err := container.hcsContainer.ProcessList()
	if err != nil {
		return nil, err
	}
	pl := make([]Summary, len(p))
	for i := range p {
		pl[i] = Summary(p[i])
	}
	return pl, nil
}

// UpdateResources updates resources for a running container.
func (clnt *client) UpdateResources(containerID string, resources Resources) error {
	// Updating resource isn't supported on Windows
	// but we should return nil for enabling updating container
	return nil
}

func (clnt *client) CreateCheckpoint(containerID string, checkpointID string, checkpointDir string, exit bool) error {
	return errors.New("Windows: Containers do not support checkpoints")
}

func (clnt *client) DeleteCheckpoint(containerID string, checkpointID string, checkpointDir string) error {
	return errors.New("Windows: Containers do not support checkpoints")
}

func (clnt *client) ListCheckpoints(containerID string, checkpointDir string) (*Checkpoints, error) {
	return nil, errors.New("Windows: Containers do not support checkpoints")
}

func (clnt *client) GetServerVersion(ctx context.Context) (*ServerVersion, error) {
	return &ServerVersion{}, nil
}
