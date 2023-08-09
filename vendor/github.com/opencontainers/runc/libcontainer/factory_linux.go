package libcontainer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"strconv"

	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/moby/sys/mountinfo"
	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups/manager"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/configs/validate"
	"github.com/opencontainers/runc/libcontainer/intelrdt"
	"github.com/opencontainers/runc/libcontainer/utils"
	"github.com/sirupsen/logrus"
)

const (
	stateFilename    = "state.json"
	execFifoFilename = "exec.fifo"
)

var idRegex = regexp.MustCompile(`^[\w+-\.]+$`)

// InitArgs returns an options func to configure a LinuxFactory with the
// provided init binary path and arguments.
func InitArgs(args ...string) func(*LinuxFactory) error {
	return func(l *LinuxFactory) (err error) {
		if len(args) > 0 {
			// Resolve relative paths to ensure that its available
			// after directory changes.
			if args[0], err = filepath.Abs(args[0]); err != nil {
				// The only error returned from filepath.Abs is
				// the one from os.Getwd, i.e. a system error.
				return err
			}
		}

		l.InitArgs = args
		return nil
	}
}

// IntelRdtfs is an options func to configure a LinuxFactory to return
// containers that use the Intel RDT "resource control" filesystem to
// create and manage Intel RDT resources (e.g., L3 cache, memory bandwidth).
func IntelRdtFs(l *LinuxFactory) error {
	if !intelrdt.IsCATEnabled() && !intelrdt.IsMBAEnabled() {
		l.NewIntelRdtManager = nil
	} else {
		l.NewIntelRdtManager = func(config *configs.Config, id string, path string) intelrdt.Manager {
			return intelrdt.NewManager(config, id, path)
		}
	}
	return nil
}

// TmpfsRoot is an option func to mount LinuxFactory.Root to tmpfs.
func TmpfsRoot(l *LinuxFactory) error {
	mounted, err := mountinfo.Mounted(l.Root)
	if err != nil {
		return err
	}
	if !mounted {
		if err := mount("tmpfs", l.Root, "", "tmpfs", 0, ""); err != nil {
			return err
		}
	}
	return nil
}

// CriuPath returns an option func to configure a LinuxFactory with the
// provided criupath
func CriuPath(criupath string) func(*LinuxFactory) error {
	return func(l *LinuxFactory) error {
		l.CriuPath = criupath
		return nil
	}
}

// New returns a linux based container factory based in the root directory and
// configures the factory with the provided option funcs.
func New(root string, options ...func(*LinuxFactory) error) (Factory, error) {
	if root != "" {
		if err := os.MkdirAll(root, 0o700); err != nil {
			return nil, err
		}
	}
	l := &LinuxFactory{
		Root:      root,
		InitPath:  "/proc/self/exe",
		InitArgs:  []string{os.Args[0], "init"},
		Validator: validate.New(),
		CriuPath:  "criu",
	}

	for _, opt := range options {
		if opt == nil {
			continue
		}
		if err := opt(l); err != nil {
			return nil, err
		}
	}
	return l, nil
}

// LinuxFactory implements the default factory interface for linux based systems.
type LinuxFactory struct {
	// Root directory for the factory to store state.
	Root string

	// InitPath is the path for calling the init responsibilities for spawning
	// a container.
	InitPath string

	// InitArgs are arguments for calling the init responsibilities for spawning
	// a container.
	InitArgs []string

	// CriuPath is the path to the criu binary used for checkpoint and restore of
	// containers.
	CriuPath string

	// New{u,g}idmapPath is the path to the binaries used for mapping with
	// rootless containers.
	NewuidmapPath string
	NewgidmapPath string

	// Validator provides validation to container configurations.
	Validator validate.Validator

	// NewIntelRdtManager returns an initialized Intel RDT manager for a single container.
	NewIntelRdtManager func(config *configs.Config, id string, path string) intelrdt.Manager
}

func (l *LinuxFactory) Create(id string, config *configs.Config) (Container, error) {
	if l.Root == "" {
		return nil, errors.New("root not set")
	}
	if err := l.validateID(id); err != nil {
		return nil, err
	}
	if err := l.Validator.Validate(config); err != nil {
		return nil, err
	}
	containerRoot, err := securejoin.SecureJoin(l.Root, id)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(containerRoot); err == nil {
		return nil, ErrExist
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	cm, err := manager.New(config.Cgroups)
	if err != nil {
		return nil, err
	}

	// Check that cgroup does not exist or empty (no processes).
	// Note for cgroup v1 this check is not thorough, as there are multiple
	// separate hierarchies, while both Exists() and GetAllPids() only use
	// one for "devices" controller (assuming others are the same, which is
	// probably true in almost all scenarios). Checking all the hierarchies
	// would be too expensive.
	if cm.Exists() {
		pids, err := cm.GetAllPids()
		// Reading PIDs can race with cgroups removal, so ignore ENOENT and ENODEV.
		if err != nil && !errors.Is(err, os.ErrNotExist) && !errors.Is(err, unix.ENODEV) {
			return nil, fmt.Errorf("unable to get cgroup PIDs: %w", err)
		}
		if len(pids) != 0 {
			// TODO: return an error.
			logrus.Warnf("container's cgroup is not empty: %d process(es) found", len(pids))
			logrus.Warn("DEPRECATED: running container in a non-empty cgroup won't be supported in runc 1.2; https://github.com/opencontainers/runc/issues/3132")
		}
	}

	// Check that cgroup is not frozen. Do not use Exists() here
	// since in cgroup v1 it only checks "devices" controller.
	st, err := cm.GetFreezerState()
	if err != nil {
		return nil, fmt.Errorf("unable to get cgroup freezer state: %w", err)
	}
	if st == configs.Frozen {
		return nil, errors.New("container's cgroup unexpectedly frozen")
	}

	if err := os.MkdirAll(containerRoot, 0o711); err != nil {
		return nil, err
	}
	if err := os.Chown(containerRoot, unix.Geteuid(), unix.Getegid()); err != nil {
		return nil, err
	}
	c := &linuxContainer{
		id:            id,
		root:          containerRoot,
		config:        config,
		initPath:      l.InitPath,
		initArgs:      l.InitArgs,
		criuPath:      l.CriuPath,
		newuidmapPath: l.NewuidmapPath,
		newgidmapPath: l.NewgidmapPath,
		cgroupManager: cm,
	}
	if l.NewIntelRdtManager != nil {
		c.intelRdtManager = l.NewIntelRdtManager(config, id, "")
	}
	c.state = &stoppedState{c: c}
	return c, nil
}

func (l *LinuxFactory) Load(id string) (Container, error) {
	if l.Root == "" {
		return nil, errors.New("root not set")
	}
	// when load, we need to check id is valid or not.
	if err := l.validateID(id); err != nil {
		return nil, err
	}
	containerRoot, err := securejoin.SecureJoin(l.Root, id)
	if err != nil {
		return nil, err
	}
	state, err := l.loadState(containerRoot)
	if err != nil {
		return nil, err
	}
	r := &nonChildProcess{
		processPid:       state.InitProcessPid,
		processStartTime: state.InitProcessStartTime,
		fds:              state.ExternalDescriptors,
	}
	cm, err := manager.NewWithPaths(state.Config.Cgroups, state.CgroupPaths)
	if err != nil {
		return nil, err
	}
	c := &linuxContainer{
		initProcess:          r,
		initProcessStartTime: state.InitProcessStartTime,
		id:                   id,
		config:               &state.Config,
		initPath:             l.InitPath,
		initArgs:             l.InitArgs,
		criuPath:             l.CriuPath,
		newuidmapPath:        l.NewuidmapPath,
		newgidmapPath:        l.NewgidmapPath,
		cgroupManager:        cm,
		root:                 containerRoot,
		created:              state.Created,
	}
	if l.NewIntelRdtManager != nil {
		c.intelRdtManager = l.NewIntelRdtManager(&state.Config, id, state.IntelRdtPath)
	}
	c.state = &loadedState{c: c}
	if err := c.refreshState(); err != nil {
		return nil, err
	}
	return c, nil
}

func (l *LinuxFactory) Type() string {
	return "libcontainer"
}

// StartInitialization loads a container by opening the pipe fd from the parent to read the configuration and state
// This is a low level implementation detail of the reexec and should not be consumed externally
func (l *LinuxFactory) StartInitialization() (err error) {
	// Get the INITPIPE.
	envInitPipe := os.Getenv("_LIBCONTAINER_INITPIPE")
	pipefd, err := strconv.Atoi(envInitPipe)
	if err != nil {
		err = fmt.Errorf("unable to convert _LIBCONTAINER_INITPIPE: %w", err)
		logrus.Error(err)
		return err
	}
	pipe := os.NewFile(uintptr(pipefd), "pipe")
	defer pipe.Close()

	defer func() {
		// We have an error during the initialization of the container's init,
		// send it back to the parent process in the form of an initError.
		if werr := writeSync(pipe, procError); werr != nil {
			fmt.Fprintln(os.Stderr, err)
			return
		}
		if werr := utils.WriteJSON(pipe, &initError{Message: err.Error()}); werr != nil {
			fmt.Fprintln(os.Stderr, err)
			return
		}
	}()

	// Only init processes have FIFOFD.
	fifofd := -1
	envInitType := os.Getenv("_LIBCONTAINER_INITTYPE")
	it := initType(envInitType)
	if it == initStandard {
		envFifoFd := os.Getenv("_LIBCONTAINER_FIFOFD")
		if fifofd, err = strconv.Atoi(envFifoFd); err != nil {
			return fmt.Errorf("unable to convert _LIBCONTAINER_FIFOFD: %w", err)
		}
	}

	var consoleSocket *os.File
	if envConsole := os.Getenv("_LIBCONTAINER_CONSOLE"); envConsole != "" {
		console, err := strconv.Atoi(envConsole)
		if err != nil {
			return fmt.Errorf("unable to convert _LIBCONTAINER_CONSOLE: %w", err)
		}
		consoleSocket = os.NewFile(uintptr(console), "console-socket")
		defer consoleSocket.Close()
	}

	logPipeFdStr := os.Getenv("_LIBCONTAINER_LOGPIPE")
	logPipeFd, err := strconv.Atoi(logPipeFdStr)
	if err != nil {
		return fmt.Errorf("unable to convert _LIBCONTAINER_LOGPIPE: %w", err)
	}

	// Get mount files (O_PATH).
	mountFds, err := parseMountFds()
	if err != nil {
		return err
	}

	// clear the current process's environment to clean any libcontainer
	// specific env vars.
	os.Clearenv()

	defer func() {
		if e := recover(); e != nil {
			if e, ok := e.(error); ok {
				err = fmt.Errorf("panic from initialization: %w, %s", e, debug.Stack())
			} else {
				//nolint:errorlint // here e is not of error type
				err = fmt.Errorf("panic from initialization: %v, %s", e, debug.Stack())
			}
		}
	}()

	i, err := newContainerInit(it, pipe, consoleSocket, fifofd, logPipeFd, mountFds)
	if err != nil {
		return err
	}

	// If Init succeeds, syscall.Exec will not return, hence none of the defers will be called.
	return i.Init()
}

func (l *LinuxFactory) loadState(root string) (*State, error) {
	stateFilePath, err := securejoin.SecureJoin(root, stateFilename)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(stateFilePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrNotExist
		}
		return nil, err
	}
	defer f.Close()
	var state *State
	if err := json.NewDecoder(f).Decode(&state); err != nil {
		return nil, err
	}
	return state, nil
}

func (l *LinuxFactory) validateID(id string) error {
	if !idRegex.MatchString(id) || string(os.PathSeparator)+id != utils.CleanPath(string(os.PathSeparator)+id) {
		return ErrInvalidID
	}

	return nil
}

// NewuidmapPath returns an option func to configure a LinuxFactory with the
// provided ..
func NewuidmapPath(newuidmapPath string) func(*LinuxFactory) error {
	return func(l *LinuxFactory) error {
		l.NewuidmapPath = newuidmapPath
		return nil
	}
}

// NewgidmapPath returns an option func to configure a LinuxFactory with the
// provided ..
func NewgidmapPath(newgidmapPath string) func(*LinuxFactory) error {
	return func(l *LinuxFactory) error {
		l.NewgidmapPath = newgidmapPath
		return nil
	}
}

func parseMountFds() ([]int, error) {
	fdsJson := os.Getenv("_LIBCONTAINER_MOUNT_FDS")
	if fdsJson == "" {
		// Always return the nil slice if no fd is present.
		return nil, nil
	}

	var mountFds []int
	if err := json.Unmarshal([]byte(fdsJson), &mountFds); err != nil {
		return nil, fmt.Errorf("Error unmarshalling _LIBCONTAINER_MOUNT_FDS: %w", err)
	}

	return mountFds, nil
}
