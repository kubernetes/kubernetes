package daemon

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/api"
	"github.com/docker/docker/daemon/events"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/daemon/execdriver/execdrivers"
	"github.com/docker/docker/daemon/graphdriver"
	_ "github.com/docker/docker/daemon/graphdriver/vfs"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/network"
	"github.com/docker/docker/graph"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/broadcastwriter"
	"github.com/docker/docker/pkg/fileutils"
	"github.com/docker/docker/pkg/graphdb"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/namesgenerator"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/sysinfo"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/truncindex"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/trust"
	"github.com/docker/libnetwork"
	"github.com/opencontainers/runc/libcontainer/netlink"
)

var (
	validContainerNameChars   = `[a-zA-Z0-9][a-zA-Z0-9_.-]`
	validContainerNamePattern = regexp.MustCompile(`^/?` + validContainerNameChars + `+$`)

	ErrSystemNotSupported = errors.New("The Docker daemon is not supported on this platform.")
)

type contStore struct {
	s map[string]*Container
	sync.Mutex
}

func (c *contStore) Add(id string, cont *Container) {
	c.Lock()
	c.s[id] = cont
	c.Unlock()
}

func (c *contStore) Get(id string) *Container {
	c.Lock()
	res := c.s[id]
	c.Unlock()
	return res
}

func (c *contStore) Delete(id string) {
	c.Lock()
	delete(c.s, id)
	c.Unlock()
}

func (c *contStore) List() []*Container {
	containers := new(History)
	c.Lock()
	for _, cont := range c.s {
		containers.Add(cont)
	}
	c.Unlock()
	containers.Sort()
	return *containers
}

type Daemon struct {
	ID               string
	repository       string
	sysInitPath      string
	containers       *contStore
	execCommands     *execStore
	graph            *graph.Graph
	repositories     *graph.TagStore
	idIndex          *truncindex.TruncIndex
	sysInfo          *sysinfo.SysInfo
	config           *Config
	containerGraph   *graphdb.Database
	driver           graphdriver.Driver
	execDriver       execdriver.Driver
	statsCollector   *statsCollector
	defaultLogConfig runconfig.LogConfig
	RegistryService  *registry.Service
	EventsService    *events.Events
	netController    libnetwork.NetworkController
	root             string
}

// Get looks for a container using the provided information, which could be
// one of the following inputs from the caller:
//  - A full container ID, which will exact match a container in daemon's list
//  - A container name, which will only exact match via the GetByName() function
//  - A partial container ID prefix (e.g. short ID) of any length that is
//    unique enough to only return a single container object
//  If none of these searches succeed, an error is returned
func (daemon *Daemon) Get(prefixOrName string) (*Container, error) {
	if containerByID := daemon.containers.Get(prefixOrName); containerByID != nil {
		// prefix is an exact match to a full container ID
		return containerByID, nil
	}

	// GetByName will match only an exact name provided; we ignore errors
	if containerByName, _ := daemon.GetByName(prefixOrName); containerByName != nil {
		// prefix is an exact match to a full container Name
		return containerByName, nil
	}

	containerId, indexError := daemon.idIndex.Get(prefixOrName)
	if indexError != nil {
		return nil, indexError
	}
	return daemon.containers.Get(containerId), nil
}

// Exists returns a true if a container of the specified ID or name exists,
// false otherwise.
func (daemon *Daemon) Exists(id string) bool {
	c, _ := daemon.Get(id)
	return c != nil
}

func (daemon *Daemon) containerRoot(id string) string {
	return filepath.Join(daemon.repository, id)
}

// Load reads the contents of a container from disk
// This is typically done at startup.
func (daemon *Daemon) load(id string) (*Container, error) {
	container := &Container{
		CommonContainer: daemon.newBaseContainer(id),
	}

	if err := container.FromDisk(); err != nil {
		return nil, err
	}

	if container.ID != id {
		return container, fmt.Errorf("Container %s is stored at %s", container.ID, id)
	}

	return container, nil
}

// Register makes a container object usable by the daemon as <container.ID>
// This is a wrapper for register
func (daemon *Daemon) Register(container *Container) error {
	return daemon.register(container, true)
}

// register makes a container object usable by the daemon as <container.ID>
func (daemon *Daemon) register(container *Container, updateSuffixarray bool) error {
	if container.daemon != nil || daemon.Exists(container.ID) {
		return fmt.Errorf("Container is already loaded")
	}
	if err := validateID(container.ID); err != nil {
		return err
	}
	if err := daemon.ensureName(container); err != nil {
		return err
	}

	container.daemon = daemon

	// Attach to stdout and stderr
	container.stderr = broadcastwriter.New()
	container.stdout = broadcastwriter.New()
	// Attach to stdin
	if container.Config.OpenStdin {
		container.stdin, container.stdinPipe = io.Pipe()
	} else {
		container.stdinPipe = ioutils.NopWriteCloser(ioutil.Discard) // Silently drop stdin
	}
	// done
	daemon.containers.Add(container.ID, container)

	// don't update the Suffixarray if we're starting up
	// we'll waste time if we update it for every container
	daemon.idIndex.Add(container.ID)

	if err := daemon.verifyVolumesInfo(container); err != nil {
		return err
	}

	if err := container.prepareMountPoints(); err != nil {
		return err
	}

	if container.IsRunning() {
		logrus.Debugf("killing old running container %s", container.ID)
		// Set exit code to 128 + SIGKILL (9) to properly represent unsuccessful exit
		container.SetStopped(&execdriver.ExitStatus{ExitCode: 137})

		// use the current driver and ensure that the container is dead x.x
		cmd := &execdriver.Command{
			ID: container.ID,
		}
		daemon.execDriver.Terminate(cmd)

		if err := container.Unmount(); err != nil {
			logrus.Debugf("unmount error %s", err)
		}
		if err := container.ToDisk(); err != nil {
			logrus.Errorf("Error saving stopped state to disk: %v", err)
		}
	}

	return nil
}

func (daemon *Daemon) ensureName(container *Container) error {
	if container.Name == "" {
		name, err := daemon.generateNewName(container.ID)
		if err != nil {
			return err
		}
		container.Name = name

		if err := container.ToDisk(); err != nil {
			logrus.Errorf("Error saving container name to disk: %v", err)
		}
	}
	return nil
}

func (daemon *Daemon) restore() error {
	type cr struct {
		container  *Container
		registered bool
	}

	var (
		debug         = (os.Getenv("DEBUG") != "" || os.Getenv("TEST") != "")
		currentDriver = daemon.driver.String()
		containers    = make(map[string]*cr)
	)

	if !debug {
		logrus.Info("Loading containers: start.")
	}
	dir, err := ioutil.ReadDir(daemon.repository)
	if err != nil {
		return err
	}

	for _, v := range dir {
		id := v.Name()
		container, err := daemon.load(id)
		if !debug && logrus.GetLevel() == logrus.InfoLevel {
			fmt.Print(".")
		}
		if err != nil {
			logrus.Errorf("Failed to load container %v: %v", id, err)
			continue
		}

		// Ignore the container if it does not support the current driver being used by the graph
		if (container.Driver == "" && currentDriver == "aufs") || container.Driver == currentDriver {
			logrus.Debugf("Loaded container %v", container.ID)

			containers[container.ID] = &cr{container: container}
		} else {
			logrus.Debugf("Cannot load container %s because it was created with another graph driver.", container.ID)
		}
	}

	if entities := daemon.containerGraph.List("/", -1); entities != nil {
		for _, p := range entities.Paths() {
			if !debug && logrus.GetLevel() == logrus.InfoLevel {
				fmt.Print(".")
			}

			e := entities[p]

			if c, ok := containers[e.ID()]; ok {
				c.registered = true
			}
		}
	}

	group := sync.WaitGroup{}
	for _, c := range containers {
		group.Add(1)

		go func(container *Container, registered bool) {
			defer group.Done()

			if !registered {
				// Try to set the default name for a container if it exists prior to links
				container.Name, err = daemon.generateNewName(container.ID)
				if err != nil {
					logrus.Debugf("Setting default id - %s", err)
				}
			}

			if err := daemon.register(container, false); err != nil {
				logrus.Debugf("Failed to register container %s: %s", container.ID, err)
			}

			// check the restart policy on the containers and restart any container with
			// the restart policy of "always"
			if daemon.config.AutoRestart && container.shouldRestart() {
				logrus.Debugf("Starting container %s", container.ID)

				if err := container.Start(); err != nil {
					logrus.Debugf("Failed to start container %s: %s", container.ID, err)
				}
			}
		}(c.container, c.registered)
	}
	group.Wait()

	if !debug {
		if logrus.GetLevel() == logrus.InfoLevel {
			fmt.Println()
		}
		logrus.Info("Loading containers: done.")
	}

	return nil
}

func (daemon *Daemon) mergeAndVerifyConfig(config *runconfig.Config, img *image.Image) error {
	if img != nil && img.Config != nil {
		if err := runconfig.Merge(config, img.Config); err != nil {
			return err
		}
	}
	if config.Entrypoint.Len() == 0 && config.Cmd.Len() == 0 {
		return fmt.Errorf("No command specified")
	}
	return nil
}

func (daemon *Daemon) generateIdAndName(name string) (string, string, error) {
	var (
		err error
		id  = stringid.GenerateRandomID()
	)

	if name == "" {
		if name, err = daemon.generateNewName(id); err != nil {
			return "", "", err
		}
		return id, name, nil
	}

	if name, err = daemon.reserveName(id, name); err != nil {
		return "", "", err
	}

	return id, name, nil
}

func (daemon *Daemon) reserveName(id, name string) (string, error) {
	if !validContainerNamePattern.MatchString(name) {
		return "", fmt.Errorf("Invalid container name (%s), only %s are allowed", name, validContainerNameChars)
	}

	if name[0] != '/' {
		name = "/" + name
	}

	if _, err := daemon.containerGraph.Set(name, id); err != nil {
		if !graphdb.IsNonUniqueNameError(err) {
			return "", err
		}

		conflictingContainer, err := daemon.GetByName(name)
		if err != nil {
			if strings.Contains(err.Error(), "Could not find entity") {
				return "", err
			}

			// Remove name and continue starting the container
			if err := daemon.containerGraph.Delete(name); err != nil {
				return "", err
			}
		} else {
			nameAsKnownByUser := strings.TrimPrefix(name, "/")
			return "", fmt.Errorf(
				"Conflict. The name %q is already in use by container %s. You have to delete (or rename) that container to be able to reuse that name.", nameAsKnownByUser,
				stringid.TruncateID(conflictingContainer.ID))
		}
	}
	return name, nil
}

func (daemon *Daemon) generateNewName(id string) (string, error) {
	var name string
	for i := 0; i < 6; i++ {
		name = namesgenerator.GetRandomName(i)
		if name[0] != '/' {
			name = "/" + name
		}

		if _, err := daemon.containerGraph.Set(name, id); err != nil {
			if !graphdb.IsNonUniqueNameError(err) {
				return "", err
			}
			continue
		}
		return name, nil
	}

	name = "/" + stringid.TruncateID(id)
	if _, err := daemon.containerGraph.Set(name, id); err != nil {
		return "", err
	}
	return name, nil
}

func (daemon *Daemon) generateHostname(id string, config *runconfig.Config) {
	// Generate default hostname
	// FIXME: the lxc template no longer needs to set a default hostname
	if config.Hostname == "" {
		config.Hostname = id[:12]
	}
}

func (daemon *Daemon) getEntrypointAndArgs(configEntrypoint *runconfig.Entrypoint, configCmd *runconfig.Command) (string, []string) {
	var (
		entrypoint string
		args       []string
	)

	cmdSlice := configCmd.Slice()
	if configEntrypoint.Len() != 0 {
		eSlice := configEntrypoint.Slice()
		entrypoint = eSlice[0]
		args = append(eSlice[1:], cmdSlice...)
	} else {
		entrypoint = cmdSlice[0]
		args = cmdSlice[1:]
	}
	return entrypoint, args
}

func (daemon *Daemon) newContainer(name string, config *runconfig.Config, imgID string) (*Container, error) {
	var (
		id  string
		err error
	)
	id, name, err = daemon.generateIdAndName(name)
	if err != nil {
		return nil, err
	}

	daemon.generateHostname(id, config)
	entrypoint, args := daemon.getEntrypointAndArgs(config.Entrypoint, config.Cmd)

	base := daemon.newBaseContainer(id)
	base.Created = time.Now().UTC()
	base.Path = entrypoint
	base.Args = args //FIXME: de-duplicate from config
	base.Config = config
	base.hostConfig = &runconfig.HostConfig{}
	base.ImageID = imgID
	base.NetworkSettings = &network.Settings{}
	base.Name = name
	base.Driver = daemon.driver.String()
	base.ExecDriver = daemon.execDriver.Name()

	container := &Container{
		CommonContainer: base,
	}

	return container, err
}

func GetFullContainerName(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("Container name cannot be empty")
	}
	if name[0] != '/' {
		name = "/" + name
	}
	return name, nil
}

func (daemon *Daemon) GetByName(name string) (*Container, error) {
	fullName, err := GetFullContainerName(name)
	if err != nil {
		return nil, err
	}
	entity := daemon.containerGraph.Get(fullName)
	if entity == nil {
		return nil, fmt.Errorf("Could not find entity for %s", name)
	}
	e := daemon.containers.Get(entity.ID())
	if e == nil {
		return nil, fmt.Errorf("Could not find container for entity id %s", entity.ID())
	}
	return e, nil
}

func (daemon *Daemon) Children(name string) (map[string]*Container, error) {
	name, err := GetFullContainerName(name)
	if err != nil {
		return nil, err
	}
	children := make(map[string]*Container)

	err = daemon.containerGraph.Walk(name, func(p string, e *graphdb.Entity) error {
		c, err := daemon.Get(e.ID())
		if err != nil {
			return err
		}
		children[p] = c
		return nil
	}, 0)

	if err != nil {
		return nil, err
	}
	return children, nil
}

func (daemon *Daemon) Parents(name string) ([]string, error) {
	name, err := GetFullContainerName(name)
	if err != nil {
		return nil, err
	}

	return daemon.containerGraph.Parents(name)
}

func (daemon *Daemon) RegisterLink(parent, child *Container, alias string) error {
	fullName := filepath.Join(parent.Name, alias)
	if !daemon.containerGraph.Exists(fullName) {
		_, err := daemon.containerGraph.Set(fullName, child.ID)
		return err
	}
	return nil
}

func NewDaemon(config *Config, registryService *registry.Service) (daemon *Daemon, err error) {
	setDefaultMtu(config)

	// Ensure we have compatible configuration options
	if err := checkConfigOptions(config); err != nil {
		return nil, err
	}

	// Do we have a disabled network?
	config.DisableBridge = isBridgeNetworkDisabled(config)

	// Verify the platform is supported as a daemon
	if runtime.GOOS != "linux" && runtime.GOOS != "windows" {
		return nil, ErrSystemNotSupported
	}

	// Validate platform-specific requirements
	if err := checkSystem(); err != nil {
		return nil, err
	}

	// set up SIGUSR1 handler on Unix-like systems, or a Win32 global event
	// on Windows to dump Go routine stacks
	setupDumpStackTrap()

	// get the canonical path to the Docker root directory
	var realRoot string
	if _, err := os.Stat(config.Root); err != nil && os.IsNotExist(err) {
		realRoot = config.Root
	} else {
		realRoot, err = fileutils.ReadSymlinkedDirectory(config.Root)
		if err != nil {
			return nil, fmt.Errorf("Unable to get the full path to root (%s): %s", config.Root, err)
		}
	}
	config.Root = realRoot
	// Create the root directory if it doesn't exists
	if err := system.MkdirAll(config.Root, 0700); err != nil && !os.IsExist(err) {
		return nil, err
	}

	// set up the tmpDir to use a canonical path
	tmp, err := tempDir(config.Root)
	if err != nil {
		return nil, fmt.Errorf("Unable to get the TempDir under %s: %s", config.Root, err)
	}
	realTmp, err := fileutils.ReadSymlinkedDirectory(tmp)
	if err != nil {
		return nil, fmt.Errorf("Unable to get the full path to the TempDir (%s): %s", tmp, err)
	}
	os.Setenv("TMPDIR", realTmp)

	// Set the default driver
	graphdriver.DefaultDriver = config.GraphDriver

	// Load storage driver
	driver, err := graphdriver.New(config.Root, config.GraphOptions)
	if err != nil {
		return nil, fmt.Errorf("error initializing graphdriver: %v", err)
	}
	logrus.Debugf("Using graph driver %s", driver)

	d := &Daemon{}
	d.driver = driver

	// Ensure the graph driver is shutdown at a later point
	defer func() {
		if err != nil {
			if err := d.Shutdown(); err != nil {
				logrus.Error(err)
			}
		}
	}()

	// Verify logging driver type
	if config.LogConfig.Type != "none" {
		if _, err := logger.GetLogDriver(config.LogConfig.Type); err != nil {
			return nil, fmt.Errorf("error finding the logging driver: %v", err)
		}
	}
	logrus.Debugf("Using default logging driver %s", config.LogConfig.Type)

	// Configure and validate the kernels security support
	if err := configureKernelSecuritySupport(config, d.driver.String()); err != nil {
		return nil, err
	}

	daemonRepo := filepath.Join(config.Root, "containers")

	if err := system.MkdirAll(daemonRepo, 0700); err != nil && !os.IsExist(err) {
		return nil, err
	}

	// Migrate the container if it is aufs and aufs is enabled
	if err := migrateIfDownlevel(d.driver, config.Root); err != nil {
		return nil, err
	}

	logrus.Debug("Creating images graph")
	g, err := graph.NewGraph(filepath.Join(config.Root, "graph"), d.driver)
	if err != nil {
		return nil, err
	}

	// Configure the volumes driver
	if err := configureVolumes(config); err != nil {
		return nil, err
	}

	trustKey, err := api.LoadOrCreateTrustKey(config.TrustKeyPath)
	if err != nil {
		return nil, err
	}

	trustDir := filepath.Join(config.Root, "trust")

	if err := system.MkdirAll(trustDir, 0700); err != nil && !os.IsExist(err) {
		return nil, err
	}
	trustService, err := trust.NewTrustStore(trustDir)
	if err != nil {
		return nil, fmt.Errorf("could not create trust store: %s", err)
	}

	eventsService := events.New()
	logrus.Debug("Creating repository list")
	tagCfg := &graph.TagStoreConfig{
		Graph:    g,
		Key:      trustKey,
		Registry: registryService,
		Events:   eventsService,
		Trust:    trustService,
	}
	repositories, err := graph.NewTagStore(filepath.Join(config.Root, "repositories-"+d.driver.String()), tagCfg)
	if err != nil {
		return nil, fmt.Errorf("Couldn't create Tag store: %s", err)
	}

	d.netController, err = initNetworkController(config)
	if err != nil {
		return nil, fmt.Errorf("Error initializing network controller: %v", err)
	}

	graphdbPath := filepath.Join(config.Root, "linkgraph.db")
	graph, err := graphdb.NewSqliteConn(graphdbPath)
	if err != nil {
		return nil, err
	}

	d.containerGraph = graph

	var sysInitPath string
	if config.ExecDriver == "lxc" {
		initPath, err := configureSysInit(config)
		if err != nil {
			return nil, err
		}
		sysInitPath = initPath
	}

	sysInfo := sysinfo.New(false)
	// Check if Devices cgroup is mounted, it is hard requirement for container security,
	// on Linux/FreeBSD.
	if runtime.GOOS != "windows" && !sysInfo.CgroupDevicesEnabled {
		return nil, fmt.Errorf("Devices cgroup isn't mounted")
	}

	ed, err := execdrivers.NewDriver(config.ExecDriver, config.ExecOptions, config.ExecRoot, config.Root, sysInitPath, sysInfo)
	if err != nil {
		return nil, err
	}

	d.ID = trustKey.PublicKey().KeyID()
	d.repository = daemonRepo
	d.containers = &contStore{s: make(map[string]*Container)}
	d.execCommands = newExecStore()
	d.graph = g
	d.repositories = repositories
	d.idIndex = truncindex.NewTruncIndex([]string{})
	d.sysInfo = sysInfo
	d.config = config
	d.sysInitPath = sysInitPath
	d.execDriver = ed
	d.statsCollector = newStatsCollector(1 * time.Second)
	d.defaultLogConfig = config.LogConfig
	d.RegistryService = registryService
	d.EventsService = eventsService
	d.root = config.Root
	go d.execCommandGC()

	if err := d.restore(); err != nil {
		return nil, err
	}

	return d, nil
}

func (daemon *Daemon) Shutdown() error {
	if daemon.containers != nil {
		group := sync.WaitGroup{}
		logrus.Debug("starting clean shutdown of all containers...")
		for _, container := range daemon.List() {
			c := container
			if c.IsRunning() {
				logrus.Debugf("stopping %s", c.ID)
				group.Add(1)

				go func() {
					defer group.Done()
					// If container failed to exit in 10 seconds of SIGTERM, then using the force
					if err := c.Stop(10); err != nil {
						logrus.Errorf("Stop container %s with error: %v", c.ID, err)
					}
					c.WaitStop(-1 * time.Second)
					logrus.Debugf("container stopped %s", c.ID)
				}()
			}
		}
		group.Wait()

		// trigger libnetwork GC only if it's initialized
		if daemon.netController != nil {
			daemon.netController.GC()
		}
	}

	if daemon.containerGraph != nil {
		if err := daemon.containerGraph.Close(); err != nil {
			logrus.Errorf("Error during container graph.Close(): %v", err)
		}
	}

	if daemon.driver != nil {
		if err := daemon.driver.Cleanup(); err != nil {
			logrus.Errorf("Error during graph storage driver.Cleanup(): %v", err)
		}
	}

	return nil
}

func (daemon *Daemon) Mount(container *Container) error {
	dir, err := daemon.driver.Get(container.ID, container.GetMountLabel())
	if err != nil {
		return fmt.Errorf("Error getting container %s from driver %s: %s", container.ID, daemon.driver, err)
	}

	if container.basefs != dir {
		// The mount path reported by the graph driver should always be trusted on Windows, since the
		// volume path for a given mounted layer may change over time.  This should only be an error
		// on non-Windows operating systems.
		if container.basefs != "" && runtime.GOOS != "windows" {
			daemon.driver.Put(container.ID)
			return fmt.Errorf("Error: driver %s is returning inconsistent paths for container %s ('%s' then '%s')",
				daemon.driver, container.ID, container.basefs, dir)
		}
	}
	container.basefs = dir
	return nil
}

func (daemon *Daemon) Unmount(container *Container) error {
	daemon.driver.Put(container.ID)
	return nil
}

func (daemon *Daemon) Run(c *Container, pipes *execdriver.Pipes, startCallback execdriver.StartCallback) (execdriver.ExitStatus, error) {
	return daemon.execDriver.Run(c.command, pipes, startCallback)
}

func (daemon *Daemon) Kill(c *Container, sig int) error {
	return daemon.execDriver.Kill(c.command, sig)
}

func (daemon *Daemon) Stats(c *Container) (*execdriver.ResourceStats, error) {
	return daemon.execDriver.Stats(c.ID)
}

func (daemon *Daemon) SubscribeToContainerStats(name string) (chan interface{}, error) {
	c, err := daemon.Get(name)
	if err != nil {
		return nil, err
	}
	ch := daemon.statsCollector.collect(c)
	return ch, nil
}

func (daemon *Daemon) UnsubscribeToContainerStats(name string, ch chan interface{}) error {
	c, err := daemon.Get(name)
	if err != nil {
		return err
	}
	daemon.statsCollector.unsubscribe(c, ch)
	return nil
}

// FIXME: this is a convenience function for integration tests
// which need direct access to daemon.graph.
// Once the tests switch to using engine and jobs, this method
// can go away.
func (daemon *Daemon) Graph() *graph.Graph {
	return daemon.graph
}

func (daemon *Daemon) Repositories() *graph.TagStore {
	return daemon.repositories
}

func (daemon *Daemon) Config() *Config {
	return daemon.config
}

func (daemon *Daemon) SystemConfig() *sysinfo.SysInfo {
	return daemon.sysInfo
}

func (daemon *Daemon) SystemInitPath() string {
	return daemon.sysInitPath
}

func (daemon *Daemon) GraphDriver() graphdriver.Driver {
	return daemon.driver
}

func (daemon *Daemon) ExecutionDriver() execdriver.Driver {
	return daemon.execDriver
}

func (daemon *Daemon) ContainerGraph() *graphdb.Database {
	return daemon.containerGraph
}

func (daemon *Daemon) ImageGetCached(imgID string, config *runconfig.Config) (*image.Image, error) {
	// Retrieve all images
	images := daemon.Graph().Map()

	// Store the tree in a map of map (map[parentId][childId])
	imageMap := make(map[string]map[string]struct{})
	for _, img := range images {
		if _, exists := imageMap[img.Parent]; !exists {
			imageMap[img.Parent] = make(map[string]struct{})
		}
		imageMap[img.Parent][img.ID] = struct{}{}
	}

	// Loop on the children of the given image and check the config
	var match *image.Image
	for elem := range imageMap[imgID] {
		img, ok := images[elem]
		if !ok {
			return nil, fmt.Errorf("unable to find image %q", elem)
		}
		if runconfig.Compare(&img.ContainerConfig, config) {
			if match == nil || match.Created.Before(img.Created) {
				match = img
			}
		}
	}
	return match, nil
}

// tempDir returns the default directory to use for temporary files.
func tempDir(rootDir string) (string, error) {
	var tmpDir string
	if tmpDir = os.Getenv("DOCKER_TMPDIR"); tmpDir == "" {
		tmpDir = filepath.Join(rootDir, "tmp")
	}
	return tmpDir, system.MkdirAll(tmpDir, 0700)
}

func (daemon *Daemon) setHostConfig(container *Container, hostConfig *runconfig.HostConfig) error {
	container.Lock()
	if err := parseSecurityOpt(container, hostConfig); err != nil {
		container.Unlock()
		return err
	}
	container.Unlock()

	// Do not lock while creating volumes since this could be calling out to external plugins
	// Don't want to block other actions, like `docker ps` because we're waiting on an external plugin
	if err := daemon.registerMountPoints(container, hostConfig); err != nil {
		return err
	}

	container.Lock()
	defer container.Unlock()
	// Register any links from the host config before starting the container
	if err := daemon.RegisterLinks(container, hostConfig); err != nil {
		return err
	}

	container.hostConfig = hostConfig
	container.toDisk()
	return nil
}

func (daemon *Daemon) newBaseContainer(id string) CommonContainer {
	return CommonContainer{
		ID:           id,
		State:        NewState(),
		MountPoints:  make(map[string]*mountPoint),
		Volumes:      make(map[string]string),
		VolumesRW:    make(map[string]bool),
		execCommands: newExecStore(),
		root:         daemon.containerRoot(id),
	}
}

func setDefaultMtu(config *Config) {
	// do nothing if the config does not have the default 0 value.
	if config.Mtu != 0 {
		return
	}
	config.Mtu = defaultNetworkMtu
	if routeMtu, err := getDefaultRouteMtu(); err == nil {
		config.Mtu = routeMtu
	}
}

var errNoDefaultRoute = errors.New("no default route was found")

// getDefaultRouteMtu returns the MTU for the default route's interface.
func getDefaultRouteMtu() (int, error) {
	routes, err := netlink.NetworkGetRoutes()
	if err != nil {
		return 0, err
	}
	for _, r := range routes {
		if r.Default {
			return r.Iface.MTU, nil
		}
	}
	return 0, errNoDefaultRoute
}
