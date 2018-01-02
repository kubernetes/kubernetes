// Package daemon exposes the functions that occur on the host server
// that the Docker daemon is running.
//
// In implementing the various functions of the daemon, there is often
// a method-specific struct for configuring the runtime behavior.
package daemon

import (
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	containerd "github.com/containerd/containerd/api/grpc/types"
	"github.com/docker/docker/api"
	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/config"
	"github.com/docker/docker/daemon/discovery"
	"github.com/docker/docker/daemon/events"
	"github.com/docker/docker/daemon/exec"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/opts"
	"github.com/sirupsen/logrus"
	// register graph drivers
	_ "github.com/docker/docker/daemon/graphdriver/register"
	"github.com/docker/docker/daemon/initlayer"
	"github.com/docker/docker/daemon/stats"
	dmetadata "github.com/docker/docker/distribution/metadata"
	"github.com/docker/docker/distribution/xfer"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/libcontainerd"
	"github.com/docker/docker/migrate/v1"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/pkg/sysinfo"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/pkg/truncindex"
	"github.com/docker/docker/plugin"
	refstore "github.com/docker/docker/reference"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/runconfig"
	volumedrivers "github.com/docker/docker/volume/drivers"
	"github.com/docker/docker/volume/local"
	"github.com/docker/docker/volume/store"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/cluster"
	nwconfig "github.com/docker/libnetwork/config"
	"github.com/docker/libtrust"
	"github.com/pkg/errors"
)

var (
	// DefaultRuntimeBinary is the default runtime to be used by
	// containerd if none is specified
	DefaultRuntimeBinary = "docker-runc"

	errSystemNotSupported = errors.New("The Docker daemon is not supported on this platform.")
)

type daemonStore struct {
	graphDriver               string
	imageRoot                 string
	imageStore                image.Store
	layerStore                layer.Store
	distributionMetadataStore dmetadata.Store
	referenceStore            refstore.Store
}

// Daemon holds information about the Docker daemon.
type Daemon struct {
	ID                    string
	repository            string
	containers            container.Store
	containersReplica     container.ViewDB
	execCommands          *exec.Store
	downloadManager       *xfer.LayerDownloadManager
	uploadManager         *xfer.LayerUploadManager
	trustKey              libtrust.PrivateKey
	idIndex               *truncindex.TruncIndex
	configStore           *config.Config
	statsCollector        *stats.Collector
	defaultLogConfig      containertypes.LogConfig
	RegistryService       registry.Service
	EventsService         *events.Events
	netController         libnetwork.NetworkController
	volumes               *store.VolumeStore
	discoveryWatcher      discovery.Reloader
	root                  string
	seccompEnabled        bool
	apparmorEnabled       bool
	shutdown              bool
	idMappings            *idtools.IDMappings
	stores                map[string]daemonStore // By container target platform
	PluginStore           *plugin.Store          // todo: remove
	pluginManager         *plugin.Manager
	linkIndex             *linkIndex
	containerd            libcontainerd.Client
	containerdRemote      libcontainerd.Remote
	defaultIsolation      containertypes.Isolation // Default isolation mode on Windows
	clusterProvider       cluster.Provider
	cluster               Cluster
	genericResources      []swarm.GenericResource
	metricsPluginListener net.Listener

	machineMemory uint64

	seccompProfile     []byte
	seccompProfilePath string

	diskUsageRunning int32
	pruneRunning     int32
	hosts            map[string]bool // hosts stores the addresses the daemon is listening on
	startupDone      chan struct{}
}

// StoreHosts stores the addresses the daemon is listening on
func (daemon *Daemon) StoreHosts(hosts []string) {
	if daemon.hosts == nil {
		daemon.hosts = make(map[string]bool)
	}
	for _, h := range hosts {
		daemon.hosts[h] = true
	}
}

// HasExperimental returns whether the experimental features of the daemon are enabled or not
func (daemon *Daemon) HasExperimental() bool {
	if daemon.configStore != nil && daemon.configStore.Experimental {
		return true
	}
	return false
}

func (daemon *Daemon) restore() error {
	containers := make(map[string]*container.Container)

	logrus.Info("Loading containers: start.")

	dir, err := ioutil.ReadDir(daemon.repository)
	if err != nil {
		return err
	}

	for _, v := range dir {
		id := v.Name()
		container, err := daemon.load(id)
		if err != nil {
			logrus.Errorf("Failed to load container %v: %v", id, err)
			continue
		}

		// Ignore the container if it does not support the current driver being used by the graph
		currentDriverForContainerPlatform := daemon.stores[container.Platform].graphDriver
		if (container.Driver == "" && currentDriverForContainerPlatform == "aufs") || container.Driver == currentDriverForContainerPlatform {
			rwlayer, err := daemon.stores[container.Platform].layerStore.GetRWLayer(container.ID)
			if err != nil {
				logrus.Errorf("Failed to load container mount %v: %v", id, err)
				continue
			}
			container.RWLayer = rwlayer
			logrus.Debugf("Loaded container %v", container.ID)

			containers[container.ID] = container
		} else {
			logrus.Debugf("Cannot load container %s because it was created with another graph driver.", container.ID)
		}
	}

	removeContainers := make(map[string]*container.Container)
	restartContainers := make(map[*container.Container]chan struct{})
	activeSandboxes := make(map[string]interface{})
	for id, c := range containers {
		if err := daemon.registerName(c); err != nil {
			logrus.Errorf("Failed to register container name %s: %s", c.ID, err)
			delete(containers, id)
			continue
		}
		// verify that all volumes valid and have been migrated from the pre-1.7 layout
		if err := daemon.verifyVolumesInfo(c); err != nil {
			// don't skip the container due to error
			logrus.Errorf("Failed to verify volumes for container '%s': %v", c.ID, err)
		}
		if err := daemon.Register(c); err != nil {
			logrus.Errorf("Failed to register container %s: %s", c.ID, err)
			delete(containers, id)
			continue
		}

		// The LogConfig.Type is empty if the container was created before docker 1.12 with default log driver.
		// We should rewrite it to use the daemon defaults.
		// Fixes https://github.com/docker/docker/issues/22536
		if c.HostConfig.LogConfig.Type == "" {
			if err := daemon.mergeAndVerifyLogConfig(&c.HostConfig.LogConfig); err != nil {
				logrus.Errorf("Failed to verify log config for container %s: %q", c.ID, err)
				continue
			}
		}
	}

	var wg sync.WaitGroup
	var mapLock sync.Mutex
	for _, c := range containers {
		wg.Add(1)
		go func(c *container.Container) {
			defer wg.Done()
			daemon.backportMountSpec(c)
			if err := daemon.checkpointAndSave(c); err != nil {
				logrus.WithError(err).WithField("container", c.ID).Error("error saving backported mountspec to disk")
			}

			daemon.setStateCounter(c)
			if c.IsRunning() || c.IsPaused() {
				c.RestartManager().Cancel() // manually start containers because some need to wait for swarm networking
				if err := daemon.containerd.Restore(c.ID, c.InitializeStdio); err != nil {
					logrus.Errorf("Failed to restore %s with containerd: %s", c.ID, err)
					return
				}

				// we call Mount and then Unmount to get BaseFs of the container
				if err := daemon.Mount(c); err != nil {
					// The mount is unlikely to fail. However, in case mount fails
					// the container should be allowed to restore here. Some functionalities
					// (like docker exec -u user) might be missing but container is able to be
					// stopped/restarted/removed.
					// See #29365 for related information.
					// The error is only logged here.
					logrus.Warnf("Failed to mount container on getting BaseFs path %v: %v", c.ID, err)
				} else {
					if err := daemon.Unmount(c); err != nil {
						logrus.Warnf("Failed to umount container on getting BaseFs path %v: %v", c.ID, err)
					}
				}

				c.ResetRestartManager(false)
				if !c.HostConfig.NetworkMode.IsContainer() && c.IsRunning() {
					options, err := daemon.buildSandboxOptions(c)
					if err != nil {
						logrus.Warnf("Failed build sandbox option to restore container %s: %v", c.ID, err)
					}
					mapLock.Lock()
					activeSandboxes[c.NetworkSettings.SandboxID] = options
					mapLock.Unlock()
				}

			}
			// fixme: only if not running
			// get list of containers we need to restart
			if !c.IsRunning() && !c.IsPaused() {
				// Do not autostart containers which
				// has endpoints in a swarm scope
				// network yet since the cluster is
				// not initialized yet. We will start
				// it after the cluster is
				// initialized.
				if daemon.configStore.AutoRestart && c.ShouldRestart() && !c.NetworkSettings.HasSwarmEndpoint {
					mapLock.Lock()
					restartContainers[c] = make(chan struct{})
					mapLock.Unlock()
				} else if c.HostConfig != nil && c.HostConfig.AutoRemove {
					mapLock.Lock()
					removeContainers[c.ID] = c
					mapLock.Unlock()
				}
			}

			c.Lock()
			if c.RemovalInProgress {
				// We probably crashed in the middle of a removal, reset
				// the flag.
				//
				// We DO NOT remove the container here as we do not
				// know if the user had requested for either the
				// associated volumes, network links or both to also
				// be removed. So we put the container in the "dead"
				// state and leave further processing up to them.
				logrus.Debugf("Resetting RemovalInProgress flag from %v", c.ID)
				c.RemovalInProgress = false
				c.Dead = true
				if err := c.CheckpointTo(daemon.containersReplica); err != nil {
					logrus.Errorf("Failed to update container %s state: %v", c.ID, err)
				}
			}
			c.Unlock()
		}(c)
	}
	wg.Wait()
	daemon.netController, err = daemon.initNetworkController(daemon.configStore, activeSandboxes)
	if err != nil {
		return fmt.Errorf("Error initializing network controller: %v", err)
	}

	// Now that all the containers are registered, register the links
	for _, c := range containers {
		if err := daemon.registerLinks(c, c.HostConfig); err != nil {
			logrus.Errorf("failed to register link for container %s: %v", c.ID, err)
		}
	}

	group := sync.WaitGroup{}
	for c, notifier := range restartContainers {
		group.Add(1)

		go func(c *container.Container, chNotify chan struct{}) {
			defer group.Done()

			logrus.Debugf("Starting container %s", c.ID)

			// ignore errors here as this is a best effort to wait for children to be
			//   running before we try to start the container
			children := daemon.children(c)
			timeout := time.After(5 * time.Second)
			for _, child := range children {
				if notifier, exists := restartContainers[child]; exists {
					select {
					case <-notifier:
					case <-timeout:
					}
				}
			}

			// Make sure networks are available before starting
			daemon.waitForNetworks(c)
			if err := daemon.containerStart(c, "", "", true); err != nil {
				logrus.Errorf("Failed to start container %s: %s", c.ID, err)
			}
			close(chNotify)
		}(c, notifier)

	}
	group.Wait()

	removeGroup := sync.WaitGroup{}
	for id := range removeContainers {
		removeGroup.Add(1)
		go func(cid string) {
			if err := daemon.ContainerRm(cid, &types.ContainerRmConfig{ForceRemove: true, RemoveVolume: true}); err != nil {
				logrus.Errorf("Failed to remove container %s: %s", cid, err)
			}
			removeGroup.Done()
		}(id)
	}
	removeGroup.Wait()

	// any containers that were started above would already have had this done,
	// however we need to now prepare the mountpoints for the rest of the containers as well.
	// This shouldn't cause any issue running on the containers that already had this run.
	// This must be run after any containers with a restart policy so that containerized plugins
	// can have a chance to be running before we try to initialize them.
	for _, c := range containers {
		// if the container has restart policy, do not
		// prepare the mountpoints since it has been done on restarting.
		// This is to speed up the daemon start when a restart container
		// has a volume and the volume driver is not available.
		if _, ok := restartContainers[c]; ok {
			continue
		} else if _, ok := removeContainers[c.ID]; ok {
			// container is automatically removed, skip it.
			continue
		}

		group.Add(1)
		go func(c *container.Container) {
			defer group.Done()
			if err := daemon.prepareMountPoints(c); err != nil {
				logrus.Error(err)
			}
		}(c)
	}

	group.Wait()

	logrus.Info("Loading containers: done.")

	return nil
}

// RestartSwarmContainers restarts any autostart container which has a
// swarm endpoint.
func (daemon *Daemon) RestartSwarmContainers() {
	group := sync.WaitGroup{}
	for _, c := range daemon.List() {
		if !c.IsRunning() && !c.IsPaused() {
			// Autostart all the containers which has a
			// swarm endpoint now that the cluster is
			// initialized.
			if daemon.configStore.AutoRestart && c.ShouldRestart() && c.NetworkSettings.HasSwarmEndpoint {
				group.Add(1)
				go func(c *container.Container) {
					defer group.Done()
					if err := daemon.containerStart(c, "", "", true); err != nil {
						logrus.Error(err)
					}
				}(c)
			}
		}

	}
	group.Wait()
}

// waitForNetworks is used during daemon initialization when starting up containers
// It ensures that all of a container's networks are available before the daemon tries to start the container.
// In practice it just makes sure the discovery service is available for containers which use a network that require discovery.
func (daemon *Daemon) waitForNetworks(c *container.Container) {
	if daemon.discoveryWatcher == nil {
		return
	}
	// Make sure if the container has a network that requires discovery that the discovery service is available before starting
	for netName := range c.NetworkSettings.Networks {
		// If we get `ErrNoSuchNetwork` here, we can assume that it is due to discovery not being ready
		// Most likely this is because the K/V store used for discovery is in a container and needs to be started
		if _, err := daemon.netController.NetworkByName(netName); err != nil {
			if _, ok := err.(libnetwork.ErrNoSuchNetwork); !ok {
				continue
			}
			// use a longish timeout here due to some slowdowns in libnetwork if the k/v store is on anything other than --net=host
			// FIXME: why is this slow???
			logrus.Debugf("Container %s waiting for network to be ready", c.Name)
			select {
			case <-daemon.discoveryWatcher.ReadyCh():
			case <-time.After(60 * time.Second):
			}
			return
		}
	}
}

func (daemon *Daemon) children(c *container.Container) map[string]*container.Container {
	return daemon.linkIndex.children(c)
}

// parents returns the names of the parent containers of the container
// with the given name.
func (daemon *Daemon) parents(c *container.Container) map[string]*container.Container {
	return daemon.linkIndex.parents(c)
}

func (daemon *Daemon) registerLink(parent, child *container.Container, alias string) error {
	fullName := path.Join(parent.Name, alias)
	if err := daemon.containersReplica.ReserveName(fullName, child.ID); err != nil {
		if err == container.ErrNameReserved {
			logrus.Warnf("error registering link for %s, to %s, as alias %s, ignoring: %v", parent.ID, child.ID, alias, err)
			return nil
		}
		return err
	}
	daemon.linkIndex.link(parent, child, fullName)
	return nil
}

// DaemonJoinsCluster informs the daemon has joined the cluster and provides
// the handler to query the cluster component
func (daemon *Daemon) DaemonJoinsCluster(clusterProvider cluster.Provider) {
	daemon.setClusterProvider(clusterProvider)
}

// DaemonLeavesCluster informs the daemon has left the cluster
func (daemon *Daemon) DaemonLeavesCluster() {
	// Daemon is in charge of removing the attachable networks with
	// connected containers when the node leaves the swarm
	daemon.clearAttachableNetworks()
	// We no longer need the cluster provider, stop it now so that
	// the network agent will stop listening to cluster events.
	daemon.setClusterProvider(nil)
	// Wait for the networking cluster agent to stop
	daemon.netController.AgentStopWait()
	// Daemon is in charge of removing the ingress network when the
	// node leaves the swarm. Wait for job to be done or timeout.
	// This is called also on graceful daemon shutdown. We need to
	// wait, because the ingress release has to happen before the
	// network controller is stopped.
	if done, err := daemon.ReleaseIngress(); err == nil {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			logrus.Warnf("timeout while waiting for ingress network removal")
		}
	} else {
		logrus.Warnf("failed to initiate ingress network removal: %v", err)
	}
}

// setClusterProvider sets a component for querying the current cluster state.
func (daemon *Daemon) setClusterProvider(clusterProvider cluster.Provider) {
	daemon.clusterProvider = clusterProvider
	daemon.netController.SetClusterProvider(clusterProvider)
}

// IsSwarmCompatible verifies if the current daemon
// configuration is compatible with the swarm mode
func (daemon *Daemon) IsSwarmCompatible() error {
	if daemon.configStore == nil {
		return nil
	}
	return daemon.configStore.IsSwarmCompatible()
}

// NewDaemon sets up everything for the daemon to be able to service
// requests from the webserver.
func NewDaemon(config *config.Config, registryService registry.Service, containerdRemote libcontainerd.Remote, pluginStore *plugin.Store) (daemon *Daemon, err error) {
	setDefaultMtu(config)

	// Ensure that we have a correct root key limit for launching containers.
	if err := ModifyRootKeyLimit(); err != nil {
		logrus.Warnf("unable to modify root key limit, number of containers could be limited by this quota: %v", err)
	}

	// Ensure we have compatible and valid configuration options
	if err := verifyDaemonSettings(config); err != nil {
		return nil, err
	}

	// Do we have a disabled network?
	config.DisableBridge = isBridgeNetworkDisabled(config)

	// Verify the platform is supported as a daemon
	if !platformSupported {
		return nil, errSystemNotSupported
	}

	// Validate platform-specific requirements
	if err := checkSystem(); err != nil {
		return nil, err
	}

	idMappings, err := setupRemappedRoot(config)
	if err != nil {
		return nil, err
	}
	rootIDs := idMappings.RootPair()
	if err := setupDaemonProcess(config); err != nil {
		return nil, err
	}

	// set up the tmpDir to use a canonical path
	tmp, err := prepareTempDir(config.Root, rootIDs)
	if err != nil {
		return nil, fmt.Errorf("Unable to get the TempDir under %s: %s", config.Root, err)
	}
	realTmp, err := getRealPath(tmp)
	if err != nil {
		return nil, fmt.Errorf("Unable to get the full path to the TempDir (%s): %s", tmp, err)
	}
	os.Setenv("TMPDIR", realTmp)

	d := &Daemon{
		configStore: config,
		startupDone: make(chan struct{}),
	}
	// Ensure the daemon is properly shutdown if there is a failure during
	// initialization
	defer func() {
		if err != nil {
			if err := d.Shutdown(); err != nil {
				logrus.Error(err)
			}
		}
	}()

	if err := d.setGenericResources(config); err != nil {
		return nil, err
	}
	// set up SIGUSR1 handler on Unix-like systems, or a Win32 global event
	// on Windows to dump Go routine stacks
	stackDumpDir := config.Root
	if execRoot := config.GetExecRoot(); execRoot != "" {
		stackDumpDir = execRoot
	}
	d.setupDumpStackTrap(stackDumpDir)

	if err := d.setupSeccompProfile(); err != nil {
		return nil, err
	}

	// Set the default isolation mode (only applicable on Windows)
	if err := d.setDefaultIsolation(); err != nil {
		return nil, fmt.Errorf("error setting default isolation mode: %v", err)
	}

	logrus.Debugf("Using default logging driver %s", config.LogConfig.Type)

	if err := configureMaxThreads(config); err != nil {
		logrus.Warnf("Failed to configure golang's threads limit: %v", err)
	}

	if err := ensureDefaultAppArmorProfile(); err != nil {
		logrus.Errorf(err.Error())
	}

	daemonRepo := filepath.Join(config.Root, "containers")
	if err := idtools.MkdirAllAndChown(daemonRepo, 0700, rootIDs); err != nil && !os.IsExist(err) {
		return nil, err
	}

	if runtime.GOOS == "windows" {
		if err := system.MkdirAll(filepath.Join(config.Root, "credentialspecs"), 0, ""); err != nil && !os.IsExist(err) {
			return nil, err
		}
	}

	// On Windows we don't support the environment variable, or a user supplied graphdriver
	// as Windows has no choice in terms of which graphdrivers to use. It's a case of
	// running Windows containers on Windows - windowsfilter, running Linux containers on Windows,
	// lcow. Unix platforms however run a single graphdriver for all containers, and it can
	// be set through an environment variable, a daemon start parameter, or chosen through
	// initialization of the layerstore through driver priority order for example.
	d.stores = make(map[string]daemonStore)
	if runtime.GOOS == "windows" {
		d.stores["windows"] = daemonStore{graphDriver: "windowsfilter"}
		if system.LCOWSupported() {
			d.stores["linux"] = daemonStore{graphDriver: "lcow"}
		}
	} else {
		driverName := os.Getenv("DOCKER_DRIVER")
		if driverName == "" {
			driverName = config.GraphDriver
		} else {
			logrus.Infof("Setting the storage driver from the $DOCKER_DRIVER environment variable (%s)", driverName)
		}
		d.stores[runtime.GOOS] = daemonStore{graphDriver: driverName} // May still be empty. Layerstore init determines instead.
	}

	d.RegistryService = registryService
	d.PluginStore = pluginStore
	logger.RegisterPluginGetter(d.PluginStore)

	metricsSockPath, err := d.listenMetricsSock()
	if err != nil {
		return nil, err
	}
	registerMetricsPluginCallback(d.PluginStore, metricsSockPath)

	// Plugin system initialization should happen before restore. Do not change order.
	d.pluginManager, err = plugin.NewManager(plugin.ManagerConfig{
		Root:               filepath.Join(config.Root, "plugins"),
		ExecRoot:           getPluginExecRoot(config.Root),
		Store:              d.PluginStore,
		Executor:           containerdRemote,
		RegistryService:    registryService,
		LiveRestoreEnabled: config.LiveRestoreEnabled,
		LogPluginEvent:     d.LogPluginEvent, // todo: make private
		AuthzMiddleware:    config.AuthzMiddleware,
	})
	if err != nil {
		return nil, errors.Wrap(err, "couldn't create plugin manager")
	}

	var graphDrivers []string
	for platform, ds := range d.stores {
		ls, err := layer.NewStoreFromOptions(layer.StoreOptions{
			StorePath:                 config.Root,
			MetadataStorePathTemplate: filepath.Join(config.Root, "image", "%s", "layerdb"),
			GraphDriver:               ds.graphDriver,
			GraphDriverOptions:        config.GraphOptions,
			IDMappings:                idMappings,
			PluginGetter:              d.PluginStore,
			ExperimentalEnabled:       config.Experimental,
			Platform:                  platform,
		})
		if err != nil {
			return nil, err
		}
		ds.graphDriver = ls.DriverName() // As layerstore may set the driver
		ds.layerStore = ls
		d.stores[platform] = ds
		graphDrivers = append(graphDrivers, ls.DriverName())
	}

	// Configure and validate the kernels security support
	if err := configureKernelSecuritySupport(config, graphDrivers); err != nil {
		return nil, err
	}

	logrus.Debugf("Max Concurrent Downloads: %d", *config.MaxConcurrentDownloads)
	lsMap := make(map[string]layer.Store)
	for platform, ds := range d.stores {
		lsMap[platform] = ds.layerStore
	}
	d.downloadManager = xfer.NewLayerDownloadManager(lsMap, *config.MaxConcurrentDownloads)
	logrus.Debugf("Max Concurrent Uploads: %d", *config.MaxConcurrentUploads)
	d.uploadManager = xfer.NewLayerUploadManager(*config.MaxConcurrentUploads)

	for platform, ds := range d.stores {
		imageRoot := filepath.Join(config.Root, "image", ds.graphDriver)
		ifs, err := image.NewFSStoreBackend(filepath.Join(imageRoot, "imagedb"))
		if err != nil {
			return nil, err
		}

		var is image.Store
		is, err = image.NewImageStore(ifs, platform, ds.layerStore)
		if err != nil {
			return nil, err
		}
		ds.imageRoot = imageRoot
		ds.imageStore = is
		d.stores[platform] = ds
	}

	// Configure the volumes driver
	volStore, err := d.configureVolumes(rootIDs)
	if err != nil {
		return nil, err
	}

	trustKey, err := api.LoadOrCreateTrustKey(config.TrustKeyPath)
	if err != nil {
		return nil, err
	}

	trustDir := filepath.Join(config.Root, "trust")

	if err := system.MkdirAll(trustDir, 0700, ""); err != nil {
		return nil, err
	}

	eventsService := events.New()

	for platform, ds := range d.stores {
		dms, err := dmetadata.NewFSMetadataStore(filepath.Join(ds.imageRoot, "distribution"), platform)
		if err != nil {
			return nil, err
		}

		rs, err := refstore.NewReferenceStore(filepath.Join(ds.imageRoot, "repositories.json"), platform)
		if err != nil {
			return nil, fmt.Errorf("Couldn't create Tag store repositories: %s", err)
		}
		ds.distributionMetadataStore = dms
		ds.referenceStore = rs
		d.stores[platform] = ds

		// No content-addressability migration on Windows as it never supported pre-CA
		if runtime.GOOS != "windows" {
			migrationStart := time.Now()
			if err := v1.Migrate(config.Root, ds.graphDriver, ds.layerStore, ds.imageStore, rs, dms); err != nil {
				logrus.Errorf("Graph migration failed: %q. Your old graph data was found to be too inconsistent for upgrading to content-addressable storage. Some of the old data was probably not upgraded. We recommend starting over with a clean storage directory if possible.", err)
			}
			logrus.Infof("Graph migration to content-addressability took %.2f seconds", time.Since(migrationStart).Seconds())
		}
	}

	// Discovery is only enabled when the daemon is launched with an address to advertise.  When
	// initialized, the daemon is registered and we can store the discovery backend as it's read-only
	if err := d.initDiscovery(config); err != nil {
		return nil, err
	}

	sysInfo := sysinfo.New(false)
	// Check if Devices cgroup is mounted, it is hard requirement for container security,
	// on Linux.
	if runtime.GOOS == "linux" && !sysInfo.CgroupDevicesEnabled {
		return nil, errors.New("Devices cgroup isn't mounted")
	}

	d.ID = trustKey.PublicKey().KeyID()
	d.repository = daemonRepo
	d.containers = container.NewMemoryStore()
	if d.containersReplica, err = container.NewViewDB(); err != nil {
		return nil, err
	}
	d.execCommands = exec.NewStore()
	d.trustKey = trustKey
	d.idIndex = truncindex.NewTruncIndex([]string{})
	d.statsCollector = d.newStatsCollector(1 * time.Second)
	d.defaultLogConfig = containertypes.LogConfig{
		Type:   config.LogConfig.Type,
		Config: config.LogConfig.Config,
	}
	d.EventsService = eventsService
	d.volumes = volStore
	d.root = config.Root
	d.idMappings = idMappings
	d.seccompEnabled = sysInfo.Seccomp
	d.apparmorEnabled = sysInfo.AppArmor

	d.linkIndex = newLinkIndex()
	d.containerdRemote = containerdRemote

	go d.execCommandGC()

	d.containerd, err = containerdRemote.Client(d)
	if err != nil {
		return nil, err
	}

	if err := d.restore(); err != nil {
		return nil, err
	}
	close(d.startupDone)

	// FIXME: this method never returns an error
	info, _ := d.SystemInfo()

	engineInfo.WithValues(
		dockerversion.Version,
		dockerversion.GitCommit,
		info.Architecture,
		info.Driver,
		info.KernelVersion,
		info.OperatingSystem,
		info.OSType,
		info.ID,
	).Set(1)
	engineCpus.Set(float64(info.NCPU))
	engineMemory.Set(float64(info.MemTotal))

	gd := ""
	for platform, ds := range d.stores {
		if len(gd) > 0 {
			gd += ", "
		}
		gd += ds.graphDriver
		if len(d.stores) > 1 {
			gd = fmt.Sprintf("%s (%s)", gd, platform)
		}
	}
	logrus.WithFields(logrus.Fields{
		"version":        dockerversion.Version,
		"commit":         dockerversion.GitCommit,
		"graphdriver(s)": gd,
	}).Info("Docker daemon")

	return d, nil
}

func (daemon *Daemon) waitForStartupDone() {
	<-daemon.startupDone
}

func (daemon *Daemon) shutdownContainer(c *container.Container) error {
	stopTimeout := c.StopTimeout()

	// If container failed to exit in stopTimeout seconds of SIGTERM, then using the force
	if err := daemon.containerStop(c, stopTimeout); err != nil {
		return fmt.Errorf("Failed to stop container %s with error: %v", c.ID, err)
	}

	// Wait without timeout for the container to exit.
	// Ignore the result.
	_ = <-c.Wait(context.Background(), container.WaitConditionNotRunning)
	return nil
}

// ShutdownTimeout returns the shutdown timeout based on the max stopTimeout of the containers,
// and is limited by daemon's ShutdownTimeout.
func (daemon *Daemon) ShutdownTimeout() int {
	// By default we use daemon's ShutdownTimeout.
	shutdownTimeout := daemon.configStore.ShutdownTimeout

	graceTimeout := 5
	if daemon.containers != nil {
		for _, c := range daemon.containers.List() {
			if shutdownTimeout >= 0 {
				stopTimeout := c.StopTimeout()
				if stopTimeout < 0 {
					shutdownTimeout = -1
				} else {
					if stopTimeout+graceTimeout > shutdownTimeout {
						shutdownTimeout = stopTimeout + graceTimeout
					}
				}
			}
		}
	}
	return shutdownTimeout
}

// Shutdown stops the daemon.
func (daemon *Daemon) Shutdown() error {
	daemon.shutdown = true
	// Keep mounts and networking running on daemon shutdown if
	// we are to keep containers running and restore them.

	if daemon.configStore.LiveRestoreEnabled && daemon.containers != nil {
		// check if there are any running containers, if none we should do some cleanup
		if ls, err := daemon.Containers(&types.ContainerListOptions{}); len(ls) != 0 || err != nil {
			// metrics plugins still need some cleanup
			daemon.cleanupMetricsPlugins()
			return nil
		}
	}

	if daemon.containers != nil {
		logrus.Debugf("start clean shutdown of all containers with a %d seconds timeout...", daemon.configStore.ShutdownTimeout)
		daemon.containers.ApplyAll(func(c *container.Container) {
			if !c.IsRunning() {
				return
			}
			logrus.Debugf("stopping %s", c.ID)
			if err := daemon.shutdownContainer(c); err != nil {
				logrus.Errorf("Stop container error: %v", err)
				return
			}
			if mountid, err := daemon.stores[c.Platform].layerStore.GetMountID(c.ID); err == nil {
				daemon.cleanupMountsByID(mountid)
			}
			logrus.Debugf("container stopped %s", c.ID)
		})
	}

	if daemon.volumes != nil {
		if err := daemon.volumes.Shutdown(); err != nil {
			logrus.Errorf("Error shutting down volume store: %v", err)
		}
	}

	for platform, ds := range daemon.stores {
		if ds.layerStore != nil {
			if err := ds.layerStore.Cleanup(); err != nil {
				logrus.Errorf("Error during layer Store.Cleanup(): %v %s", err, platform)
			}
		}
	}

	// If we are part of a cluster, clean up cluster's stuff
	if daemon.clusterProvider != nil {
		logrus.Debugf("start clean shutdown of cluster resources...")
		daemon.DaemonLeavesCluster()
	}

	daemon.cleanupMetricsPlugins()

	// Shutdown plugins after containers and layerstore. Don't change the order.
	daemon.pluginShutdown()

	// trigger libnetwork Stop only if it's initialized
	if daemon.netController != nil {
		daemon.netController.Stop()
	}

	if err := daemon.cleanupMounts(); err != nil {
		return err
	}

	return nil
}

// Mount sets container.BaseFS
// (is it not set coming in? why is it unset?)
func (daemon *Daemon) Mount(container *container.Container) error {
	dir, err := container.RWLayer.Mount(container.GetMountLabel())
	if err != nil {
		return err
	}
	logrus.Debugf("container mounted via layerStore: %v", dir)

	if container.BaseFS != dir {
		// The mount path reported by the graph driver should always be trusted on Windows, since the
		// volume path for a given mounted layer may change over time.  This should only be an error
		// on non-Windows operating systems.
		if container.BaseFS != "" && runtime.GOOS != "windows" {
			daemon.Unmount(container)
			return fmt.Errorf("Error: driver %s is returning inconsistent paths for container %s ('%s' then '%s')",
				daemon.GraphDriverName(container.Platform), container.ID, container.BaseFS, dir)
		}
	}
	container.BaseFS = dir // TODO: combine these fields
	return nil
}

// Unmount unsets the container base filesystem
func (daemon *Daemon) Unmount(container *container.Container) error {
	if err := container.RWLayer.Unmount(); err != nil {
		logrus.Errorf("Error unmounting container %s: %s", container.ID, err)
		return err
	}

	return nil
}

// Subnets return the IPv4 and IPv6 subnets of networks that are manager by Docker.
func (daemon *Daemon) Subnets() ([]net.IPNet, []net.IPNet) {
	var v4Subnets []net.IPNet
	var v6Subnets []net.IPNet

	managedNetworks := daemon.netController.Networks()

	for _, managedNetwork := range managedNetworks {
		v4infos, v6infos := managedNetwork.Info().IpamInfo()
		for _, info := range v4infos {
			if info.IPAMData.Pool != nil {
				v4Subnets = append(v4Subnets, *info.IPAMData.Pool)
			}
		}
		for _, info := range v6infos {
			if info.IPAMData.Pool != nil {
				v6Subnets = append(v6Subnets, *info.IPAMData.Pool)
			}
		}
	}

	return v4Subnets, v6Subnets
}

// GraphDriverName returns the name of the graph driver used by the layer.Store
func (daemon *Daemon) GraphDriverName(platform string) string {
	return daemon.stores[platform].layerStore.DriverName()
}

// prepareTempDir prepares and returns the default directory to use
// for temporary files.
// If it doesn't exist, it is created. If it exists, its content is removed.
func prepareTempDir(rootDir string, rootIDs idtools.IDPair) (string, error) {
	var tmpDir string
	if tmpDir = os.Getenv("DOCKER_TMPDIR"); tmpDir == "" {
		tmpDir = filepath.Join(rootDir, "tmp")
		newName := tmpDir + "-old"
		if err := os.Rename(tmpDir, newName); err == nil {
			go func() {
				if err := os.RemoveAll(newName); err != nil {
					logrus.Warnf("failed to delete old tmp directory: %s", newName)
				}
			}()
		} else {
			logrus.Warnf("failed to rename %s for background deletion: %s. Deleting synchronously", tmpDir, err)
			if err := os.RemoveAll(tmpDir); err != nil {
				logrus.Warnf("failed to delete old tmp directory: %s", tmpDir)
			}
		}
	}
	// We don't remove the content of tmpdir if it's not the default,
	// it may hold things that do not belong to us.
	return tmpDir, idtools.MkdirAllAndChown(tmpDir, 0700, rootIDs)
}

func (daemon *Daemon) setupInitLayer(initPath string) error {
	rootIDs := daemon.idMappings.RootPair()
	return initlayer.Setup(initPath, rootIDs)
}

func (daemon *Daemon) setGenericResources(conf *config.Config) error {
	genericResources, err := opts.ParseGenericResources(conf.NodeGenericResources)
	if err != nil {
		return err
	}

	daemon.genericResources = genericResources

	return nil
}

func setDefaultMtu(conf *config.Config) {
	// do nothing if the config does not have the default 0 value.
	if conf.Mtu != 0 {
		return
	}
	conf.Mtu = config.DefaultNetworkMtu
}

func (daemon *Daemon) configureVolumes(rootIDs idtools.IDPair) (*store.VolumeStore, error) {
	volumesDriver, err := local.New(daemon.configStore.Root, rootIDs)
	if err != nil {
		return nil, err
	}

	volumedrivers.RegisterPluginGetter(daemon.PluginStore)

	if !volumedrivers.Register(volumesDriver, volumesDriver.Name()) {
		return nil, errors.New("local volume driver could not be registered")
	}
	return store.New(daemon.configStore.Root)
}

// IsShuttingDown tells whether the daemon is shutting down or not
func (daemon *Daemon) IsShuttingDown() bool {
	return daemon.shutdown
}

// initDiscovery initializes the discovery watcher for this daemon.
func (daemon *Daemon) initDiscovery(conf *config.Config) error {
	advertise, err := config.ParseClusterAdvertiseSettings(conf.ClusterStore, conf.ClusterAdvertise)
	if err != nil {
		if err == discovery.ErrDiscoveryDisabled {
			return nil
		}
		return err
	}

	conf.ClusterAdvertise = advertise
	discoveryWatcher, err := discovery.Init(conf.ClusterStore, conf.ClusterAdvertise, conf.ClusterOpts)
	if err != nil {
		return fmt.Errorf("discovery initialization failed (%v)", err)
	}

	daemon.discoveryWatcher = discoveryWatcher
	return nil
}

func isBridgeNetworkDisabled(conf *config.Config) bool {
	return conf.BridgeConfig.Iface == config.DisableNetworkBridge
}

func (daemon *Daemon) networkOptions(dconfig *config.Config, pg plugingetter.PluginGetter, activeSandboxes map[string]interface{}) ([]nwconfig.Option, error) {
	options := []nwconfig.Option{}
	if dconfig == nil {
		return options, nil
	}

	options = append(options, nwconfig.OptionExperimental(dconfig.Experimental))
	options = append(options, nwconfig.OptionDataDir(dconfig.Root))
	options = append(options, nwconfig.OptionExecRoot(dconfig.GetExecRoot()))

	dd := runconfig.DefaultDaemonNetworkMode()
	dn := runconfig.DefaultDaemonNetworkMode().NetworkName()
	options = append(options, nwconfig.OptionDefaultDriver(string(dd)))
	options = append(options, nwconfig.OptionDefaultNetwork(dn))

	if strings.TrimSpace(dconfig.ClusterStore) != "" {
		kv := strings.Split(dconfig.ClusterStore, "://")
		if len(kv) != 2 {
			return nil, errors.New("kv store daemon config must be of the form KV-PROVIDER://KV-URL")
		}
		options = append(options, nwconfig.OptionKVProvider(kv[0]))
		options = append(options, nwconfig.OptionKVProviderURL(kv[1]))
	}
	if len(dconfig.ClusterOpts) > 0 {
		options = append(options, nwconfig.OptionKVOpts(dconfig.ClusterOpts))
	}

	if daemon.discoveryWatcher != nil {
		options = append(options, nwconfig.OptionDiscoveryWatcher(daemon.discoveryWatcher))
	}

	if dconfig.ClusterAdvertise != "" {
		options = append(options, nwconfig.OptionDiscoveryAddress(dconfig.ClusterAdvertise))
	}

	options = append(options, nwconfig.OptionLabels(dconfig.Labels))
	options = append(options, driverOptions(dconfig)...)

	if daemon.configStore != nil && daemon.configStore.LiveRestoreEnabled && len(activeSandboxes) != 0 {
		options = append(options, nwconfig.OptionActiveSandboxes(activeSandboxes))
	}

	if pg != nil {
		options = append(options, nwconfig.OptionPluginGetter(pg))
	}

	options = append(options, nwconfig.OptionNetworkControlPlaneMTU(dconfig.NetworkControlPlaneMTU))

	return options, nil
}

func copyBlkioEntry(entries []*containerd.BlkioStatsEntry) []types.BlkioStatEntry {
	out := make([]types.BlkioStatEntry, len(entries))
	for i, re := range entries {
		out[i] = types.BlkioStatEntry{
			Major: re.Major,
			Minor: re.Minor,
			Op:    re.Op,
			Value: re.Value,
		}
	}
	return out
}

// GetCluster returns the cluster
func (daemon *Daemon) GetCluster() Cluster {
	return daemon.cluster
}

// SetCluster sets the cluster
func (daemon *Daemon) SetCluster(cluster Cluster) {
	daemon.cluster = cluster
}

func (daemon *Daemon) pluginShutdown() {
	manager := daemon.pluginManager
	// Check for a valid manager object. In error conditions, daemon init can fail
	// and shutdown called, before plugin manager is initialized.
	if manager != nil {
		manager.Shutdown()
	}
}

// PluginManager returns current pluginManager associated with the daemon
func (daemon *Daemon) PluginManager() *plugin.Manager { // set up before daemon to avoid this method
	return daemon.pluginManager
}

// PluginGetter returns current pluginStore associated with the daemon
func (daemon *Daemon) PluginGetter() *plugin.Store {
	return daemon.PluginStore
}

// CreateDaemonRoot creates the root for the daemon
func CreateDaemonRoot(config *config.Config) error {
	// get the canonical path to the Docker root directory
	var realRoot string
	if _, err := os.Stat(config.Root); err != nil && os.IsNotExist(err) {
		realRoot = config.Root
	} else {
		realRoot, err = getRealPath(config.Root)
		if err != nil {
			return fmt.Errorf("Unable to get the full path to root (%s): %s", config.Root, err)
		}
	}

	idMappings, err := setupRemappedRoot(config)
	if err != nil {
		return err
	}
	return setupDaemonRoot(config, realRoot, idMappings.RootPair())
}

// checkpointAndSave grabs a container lock to safely call container.CheckpointTo
func (daemon *Daemon) checkpointAndSave(container *container.Container) error {
	container.Lock()
	defer container.Unlock()
	if err := container.CheckpointTo(daemon.containersReplica); err != nil {
		return fmt.Errorf("Error saving container state: %v", err)
	}
	return nil
}

// because the CLI sends a -1 when it wants to unset the swappiness value
// we need to clear it on the server side
func fixMemorySwappiness(resources *containertypes.Resources) {
	if resources.MemorySwappiness != nil && *resources.MemorySwappiness == -1 {
		resources.MemorySwappiness = nil
	}
}
