/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package service

import (
	"io"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	log "github.com/golang/glog"
	bindings "github.com/mesos/mesos-go/executor"
	"k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	"k8s.io/kubernetes/contrib/mesos/pkg/redirfd"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kconfig "k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util"
	utilio "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/oom"

	"github.com/spf13/pflag"
)

const (
	// if we don't use this source then the kubelet will do funny, mirror things.
	// @see ConfigSourceAnnotationKey
	MESOS_CFG_SOURCE = kubelet.ApiserverSource
)

type KubeletExecutorServer struct {
	*app.KubeletServer
	SuicideTimeout time.Duration
	ShutdownFD     int
	ShutdownFIFO   string
}

func NewKubeletExecutorServer() *KubeletExecutorServer {
	k := &KubeletExecutorServer{
		KubeletServer:  app.NewKubeletServer(),
		SuicideTimeout: config.DefaultSuicideTimeout,
	}
	if pwd, err := os.Getwd(); err != nil {
		log.Warningf("failed to determine current directory: %v", err)
	} else {
		k.RootDirectory = pwd // mesos sandbox dir
	}
	k.Address = net.ParseIP(defaultBindingAddress())
	k.ShutdownFD = -1 // indicates unspecified FD

	return k
}

func (s *KubeletExecutorServer) AddFlags(fs *pflag.FlagSet) {
	s.KubeletServer.AddFlags(fs)
	fs.DurationVar(&s.SuicideTimeout, "suicide-timeout", s.SuicideTimeout, "Self-terminate after this period of inactivity. Zero disables suicide watch.")
	fs.IntVar(&s.ShutdownFD, "shutdown-fd", s.ShutdownFD, "File descriptor used to signal shutdown to external watchers, requires shutdown-fifo flag")
	fs.StringVar(&s.ShutdownFIFO, "shutdown-fifo", s.ShutdownFIFO, "FIFO used to signal shutdown to external watchers, requires shutdown-fd flag")
}

// returns a Closer that should be closed to signal impending shutdown, but only if ShutdownFD
// and ShutdownFIFO were specified. if they are specified, then this func blocks until there's
// a reader on the FIFO stream.
func (s *KubeletExecutorServer) syncExternalShutdownWatcher() (io.Closer, error) {
	if s.ShutdownFD == -1 || s.ShutdownFIFO == "" {
		return nil, nil
	}
	// redirfd -w n fifo ...  # (blocks until the fifo is read)
	log.Infof("blocked, waiting for shutdown reader for FD %d FIFO at %s", s.ShutdownFD, s.ShutdownFIFO)
	return redirfd.Write.Redirect(true, false, redirfd.FileDescriptor(s.ShutdownFD), s.ShutdownFIFO)
}

// Run runs the specified KubeletExecutorServer.
func (s *KubeletExecutorServer) Run(hks hyperkube.Interface, _ []string) error {
	rand.Seed(time.Now().UTC().UnixNano())

	oomAdjuster := oom.NewOomAdjuster()
	if err := oomAdjuster.ApplyOomScoreAdj(0, s.OOMScoreAdj); err != nil {
		log.Info(err)
	}

	// empty string for the docker and system containers (= cgroup paths). This
	// stops the kubelet taking any control over other system processes.
	s.SystemContainer = ""
	s.DockerDaemonContainer = ""

	// create apiserver client
	var apiclient *client.Client
	clientConfig, err := s.CreateAPIServerClientConfig()
	if err == nil {
		apiclient, err = client.New(clientConfig)
	}
	if err != nil {
		// required for k8sm since we need to send api.Binding information
		// back to the apiserver
		log.Fatalf("No API client: %v", err)
	}

	log.Infof("Using root directory: %v", s.RootDirectory)
	credentialprovider.SetPreferredDockercfgPath(s.RootDirectory)

	shutdownCloser, err := s.syncExternalShutdownWatcher()
	if err != nil {
		return err
	}

	cAdvisorInterface, err := cadvisor.New(s.CAdvisorPort)
	if err != nil {
		return err
	}

	imageGCPolicy := kubelet.ImageGCPolicy{
		HighThresholdPercent: s.ImageGCHighThresholdPercent,
		LowThresholdPercent:  s.ImageGCLowThresholdPercent,
	}

	diskSpacePolicy := kubelet.DiskSpacePolicy{
		DockerFreeDiskMB: s.LowDiskSpaceThresholdMB,
		RootFreeDiskMB:   s.LowDiskSpaceThresholdMB,
	}

	//TODO(jdef) intentionally NOT initializing a cloud provider here since:
	//(a) the kubelet doesn't actually use it
	//(b) we don't need to create N-kubelet connections to zookeeper for no good reason
	//cloud := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
	//log.Infof("Successfully initialized cloud provider: %q from the config file: %q\n", s.CloudProvider, s.CloudConfigFile)

	hostNetworkSources, err := kubelet.GetValidatedSources(strings.Split(s.HostNetworkSources, ","))
	if err != nil {
		return err
	}

	tlsOptions, err := s.InitializeTLS()
	if err != nil {
		return err
	}
	mounter := mount.New()
	if s.Containerized {
		log.V(2).Info("Running kubelet in containerized mode (experimental)")
		mounter = &mount.NsenterMounter{}
	}

	var writer utilio.Writer = &utilio.StdWriter{}
	var dockerExecHandler dockertools.ExecHandler
	switch s.DockerExecHandlerName {
	case "native":
		dockerExecHandler = &dockertools.NativeExecHandler{}
	case "nsenter":
		writer = &utilio.NsenterWriter{}
		dockerExecHandler = &dockertools.NsenterExecHandler{}
	default:
		log.Warningf("Unknown Docker exec handler %q; defaulting to native", s.DockerExecHandlerName)
		dockerExecHandler = &dockertools.NativeExecHandler{}
	}

	kcfg := app.KubeletConfig{
		Address:            s.Address,
		AllowPrivileged:    s.AllowPrivileged,
		HostNetworkSources: hostNetworkSources,
		HostnameOverride:   s.HostnameOverride,
		RootDirectory:      s.RootDirectory,
		// ConfigFile: ""
		// ManifestURL: ""
		FileCheckFrequency: s.FileCheckFrequency,
		// HTTPCheckFrequency
		PodInfraContainerImage:  s.PodInfraContainerImage,
		SyncFrequency:           s.SyncFrequency,
		RegistryPullQPS:         s.RegistryPullQPS,
		RegistryBurst:           s.RegistryBurst,
		MinimumGCAge:            s.MinimumGCAge,
		MaxPerPodContainerCount: s.MaxPerPodContainerCount,
		MaxContainerCount:       s.MaxContainerCount,
		RegisterNode:            s.RegisterNode,
		// StandaloneMode: false
		ClusterDomain:                  s.ClusterDomain,
		ClusterDNS:                     s.ClusterDNS,
		Runonce:                        s.RunOnce,
		Port:                           s.Port,
		ReadOnlyPort:                   s.ReadOnlyPort,
		CAdvisorInterface:              cAdvisorInterface,
		EnableServer:                   s.EnableServer,
		EnableDebuggingHandlers:        s.EnableDebuggingHandlers,
		DockerClient:                   dockertools.ConnectToDockerOrDie(s.DockerEndpoint),
		KubeClient:                     apiclient,
		MasterServiceNamespace:         s.MasterServiceNamespace,
		VolumePlugins:                  app.ProbeVolumePlugins(),
		NetworkPlugins:                 app.ProbeNetworkPlugins(s.NetworkPluginDir),
		NetworkPluginName:              s.NetworkPluginName,
		StreamingConnectionIdleTimeout: s.StreamingConnectionIdleTimeout,
		TLSOptions:                     tlsOptions,
		ImageGCPolicy:                  imageGCPolicy,
		DiskSpacePolicy:                diskSpacePolicy,
		Cloud:                          nil, // TODO(jdef) Cloud, specifying null here because we don't want all kubelets polling mesos-master; need to account for this in the cloudprovider impl
		NodeStatusUpdateFrequency: s.NodeStatusUpdateFrequency,
		ResourceContainer:         s.ResourceContainer,
		CgroupRoot:                s.CgroupRoot,
		ContainerRuntime:          s.ContainerRuntime,
		Mounter:                   mounter,
		DockerDaemonContainer:     s.DockerDaemonContainer,
		SystemContainer:           s.SystemContainer,
		ConfigureCBR0:             s.ConfigureCBR0,
		MaxPods:                   s.MaxPods,
		DockerExecHandler:         dockerExecHandler,
		ResolverConfig:            s.ResolverConfig,
		CPUCFSQuota:               s.CPUCFSQuota,
		Writer:                    writer,
		MaxOpenFiles:              s.MaxOpenFiles,
	}

	kcfg.NodeName = kcfg.Hostname

	err = app.RunKubelet(&kcfg, app.KubeletBuilder(func(kc *app.KubeletConfig) (app.KubeletBootstrap, *kconfig.PodConfig, error) {
		return s.createAndInitKubelet(kc, hks, clientConfig, shutdownCloser)
	}))
	if err != nil {
		return err
	}

	if s.HealthzPort > 0 {
		healthz.DefaultHealthz()
		go util.Until(func() {
			err := http.ListenAndServe(net.JoinHostPort(s.HealthzBindAddress.String(), strconv.Itoa(s.HealthzPort)), nil)
			if err != nil {
				log.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second, util.NeverStop)
	}

	// block until executor is shut down or commits shutdown
	select {}
}

func defaultBindingAddress() string {
	libProcessIP := os.Getenv("LIBPROCESS_IP")
	if libProcessIP == "" {
		return "0.0.0.0"
	} else {
		return libProcessIP
	}
}

func (ks *KubeletExecutorServer) createAndInitKubelet(
	kc *app.KubeletConfig,
	hks hyperkube.Interface,
	clientConfig *client.Config,
	shutdownCloser io.Closer,
) (app.KubeletBootstrap, *kconfig.PodConfig, error) {

	// TODO(k8s): block until all sources have delivered at least one update to the channel, or break the sync loop
	// up into "per source" synchronizations
	// TODO(k8s): KubeletConfig.KubeClient should be a client interface, but client interface misses certain methods
	// used by kubelet. Since NewMainKubelet expects a client interface, we need to make sure we are not passing
	// a nil pointer to it when what we really want is a nil interface.
	var kubeClient client.Interface
	if kc.KubeClient == nil {
		kubeClient = nil
	} else {
		kubeClient = kc.KubeClient
	}

	gcPolicy := kubelet.ContainerGCPolicy{
		MinAge:             kc.MinimumGCAge,
		MaxPerPodContainer: kc.MaxPerPodContainerCount,
		MaxContainers:      kc.MaxContainerCount,
	}

	pc := kconfig.NewPodConfig(kconfig.PodConfigNotificationIncremental, kc.Recorder)
	updates := pc.Channel(MESOS_CFG_SOURCE)

	klet, err := kubelet.NewMainKubelet(
		kc.Hostname,
		kc.NodeName,
		kc.DockerClient,
		kubeClient,
		kc.RootDirectory,
		kc.PodInfraContainerImage,
		kc.SyncFrequency,
		float32(kc.RegistryPullQPS),
		kc.RegistryBurst,
		kc.EventRecordQPS,
		kc.EventBurst,
		gcPolicy,
		pc.SeenAllSources,
		kc.RegisterNode,
		kc.StandaloneMode,
		kc.ClusterDomain,
		net.IP(kc.ClusterDNS),
		kc.MasterServiceNamespace,
		kc.VolumePlugins,
		kc.NetworkPlugins,
		kc.NetworkPluginName,
		kc.StreamingConnectionIdleTimeout,
		kc.Recorder,
		kc.CAdvisorInterface,
		kc.ImageGCPolicy,
		kc.DiskSpacePolicy,
		kc.Cloud,
		kc.NodeStatusUpdateFrequency,
		kc.ResourceContainer,
		kc.OSInterface,
		kc.CgroupRoot,
		kc.ContainerRuntime,
		kc.RktPath,
		kc.RktStage1Image,
		kc.Mounter,
		kc.Writer,
		kc.DockerDaemonContainer,
		kc.SystemContainer,
		kc.ConfigureCBR0,
		kc.PodCIDR,
		kc.MaxPods,
		kc.DockerExecHandler,
		kc.ResolverConfig,
		kc.CPUCFSQuota,
		&api.NodeDaemonEndpoints{
			KubeletEndpoint: api.DaemonEndpoint{Port: int(kc.Port)},
		},
		true, // Serialize Image pulls
	)
	if err != nil {
		return nil, nil, err
	}

	//TODO(jdef) either configure Watch here with something useful, or else
	// get rid of it from executor.Config
	kubeletFinished := make(chan struct{})
	staticPodsConfigPath := filepath.Join(kc.RootDirectory, "static-pods")
	exec := executor.New(executor.Config{
		Kubelet:         klet,
		Updates:         updates,
		SourceName:      MESOS_CFG_SOURCE,
		APIClient:       kc.KubeClient,
		Docker:          kc.DockerClient,
		SuicideTimeout:  ks.SuicideTimeout,
		KubeletFinished: kubeletFinished,
		ShutdownAlert: func() {
			if shutdownCloser != nil {
				if e := shutdownCloser.Close(); e != nil {
					log.Warningf("failed to signal shutdown to external watcher: %v", e)
				}
			}
		},
		ExitFunc: os.Exit,
		PodStatusFunc: func(_ executor.KubeletInterface, pod *api.Pod) (*api.PodStatus, error) {
			return klet.GetRuntime().GetPodStatus(pod)
		},
		StaticPodsConfigPath: staticPodsConfigPath,
		PodLW:                cache.NewListWatchFromClient(kc.KubeClient, "pods", api.NamespaceAll, fields.OneTermEqualSelector(client.PodHost, kc.NodeName)),
	})

	go exec.InitializeStaticPodsSource(func() {
		// Create file source only when we are called back. Otherwise, it is never marked unseen.
		fileSourceUpdates := pc.Channel(kubelet.FileSource)

		kconfig.NewSourceFile(staticPodsConfigPath, kc.Hostname, kc.FileCheckFrequency, fileSourceUpdates)
	})

	k := &kubeletExecutor{
		Kubelet:         klet,
		address:         ks.Address,
		dockerClient:    kc.DockerClient,
		hks:             hks,
		kubeletFinished: kubeletFinished,
		executorDone:    exec.Done(),
		clientConfig:    clientConfig,
	}

	dconfig := bindings.DriverConfig{
		Executor:         exec,
		HostnameOverride: ks.HostnameOverride,
		BindingAddress:   ks.Address,
	}
	if driver, err := bindings.NewMesosExecutorDriver(dconfig); err != nil {
		log.Fatalf("failed to create executor driver: %v", err)
	} else {
		k.driver = driver
	}

	log.V(2).Infof("Initialize executor driver...")

	k.BirthCry()
	exec.Init(k.driver)

	k.StartGarbageCollection()

	return k, pc, nil
}

// kubelet decorator
type kubeletExecutor struct {
	*kubelet.Kubelet
	initialize      sync.Once
	driver          bindings.ExecutorDriver
	address         net.IP
	dockerClient    dockertools.DockerInterface
	hks             hyperkube.Interface
	kubeletFinished chan struct{}   // closed once kubelet.Run() returns
	executorDone    <-chan struct{} // from KubeletExecutor.Done()
	clientConfig    *client.Config
}

func (kl *kubeletExecutor) ListenAndServe(address net.IP, port uint, tlsOptions *kubelet.TLSOptions, enableDebuggingHandlers bool) {
	// this func could be called many times, depending how often the HTTP server crashes,
	// so only execute certain initialization procs once
	kl.initialize.Do(func() {
		go func() {
			if _, err := kl.driver.Run(); err != nil {
				log.Fatalf("executor driver failed: %v", err)
			}
			log.Info("executor Run completed")
		}()
	})
	log.Infof("Starting kubelet server...")
	kubelet.ListenAndServeKubeletServer(kl, address, port, tlsOptions, enableDebuggingHandlers)
}

// runs the main kubelet loop, closing the kubeletFinished chan when the loop exits.
// never returns.
func (kl *kubeletExecutor) Run(updates <-chan kubelet.PodUpdate) {
	defer func() {
		close(kl.kubeletFinished)
		util.HandleCrash()
		log.Infoln("kubelet run terminated") //TODO(jdef) turn down verbosity
		// important: never return! this is in our contract
		select {}
	}()

	// push updates through a closable pipe. when the executor indicates shutdown
	// via Done() we want to stop the Kubelet from processing updates.
	pipe := make(chan kubelet.PodUpdate)
	go func() {
		// closing pipe will cause our patched kubelet's syncLoop() to exit
		defer close(pipe)
	pipeLoop:
		for {
			select {
			case <-kl.executorDone:
				break pipeLoop
			default:
				select {
				case u := <-updates:
					select {
					case pipe <- u: // noop
					case <-kl.executorDone:
						break pipeLoop
					}
				case <-kl.executorDone:
					break pipeLoop
				}
			}
		}
	}()

	// we expect that Run() will complete after the pipe is closed and the
	// kubelet's syncLoop() has finished processing its backlog, which hopefully
	// will not take very long. Peeking into the future (current k8s master) it
	// seems that the backlog has grown from 1 to 50 -- this may negatively impact
	// us going forward, time will tell.
	util.Until(func() { kl.Kubelet.Run(pipe) }, 0, kl.executorDone)

	//TODO(jdef) revisit this if/when executor failover lands
	// Force kubelet to delete all pods.
	kl.HandlePodDeletions(kl.GetPods())
}
