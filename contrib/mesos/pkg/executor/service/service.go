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
	"bufio"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/cmd/kubelet/app"
	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/executor"
	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/executor/config"
	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/hyperkube"
	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/redirfd"
	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	kconfig "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	log "github.com/golang/glog"
	"github.com/kardianos/osext"
	bindings "github.com/mesos/mesos-go/executor"

	"github.com/spf13/pflag"
)

const (
	// if we don't use this source then the kubelet will do funny, mirror things.
	// @see ConfigSourceAnnotationKey
	MESOS_CFG_SOURCE = kubelet.ApiserverSource
)

type KubeletExecutorServer struct {
	*app.KubeletServer
	RunProxy       bool
	ProxyLogV      int
	ProxyExec      string
	ProxyLogfile   string
	ProxyBindall   bool
	SuicideTimeout time.Duration
	ShutdownFD     int
	ShutdownFIFO   string
}

func NewKubeletExecutorServer() *KubeletExecutorServer {
	k := &KubeletExecutorServer{
		KubeletServer:  app.NewKubeletServer(),
		RunProxy:       true,
		ProxyExec:      "./kube-proxy",
		ProxyLogfile:   "./proxy-log",
		SuicideTimeout: config.DefaultSuicideTimeout,
	}
	if pwd, err := os.Getwd(); err != nil {
		log.Warningf("failed to determine current directory: %v", err)
	} else {
		k.RootDirectory = pwd // mesos sandbox dir
	}
	k.Address = util.IP(net.ParseIP(defaultBindingAddress()))
	k.ShutdownFD = -1 // indicates unspecified FD
	return k
}

func NewHyperKubeletExecutorServer() *KubeletExecutorServer {
	s := NewKubeletExecutorServer()

	// cache this for later use
	binary, err := osext.Executable()
	if err != nil {
		log.Fatalf("failed to determine currently running executable: %v", err)
	}

	s.ProxyExec = binary
	return s
}

func (s *KubeletExecutorServer) addCoreFlags(fs *pflag.FlagSet) {
	s.KubeletServer.AddFlags(fs)
	fs.BoolVar(&s.RunProxy, "run-proxy", s.RunProxy, "Maintain a running kube-proxy instance as a child proc of this kubelet-executor.")
	fs.IntVar(&s.ProxyLogV, "proxy-logv", s.ProxyLogV, "Log verbosity of the child kube-proxy.")
	fs.StringVar(&s.ProxyLogfile, "proxy-logfile", s.ProxyLogfile, "Path to the kube-proxy log file.")
	fs.BoolVar(&s.ProxyBindall, "proxy-bindall", s.ProxyBindall, "When true will cause kube-proxy to bind to 0.0.0.0.")
	fs.DurationVar(&s.SuicideTimeout, "suicide-timeout", s.SuicideTimeout, "Self-terminate after this period of inactivity. Zero disables suicide watch.")
	fs.IntVar(&s.ShutdownFD, "shutdown-fd", s.ShutdownFD, "File descriptor used to signal shutdown to external watchers, requires shutdown-fifo flag")
	fs.StringVar(&s.ShutdownFIFO, "shutdown-fifo", s.ShutdownFIFO, "FIFO used to signal shutdown to external watchers, requires shutdown-fd flag")
}

func (s *KubeletExecutorServer) AddStandaloneFlags(fs *pflag.FlagSet) {
	s.addCoreFlags(fs)
	fs.StringVar(&s.ProxyExec, "proxy-exec", s.ProxyExec, "Path to the kube-proxy executable.")
}

func (s *KubeletExecutorServer) AddHyperkubeFlags(fs *pflag.FlagSet) {
	s.addCoreFlags(fs)
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

	if err := util.ApplyOomScoreAdj(0, s.OOMScoreAdj); err != nil {
		log.Info(err)
	}

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

	cadvisorInterface, err := cadvisor.New(s.CadvisorPort)
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

	var dockerExecHandler dockertools.ExecHandler
	switch s.DockerExecHandlerName {
	case "native":
		dockerExecHandler = &dockertools.NativeExecHandler{}
	case "nsenter":
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
		CadvisorInterface:              cadvisorInterface,
		EnableServer:                   s.EnableServer,
		EnableDebuggingHandlers:        s.EnableDebuggingHandlers,
		DockerClient:                   dockertools.ConnectToDockerOrDie(s.DockerEndpoint),
		KubeClient:                     apiclient,
		MasterServiceNamespace:         s.MasterServiceNamespace,
		VolumePlugins:                  app.ProbeVolumePlugins(),
		NetworkPlugins:                 app.ProbeNetworkPlugins(),
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
		go util.Forever(func() {
			err := http.ListenAndServe(net.JoinHostPort(s.HealthzBindAddress.String(), strconv.Itoa(s.HealthzPort)), nil)
			if err != nil {
				log.Errorf("Starting health server failed: %v", err)
			}
		}, 5*time.Second)
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

	pc := kconfig.NewPodConfig(kconfig.PodConfigNotificationSnapshotAndUpdates, kc.Recorder)
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
		kc.CadvisorInterface,
		kc.ImageGCPolicy,
		kc.DiskSpacePolicy,
		kc.Cloud,
		kc.NodeStatusUpdateFrequency,
		kc.ResourceContainer,
		kc.OSInterface,
		kc.CgroupRoot,
		kc.ContainerRuntime,
		kc.Mounter,
		kc.DockerDaemonContainer,
		kc.SystemContainer,
		kc.ConfigureCBR0,
		kc.MaxPods,
		kc.DockerExecHandler,
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
	})

	go exec.InitializeStaticPodsSource(func() {
		// Create file source only when we are called back. Otherwise, it is never marked unseen.
		fileSourceUpdates := pc.Channel(kubelet.FileSource)

		kconfig.NewSourceFile(staticPodsConfigPath, kc.Hostname, kc.FileCheckFrequency, fileSourceUpdates)
	})

	k := &kubeletExecutor{
		Kubelet:         klet,
		runProxy:        ks.RunProxy,
		proxyLogV:       ks.ProxyLogV,
		proxyExec:       ks.ProxyExec,
		proxyLogfile:    ks.ProxyLogfile,
		proxyBindall:    ks.ProxyBindall,
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
		BindingAddress:   net.IP(ks.Address),
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
	runProxy        bool
	proxyLogV       int
	proxyExec       string
	proxyLogfile    string
	proxyBindall    bool
	address         util.IP
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
		if kl.runProxy {
			go runtime.Until(kl.runProxyService, 5*time.Second, kl.executorDone)
		}
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

// this function blocks as long as the proxy service is running; intended to be
// executed asynchronously.
func (kl *kubeletExecutor) runProxyService() {

	log.Infof("Starting proxy process...")

	const KM_PROXY = "proxy" //TODO(jdef) constant should be shared with km package
	args := []string{}

	if kl.hks.FindServer(KM_PROXY) {
		args = append(args, KM_PROXY)
		log.V(1).Infof("attempting to using km proxy service")
	} else if _, err := os.Stat(kl.proxyExec); os.IsNotExist(err) {
		log.Errorf("failed to locate proxy executable at '%v' and km not present: %v", kl.proxyExec, err)
		return
	}

	bindAddress := "0.0.0.0"
	if !kl.proxyBindall {
		bindAddress = kl.address.String()
	}
	args = append(args,
		fmt.Sprintf("--bind-address=%s", bindAddress),
		fmt.Sprintf("--v=%d", kl.proxyLogV),
		"--logtostderr=true",
	)

	// add client.Config args here. proxy still calls client.BindClientConfigFlags
	appendStringArg := func(name, value string) {
		if value != "" {
			args = append(args, fmt.Sprintf("--%s=%s", name, value))
		}
	}
	appendStringArg("master", kl.clientConfig.Host)
	/* TODO(jdef) move these flags to a config file pointed to by --kubeconfig
	appendStringArg("api-version", kl.clientConfig.Version)
	appendStringArg("client-certificate", kl.clientConfig.CertFile)
	appendStringArg("client-key", kl.clientConfig.KeyFile)
	appendStringArg("certificate-authority", kl.clientConfig.CAFile)
	args = append(args, fmt.Sprintf("--insecure-skip-tls-verify=%t", kl.clientConfig.Insecure))
	*/

	log.Infof("Spawning process executable %s with args '%+v'", kl.proxyExec, args)

	cmd := exec.Command(kl.proxyExec, args...)
	if _, err := cmd.StdoutPipe(); err != nil {
		log.Fatal(err)
	}

	proxylogs, err := cmd.StderrPipe()
	if err != nil {
		log.Fatal(err)
	}

	//TODO(jdef) append instead of truncate? what if the disk is full?
	logfile, err := os.Create(kl.proxyLogfile)
	if err != nil {
		log.Fatal(err)
	}
	defer logfile.Close()

	ch := make(chan struct{})
	go func() {
		defer func() {
			select {
			case <-ch:
				log.Infof("killing proxy process..")
				if err = cmd.Process.Kill(); err != nil {
					log.Errorf("failed to kill proxy process: %v", err)
				}
			default:
			}
		}()

		writer := bufio.NewWriter(logfile)
		defer writer.Flush()

		<-ch
		written, err := io.Copy(writer, proxylogs)
		if err != nil {
			log.Errorf("error writing data to proxy log: %v", err)
		}

		log.Infof("wrote %d bytes to proxy log", written)
	}()

	// if the proxy fails to start then we exit the executor, otherwise
	// wait for the proxy process to end (and release resources after).
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}
	close(ch)
	if err := cmd.Wait(); err != nil {
		log.Error(err)
	}
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
	err := kl.SyncPods([]*api.Pod{}, nil, nil, time.Now())
	if err != nil {
		log.Errorf("failed to cleanly remove all pods and associated state: %v", err)
	}
}
