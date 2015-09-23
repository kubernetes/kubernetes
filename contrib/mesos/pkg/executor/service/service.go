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
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	log "github.com/golang/glog"
	bindings "github.com/mesos/mesos-go/executor"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet"
	kconfig "k8s.io/kubernetes/pkg/kubelet/config"
	kubeletContainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util"
)

const (
	// if we don't use this source then the kubelet will do funny, mirror things.
	// @see ConfigSourceAnnotationKey
	MESOS_CFG_SOURCE = kubetypes.ApiserverSource
)

type KubeletExecutorServer struct {
	*app.KubeletServer
	SuicideTimeout    time.Duration
	LaunchGracePeriod time.Duration

	kletLock sync.Mutex // TODO(sttts): remove necessity to access the kubelet from the executor
	klet     *kubelet.Kubelet
}

func NewKubeletExecutorServer() *KubeletExecutorServer {
	k := &KubeletExecutorServer{
		KubeletServer:     app.NewKubeletServer(),
		SuicideTimeout:    config.DefaultSuicideTimeout,
		LaunchGracePeriod: config.DefaultLaunchGracePeriod,
	}
	if pwd, err := os.Getwd(); err != nil {
		log.Warningf("failed to determine current directory: %v", err)
	} else {
		k.RootDirectory = pwd // mesos sandbox dir
	}
	k.Address = net.ParseIP(defaultBindingAddress())

	return k
}

func (s *KubeletExecutorServer) AddFlags(fs *pflag.FlagSet) {
	s.KubeletServer.AddFlags(fs)
	fs.DurationVar(&s.SuicideTimeout, "suicide-timeout", s.SuicideTimeout, "Self-terminate after this period of inactivity. Zero disables suicide watch.")
	fs.DurationVar(&s.LaunchGracePeriod, "mesos-launch-grace-period", s.LaunchGracePeriod, "Launch grace period after which launching tasks will be cancelled. Zero disables launch cancellation.")
}

func (s *KubeletExecutorServer) runExecutor(execUpdates chan<- kubetypes.PodUpdate, kubeletFinished <-chan struct{},
	staticPodsConfigPath string, apiclient *client.Client) error {
	exec := executor.New(executor.Config{
		Updates:         execUpdates,
		APIClient:       apiclient,
		Docker:          dockertools.ConnectToDockerOrDie(s.DockerEndpoint),
		SuicideTimeout:  s.SuicideTimeout,
		KubeletFinished: kubeletFinished,
		ExitFunc:        os.Exit,
		PodStatusFunc: func(pod *api.Pod) (*api.PodStatus, error) {
			s.kletLock.Lock()
			defer s.kletLock.Unlock()

			if s.klet == nil {
				return nil, fmt.Errorf("PodStatucFunc called before kubelet is initialized")
			}

			status, err := s.klet.GetRuntime().GetPodStatus(pod)
			if err != nil {
				return nil, err
			}

			status.Phase = kubelet.GetPhase(&pod.Spec, status.ContainerStatuses)
			hostIP, err := s.klet.GetHostIP()
			if err != nil {
				log.Errorf("Cannot get host IP: %v", err)
			} else {
				status.HostIP = hostIP.String()
			}
			return status, nil
		},
		StaticPodsConfigPath: staticPodsConfigPath,
		PodLW: cache.NewListWatchFromClient(apiclient, "pods", api.NamespaceAll,
			fields.OneTermEqualSelector(client.PodHost, s.HostnameOverride),
		),
	})

	// initialize driver and initialize the executor with it
	dconfig := bindings.DriverConfig{
		Executor:         exec,
		HostnameOverride: s.HostnameOverride,
		BindingAddress:   s.Address,
	}
	driver, err := bindings.NewMesosExecutorDriver(dconfig)
	if err != nil {
		return fmt.Errorf("failed to create executor driver: %v", err)
	}
	log.V(2).Infof("Initialize executor driver...")
	exec.Init(driver)

	// start the driver
	go func() {
		if _, err := driver.Run(); err != nil {
			log.Fatalf("executor driver failed: %v", err)
		}
		log.Info("executor Run completed")
	}()

	return nil
}

func (s *KubeletExecutorServer) runKubelet(execUpdates <-chan kubetypes.PodUpdate, kubeletFinished chan<- struct{},
	staticPodsConfigPath string, apiclient *client.Client) error {
	kcfg, err := s.UnsecuredKubeletConfig()
	if err == nil {
		kcfg.Builder = func(kc *app.KubeletConfig) (app.KubeletBootstrap, *kconfig.PodConfig, error) {
			return s.createAndInitKubelet(kc, staticPodsConfigPath, execUpdates, kubeletFinished)
		}
		kcfg.DockerDaemonContainer = "" // don't move the docker daemon into a cgroup
		kcfg.Hostname = kcfg.HostnameOverride
		kcfg.KubeClient = apiclient
		kcfg.NodeName = kcfg.HostnameOverride
		kcfg.StandaloneMode = false
		kcfg.SystemContainer = "" // don't take control over other system processes.
		err = s.KubeletServer.Run(kcfg)
	}

	if err != nil {
		close(kubeletFinished)
	}
	return err
}

// Run runs the specified KubeletExecutorServer.
func (s *KubeletExecutorServer) Run(hks hyperkube.Interface, _ []string) error {
	// create shared channels
	kubeletFinished := make(chan struct{})
	execUpdates := make(chan kubetypes.PodUpdate, 1)

	// create static pods directory
	staticPodsConfigPath := filepath.Join(s.RootDirectory, "static-pods")
	err := os.Mkdir(staticPodsConfigPath, 0755)
	if err != nil {
		return err
	}

	// create apiserver client
	var apiclient *client.Client
	clientConfig, err := s.CreateAPIServerClientConfig()
	if err == nil {
		apiclient, err = client.New(clientConfig)
	}
	if err != nil {
		// required for k8sm since we need to send api.Binding information back to the apiserver
		log.Fatalf("No API client: %v", err)
	}

	// start executor
	err = s.runExecutor(execUpdates, kubeletFinished, staticPodsConfigPath, apiclient)
	if err != nil {
		return err
	}

	// start kubelet, blocking
	return s.runKubelet(execUpdates, kubeletFinished, staticPodsConfigPath, apiclient)
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
	staticPodsConfigPath string,
	execUpdates <-chan kubetypes.PodUpdate,
	kubeletDone chan<- struct{},
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

	gcPolicy := kubeletContainer.ContainerGCPolicy{
		MinAge:             kc.MinimumGCAge,
		MaxPerPodContainer: kc.MaxPerPodContainerCount,
		MaxContainers:      kc.MaxContainerCount,
	}

	// create main pod source
	pc := kconfig.NewPodConfig(kconfig.PodConfigNotificationIncremental, kc.Recorder)
	updates := pc.Channel(MESOS_CFG_SOURCE)
	executorDone := make(chan struct{})
	go func() {
		// execUpdates will be closed by the executor on shutdown
		defer close(executorDone)

		for u := range execUpdates {
			u.Source = MESOS_CFG_SOURCE
			updates <- u
		}
	}()

	// create static-pods directory file source
	log.V(2).Infof("initializing static pods source factory, configured at path %q", staticPodsConfigPath)
	fileSourceUpdates := pc.Channel(kubetypes.FileSource)
	kconfig.NewSourceFile(staticPodsConfigPath, kc.Hostname, kc.FileCheckFrequency, fileSourceUpdates)

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
		kc.RegisterSchedulable,
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
		kc.ReconcileCIDR,
		kc.MaxPods,
		kc.DockerExecHandler,
		kc.ResolverConfig,
		kc.CPUCFSQuota,
		&api.NodeDaemonEndpoints{
			KubeletEndpoint: api.DaemonEndpoint{Port: int(kc.Port)},
		},
		kc.OOMAdjuster,
	)
	if err != nil {
		return nil, nil, err
	}

	ks.kletLock.Lock()
	ks.klet = klet
	ks.kletLock.Unlock()

	k := &kubeletExecutor{
		Kubelet:      ks.klet,
		address:      ks.Address,
		dockerClient: kc.DockerClient,
		kubeletDone:  kubeletDone,
		executorDone: executorDone,
	}

	k.BirthCry()
	k.StartGarbageCollection()

	return k, pc, nil
}

// kubelet decorator
type kubeletExecutor struct {
	*kubelet.Kubelet
	address      net.IP
	dockerClient dockertools.DockerInterface
	kubeletDone  chan<- struct{} // closed once kubelet.Run() returns
	executorDone <-chan struct{} // closed when executor terminates
}

func (kl *kubeletExecutor) ListenAndServe(address net.IP, port uint, tlsOptions *kubelet.TLSOptions, auth kubelet.AuthInterface, enableDebuggingHandlers bool) {
	log.Infof("Starting kubelet server...")
	kubelet.ListenAndServeKubeletServer(kl, address, port, tlsOptions, auth, enableDebuggingHandlers)
}

// runs the main kubelet loop, closing the kubeletFinished chan when the loop exits.
// never returns.
func (kl *kubeletExecutor) Run(mergedUpdates <-chan kubetypes.PodUpdate) {
	defer func() {
		close(kl.kubeletDone)
		util.HandleCrash()
		log.Infoln("kubelet run terminated") //TODO(jdef) turn down verbosity
		// important: never return! this is in our contract
		select {}
	}()

	// push merged updates into another, closable update channel which is closed
	// when the executor shuts down.
	closableUpdates := make(chan kubetypes.PodUpdate)
	go func() {
		// closing closableUpdates will cause our patched kubelet's syncLoop() to exit
		defer close(closableUpdates)
	pipeLoop:
		for {
			select {
			case <-kl.executorDone:
				break pipeLoop
			default:
				select {
				case u := <-mergedUpdates:
					select {
					case closableUpdates <- u: // noop
					case <-kl.executorDone:
						break pipeLoop
					}
				case <-kl.executorDone:
					break pipeLoop
				}
			}
		}
	}()

	// we expect that Run() will complete after closableUpdates is closed and the
	// kubelet's syncLoop() has finished processing its backlog, which hopefully
	// will not take very long. Peeking into the future (current k8s master) it
	// seems that the backlog has grown from 1 to 50 -- this may negatively impact
	// us going forward, time will tell.
	util.Until(func() { kl.Kubelet.Run(closableUpdates) }, 0, kl.executorDone)

	//TODO(jdef) revisit this if/when executor failover lands
	// Force kubelet to delete all pods.
	kl.HandlePodDeletions(kl.GetPods())
}

