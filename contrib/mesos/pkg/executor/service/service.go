/*
Copyright 2015 The Kubernetes Authors.

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
	"time"

	log "github.com/golang/glog"
	bindings "github.com/mesos/mesos-go/executor"
	"github.com/spf13/pflag"
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/service/podsource"
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kconfig "k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// TODO(jdef): passing the value of envContainerID to all docker containers instantiated
// through the kubelet is part of a strategy to enable orphan container GC; this can all
// be ripped out once we have a kubelet runtime that leverages Mesos native containerization.

// envContainerID is the name of the environment variable that contains the
// Mesos-assigned container ID of the Executor.
const envContainerID = "MESOS_EXECUTOR_CONTAINER_UUID"

type KubeletExecutorServer struct {
	*options.KubeletServer
	SuicideTimeout    time.Duration
	LaunchGracePeriod time.Duration

	containerID string
}

func NewKubeletExecutorServer() *KubeletExecutorServer {
	k := &KubeletExecutorServer{
		KubeletServer:     options.NewKubeletServer(),
		SuicideTimeout:    config.DefaultSuicideTimeout,
		LaunchGracePeriod: config.DefaultLaunchGracePeriod,
	}
	if pwd, err := os.Getwd(); err != nil {
		log.Warningf("failed to determine current directory: %v", err)
	} else {
		k.RootDirectory = pwd // mesos sandbox dir
	}
	k.Address = defaultBindingAddress()

	return k
}

func (s *KubeletExecutorServer) AddFlags(fs *pflag.FlagSet) {
	s.KubeletServer.AddFlags(fs)
	fs.DurationVar(&s.SuicideTimeout, "suicide-timeout", s.SuicideTimeout, "Self-terminate after this period of inactivity. Zero disables suicide watch.")
	fs.DurationVar(&s.LaunchGracePeriod, "mesos-launch-grace-period", s.LaunchGracePeriod, "Launch grace period after which launching tasks will be cancelled. Zero disables launch cancellation.")
}

func (s *KubeletExecutorServer) runExecutor(
	nodeInfos chan<- executor.NodeInfo,
	kubeletFinished <-chan struct{},
	staticPodsConfigPath string,
	apiclient *clientset.Clientset,
	registry executor.Registry,
) (<-chan struct{}, error) {
	staticPodFilters := podutil.Filters{
		// annotate the pod with BindingHostKey so that the scheduler will ignore the pod
		// once it appears in the pod registry. the stock kubelet sets the pod host in order
		// to accomplish the same; we do this because the k8sm scheduler works differently.
		podutil.Annotator(map[string]string{
			meta.BindingHostKey: s.HostnameOverride,
		}),
	}
	if s.containerID != "" {
		// tag all pod containers with the containerID so that they can be properly GC'd by Mesos
		staticPodFilters = append(staticPodFilters, podutil.Environment([]api.EnvVar{
			{Name: envContainerID, Value: s.containerID},
		}))
	}
	exec := executor.New(executor.Config{
		Registry:        registry,
		APIClient:       apiclient,
		Docker:          dockertools.ConnectToDockerOrDie(s.DockerEndpoint, 0),
		SuicideTimeout:  s.SuicideTimeout,
		KubeletFinished: kubeletFinished,
		ExitFunc:        os.Exit,
		NodeInfos:       nodeInfos,
		Options: []executor.Option{
			executor.StaticPods(staticPodsConfigPath, staticPodFilters),
		},
	})

	// initialize driver and initialize the executor with it
	dconfig := bindings.DriverConfig{
		Executor:         exec,
		HostnameOverride: s.HostnameOverride,
		BindingAddress:   net.ParseIP(s.Address),
	}
	driver, err := bindings.NewMesosExecutorDriver(dconfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create executor driver: %v", err)
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

	return exec.Done(), nil
}

func (s *KubeletExecutorServer) runKubelet(
	nodeInfos <-chan executor.NodeInfo,
	kubeletDone chan<- struct{},
	staticPodsConfigPath string,
	apiclient *clientset.Clientset,
	podLW *cache.ListWatch,
	registry executor.Registry,
	executorDone <-chan struct{},
) (err error) {
	defer func() {
		if err != nil {
			// close the channel here. When Run returns without error, the executorKubelet is
			// responsible to do this. If it returns with an error, we are responsible here.
			close(kubeletDone)
		}
	}()

	kubeDeps, err := kubeletapp.UnsecuredKubeletDeps(s.KubeletServer)
	if err != nil {
		return err
	}

	// apply Mesos specific settings
	kubeDeps.Builder = func(kubeCfg *componentconfig.KubeletConfiguration, kubeDeps *kubelet.KubeletDeps, standaloneMode bool) (kubelet.KubeletBootstrap, error) {
		k, err := kubeletapp.CreateAndInitKubelet(kubeCfg, kubeDeps, standaloneMode)
		if err != nil {
			return k, err
		}

		// decorate kubelet such that it shuts down when the executor is
		decorated := &executorKubelet{
			Kubelet:      k.(*kubelet.Kubelet),
			kubeletDone:  kubeletDone,
			executorDone: executorDone,
		}

		return decorated, nil
	}
	s.RuntimeCgroups = "" // don't move the docker daemon into a cgroup
	kubeDeps.KubeClient = apiclient

	// taken from KubeletServer#Run(*KubeletConfig)
	eventClientConfig, err := kubeletapp.CreateAPIServerClientConfig(s.KubeletServer)
	if err != nil {
		return err
	}

	// make a separate client for events
	eventClientConfig.QPS = float32(s.EventRecordQPS)
	eventClientConfig.Burst = int(s.EventBurst)
	kubeDeps.EventClient, err = clientset.NewForConfig(eventClientConfig)
	if err != nil {
		return err
	}

	kubeDeps.PodConfig = kconfig.NewPodConfig(kconfig.PodConfigNotificationIncremental, kubeDeps.Recorder) // override the default pod source

	s.SystemCgroups = "" // don't take control over other system processes.

	if kubeDeps.Cloud != nil {
		// fail early and hard because having the cloud provider loaded would go unnoticed,
		// but break bigger cluster because accessing the state.json from every slave kills the master.
		panic("cloud provider must not be set")
	}

	// create custom cAdvisor interface which return the resource values that Mesos reports
	ni := <-nodeInfos
	cAdvisorInterface, err := NewMesosCadvisor(ni.Cores, ni.Mem, uint(s.CAdvisorPort), s.ContainerRuntime)
	if err != nil {
		return err
	}

	kubeDeps.CAdvisorInterface = cAdvisorInterface
	kubeDeps.ContainerManager, err = cm.NewContainerManager(kubeDeps.Mounter, cAdvisorInterface, cm.NodeConfig{
		RuntimeCgroupsName: s.RuntimeCgroups,
		SystemCgroupsName:  s.SystemCgroups,
		KubeletCgroupsName: s.KubeletCgroups,
		ContainerRuntime:   s.ContainerRuntime,
	})
	if err != nil {
		return err
	}

	go func() {
		for ni := range nodeInfos {
			// TODO(sttts): implement with MachineAllocable mechanism when https://github.com/kubernetes/kubernetes/issues/13984 is finished
			log.V(3).Infof("ignoring updated node resources: %v", ni)
		}
	}()

	// create main pod source, it will stop generating events once executorDone is closed
	var containerOptions []podsource.Option
	if s.containerID != "" {
		// tag all pod containers with the containerID so that they can be properly GC'd by Mesos
		containerOptions = append(containerOptions, podsource.ContainerEnvOverlay([]api.EnvVar{
			{Name: envContainerID, Value: s.containerID},
		}))
		kubeDeps.ContainerRuntimeOptions = append(kubeDeps.ContainerRuntimeOptions,
			dockertools.PodInfraContainerEnv(map[string]string{
				envContainerID: s.containerID,
			}))
	}

	podsource.Mesos(executorDone, kubeDeps.PodConfig.Channel(podsource.MesosSource), podLW, registry, containerOptions...)

	// create static-pods directory file source
	log.V(2).Infof("initializing static pods source factory, configured at path %q", staticPodsConfigPath)
	fileSourceUpdates := kubeDeps.PodConfig.Channel(kubetypes.FileSource)
	kconfig.NewSourceFile(staticPodsConfigPath, s.HostnameOverride, s.FileCheckFrequency.Duration, fileSourceUpdates)

	// run the kubelet
	// NOTE: because kubeDeps != nil holds, the upstream Run function will not
	//       initialize the cloud provider. We explicitly wouldn't want
	//       that because then every kubelet instance would query the master
	//       state.json which does not scale.
	s.KubeletServer.LockFilePath = "" // disable lock file
	err = kubeletapp.Run(s.KubeletServer, kubeDeps)
	return
}

// Run runs the specified KubeletExecutorServer.
func (s *KubeletExecutorServer) Run(hks hyperkube.Interface, _ []string) error {
	// create shared channels
	kubeletFinished := make(chan struct{})
	nodeInfos := make(chan executor.NodeInfo, 1)

	// create static pods directory
	staticPodsConfigPath := filepath.Join(s.RootDirectory, "static-pods")
	err := os.Mkdir(staticPodsConfigPath, 0750)
	if err != nil {
		return err
	}

	// we're expecting that either Mesos or the minion process will set this for us
	s.containerID = os.Getenv(envContainerID)
	if s.containerID == "" {
		log.Warningf("missing expected environment variable %q", envContainerID)
	}

	// create apiserver client
	var apiclient *clientset.Clientset
	clientConfig, err := kubeletapp.CreateAPIServerClientConfig(s.KubeletServer)
	if err == nil {
		apiclient, err = clientset.NewForConfig(clientConfig)
	}
	if err != nil {
		// required for k8sm since we need to send api.Binding information back to the apiserver
		return fmt.Errorf("cannot create API client: %v", err)
	}

	var (
		pw = cache.NewListWatchFromClient(apiclient.CoreClient, "pods", api.NamespaceAll,
			fields.OneTermEqualSelector(api.PodHostField, s.HostnameOverride),
		)
		reg = executor.NewRegistry(apiclient)
	)

	// start executor
	var executorDone <-chan struct{}
	executorDone, err = s.runExecutor(nodeInfos, kubeletFinished, staticPodsConfigPath, apiclient, reg)
	if err != nil {
		return err
	}

	// start kubelet, blocking
	return s.runKubelet(nodeInfos, kubeletFinished, staticPodsConfigPath, apiclient, pw, reg, executorDone)
}

func defaultBindingAddress() string {
	libProcessIP := os.Getenv("LIBPROCESS_IP")
	if libProcessIP == "" {
		return "0.0.0.0"
	} else {
		return libProcessIP
	}
}
