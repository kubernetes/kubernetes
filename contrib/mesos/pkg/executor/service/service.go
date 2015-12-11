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
	"time"

	log "github.com/golang/glog"
	bindings "github.com/mesos/mesos-go/executor"
	"github.com/spf13/pflag"
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor"
	"k8s.io/kubernetes/contrib/mesos/pkg/executor/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kconfig "k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

type KubeletExecutorServer struct {
	*kubeletapp.KubeletServer
	SuicideTimeout    time.Duration
	LaunchGracePeriod time.Duration

	// TODO(sttts): remove necessity to access the kubelet from the executor

	klet      *kubelet.Kubelet // once set, immutable
	kletReady chan struct{}    // once closed, klet is guaranteed to be valid and concurrently readable
}

func NewKubeletExecutorServer() *KubeletExecutorServer {
	k := &KubeletExecutorServer{
		KubeletServer:     kubeletapp.NewKubeletServer(),
		SuicideTimeout:    config.DefaultSuicideTimeout,
		LaunchGracePeriod: config.DefaultLaunchGracePeriod,
		kletReady:         make(chan struct{}),
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

func (s *KubeletExecutorServer) runExecutor(
	execUpdates chan<- kubetypes.PodUpdate,
	nodeInfos chan<- executor.NodeInfo,
	kubeletFinished <-chan struct{},
	staticPodsConfigPath string,
	apiclient *client.Client,
	podLW *cache.ListWatch,
) error {
	exec := executor.New(executor.Config{
		Updates:         execUpdates,
		APIClient:       apiclient,
		Docker:          dockertools.ConnectToDockerOrDie(s.DockerEndpoint),
		SuicideTimeout:  s.SuicideTimeout,
		KubeletFinished: kubeletFinished,
		ExitFunc:        os.Exit,
		PodStatusFunc: func(pod *api.Pod) (*api.PodStatus, error) {
			select {
			case <-s.kletReady:
			default:
				return nil, fmt.Errorf("PodStatucFunc called before kubelet is initialized")
			}

			status, err := s.klet.GetRuntime().GetAPIPodStatus(pod)
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
		PodLW:                podLW,
		NodeInfos:            nodeInfos,
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

func (s *KubeletExecutorServer) runKubelet(
	execUpdates <-chan kubetypes.PodUpdate,
	nodeInfos <-chan executor.NodeInfo,
	kubeletDone chan<- struct{},
	staticPodsConfigPath string,
	apiclient *client.Client,
	podLW *cache.ListWatch,
) (err error) {
	defer func() {
		if err != nil {
			// close the channel here. When Run returns without error, the executorKubelet is
			// responsible to do this. If it returns with an error, we are responsible here.
			close(kubeletDone)
		}
	}()

	kcfg, err := s.UnsecuredKubeletConfig()
	if err != nil {
		return err
	}

	// apply Mesos specific settings
	executorDone := make(chan struct{})
	kcfg.Builder = func(kc *kubeletapp.KubeletConfig) (kubeletapp.KubeletBootstrap, *kconfig.PodConfig, error) {
		k, pc, err := kubeletapp.CreateAndInitKubelet(kc)
		if err != nil {
			return k, pc, err
		}

		s.klet = k.(*kubelet.Kubelet)
		close(s.kletReady) // intentionally crash if this is called more than once

		// decorate kubelet such that it shuts down when the executor is
		decorated := &executorKubelet{
			Kubelet:      s.klet,
			kubeletDone:  kubeletDone,
			executorDone: executorDone,
		}

		return decorated, pc, nil
	}
	kcfg.DockerDaemonContainer = "" // don't move the docker daemon into a cgroup
	kcfg.Hostname = kcfg.HostnameOverride
	kcfg.KubeClient = apiclient

	// taken from KubeletServer#Run(*KubeletConfig)
	eventClientConfig, err := s.CreateAPIServerClientConfig()
	if err != nil {
		return err
	}

	// make a separate client for events
	eventClientConfig.QPS = s.EventRecordQPS
	eventClientConfig.Burst = s.EventBurst
	kcfg.EventClient, err = client.New(eventClientConfig)
	if err != nil {
		return err
	}

	kcfg.NodeName = kcfg.HostnameOverride
	kcfg.PodConfig = kconfig.NewPodConfig(kconfig.PodConfigNotificationIncremental, kcfg.Recorder) // override the default pod source
	kcfg.StandaloneMode = false
	kcfg.SystemContainer = "" // don't take control over other system processes.
	if kcfg.Cloud != nil {
		// fail early and hard because having the cloud provider loaded would go unnoticed,
		// but break bigger cluster because accessing the state.json from every slave kills the master.
		panic("cloud provider must not be set")
	}

	// create custom cAdvisor interface which return the resource values that Mesos reports
	ni := <-nodeInfos
	cAdvisorInterface, err := NewMesosCadvisor(ni.Cores, ni.Mem, s.CAdvisorPort)
	if err != nil {
		return err
	}

	kcfg.CAdvisorInterface = cAdvisorInterface
	kcfg.ContainerManager, err = cm.NewContainerManager(kcfg.Mounter, cAdvisorInterface)
	if err != nil {
		return err
	}

	go func() {
		for ni := range nodeInfos {
			// TODO(sttts): implement with MachineAllocable mechanism when https://github.com/kubernetes/kubernetes/issues/13984 is finished
			log.V(3).Infof("ignoring updated node resources: %v", ni)
		}
	}()

	// create main pod source, it will close executorDone when the executor updates stop flowing
	newSourceMesos(executorDone, execUpdates, kcfg.PodConfig.Channel(mesosSource), podLW)

	// create static-pods directory file source
	log.V(2).Infof("initializing static pods source factory, configured at path %q", staticPodsConfigPath)
	fileSourceUpdates := kcfg.PodConfig.Channel(kubetypes.FileSource)
	kconfig.NewSourceFile(staticPodsConfigPath, kcfg.HostnameOverride, kcfg.FileCheckFrequency, fileSourceUpdates)

	// run the kubelet, until execUpdates is closed
	// NOTE: because kcfg != nil holds, the upstream Run function will not
	//       initialize the cloud provider. We explicitly wouldn't want
	//       that because then every kubelet instance would query the master
	//       state.json which does not scale.
	err = s.KubeletServer.Run(kcfg)

	return
}

// Run runs the specified KubeletExecutorServer.
func (s *KubeletExecutorServer) Run(hks hyperkube.Interface, _ []string) error {
	// create shared channels
	kubeletFinished := make(chan struct{})
	execUpdates := make(chan kubetypes.PodUpdate, 1)
	nodeInfos := make(chan executor.NodeInfo, 1)

	// create static pods directory
	staticPodsConfigPath := filepath.Join(s.RootDirectory, "static-pods")
	err := os.Mkdir(staticPodsConfigPath, 0750)
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
		return fmt.Errorf("cannot create API client: %v", err)
	}

	pw := cache.NewListWatchFromClient(apiclient, "pods", api.NamespaceAll,
		fields.OneTermEqualSelector(client.PodHost, s.HostnameOverride),
	)

	// start executor
	err = s.runExecutor(execUpdates, nodeInfos, kubeletFinished, staticPodsConfigPath, apiclient, pw)
	if err != nil {
		return err
	}

	// start kubelet, blocking
	return s.runKubelet(execUpdates, nodeInfos, kubeletFinished, staticPodsConfigPath, apiclient, pw)
}

func defaultBindingAddress() string {
	libProcessIP := os.Getenv("LIBPROCESS_IP")
	if libProcessIP == "" {
		return "0.0.0.0"
	} else {
		return libProcessIP
	}
}
