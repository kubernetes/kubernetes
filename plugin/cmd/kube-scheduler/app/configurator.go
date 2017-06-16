/*
Copyright 2017 The Kubernetes Authors.

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

package app

import (
	"fmt"
	"io/ioutil"
	"os"

	appsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/apps/v1beta1"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/core/v1"
	extensionsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/extensions/v1beta1"
	"k8s.io/kubernetes/plugin/cmd/kube-scheduler/app/options"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"

	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"

	clientv1 "k8s.io/client-go/pkg/api/v1"

	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api/latest"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/golang/glog"
)

func createRecorder(kubecli *clientset.Clientset, s *options.SchedulerServer) record.EventRecorder {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubecli.Core().RESTClient()).Events("")})
	return eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: s.SchedulerName})
}

func createClient(s *options.SchedulerServer) (*clientset.Clientset, error) {
	kubeconfig, err := clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("unable to build config from flags: %v", err)
	}

	kubeconfig.ContentType = s.ContentType
	// Override kubeconfig qps/burst settings from flags
	kubeconfig.QPS = s.KubeAPIQPS
	kubeconfig.Burst = int(s.KubeAPIBurst)

	cli, err := clientset.NewForConfig(restclient.AddUserAgent(kubeconfig, "leader-election"))
	if err != nil {
		return nil, fmt.Errorf("invalid API configuration: %v", err)
	}
	return cli, nil
}

// CreateScheduler encapsulates the entire creation of a runnable scheduler.
func CreateScheduler(
	s *options.SchedulerServer,
	kubecli *clientset.Clientset,
	nodeInformer coreinformers.NodeInformer,
	podInformer coreinformers.PodInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	replicationControllerInformer coreinformers.ReplicationControllerInformer,
	replicaSetInformer extensionsinformers.ReplicaSetInformer,
	statefulSetInformer appsinformers.StatefulSetInformer,
	serviceInformer coreinformers.ServiceInformer,
	recorder record.EventRecorder,
) (*scheduler.Scheduler, error) {
	configurator := factory.NewConfigFactory(
		s.SchedulerName,
		kubecli,
		nodeInformer,
		podInformer,
		pvInformer,
		pvcInformer,
		replicationControllerInformer,
		replicaSetInformer,
		statefulSetInformer,
		serviceInformer,
		s.HardPodAffinitySymmetricWeight,
	)

	// Rebuild the configurator with a default Create(...) method.
	configurator = &schedulerConfigurator{
		configurator,
		s.PolicyConfigFile,
		s.AlgorithmProvider,
		s.PolicyConfigMapName,
		s.PolicyConfigMapNamespace,
		s.UseLegacyPolicyConfig,
	}

	return scheduler.NewFromConfigurator(configurator, func(cfg *scheduler.Config) {
		cfg.Recorder = recorder
	})
}

// schedulerConfigurator is an interface wrapper that provides a way to create
// a scheduler from a user provided config file or ConfigMap object.
type schedulerConfigurator struct {
	scheduler.Configurator
	policyFile               string
	algorithmProvider        string
	policyConfigMap          string
	policyConfigMapNamespace string
	useLegacyPolicyConfig    bool
}

// getSchedulerPolicyConfig finds and decodes scheduler's policy config. If no
// such policy is found, it returns nil, nil.
func (sc schedulerConfigurator) getSchedulerPolicyConfig() (*schedulerapi.Policy, error) {
	var configData []byte
	var policyConfigMapFound bool
	var policy schedulerapi.Policy

	// If not in legacy mode, try to find policy ConfigMap.
	if !sc.useLegacyPolicyConfig && len(sc.policyConfigMap) != 0 {
		namespace := sc.policyConfigMapNamespace
		policyConfigMap, err := sc.GetClient().CoreV1().ConfigMaps(namespace).Get(sc.policyConfigMap, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("Error getting scheduler policy ConfigMap: %v.", err)
		}
		if policyConfigMap != nil {
			var configString string
			configString, policyConfigMapFound = policyConfigMap.Data[options.SchedulerPolicyConfigMapKey]
			if !policyConfigMapFound {
				return nil, fmt.Errorf("No element with key = '%v' is found in the ConfigMap 'Data'.", options.SchedulerPolicyConfigMapKey)
			}
			glog.V(5).Infof("Scheduler policy ConfigMap: %v", configString)
			configData = []byte(configString)
		}
	}

	// If we are in legacy mode or ConfigMap name is empty, try to use policy
	// config file.
	if !policyConfigMapFound {
		if _, err := os.Stat(sc.policyFile); err != nil {
			// No config file is found.
			return nil, nil
		}
		var err error
		configData, err = ioutil.ReadFile(sc.policyFile)
		if err != nil {
			return nil, fmt.Errorf("unable to read policy config: %v", err)
		}
	}

	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		return nil, fmt.Errorf("invalid configuration: %v", err)
	}
	return &policy, nil
}

// Create implements the interface for the Configurator, hence it is exported
// even though the struct is not.
func (sc schedulerConfigurator) Create() (*scheduler.Config, error) {
	policy, err := sc.getSchedulerPolicyConfig()
	if err != nil {
		return nil, err
	}
	// If no policy is found, create scheduler from algorithm provider.
	if policy == nil {
		if sc.Configurator != nil {
			return sc.Configurator.CreateFromProvider(sc.algorithmProvider)
		}
		return nil, fmt.Errorf("Configurator was nil")
	}

	return sc.CreateFromConfig(*policy)
}
