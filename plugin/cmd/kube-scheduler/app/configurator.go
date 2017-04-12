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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"

	"k8s.io/client-go/tools/cache"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/plugin/cmd/kube-scheduler/app/options"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api/latest"
	"k8s.io/kubernetes/plugin/pkg/scheduler/api/validation"
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
  informerFactory informers.SharedInformerFactory,
	recorder record.EventRecorder,
) (*scheduler.Scheduler, error) {
	configurator := factory.NewConfigFactory(
		s.SchedulerName,
		kubecli,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		s.HardPodAffinitySymmetricWeight,
	)

	// Rebuild the configurator with a default Create(...) method.
	schedConfigurator := &schedulerConfigurator{
		configurator,
		s.PolicyConfigFile,
		s.AlgorithmProvider,
		s.PolicyConfigMapName,
		s.PolicyConfigMapNamespace,
		s.UseLegacyPolicyConfig,
		nil,
	}

	scheduler, err := scheduler.NewFromConfigurator(schedConfigurator, func(cfg *scheduler.Config) {
		cfg.Recorder = recorder
		schedConfigurator.schedulerConfig = cfg
	})

	// Install event handlers for changes to the scheduler's ConfigMap
	if !s.UseLegacyPolicyConfig && len(s.PolicyConfigMapName) != 0 {
		schedConfigurator.SetupPolicyConfigMapEventHandlers(kubecli, informerFactory)
	}

	return scheduler, err
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
	schedulerConfig          *scheduler.Config
}

// getSchedulerPolicyConfig finds and decodes scheduler's policy config. If no
// such policy is found, it returns nil, nil.
func (sc *schedulerConfigurator) getSchedulerPolicyConfig() (*schedulerapi.Policy, error) {
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

func (sc *schedulerConfigurator) SetupPolicyConfigMapEventHandlers(client clientset.Interface, informerFactory informers.SharedInformerFactory) {
	// selector targets only the scheduler's policy ConfigMap.
	selector := cache.NewListWatchFromClient(client.CoreV1().RESTClient(), "configmaps", sc.policyConfigMapNamespace, fields.OneTermEqualSelector(api.ObjectNameField, string(sc.policyConfigMap)))

	sharedIndexInformer := informerFactory.InformerFor(&v1.ConfigMap{}, func(client clientset.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
		sharedIndexInformer := cache.NewSharedIndexInformer(
			selector,
			&v1.ConfigMap{},
			resyncPeriod,
			cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		)
		return sharedIndexInformer
	})

	sharedIndexInformer.AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs {
			AddFunc:    nil, // scheduler aborts if its policy ConfigMap does not exist. So, we do not need and "add" handler.
			UpdateFunc: sc.updatePolicyConfigMap,
			DeleteFunc: sc.deletePolicyConfigMap,
		},
		0,
	)
}

// verifyNewPolicyConfig verifies that the new object received by the ConfigMap watch is a valid policy config.
func verifyNewPolicyConfig(obj interface{}) error {
	newConfig, ok := obj.(*v1.ConfigMap)
	if !ok {
		return fmt.Errorf("cannot convert obj to *v1.ConfigMap: %v", obj)
	}
	newPolicy, ok := newConfig.Data[options.SchedulerPolicyConfigMapKey]
	if !ok {
		return fmt.Errorf("No element with key = '%v' is found in the ConfigMap.Data", options.SchedulerPolicyConfigMapKey)
	}
	configData := []byte(newPolicy)
	var policy schedulerapi.Policy
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		return fmt.Errorf("invalid scheduler policy configuration: %v", err)
	}
	// validate the policy configuration
	if err := validation.ValidatePolicy(policy); err != nil {
		return fmt.Errorf("The new scheduler policy is invalid. Will keep using the old policy! Error: %v", err)
	}
	return nil
}

func (sc *schedulerConfigurator) addPolicyConfigMap(obj interface{}) {
	glog.Info("Received a request to add a scheduler policy config.")
	err := verifyNewPolicyConfig(obj)
	if err != nil {
		glog.Error(err)
		return
	}
	// If things are in order, kill the scheduler to apply the new config.
	sc.KillScheduler()
}

func (sc *schedulerConfigurator) updatePolicyConfigMap(oldObj, newObj interface{}) {
	glog.Info("Received an update to the scheduler policy config.")
	_, ok := oldObj.(*v1.ConfigMap)
	if !ok {
		glog.Errorf("cannot convert oldObj to *v1.ConfigMap: %v", oldObj)
		return
	}
	err := verifyNewPolicyConfig(newObj)
	if err != nil {
		glog.Error(err)
		return
	}
	// If things are in order, kill the scheduler to apply the new config.
	sc.KillScheduler()
}


func (sc *schedulerConfigurator) deletePolicyConfigMap(obj interface{}) {
	glog.Info("Scheduler's policy ConfigMap is deleted.")
	switch t := obj.(type) {
	case *v1.ConfigMap:  // Nothing is needed. Jump out of the switch.
	case cache.DeletedFinalStateUnknown:
		_, ok := t.Obj.(*v1.ConfigMap)
		if !ok {
			glog.Errorf("cannot convert to *v1.ConfigMap: %v", t.Obj)
			return
		}
	default:
		glog.Errorf("cannot convert to *v1.ConfigMap: %v", t)
		return
	}
	sc.KillScheduler()
}

// schedulerKillFunc is a function that kills the scheduler. It is here mainly for testability. Tests set it to a function to perform an action that can be verified in tests instead of the default behavior which causes the scheduler to die.
var SchedulerKillFunc func() = nil

func (sc *schedulerConfigurator) KillScheduler() {
	if SchedulerKillFunc != nil {
		SchedulerKillFunc()
	} else {
		if sc.schedulerConfig != nil {
			glog.Infof("Scheduler is going to die (and restarted) in order to update its policy.")
			close(sc.schedulerConfig.StopEverything)
			// The sleep is only to allow cleanups to happen.
			time.Sleep(2 * time.Second)
			glog.Flush()
			os.Exit(0)
		}
		glog.Infof("Scheduler is not going to exit, as it doesn't seem to be initialized yet.")
	}
}
