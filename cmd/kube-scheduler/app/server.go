/*
Copyright 2014 The Kubernetes Authors.

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

// Package app implements a Server object for running the scheduler.
package app

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"reflect"
	goruntime "runtime"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	componentconfigv1alpha1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithmprovider"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/pkg/scheduler/api/latest"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/version/verflag"

	"k8s.io/kubernetes/pkg/scheduler/factory"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// SchedulerServer has all the context and params needed to run a Scheduler
type Options struct {
	// ConfigFile is the location of the scheduler server's configuration file.
	ConfigFile string

	// config is the scheduler server's configuration object.
	config *componentconfig.KubeSchedulerConfiguration

	scheme *runtime.Scheme
	codecs serializer.CodecFactory

	// The fields below here are placeholders for flags that can't be directly
	// mapped into componentconfig.KubeSchedulerConfiguration.
	//
	// TODO remove these fields once the deprecated flags are removed.

	// master is the address of the Kubernetes API server (overrides any
	// value in kubeconfig).
	master                   string
	healthzAddress           string
	healthzPort              int32
	policyConfigFile         string
	policyConfigMapName      string
	policyConfigMapNamespace string
	useLegacyPolicyConfig    bool
	algorithmProvider        string
}

// AddFlags adds flags for a specific SchedulerServer to the specified FlagSet
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file.")

	// All flags below here are deprecated and will eventually be removed.

	fs.Int32Var(&o.healthzPort, "port", ports.SchedulerPort, "The port that the scheduler's http service runs on")
	fs.StringVar(&o.healthzAddress, "address", o.healthzAddress, "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&o.algorithmProvider, "algorithm-provider", o.algorithmProvider, "The scheduling algorithm provider to use, one of: "+factory.ListAlgorithmProviders())
	fs.StringVar(&o.policyConfigFile, "policy-config-file", o.policyConfigFile, "File with scheduler policy configuration. This file is used if policy ConfigMap is not provided or --use-legacy-policy-config==true")
	usage := fmt.Sprintf("Name of the ConfigMap object that contains scheduler's policy configuration. It must exist in the system namespace before scheduler initialization if --use-legacy-policy-config==false. The config must be provided as the value of an element in 'Data' map with the key='%v'", componentconfig.SchedulerPolicyConfigMapKey)
	fs.StringVar(&o.policyConfigMapName, "policy-configmap", o.policyConfigMapName, usage)
	fs.StringVar(&o.policyConfigMapNamespace, "policy-configmap-namespace", o.policyConfigMapNamespace, "The namespace where policy ConfigMap is located. The system namespace will be used if this is not provided or is empty.")
	fs.BoolVar(&o.useLegacyPolicyConfig, "use-legacy-policy-config", false, "When set to true, scheduler will ignore policy ConfigMap and uses policy config file")
	fs.BoolVar(&o.config.EnableProfiling, "profiling", o.config.EnableProfiling, "Enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&o.config.EnableContentionProfiling, "contention-profiling", o.config.EnableContentionProfiling, "Enable lock contention profiling, if profiling is enabled")
	fs.StringVar(&o.master, "master", o.master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&o.config.ClientConnection.KubeConfigFile, "kubeconfig", o.config.ClientConnection.KubeConfigFile, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&o.config.ClientConnection.ContentType, "kube-api-content-type", o.config.ClientConnection.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&o.config.ClientConnection.QPS, "kube-api-qps", o.config.ClientConnection.QPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&o.config.ClientConnection.Burst, "kube-api-burst", o.config.ClientConnection.Burst, "Burst to use while talking with kubernetes apiserver")
	fs.StringVar(&o.config.SchedulerName, "scheduler-name", o.config.SchedulerName, "Name of the scheduler, used to select which pods will be processed by this scheduler, based on pod's \"spec.SchedulerName\".")
	fs.StringVar(&o.config.LeaderElection.LockObjectNamespace, "lock-object-namespace", o.config.LeaderElection.LockObjectNamespace, "Define the namespace of the lock object.")
	fs.StringVar(&o.config.LeaderElection.LockObjectName, "lock-object-name", o.config.LeaderElection.LockObjectName, "Define the name of the lock object.")
	fs.Int32Var(&o.config.HardPodAffinitySymmetricWeight, "hard-pod-affinity-symmetric-weight", o.config.HardPodAffinitySymmetricWeight,
		"RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule corresponding "+
			"to every RequiredDuringScheduling affinity rule. --hard-pod-affinity-symmetric-weight represents the weight of implicit PreferredDuringScheduling affinity rule.")
	fs.MarkDeprecated("hard-pod-affinity-symmetric-weight", "This option was moved to the policy configuration file")
	fs.StringVar(&o.config.FailureDomains, "failure-domains", o.config.FailureDomains, "Indicate the \"all topologies\" set for an empty topologyKey when it's used for PreferredDuringScheduling pod anti-affinity.")
	fs.MarkDeprecated("failure-domains", "Doesn't have any effect. Will be removed in future version.")
	leaderelectionconfig.BindFlags(&o.config.LeaderElection.LeaderElectionConfiguration, fs)
	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

func NewOptions() (*Options, error) {
	o := &Options{
		config: new(componentconfig.KubeSchedulerConfiguration),
	}

	o.scheme = runtime.NewScheme()
	o.codecs = serializer.NewCodecFactory(o.scheme)

	if err := componentconfig.AddToScheme(o.scheme); err != nil {
		return nil, err
	}
	if err := componentconfigv1alpha1.AddToScheme(o.scheme); err != nil {
		return nil, err
	}

	return o, nil
}

func (o *Options) Complete() error {
	if len(o.ConfigFile) == 0 {
		glog.Warning("WARNING: all flags other than --config are deprecated. Please begin using a config file ASAP.")
		o.applyDeprecatedHealthzAddressToConfig()
		o.applyDeprecatedHealthzPortToConfig()
		o.applyDeprecatedAlgorithmSourceOptionsToConfig()
	}

	return nil
}

// applyDeprecatedHealthzAddressToConfig sets o.config.HealthzBindAddress and
// o.config.MetricsBindAddress from flags passed on the command line based on
// the following rules:
//
// 1. If --address is empty, leave the config as-is.
// 2. Otherwise, use the value of --address for the address portion of
//    o.config.HealthzBindAddress
func (o *Options) applyDeprecatedHealthzAddressToConfig() {
	if len(o.healthzAddress) == 0 {
		return
	}

	_, port, err := net.SplitHostPort(o.config.HealthzBindAddress)
	if err != nil {
		glog.Fatalf("invalid healthz bind address %q: %v", o.config.HealthzBindAddress, err)
	}
	o.config.HealthzBindAddress = net.JoinHostPort(o.healthzAddress, port)
	o.config.MetricsBindAddress = net.JoinHostPort(o.healthzAddress, port)
}

// applyDeprecatedHealthzPortToConfig sets o.config.HealthzBindAddress and
// o.config.MetricsBindAddress from flags passed on the command line based on
// the following rules:
//
// 1. If --port is -1, disable the healthz server.
// 2. Otherwise, use the value of --port for the port portion of
//    o.config.HealthzBindAddress
func (o *Options) applyDeprecatedHealthzPortToConfig() {
	if o.healthzPort == -1 {
		o.config.HealthzBindAddress = ""
		return
	}

	host, _, err := net.SplitHostPort(o.config.HealthzBindAddress)
	if err != nil {
		glog.Fatalf("invalid healthz bind address %q: %v", o.config.HealthzBindAddress, err)
	}
	o.config.HealthzBindAddress = net.JoinHostPort(host, strconv.Itoa(int(o.healthzPort)))
	o.config.MetricsBindAddress = net.JoinHostPort(host, strconv.Itoa(int(o.healthzPort)))
}

// applyDeprecatedAlgorithmSourceOptionsToConfig sets o.config.AlgorithmSource from
// flags passed on the command line in the following precedence order:
//
// 1. --use-legacy-policy-config to use a policy file.
// 2. --policy-configmap to use a policy config map value.
// 3. --algorithm-provider to use a named algorithm provider.
func (o *Options) applyDeprecatedAlgorithmSourceOptionsToConfig() {
	switch {
	case o.useLegacyPolicyConfig:
		o.config.AlgorithmSource = componentconfig.SchedulerAlgorithmSource{
			Policy: &componentconfig.SchedulerPolicySource{
				File: &componentconfig.SchedulerPolicyFileSource{
					Path: o.policyConfigFile,
				},
			},
		}
	case len(o.policyConfigMapName) > 0:
		o.config.AlgorithmSource = componentconfig.SchedulerAlgorithmSource{
			Policy: &componentconfig.SchedulerPolicySource{
				ConfigMap: &componentconfig.SchedulerPolicyConfigMapSource{
					Name:      o.policyConfigMapName,
					Namespace: o.policyConfigMapNamespace,
				},
			},
		}
	case len(o.algorithmProvider) > 0:
		o.config.AlgorithmSource = componentconfig.SchedulerAlgorithmSource{
			Provider: &o.algorithmProvider,
		}
	}
}

// Validate validates all the required options.
func (o *Options) Validate(args []string) error {
	if len(args) != 0 {
		return errors.New("no arguments are supported")
	}

	return nil
}

// loadConfigFromFile loads the contents of file and decodes it as a
// KubeSchedulerConfiguration object.
func (o *Options) loadConfigFromFile(file string) (*componentconfig.KubeSchedulerConfiguration, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return o.loadConfig(data)
}

// loadConfig decodes data as a KubeSchedulerConfiguration object.
func (o *Options) loadConfig(data []byte) (*componentconfig.KubeSchedulerConfiguration, error) {
	configObj, gvk, err := o.codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	config, ok := configObj.(*componentconfig.KubeSchedulerConfiguration)
	if !ok {
		return nil, fmt.Errorf("got unexpected config type: %v", gvk)
	}
	return config, nil
}

func (o *Options) ApplyDefaults(in *componentconfig.KubeSchedulerConfiguration) (*componentconfig.KubeSchedulerConfiguration, error) {
	external, err := o.scheme.ConvertToVersion(in, componentconfigv1alpha1.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	o.scheme.Default(external)

	internal, err := o.scheme.ConvertToVersion(external, componentconfig.SchemeGroupVersion)
	if err != nil {
		return nil, err
	}

	out := internal.(*componentconfig.KubeSchedulerConfiguration)

	return out, nil
}

func (o *Options) Run() error {
	config := o.config

	if len(o.ConfigFile) > 0 {
		if c, err := o.loadConfigFromFile(o.ConfigFile); err != nil {
			return err
		} else {
			config = c
		}
	}

	// Apply algorithms based on feature gates.
	// TODO: make configurable?
	algorithmprovider.ApplyFeatureGates()

	server, err := NewSchedulerServer(config, o.master)
	if err != nil {
		return err
	}

	stop := make(chan struct{})
	return server.Run(stop)
}

// NewSchedulerCommand creates a *cobra.Command object with default parameters
func NewSchedulerCommand() *cobra.Command {
	opts, err := NewOptions()
	if err != nil {
		glog.Fatalf("unable to initialize command options: %v", err)
	}

	cmd := &cobra.Command{
		Use: "kube-scheduler",
		Long: `The Kubernetes scheduler is a policy-rich, topology-aware,
workload-specific function that significantly impacts availability, performance,
and capacity. The scheduler needs to take into account individual and collective
resource requirements, quality of service requirements, hardware/software/policy
constraints, affinity and anti-affinity specifications, data locality, inter-workload
interference, deadlines, and so on. Workload-specific requirements will be exposed
through the API as necessary.`,
		Run: func(cmd *cobra.Command, args []string) {
			verflag.PrintAndExitIfRequested()
			cmdutil.CheckErr(opts.Complete())
			cmdutil.CheckErr(opts.Validate(args))
			cmdutil.CheckErr(opts.Run())
		},
	}

	opts.config, err = opts.ApplyDefaults(opts.config)
	if err != nil {
		glog.Fatalf("unable to apply config defaults: %v", err)
	}

	opts.AddFlags(pflag.CommandLine)

	cmd.MarkFlagFilename("config", "yaml", "yml", "json")

	return cmd
}

// SchedulerServer represents all the parameters required to start the
// Kubernetes scheduler server.
type SchedulerServer struct {
	SchedulerName                  string
	Client                         clientset.Interface
	InformerFactory                informers.SharedInformerFactory
	PodInformer                    coreinformers.PodInformer
	AlgorithmSource                componentconfig.SchedulerAlgorithmSource
	HardPodAffinitySymmetricWeight int32
	EventClient                    v1core.EventsGetter
	Recorder                       record.EventRecorder
	Broadcaster                    record.EventBroadcaster
	// LeaderElection is optional.
	LeaderElection *leaderelection.LeaderElectionConfig
	// HealthzServer is optional.
	HealthzServer *http.Server
	// MetricsServer is optional.
	MetricsServer *http.Server
}

// NewSchedulerServer creates a runnable SchedulerServer from configuration.
func NewSchedulerServer(config *componentconfig.KubeSchedulerConfiguration, master string) (*SchedulerServer, error) {
	if config == nil {
		return nil, errors.New("config is required")
	}

	// Configz registration.
	if c, err := configz.New("componentconfig"); err == nil {
		c.Set(config)
	} else {
		return nil, fmt.Errorf("unable to register configz: %s", err)
	}

	// Prepare some Kube clients.
	client, leaderElectionClient, eventClient, err := createClients(config.ClientConnection, master)
	if err != nil {
		return nil, err
	}

	// Prepare event clients.
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: config.SchedulerName})

	// Set up leader election if enabled.
	var leaderElectionConfig *leaderelection.LeaderElectionConfig
	if config.LeaderElection.LeaderElect {
		leaderElectionConfig, err = makeLeaderElectionConfig(config.LeaderElection, leaderElectionClient, recorder)
		if err != nil {
			return nil, err
		}
	}

	// Prepare a healthz server. If the metrics bind address is the same as the
	// healthz bind address, consolidate the servers into one.
	var healthzServer *http.Server
	if len(config.HealthzBindAddress) != 0 {
		healthzServer = makeHealthzServer(config)
	}

	// Prepare a separate metrics server only if the bind address differs from the
	// healthz bind address.
	var metricsServer *http.Server
	if len(config.MetricsBindAddress) > 0 && config.HealthzBindAddress != config.MetricsBindAddress {
		metricsServer = makeMetricsServer(config)
	}

	return &SchedulerServer{
		SchedulerName:                  config.SchedulerName,
		Client:                         client,
		InformerFactory:                informers.NewSharedInformerFactory(client, 0),
		PodInformer:                    factory.NewPodInformer(client, 0, config.SchedulerName),
		AlgorithmSource:                config.AlgorithmSource,
		HardPodAffinitySymmetricWeight: config.HardPodAffinitySymmetricWeight,
		EventClient:                    eventClient,
		Recorder:                       recorder,
		Broadcaster:                    eventBroadcaster,
		LeaderElection:                 leaderElectionConfig,
		HealthzServer:                  healthzServer,
		MetricsServer:                  metricsServer,
	}, nil
}

// makeLeaderElectionConfig builds a leader election configuration. It will
// create a new resource lock associated with the configuration.
func makeLeaderElectionConfig(config componentconfig.KubeSchedulerLeaderElectionConfiguration, client clientset.Interface, recorder record.EventRecorder) (*leaderelection.LeaderElectionConfig, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return nil, fmt.Errorf("unable to get hostname: %v", err)
	}

	rl, err := resourcelock.New(config.ResourceLock,
		config.LockObjectNamespace,
		config.LockObjectName,
		client.CoreV1(),
		resourcelock.ResourceLockConfig{
			Identity:      hostname,
			EventRecorder: recorder,
		})
	if err != nil {
		return nil, fmt.Errorf("couldn't create resource lock: %v", err)
	}

	return &leaderelection.LeaderElectionConfig{
		Lock:          rl,
		LeaseDuration: config.LeaseDuration.Duration,
		RenewDeadline: config.RenewDeadline.Duration,
		RetryPeriod:   config.RetryPeriod.Duration,
	}, nil
}

// makeHealthzServer creates a healthz server from the config, and will also
// embed the metrics handler if the healthz and metrics address configurations
// are the same.
func makeHealthzServer(config *componentconfig.KubeSchedulerConfiguration) *http.Server {
	mux := http.NewServeMux()
	healthz.InstallHandler(mux)
	if config.HealthzBindAddress == config.MetricsBindAddress {
		configz.InstallHandler(mux)
		mux.Handle("/metrics", prometheus.Handler())
	}
	if config.EnableProfiling {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		if config.EnableContentionProfiling {
			goruntime.SetBlockProfileRate(1)
		}
	}
	return &http.Server{
		Addr:    config.HealthzBindAddress,
		Handler: mux,
	}
}

// makeMetricsServer builds a metrics server from the config.
func makeMetricsServer(config *componentconfig.KubeSchedulerConfiguration) *http.Server {
	mux := http.NewServeMux()
	configz.InstallHandler(mux)
	mux.Handle("/metrics", prometheus.Handler())
	if config.EnableProfiling {
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		if config.EnableContentionProfiling {
			goruntime.SetBlockProfileRate(1)
		}
	}
	return &http.Server{
		Addr:    config.MetricsBindAddress,
		Handler: mux,
	}
}

// createClients creates a kube client and an event client from the given config and masterOverride.
// TODO remove masterOverride when CLI flags are removed.
func createClients(config componentconfig.ClientConnectionConfiguration, masterOverride string) (clientset.Interface, clientset.Interface, v1core.EventsGetter, error) {
	if len(config.KubeConfigFile) == 0 && len(masterOverride) == 0 {
		glog.Warningf("Neither --kubeconfig nor --master was specified. Using default API client. This might not work.")
	}

	// This creates a client, first loading any specified kubeconfig
	// file, and then overriding the Master flag, if non-empty.
	kubeConfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: config.KubeConfigFile},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: masterOverride}}).ClientConfig()
	if err != nil {
		return nil, nil, nil, err
	}

	kubeConfig.AcceptContentTypes = config.AcceptContentTypes
	kubeConfig.ContentType = config.ContentType
	kubeConfig.QPS = config.QPS
	//TODO make config struct use int instead of int32?
	kubeConfig.Burst = int(config.Burst)

	client, err := clientset.NewForConfig(restclient.AddUserAgent(kubeConfig, "scheduler"))
	if err != nil {
		return nil, nil, nil, err
	}

	leaderElectionClient, err := clientset.NewForConfig(restclient.AddUserAgent(kubeConfig, "leader-election"))
	if err != nil {
		return nil, nil, nil, err
	}

	eventClient, err := clientset.NewForConfig(kubeConfig)
	if err != nil {
		return nil, nil, nil, err
	}

	return client, leaderElectionClient, eventClient.CoreV1(), nil
}

// Run runs the SchedulerServer. This should never exit.
func (s *SchedulerServer) Run(stop chan struct{}) error {
	// To help debugging, immediately log version
	glog.Infof("Version: %+v", version.Get())

	// Build a scheduler config from the provided algorithm source.
	schedulerConfig, err := s.SchedulerConfig()
	if err != nil {
		return err
	}

	// Create the scheduler.
	sched := scheduler.NewFromConfig(schedulerConfig)

	// Prepare the event broadcaster.
	if !reflect.ValueOf(s.Broadcaster).IsNil() && !reflect.ValueOf(s.EventClient).IsNil() {
		s.Broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: s.EventClient.Events("")})
	}

	// Start up the healthz server.
	if s.HealthzServer != nil {
		go wait.Until(func() {
			glog.Infof("starting healthz server on %v", s.HealthzServer.Addr)
			err := s.HealthzServer.ListenAndServe()
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("failed to start healthz server: %v", err))
			}
		}, 5*time.Second, stop)
	}

	// Start up the metrics server.
	if s.MetricsServer != nil {
		go wait.Until(func() {
			glog.Infof("starting metrics server on %v", s.MetricsServer.Addr)
			err := s.MetricsServer.ListenAndServe()
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("failed to start metrics server: %v", err))
			}
		}, 5*time.Second, stop)
	}

	// Start all informers.
	go s.PodInformer.Informer().Run(stop)
	s.InformerFactory.Start(stop)

	// Wait for all caches to sync before scheduling.
	s.InformerFactory.WaitForCacheSync(stop)
	controller.WaitForCacheSync("scheduler", stop, s.PodInformer.Informer().HasSynced)

	// Prepare a reusable run function.
	run := func(stopCh <-chan struct{}) {
		sched.Run()
		<-stopCh
	}

	// If leader election is enabled, run via LeaderElector until done and exit.
	if s.LeaderElection != nil {
		s.LeaderElection.Callbacks = leaderelection.LeaderCallbacks{
			OnStartedLeading: run,
			OnStoppedLeading: func() {
				utilruntime.HandleError(fmt.Errorf("lost master"))
			},
		}
		leaderElector, err := leaderelection.NewLeaderElector(*s.LeaderElection)
		if err != nil {
			return fmt.Errorf("couldn't create leader elector: %v", err)
		}

		leaderElector.Run()

		return fmt.Errorf("lost lease")
	}

	// Leader election is disabled, so run inline until done.
	run(stop)
	return fmt.Errorf("finished without leader elect")
}

// SchedulerConfig creates the scheduler configuration. This is exposed for use
// by tests.
func (s *SchedulerServer) SchedulerConfig() (*scheduler.Config, error) {
	var storageClassInformer storageinformers.StorageClassInformer
	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		storageClassInformer = s.InformerFactory.Storage().V1().StorageClasses()
	}

	// Set up the configurator which can create schedulers from configs.
	configurator := factory.NewConfigFactory(
		s.SchedulerName,
		s.Client,
		s.InformerFactory.Core().V1().Nodes(),
		s.PodInformer,
		s.InformerFactory.Core().V1().PersistentVolumes(),
		s.InformerFactory.Core().V1().PersistentVolumeClaims(),
		s.InformerFactory.Core().V1().ReplicationControllers(),
		s.InformerFactory.Extensions().V1beta1().ReplicaSets(),
		s.InformerFactory.Apps().V1beta1().StatefulSets(),
		s.InformerFactory.Core().V1().Services(),
		s.InformerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		storageClassInformer,
		s.HardPodAffinitySymmetricWeight,
		utilfeature.DefaultFeatureGate.Enabled(features.EnableEquivalenceClassCache),
	)

	source := s.AlgorithmSource
	var config *scheduler.Config
	switch {
	case source.Provider != nil:
		// Create the config from a named algorithm provider.
		sc, err := configurator.CreateFromProvider(*source.Provider)
		if err != nil {
			return nil, fmt.Errorf("couldn't create scheduler using provider %q: %v", *source.Provider, err)
		}
		config = sc
	case source.Policy != nil:
		// Create the config from a user specified policy source.
		policy := &schedulerapi.Policy{}
		switch {
		case source.Policy.File != nil:
			// Use a policy serialized in a file.
			policyFile := source.Policy.File.Path
			_, err := os.Stat(policyFile)
			if err != nil {
				return nil, fmt.Errorf("missing policy config file %s", policyFile)
			}
			data, err := ioutil.ReadFile(policyFile)
			if err != nil {
				return nil, fmt.Errorf("couldn't read policy config: %v", err)
			}
			err = runtime.DecodeInto(latestschedulerapi.Codec, []byte(data), policy)
			if err != nil {
				return nil, fmt.Errorf("invalid policy: %v", err)
			}
		case source.Policy.ConfigMap != nil:
			// Use a policy serialized in a config map value.
			policyRef := source.Policy.ConfigMap
			policyConfigMap, err := s.Client.CoreV1().ConfigMaps(policyRef.Namespace).Get(policyRef.Name, metav1.GetOptions{})
			if err != nil {
				return nil, fmt.Errorf("couldn't get policy config map %s/%s: %v", policyRef.Namespace, policyRef.Name, err)
			}
			data, found := policyConfigMap.Data[componentconfig.SchedulerPolicyConfigMapKey]
			if !found {
				return nil, fmt.Errorf("missing policy config map value at key %q", componentconfig.SchedulerPolicyConfigMapKey)
			}
			err = runtime.DecodeInto(latestschedulerapi.Codec, []byte(data), policy)
			if err != nil {
				return nil, fmt.Errorf("invalid policy: %v", err)
			}
		}
		sc, err := configurator.CreateFromConfig(*policy)
		if err != nil {
			return nil, fmt.Errorf("couldn't create scheduler from policy: %v", err)
		}
		config = sc
	default:
		return nil, fmt.Errorf("unsupported algorithm source: %v", source)
	}
	// Additional tweaks to the config produced by the configurator.
	config.Recorder = s.Recorder
	return config, nil
}
