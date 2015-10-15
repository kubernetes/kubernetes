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
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/user"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	"github.com/kardianos/osext"
	"github.com/mesos/mesos-go/auth"
	"github.com/mesos/mesos-go/auth/sasl"
	"github.com/mesos/mesos-go/auth/sasl/mech"
	mesos "github.com/mesos/mesos-go/mesosproto"
	mutil "github.com/mesos/mesos-go/mesosutil"
	bindings "github.com/mesos/mesos-go/scheduler"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/spf13/pflag"
	"golang.org/x/net/context"

	"k8s.io/kubernetes/contrib/mesos/pkg/archive"
	"k8s.io/kubernetes/contrib/mesos/pkg/election"
	execcfg "k8s.io/kubernetes/contrib/mesos/pkg/executor/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	minioncfg "k8s.io/kubernetes/contrib/mesos/pkg/minion/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/profile"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	schedcfg "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/ha"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/metrics"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	mresource "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resource"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/uid"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	clientauth "k8s.io/kubernetes/pkg/client/unversioned/auth"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/master/ports"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
)

const (
	defaultMesosMaster       = "localhost:5050"
	defaultMesosUser         = "root" // should have privs to execute docker and iptables commands
	defaultReconcileInterval = 300    // 5m default task reconciliation interval
	defaultReconcileCooldown = 15 * time.Second
	defaultFrameworkName     = "Kubernetes"
	defaultExecutorCPUs      = mresource.CPUShares(0.25)  // initial CPU allocated for executor
	defaultExecutorMem       = mresource.MegaBytes(128.0) // initial memory allocated for executor
)

type SchedulerServer struct {
	Port                int
	Address             net.IP
	EnableProfiling     bool
	AuthPath            string
	APIServerList       []string
	EtcdServerList      []string
	EtcdConfigFile      string
	AllowPrivileged     bool
	ExecutorPath        string
	ProxyPath           string
	MesosMaster         string
	MesosUser           string
	MesosRole           string
	MesosAuthPrincipal  string
	MesosAuthSecretFile string
	MesosCgroupPrefix   string
	MesosExecutorCPUs   mresource.CPUShares
	MesosExecutorMem    mresource.MegaBytes
	Checkpoint          bool
	FailoverTimeout     float64

	ExecutorLogV           int
	ExecutorBindall        bool
	ExecutorSuicideTimeout time.Duration

	RunProxy     bool
	ProxyBindall bool
	ProxyLogV    int

	MinionPathOverride    string
	MinionLogMaxSize      resource.Quantity
	MinionLogMaxBackups   int
	MinionLogMaxAgeInDays int

	MesosAuthProvider             string
	DriverPort                    uint
	HostnameOverride              string
	ReconcileInterval             int64
	ReconcileCooldown             time.Duration
	DefaultContainerCPULimit      mresource.CPUShares
	DefaultContainerMemLimit      mresource.MegaBytes
	SchedulerConfigFileName       string
	Graceful                      bool
	FrameworkName                 string
	FrameworkWebURI               string
	HA                            bool
	AdvertisedAddress             string
	ServiceAddress                net.IP
	HADomain                      string
	KMPath                        string
	ClusterDNS                    net.IP
	ClusterDomain                 string
	KubeletRootDirectory          string
	KubeletDockerEndpoint         string
	KubeletPodInfraContainerImage string
	KubeletCadvisorPort           uint
	KubeletHostNetworkSources     string
	KubeletSyncFrequency          time.Duration
	KubeletNetworkPluginName      string
	StaticPodsConfigPath          string
	DockerCfgPath                 string
	ContainPodResources           bool
	AccountForPodResources        bool

	executable  string // path to the binary running this service
	client      *client.Client
	driver      bindings.SchedulerDriver
	driverMutex sync.RWMutex
	mux         *http.ServeMux
}

// useful for unit testing specific funcs
type schedulerProcessInterface interface {
	End() <-chan struct{}
	Failover() <-chan struct{}
	Terminal() <-chan struct{}
}

// NewSchedulerServer creates a new SchedulerServer with default parameters
func NewSchedulerServer() *SchedulerServer {
	s := SchedulerServer{
		Port:            ports.SchedulerPort,
		Address:         net.ParseIP("127.0.0.1"),
		FailoverTimeout: time.Duration((1 << 62) - 1).Seconds(),

		RunProxy:                 true,
		ExecutorSuicideTimeout:   execcfg.DefaultSuicideTimeout,
		DefaultContainerCPULimit: mresource.DefaultDefaultContainerCPULimit,
		DefaultContainerMemLimit: mresource.DefaultDefaultContainerMemLimit,

		MinionLogMaxSize:      minioncfg.DefaultLogMaxSize(),
		MinionLogMaxBackups:   minioncfg.DefaultLogMaxBackups,
		MinionLogMaxAgeInDays: minioncfg.DefaultLogMaxAgeInDays,

		MesosAuthProvider:      sasl.ProviderName,
		MesosCgroupPrefix:      minioncfg.DefaultCgroupPrefix,
		MesosMaster:            defaultMesosMaster,
		MesosUser:              defaultMesosUser,
		MesosExecutorCPUs:      defaultExecutorCPUs,
		MesosExecutorMem:       defaultExecutorMem,
		ReconcileInterval:      defaultReconcileInterval,
		ReconcileCooldown:      defaultReconcileCooldown,
		Checkpoint:             true,
		FrameworkName:          defaultFrameworkName,
		HA:                     false,
		mux:                    http.NewServeMux(),
		KubeletCadvisorPort:    4194, // copied from github.com/GoogleCloudPlatform/kubernetes/blob/release-0.14/cmd/kubelet/app/server.go
		KubeletSyncFrequency:   10 * time.Second,
		ContainPodResources:    true,
		AccountForPodResources: true,
	}
	// cache this for later use. also useful in case the original binary gets deleted, e.g.
	// during upgrades, development deployments, etc.
	if filename, err := osext.Executable(); err != nil {
		log.Fatalf("failed to determine path to currently running executable: %v", err)
	} else {
		s.executable = filename
		s.KMPath = filename
	}

	return &s
}

func (s *SchedulerServer) addCoreFlags(fs *pflag.FlagSet) {
	fs.IntVar(&s.Port, "port", s.Port, "The port that the scheduler's http service runs on")
	fs.IPVar(&s.Address, "address", s.Address, "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	fs.BoolVar(&s.EnableProfiling, "profiling", s.EnableProfiling, "Enable profiling via web interface host:port/debug/pprof/")
	fs.StringSliceVar(&s.APIServerList, "api-servers", s.APIServerList, "List of Kubernetes API servers for publishing events, and reading pods and services. (ip:port), comma separated.")
	fs.StringVar(&s.AuthPath, "auth-path", s.AuthPath, "Path to .kubernetes_auth file, specifying how to authenticate to API server.")
	fs.StringSliceVar(&s.EtcdServerList, "etcd-servers", s.EtcdServerList, "List of etcd servers to watch (http://ip:port), comma separated. Mutually exclusive with --etcd-config")
	fs.StringVar(&s.EtcdConfigFile, "etcd-config", s.EtcdConfigFile, "The config file for the etcd client. Mutually exclusive with --etcd-servers.")
	fs.BoolVar(&s.AllowPrivileged, "allow-privileged", s.AllowPrivileged, "If true, allow privileged containers.")
	fs.StringVar(&s.ClusterDomain, "cluster-domain", s.ClusterDomain, "Domain for this cluster.  If set, kubelet will configure all containers to search this domain in addition to the host's search domains")
	fs.IPVar(&s.ClusterDNS, "cluster-dns", s.ClusterDNS, "IP address for a cluster DNS server. If set, kubelet will configure all containers to use this for DNS resolution in addition to the host's DNS servers")
	fs.StringVar(&s.StaticPodsConfigPath, "static-pods-config", s.StaticPodsConfigPath, "Path for specification of static pods. Path should point to dir containing the staticPods configuration files. Defaults to none.")

	fs.StringVar(&s.MesosMaster, "mesos-master", s.MesosMaster, "Location of the Mesos master. The format is a comma-delimited list of of hosts like zk://host1:port,host2:port/mesos. If using ZooKeeper, pay particular attention to the leading zk:// and trailing /mesos! If not using ZooKeeper, standard URLs like http://localhost are also acceptable.")
	fs.StringVar(&s.MesosUser, "mesos-user", s.MesosUser, "Mesos user for this framework, defaults to root.")
	fs.StringVar(&s.MesosRole, "mesos-role", s.MesosRole, "Mesos role for this framework, defaults to none.")
	fs.StringVar(&s.MesosAuthPrincipal, "mesos-authentication-principal", s.MesosAuthPrincipal, "Mesos authentication principal.")
	fs.StringVar(&s.MesosAuthSecretFile, "mesos-authentication-secret-file", s.MesosAuthSecretFile, "Mesos authentication secret file.")
	fs.StringVar(&s.MesosAuthProvider, "mesos-authentication-provider", s.MesosAuthProvider, fmt.Sprintf("Authentication provider to use, default is SASL that supports mechanisms: %+v", mech.ListSupported()))
	fs.StringVar(&s.DockerCfgPath, "dockercfg-path", s.DockerCfgPath, "Path to a dockercfg file that will be used by the docker instance of the minions.")
	fs.StringVar(&s.MesosCgroupPrefix, "mesos-cgroup-prefix", s.MesosCgroupPrefix, "The cgroup prefix concatenated with MESOS_DIRECTORY must give the executor cgroup set by Mesos")
	fs.Var(&s.MesosExecutorCPUs, "mesos-executor-cpus", "Initial CPU shares to allocate for each Mesos executor container.")
	fs.Var(&s.MesosExecutorMem, "mesos-executor-mem", "Initial memory (MB) to allocate for each Mesos executor container.")
	fs.BoolVar(&s.Checkpoint, "checkpoint", s.Checkpoint, "Enable/disable checkpointing for the kubernetes-mesos framework.")
	fs.Float64Var(&s.FailoverTimeout, "failover-timeout", s.FailoverTimeout, fmt.Sprintf("Framework failover timeout, in sec."))
	fs.UintVar(&s.DriverPort, "driver-port", s.DriverPort, "Port that the Mesos scheduler driver process should listen on.")
	fs.StringVar(&s.HostnameOverride, "hostname-override", s.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	fs.Int64Var(&s.ReconcileInterval, "reconcile-interval", s.ReconcileInterval, "Interval at which to execute task reconciliation, in sec. Zero disables.")
	fs.DurationVar(&s.ReconcileCooldown, "reconcile-cooldown", s.ReconcileCooldown, "Minimum rest period between task reconciliation operations.")
	fs.StringVar(&s.SchedulerConfigFileName, "scheduler-config", s.SchedulerConfigFileName, "An ini-style configuration file with low-level scheduler settings.")
	fs.BoolVar(&s.Graceful, "graceful", s.Graceful, "Indicator of a graceful failover, intended for internal use only.")
	fs.BoolVar(&s.HA, "ha", s.HA, "Run the scheduler in high availability mode with leader election. All peers should be configured exactly the same.")
	fs.StringVar(&s.FrameworkName, "framework-name", s.FrameworkName, "The framework name to register with Mesos.")
	fs.StringVar(&s.FrameworkWebURI, "framework-weburi", s.FrameworkWebURI, "A URI that points to a web-based interface for interacting with the framework.")
	fs.StringVar(&s.AdvertisedAddress, "advertised-address", s.AdvertisedAddress, "host:port address that is advertised to clients. May be used to construct artifact download URIs.")
	fs.IPVar(&s.ServiceAddress, "service-address", s.ServiceAddress, "The service portal IP address that the scheduler should register with (if unset, chooses randomly)")
	fs.Var(&s.DefaultContainerCPULimit, "default-container-cpu-limit", "Containers without a CPU resource limit are admitted this much CPU shares")
	fs.Var(&s.DefaultContainerMemLimit, "default-container-mem-limit", "Containers without a memory resource limit are admitted this much amount of memory in MB")
	fs.BoolVar(&s.ContainPodResources, "contain-pod-resources", s.ContainPodResources, "Reparent pod containers into mesos cgroups; disable if you're having strange mesos/docker/systemd interactions.")
	fs.BoolVar(&s.AccountForPodResources, "account-for-pod-resources", s.AccountForPodResources, "Allocate pod CPU and memory resources from offers (Default: true)")

	fs.IntVar(&s.ExecutorLogV, "executor-logv", s.ExecutorLogV, "Logging verbosity of spawned minion and executor processes.")
	fs.BoolVar(&s.ExecutorBindall, "executor-bindall", s.ExecutorBindall, "When true will set -address of the executor to 0.0.0.0.")
	fs.DurationVar(&s.ExecutorSuicideTimeout, "executor-suicide-timeout", s.ExecutorSuicideTimeout, "Executor self-terminates after this period of inactivity. Zero disables suicide watch.")

	fs.BoolVar(&s.ProxyBindall, "proxy-bindall", s.ProxyBindall, "When true pass -proxy-bindall to the executor.")
	fs.BoolVar(&s.RunProxy, "run-proxy", s.RunProxy, "Run the kube-proxy as a side process of the executor.")
	fs.IntVar(&s.ProxyLogV, "proxy-logv", s.ProxyLogV, "Logging verbosity of spawned minion proxy processes.")

	fs.StringVar(&s.MinionPathOverride, "minion-path-override", s.MinionPathOverride, "Override the PATH in the environment of the minion sub-processes.")
	fs.Var(resource.NewQuantityFlagValue(&s.MinionLogMaxSize), "minion-max-log-size", "Maximum log file size for the executor and proxy before rotation")
	fs.IntVar(&s.MinionLogMaxAgeInDays, "minion-max-log-age", s.MinionLogMaxAgeInDays, "Maximum log file age of the executor and proxy in days")
	fs.IntVar(&s.MinionLogMaxBackups, "minion-max-log-backups", s.MinionLogMaxBackups, "Maximum log file backups of the executor and proxy to keep after rotation")

	fs.StringVar(&s.KubeletRootDirectory, "kubelet-root-dir", s.KubeletRootDirectory, "Directory path for managing kubelet files (volume mounts,etc). Defaults to executor sandbox.")
	fs.StringVar(&s.KubeletDockerEndpoint, "kubelet-docker-endpoint", s.KubeletDockerEndpoint, "If non-empty, kubelet will use this for the docker endpoint to communicate with.")
	fs.StringVar(&s.KubeletPodInfraContainerImage, "kubelet-pod-infra-container-image", s.KubeletPodInfraContainerImage, "The image whose network/ipc namespaces containers in each pod will use.")
	fs.UintVar(&s.KubeletCadvisorPort, "kubelet-cadvisor-port", s.KubeletCadvisorPort, "The port of the kubelet's local cAdvisor endpoint")
	fs.StringVar(&s.KubeletHostNetworkSources, "kubelet-host-network-sources", s.KubeletHostNetworkSources, "Comma-separated list of sources from which the Kubelet allows pods to use of host network. For all sources use \"*\" [default=\"file\"]")
	fs.DurationVar(&s.KubeletSyncFrequency, "kubelet-sync-frequency", s.KubeletSyncFrequency, "Max period between synchronizing running containers and config")
	fs.StringVar(&s.KubeletNetworkPluginName, "kubelet-network-plugin", s.KubeletNetworkPluginName, "<Warning: Alpha feature> The name of the network plugin to be invoked for various events in kubelet/pod lifecycle")

	//TODO(jdef) support this flag once we have a better handle on mesos-dns and k8s DNS integration
	//fs.StringVar(&s.HADomain, "ha-domain", s.HADomain, "Domain of the HA scheduler service, only used in HA mode. If specified may be used to construct artifact download URIs.")
}

func (s *SchedulerServer) AddStandaloneFlags(fs *pflag.FlagSet) {
	s.addCoreFlags(fs)
	fs.StringVar(&s.ExecutorPath, "executor-path", s.ExecutorPath, "Location of the kubernetes executor executable")
}

func (s *SchedulerServer) AddHyperkubeFlags(fs *pflag.FlagSet) {
	s.addCoreFlags(fs)
	fs.StringVar(&s.KMPath, "km-path", s.KMPath, "Location of the km executable, may be a URI or an absolute file path.")
}

// returns (downloadURI, basename(path))
func (s *SchedulerServer) serveFrameworkArtifact(path string) (string, string) {
	pathSplit := strings.Split(path, "/")

	var basename string
	if len(pathSplit) > 0 {
		basename = pathSplit[len(pathSplit)-1]
	} else {
		basename = path
	}

	return s.serveFrameworkArtifactWithFilename(path, basename), basename
}

// returns downloadURI
func (s *SchedulerServer) serveFrameworkArtifactWithFilename(path string, filename string) string {
	serveFile := func(pattern string, filepath string) {
		s.mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
			http.ServeFile(w, r, filepath)
		})
	}

	serveFile("/"+filename, path)

	hostURI := ""
	if s.AdvertisedAddress != "" {
		hostURI = fmt.Sprintf("http://%s/%s", s.AdvertisedAddress, filename)
	} else if s.HA && s.HADomain != "" {
		hostURI = fmt.Sprintf("http://%s.%s:%d/%s", SCHEDULER_SERVICE_NAME, s.HADomain, ports.SchedulerPort, filename)
	} else {
		hostURI = fmt.Sprintf("http://%s:%d/%s", s.Address.String(), s.Port, filename)
	}
	log.V(2).Infof("Hosting artifact '%s' at '%s'", filename, hostURI)

	return hostURI
}

func (s *SchedulerServer) prepareExecutorInfo(hks hyperkube.Interface) (*mesos.ExecutorInfo, *uid.UID, error) {
	ci := &mesos.CommandInfo{
		Shell: proto.Bool(false),
	}

	if s.ExecutorPath != "" {
		uri, executorCmd := s.serveFrameworkArtifact(s.ExecutorPath)
		ci.Uris = append(ci.Uris, &mesos.CommandInfo_URI{Value: proto.String(uri), Executable: proto.Bool(true)})
		ci.Value = proto.String(fmt.Sprintf("./%s", executorCmd))
	} else if !hks.FindServer(hyperkube.CommandMinion) {
		return nil, nil, fmt.Errorf("either run this scheduler via km or else --executor-path is required")
	} else {
		if strings.Index(s.KMPath, "://") > 0 {
			// URI could point directly to executable, e.g. hdfs:///km
			// or else indirectly, e.g. http://acmestorage/tarball.tgz
			// so we assume that for this case the command will always "km"
			ci.Uris = append(ci.Uris, &mesos.CommandInfo_URI{Value: proto.String(s.KMPath), Executable: proto.Bool(true)})
			ci.Value = proto.String("./km") // TODO(jdef) extract constant
		} else if s.KMPath != "" {
			uri, kmCmd := s.serveFrameworkArtifact(s.KMPath)
			ci.Uris = append(ci.Uris, &mesos.CommandInfo_URI{Value: proto.String(uri), Executable: proto.Bool(true)})
			ci.Value = proto.String(fmt.Sprintf("./%s", kmCmd))
		} else {
			uri, kmCmd := s.serveFrameworkArtifact(s.executable)
			ci.Uris = append(ci.Uris, &mesos.CommandInfo_URI{Value: proto.String(uri), Executable: proto.Bool(true)})
			ci.Value = proto.String(fmt.Sprintf("./%s", kmCmd))
		}
		ci.Arguments = append(ci.Arguments, hyperkube.CommandMinion)

		ci.Arguments = append(ci.Arguments, fmt.Sprintf("--run-proxy=%v", s.RunProxy))
		ci.Arguments = append(ci.Arguments, fmt.Sprintf("--proxy-bindall=%v", s.ProxyBindall))
		ci.Arguments = append(ci.Arguments, fmt.Sprintf("--proxy-logv=%d", s.ProxyLogV))

		ci.Arguments = append(ci.Arguments, fmt.Sprintf("--path-override=%s", s.MinionPathOverride))
		ci.Arguments = append(ci.Arguments, fmt.Sprintf("--max-log-size=%v", s.MinionLogMaxSize.String()))
		ci.Arguments = append(ci.Arguments, fmt.Sprintf("--max-log-backups=%d", s.MinionLogMaxBackups))
		ci.Arguments = append(ci.Arguments, fmt.Sprintf("--max-log-age=%d", s.MinionLogMaxAgeInDays))
	}

	if s.DockerCfgPath != "" {
		uri := s.serveFrameworkArtifactWithFilename(s.DockerCfgPath, ".dockercfg")
		ci.Uris = append(ci.Uris, &mesos.CommandInfo_URI{Value: proto.String(uri), Executable: proto.Bool(false), Extract: proto.Bool(false)})
	}

	//TODO(jdef): provide some way (env var?) for users to customize executor config
	//TODO(jdef): set -address to 127.0.0.1 if `address` is 127.0.0.1

	apiServerArgs := strings.Join(s.APIServerList, ",")
	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--api-servers=%s", apiServerArgs))
	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--v=%d", s.ExecutorLogV)) // this also applies to the minion
	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--allow-privileged=%t", s.AllowPrivileged))
	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--suicide-timeout=%v", s.ExecutorSuicideTimeout))

	if s.ExecutorBindall {
		//TODO(jdef) determine whether hostname-override is really needed for bindall because
		//it conflicts with kubelet node status checks/updates
		//ci.Arguments = append(ci.Arguments, "--hostname-override=0.0.0.0")
		ci.Arguments = append(ci.Arguments, "--address=0.0.0.0")
	}

	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--mesos-cgroup-prefix=%v", s.MesosCgroupPrefix))
	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--cadvisor-port=%v", s.KubeletCadvisorPort))
	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--sync-frequency=%v", s.KubeletSyncFrequency))
	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--contain-pod-resources=%t", s.ContainPodResources))
	ci.Arguments = append(ci.Arguments, fmt.Sprintf("--enable-debugging-handlers=%t", s.EnableProfiling))

	if s.AuthPath != "" {
		//TODO(jdef) should probably support non-local files, e.g. hdfs:///some/config/file
		uri, basename := s.serveFrameworkArtifact(s.AuthPath)
		ci.Uris = append(ci.Uris, &mesos.CommandInfo_URI{Value: proto.String(uri)})
		ci.Arguments = append(ci.Arguments, fmt.Sprintf("--auth-path=%s", basename))
	}
	appendOptional := func(name string, value string) {
		if value != "" {
			ci.Arguments = append(ci.Arguments, fmt.Sprintf("--%s=%s", name, value))
		}
	}
	if s.ClusterDNS != nil {
		appendOptional("cluster-dns", s.ClusterDNS.String())
	}
	appendOptional("cluster-domain", s.ClusterDomain)
	appendOptional("root-dir", s.KubeletRootDirectory)
	appendOptional("docker-endpoint", s.KubeletDockerEndpoint)
	appendOptional("pod-infra-container-image", s.KubeletPodInfraContainerImage)
	appendOptional("host-network-sources", s.KubeletHostNetworkSources)
	appendOptional("network-plugin", s.KubeletNetworkPluginName)

	log.V(1).Infof("prepared executor command %q with args '%+v'", ci.GetValue(), ci.Arguments)

	// Create mesos scheduler driver.
	execInfo := &mesos.ExecutorInfo{
		Command: ci,
		Name:    proto.String(execcfg.DefaultInfoName),
		Source:  proto.String(execcfg.DefaultInfoSource),
	}

	// Check for staticPods
	var staticPodCPUs, staticPodMem float64
	if s.StaticPodsConfigPath != "" {
		bs, paths, err := archive.ZipDir(s.StaticPodsConfigPath)
		if err != nil {
			return nil, nil, err
		}

		// try to read pod files and sum resources
		// TODO(sttts): don't terminate when static pods are broken, but skip them
		// TODO(sttts): add a directory watch and tell running executors about updates
		for _, podPath := range paths {
			podJson, err := ioutil.ReadFile(podPath)
			if err != nil {
				return nil, nil, fmt.Errorf("error reading static pod spec: %v", err)
			}

			pod := api.Pod{}
			err = json.Unmarshal(podJson, &pod)
			if err != nil {
				return nil, nil, fmt.Errorf("error parsing static pod spec at %v: %v", podPath, err)
			}

			// TODO(sttts): allow unlimited static pods as well and patch in the default resource limits
			unlimitedCPU := mresource.LimitPodCPU(&pod, s.DefaultContainerCPULimit)
			unlimitedMem := mresource.LimitPodMem(&pod, s.DefaultContainerMemLimit)
			if unlimitedCPU {
				return nil, nil, fmt.Errorf("found static pod without limit on cpu resources: %v", podPath)
			}
			if unlimitedMem {
				return nil, nil, fmt.Errorf("found static pod without limit on memory resources: %v", podPath)
			}

			cpu := mresource.PodCPULimit(&pod)
			mem := mresource.PodMemLimit(&pod)
			log.V(2).Infof("reserving %.2f cpu shares and %.2f MB of memory to static pod %s", cpu, mem, pod.Name)

			staticPodCPUs += float64(cpu)
			staticPodMem += float64(mem)
		}

		// pass zipped pod spec to executor
		execInfo.Data = bs
	}

	execInfo.Resources = []*mesos.Resource{
		mutil.NewScalarResource("cpus", float64(s.MesosExecutorCPUs)+staticPodCPUs),
		mutil.NewScalarResource("mem", float64(s.MesosExecutorMem)+staticPodMem),
	}

	// calculate ExecutorInfo hash to be used for validating compatibility
	// of ExecutorInfo's generated by other HA schedulers.
	ehash := hashExecutorInfo(execInfo)
	eid := uid.New(ehash, execcfg.DefaultInfoID)
	execInfo.ExecutorId = &mesos.ExecutorID{Value: proto.String(eid.String())}

	return execInfo, eid, nil
}

// TODO(jdef): hacked from kubelet/server/server.go
// TODO(k8s): replace this with clientcmd
func (s *SchedulerServer) createAPIServerClient() (*client.Client, error) {
	authInfo, err := clientauth.LoadFromFile(s.AuthPath)
	if err != nil {
		log.Warningf("Could not load kubernetes auth path: %v. Continuing with defaults.", err)
	}
	if authInfo == nil {
		// authInfo didn't load correctly - continue with defaults.
		authInfo = &clientauth.Info{}
	}
	clientConfig, err := authInfo.MergeWithConfig(client.Config{})
	if err != nil {
		return nil, err
	}
	if len(s.APIServerList) < 1 {
		return nil, fmt.Errorf("no api servers specified")
	}
	// TODO: adapt Kube client to support LB over several servers
	if len(s.APIServerList) > 1 {
		log.Infof("Multiple api servers specified.  Picking first one")
	}
	clientConfig.Host = s.APIServerList[0]
	c, err := client.New(&clientConfig)
	if err != nil {
		return nil, err
	}
	return c, nil
}

func (s *SchedulerServer) setDriver(driver bindings.SchedulerDriver) {
	s.driverMutex.Lock()
	defer s.driverMutex.Unlock()
	s.driver = driver
}

func (s *SchedulerServer) getDriver() (driver bindings.SchedulerDriver) {
	s.driverMutex.RLock()
	defer s.driverMutex.RUnlock()
	return s.driver
}

func (s *SchedulerServer) Run(hks hyperkube.Interface, _ []string) error {
	// get scheduler low-level config
	sc := schedcfg.CreateDefaultConfig()
	if s.SchedulerConfigFileName != "" {
		f, err := os.Open(s.SchedulerConfigFileName)
		if err != nil {
			log.Fatalf("Cannot open scheduler config file: %v", err)
		}

		err = sc.Read(bufio.NewReader(f))
		if err != nil {
			log.Fatalf("Invalid scheduler config file: %v", err)
		}
	}

	schedulerProcess, driverFactory, etcdClient, eid := s.bootstrap(hks, sc)

	if s.EnableProfiling {
		profile.InstallHandler(s.mux)
	}
	go runtime.Until(func() {
		log.V(1).Info("Starting HTTP interface")
		log.Error(http.ListenAndServe(net.JoinHostPort(s.Address.String(), strconv.Itoa(s.Port)), s.mux))
	}, sc.HttpBindInterval.Duration, schedulerProcess.Terminal())

	if s.HA {
		validation := ha.ValidationFunc(validateLeadershipTransition)
		srv := ha.NewCandidate(schedulerProcess, driverFactory, validation)
		path := fmt.Sprintf(meta.DefaultElectionFormat, s.FrameworkName)
		sid := uid.New(eid.Group(), "").String()
		log.Infof("registering for election at %v with id %v", path, sid)
		go election.Notify(election.NewEtcdMasterElector(etcdClient), path, sid, srv, nil)
	} else {
		log.Infoln("self-electing in non-HA mode")
		schedulerProcess.Elect(driverFactory)
	}
	return s.awaitFailover(schedulerProcess, func() error { return s.failover(s.getDriver(), hks) })
}

// watch the scheduler process for failover signals and properly handle such. may never return.
func (s *SchedulerServer) awaitFailover(schedulerProcess schedulerProcessInterface, handler func() error) error {

	// we only want to return the first error (if any), everyone else can block forever
	errCh := make(chan error, 1)
	doFailover := func() error {
		// we really don't expect handler to return, if it does something went seriously wrong
		err := handler()
		if err != nil {
			defer schedulerProcess.End()
			err = fmt.Errorf("failover failed, scheduler will terminate: %v", err)
		}
		return err
	}

	// guard for failover signal processing, first signal processor wins
	failoverLatch := &runtime.Latch{}
	runtime.On(schedulerProcess.Terminal(), func() {
		if !failoverLatch.Acquire() {
			log.V(1).Infof("scheduler process ending, already failing over")
			select {}
		}
		var err error
		defer func() { errCh <- err }()
		select {
		case <-schedulerProcess.Failover():
			err = doFailover()
		default:
			if s.HA {
				err = fmt.Errorf("ha scheduler exiting instead of failing over")
			} else {
				log.Infof("exiting scheduler")
			}
		}
	})
	runtime.OnOSSignal(makeFailoverSigChan(), func(_ os.Signal) {
		if !failoverLatch.Acquire() {
			log.V(1).Infof("scheduler process signalled, already failing over")
			select {}
		}
		errCh <- doFailover()
	})
	return <-errCh
}

func validateLeadershipTransition(desired, current string) {
	log.Infof("validating leadership transition")
	d := uid.Parse(desired).Group()
	c := uid.Parse(current).Group()
	if d == 0 {
		// should *never* happen, but..
		log.Fatalf("illegal scheduler UID: %q", desired)
	}
	if d != c && c != 0 {
		log.Fatalf("desired scheduler group (%x) != current scheduler group (%x)", d, c)
	}
}

// hacked from https://github.com/GoogleCloudPlatform/kubernetes/blob/release-0.14/cmd/kube-apiserver/app/server.go
func newEtcd(etcdConfigFile string, etcdServerList []string) (client tools.EtcdClient, err error) {
	if etcdConfigFile != "" {
		client, err = etcd.NewClientFromFile(etcdConfigFile)
	} else {
		client = etcd.NewClient(etcdServerList)
	}
	return
}

func (s *SchedulerServer) bootstrap(hks hyperkube.Interface, sc *schedcfg.Config) (*ha.SchedulerProcess, ha.DriverFactory, tools.EtcdClient, *uid.UID) {

	s.FrameworkName = strings.TrimSpace(s.FrameworkName)
	if s.FrameworkName == "" {
		log.Fatalf("framework-name must be a non-empty string")
	}
	s.FrameworkWebURI = strings.TrimSpace(s.FrameworkWebURI)

	metrics.Register()
	runtime.Register()
	s.mux.Handle("/metrics", prometheus.Handler())
	healthz.InstallHandler(s.mux)

	if (s.EtcdConfigFile != "" && len(s.EtcdServerList) != 0) || (s.EtcdConfigFile == "" && len(s.EtcdServerList) == 0) {
		log.Fatalf("specify either --etcd-servers or --etcd-config")
	}

	if len(s.APIServerList) < 1 {
		log.Fatal("No api servers specified.")
	}

	client, err := s.createAPIServerClient()
	if err != nil {
		log.Fatalf("Unable to make apiserver client: %v", err)
	}
	s.client = client

	if s.ReconcileCooldown < defaultReconcileCooldown {
		s.ReconcileCooldown = defaultReconcileCooldown
		log.Warningf("user-specified reconcile cooldown too small, defaulting to %v", s.ReconcileCooldown)
	}

	executor, eid, err := s.prepareExecutorInfo(hks)
	if err != nil {
		log.Fatalf("misconfigured executor: %v", err)
	}

	// TODO(jdef): remove the dependency on etcd as soon as
	// (1) the generic config store is available for the FrameworkId storage
	// (2) the generic master election is provided by the apiserver
	// Compare docs/proposals/high-availability.md
	etcdClient, err := newEtcd(s.EtcdConfigFile, s.EtcdServerList)
	if err != nil {
		log.Fatalf("misconfigured etcd: %v", err)
	}

	as := scheduler.NewAllocationStrategy(
		podtask.DefaultPredicate,
		podtask.NewDefaultProcurement(s.DefaultContainerCPULimit, s.DefaultContainerMemLimit))

	// downgrade allocation strategy if user disables "account-for-pod-resources"
	if !s.AccountForPodResources {
		as = scheduler.NewAllocationStrategy(
			podtask.DefaultMinimalPredicate,
			podtask.DefaultMinimalProcurement)
	}

	fcfs := scheduler.NewFCFSPodScheduler(as)
	mesosPodScheduler := scheduler.New(scheduler.Config{
		Schedcfg:          *sc,
		Executor:          executor,
		Scheduler:         fcfs,
		Client:            client,
		EtcdClient:        etcdClient,
		FailoverTimeout:   s.FailoverTimeout,
		ReconcileInterval: s.ReconcileInterval,
		ReconcileCooldown: s.ReconcileCooldown,
	})

	masterUri := s.MesosMaster
	info, cred, err := s.buildFrameworkInfo()
	if err != nil {
		log.Fatalf("Misconfigured mesos framework: %v", err)
	}

	schedulerProcess := ha.New(mesosPodScheduler)
	dconfig := &bindings.DriverConfig{
		Scheduler:        schedulerProcess,
		Framework:        info,
		Master:           masterUri,
		Credential:       cred,
		BindingAddress:   s.Address,
		BindingPort:      uint16(s.DriverPort),
		HostnameOverride: s.HostnameOverride,
		WithAuthContext: func(ctx context.Context) context.Context {
			ctx = auth.WithLoginProvider(ctx, s.MesosAuthProvider)
			ctx = sasl.WithBindingAddress(ctx, s.Address)
			return ctx
		},
	}

	kpl := scheduler.NewPlugin(mesosPodScheduler.NewDefaultPluginConfig(schedulerProcess.Terminal(), s.mux))
	runtime.On(mesosPodScheduler.Registration(), func() { kpl.Run(schedulerProcess.Terminal()) })
	runtime.On(mesosPodScheduler.Registration(), s.newServiceWriter(schedulerProcess.Terminal()))

	driverFactory := ha.DriverFactory(func() (drv bindings.SchedulerDriver, err error) {
		log.V(1).Infoln("performing deferred initialization")
		if err = mesosPodScheduler.Init(schedulerProcess.Master(), kpl, s.mux); err != nil {
			return nil, fmt.Errorf("failed to initialize pod scheduler: %v", err)
		}
		log.V(1).Infoln("deferred init complete")
		// defer obtaining framework ID to prevent multiple schedulers
		// from overwriting each other's framework IDs
		dconfig.Framework.Id, err = s.fetchFrameworkID(etcdClient)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch framework ID from etcd: %v", err)
		}
		log.V(1).Infoln("constructing mesos scheduler driver")
		drv, err = bindings.NewMesosSchedulerDriver(*dconfig)
		if err != nil {
			return nil, fmt.Errorf("failed to construct scheduler driver: %v", err)
		}
		log.V(1).Infoln("constructed mesos scheduler driver:", drv)
		s.setDriver(drv)
		return drv, nil
	})

	return schedulerProcess, driverFactory, etcdClient, eid
}

func (s *SchedulerServer) failover(driver bindings.SchedulerDriver, hks hyperkube.Interface) error {
	if driver != nil {
		stat, err := driver.Stop(true)
		if stat != mesos.Status_DRIVER_STOPPED {
			return fmt.Errorf("failed to stop driver for failover, received unexpected status code: %v", stat)
		} else if err != nil {
			return err
		}
	}

	// there's no guarantee that all goroutines are actually programmed intelligently with 'done'
	// signals, so we'll need to restart if we want to really stop everything

	// run the same command that we were launched with
	//TODO(jdef) assumption here is that the sheduler is the only service running in this process, we should probably validate that somehow
	args := []string{}
	flags := pflag.CommandLine
	if hks != nil {
		args = append(args, hks.Name())
		flags = hks.Flags()
	}
	flags.Visit(func(flag *pflag.Flag) {
		if flag.Name != "api-servers" && flag.Name != "etcd-servers" {
			args = append(args, fmt.Sprintf("--%s=%s", flag.Name, flag.Value.String()))
		}
	})
	if !s.Graceful {
		args = append(args, "--graceful")
	}
	if len(s.APIServerList) > 0 {
		args = append(args, "--api-servers="+strings.Join(s.APIServerList, ","))
	}
	if len(s.EtcdServerList) > 0 {
		args = append(args, "--etcd-servers="+strings.Join(s.EtcdServerList, ","))
	}
	args = append(args, flags.Args()...)

	log.V(1).Infof("spawning scheduler for graceful failover: %s %+v", s.executable, args)

	cmd := exec.Command(s.executable, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = makeDisownedProcAttr()

	// TODO(jdef) pass in a pipe FD so that we can block, waiting for the child proc to be ready
	//cmd.ExtraFiles = []*os.File{}

	exitcode := 0
	log.Flush() // TODO(jdef) it would be really nice to ensure that no one else in our process was still logging
	if err := cmd.Start(); err != nil {
		//log to stdtout here to avoid conflicts with normal stderr logging
		fmt.Fprintf(os.Stdout, "failed to spawn failover process: %v\n", err)
		os.Exit(1)
	}
	os.Exit(exitcode)
	select {} // will never reach here
}

func (s *SchedulerServer) buildFrameworkInfo() (info *mesos.FrameworkInfo, cred *mesos.Credential, err error) {
	username, err := s.getUsername()
	if err != nil {
		return nil, nil, err
	}
	log.V(2).Infof("Framework configured with mesos user %v", username)
	info = &mesos.FrameworkInfo{
		Name:       proto.String(s.FrameworkName),
		User:       proto.String(username),
		Checkpoint: proto.Bool(s.Checkpoint),
	}
	if s.FrameworkWebURI != "" {
		info.WebuiUrl = proto.String(s.FrameworkWebURI)
	}
	if s.FailoverTimeout > 0 {
		info.FailoverTimeout = proto.Float64(s.FailoverTimeout)
	}
	if s.MesosRole != "" {
		info.Role = proto.String(s.MesosRole)
	}
	if s.MesosAuthPrincipal != "" {
		info.Principal = proto.String(s.MesosAuthPrincipal)
		if s.MesosAuthSecretFile == "" {
			return nil, nil, errors.New("authentication principal specified without the required credentials file")
		}
		secret, err := ioutil.ReadFile(s.MesosAuthSecretFile)
		if err != nil {
			return nil, nil, err
		}
		cred = &mesos.Credential{
			Principal: proto.String(s.MesosAuthPrincipal),
			Secret:    secret,
		}
	}
	return
}

func (s *SchedulerServer) fetchFrameworkID(client tools.EtcdClient) (*mesos.FrameworkID, error) {
	if s.FailoverTimeout > 0 {
		if response, err := client.Get(meta.FrameworkIDKey, false, false); err != nil {
			if !etcdstorage.IsEtcdNotFound(err) {
				return nil, fmt.Errorf("unexpected failure attempting to load framework ID from etcd: %v", err)
			}
			log.V(1).Infof("did not find framework ID in etcd")
		} else if response.Node.Value != "" {
			log.Infof("configuring FrameworkInfo with Id found in etcd: '%s'", response.Node.Value)
			return mutil.NewFrameworkID(response.Node.Value), nil
		}
	} else {
		//TODO(jdef) this seems like a totally hackish way to clean up the framework ID
		if _, err := client.Delete(meta.FrameworkIDKey, true); err != nil {
			if !etcdstorage.IsEtcdNotFound(err) {
				return nil, fmt.Errorf("failed to delete framework ID from etcd: %v", err)
			}
			log.V(1).Infof("nothing to delete: did not find framework ID in etcd")
		}
	}
	return nil, nil
}

func (s *SchedulerServer) getUsername() (username string, err error) {
	username = s.MesosUser
	if username == "" {
		if u, err := user.Current(); err == nil {
			username = u.Username
			if username == "" {
				username = defaultMesosUser
			}
		}
	}
	return
}
