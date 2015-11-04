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

package minion

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/signal"
	"path"
	"strings"
	"syscall"

	exservice "k8s.io/kubernetes/contrib/mesos/pkg/executor/service"
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	"k8s.io/kubernetes/contrib/mesos/pkg/minion/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/minion/tasks"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	log "github.com/golang/glog"
	"github.com/kardianos/osext"
	"github.com/spf13/pflag"
	"gopkg.in/natefinch/lumberjack.v2"
)

const (
	proxyLogFilename    = "proxy.log"
	executorLogFilename = "executor.log"
)

type MinionServer struct {
	// embed the executor server to be able to use its flags
	// TODO(sttts): get rid of this mixing of the minion and the executor server with a multiflags implementation for km
	KubeletExecutorServer *exservice.KubeletExecutorServer

	privateMountNS bool
	hks            hyperkube.Interface
	clientConfig   *client.Config
	kmBinary       string
	tasks          []*tasks.Task

	pathOverride        string // the PATH environment for the sub-processes
	cgroupPrefix        string // e.g. mesos
	cgroupRoot          string // the cgroupRoot that we pass to the kubelet-executor, depends on containPodResources
	mesosCgroup         string // discovered mesos cgroup root, e.g. /mesos/{container-id}
	containPodResources bool

	logMaxSize      resource.Quantity
	logMaxBackups   int
	logMaxAgeInDays int
	logVerbosity    int32 // see glog.Level

	runProxy     bool
	proxyLogV    int
	proxyBindall bool
}

// NewMinionServer creates the MinionServer struct with default values to be used by hyperkube
func NewMinionServer() *MinionServer {
	s := &MinionServer{
		KubeletExecutorServer: exservice.NewKubeletExecutorServer(),
		privateMountNS:        false, // disabled until Docker supports customization of the parent mount namespace
		cgroupPrefix:          config.DefaultCgroupPrefix,
		containPodResources:   true,
		logMaxSize:            config.DefaultLogMaxSize(),
		logMaxBackups:         config.DefaultLogMaxBackups,
		logMaxAgeInDays:       config.DefaultLogMaxAgeInDays,
		runProxy:              true,
	}

	// cache this for later use
	binary, err := osext.Executable()
	if err != nil {
		log.Fatalf("failed to determine currently running executable: %v", err)
	}
	s.kmBinary = binary

	return s
}

// filterArgsByFlagSet returns a list of args which are parsed by the given flag set
// and another list with those which do not match
func filterArgsByFlagSet(args []string, flags *pflag.FlagSet) ([]string, []string) {
	matched := []string{}
	notMatched := []string{}
	for _, arg := range args {
		err := flags.Parse([]string{arg})
		if err != nil {
			notMatched = append(notMatched, arg)
		} else {
			matched = append(matched, arg)
		}
	}
	return matched, notMatched
}

func findMesosCgroup(prefix string) string {
	// derive our cgroup from MESOS_DIRECTORY environment
	mesosDir := os.Getenv("MESOS_DIRECTORY")
	if mesosDir == "" {
		log.V(2).Infof("cannot derive executor's cgroup because MESOS_DIRECTORY is empty")
		return ""
	}

	containerId := path.Base(mesosDir)
	if containerId == "" {
		log.V(2).Infof("cannot derive executor's cgroup from MESOS_DIRECTORY=%q", mesosDir)
		return ""
	}
	return path.Join("/", prefix, containerId)
}

func (ms *MinionServer) launchProxyServer() {
	bindAddress := "0.0.0.0"
	if !ms.proxyBindall {
		bindAddress = ms.KubeletExecutorServer.Address.String()
	}
	args := []string{
		fmt.Sprintf("--bind-address=%s", bindAddress),
		fmt.Sprintf("--v=%d", ms.proxyLogV),
		"--logtostderr=true",
		"--resource-container=" + path.Join("/", ms.mesosCgroup, "kube-proxy"),
	}

	if ms.clientConfig.Host != "" {
		args = append(args, fmt.Sprintf("--master=%s", ms.clientConfig.Host))
	}
	if ms.KubeletExecutorServer.HostnameOverride != "" {
		args = append(args, fmt.Sprintf("--hostname-override=%s", ms.KubeletExecutorServer.HostnameOverride))
	}

	ms.launchHyperkubeServer(hyperkube.CommandProxy, args, proxyLogFilename, nil)
}

// launchExecutorServer returns a chan that closes upon kubelet-executor death. since the kubelet-
// executor doesn't support failover right now, the right thing to do is to fail completely since all
// pods will be lost upon restart and we want mesos to recover the resources from them.
func (ms *MinionServer) launchExecutorServer() <-chan struct{} {
	allArgs := os.Args[1:]

	// filter out minion flags, leaving those for the executor
	executorFlags := pflag.NewFlagSet("executor", pflag.ContinueOnError)
	executorFlags.SetOutput(ioutil.Discard)
	ms.AddExecutorFlags(executorFlags)
	executorArgs, _ := filterArgsByFlagSet(allArgs, executorFlags)

	executorArgs = append(executorArgs, "--resource-container="+path.Join("/", ms.mesosCgroup, "kubelet"))
	if ms.cgroupRoot != "" {
		executorArgs = append(executorArgs, "--cgroup-root="+ms.cgroupRoot)
	}

	// run executor and quit minion server when this exits cleanly
	execDied := make(chan struct{})
	decorator := func(t *tasks.Task) *tasks.Task {
		t.Finished = func(_ bool) bool {
			// this func implements the task.finished spec, so when the executor exits
			// we return false to indicate that it should not be restarted. we also
			// close execDied to signal interested listeners.
			close(execDied)
			return false
		}
		// since we only expect to die once, and there is no restart; don't delay any longer than needed
		t.RestartDelay = 0
		return t
	}
	ms.launchHyperkubeServer(hyperkube.CommandExecutor, executorArgs, executorLogFilename, decorator)
	return execDied
}

func (ms *MinionServer) launchHyperkubeServer(server string, args []string, logFileName string, decorator func(*tasks.Task) *tasks.Task) {
	log.V(2).Infof("Spawning hyperkube %v with args '%+v'", server, args)

	kmArgs := append([]string{server}, args...)
	maxSize := ms.logMaxSize.Value()
	if maxSize > 0 {
		// convert to MB
		maxSize = maxSize / 1024 / 1024
		if maxSize == 0 {
			log.Warning("maximal log file size is rounded to 1 MB")
			maxSize = 1
		}
	}

	writerFunc := func() io.WriteCloser {
		return &lumberjack.Logger{
			Filename:   logFileName,
			MaxSize:    int(maxSize),
			MaxBackups: ms.logMaxBackups,
			MaxAge:     ms.logMaxAgeInDays,
		}
	}

	// use given environment, but add /usr/sbin and $SANDBOX/bin to the path for the iptables binary used in kube-proxy
	var kmEnv []string
	env := os.Environ()
	kmEnv = make([]string, 0, len(env))
	for _, e := range env {
		if !strings.HasPrefix(e, "PATH=") {
			kmEnv = append(kmEnv, e)
		} else {
			if ms.pathOverride != "" {
				e = "PATH=" + ms.pathOverride
			}
			pwd, err := os.Getwd()
			if err != nil {
				panic(fmt.Errorf("Cannot get current directory: %v", err))
			}
			kmEnv = append(kmEnv, fmt.Sprintf("%s:%s", e, path.Join(pwd, "bin")))
		}
	}

	t := tasks.New(server, ms.kmBinary, kmArgs, kmEnv, writerFunc)
	if decorator != nil {
		t = decorator(t)
	}
	go t.Start()
	ms.tasks = append(ms.tasks, t)
}

// runs the main kubelet loop, closing the kubeletFinished chan when the loop exits.
// never returns.
func (ms *MinionServer) Run(hks hyperkube.Interface, _ []string) error {
	if ms.privateMountNS {
		// only the Linux version will do anything
		enterPrivateMountNamespace()
	}

	// create apiserver client
	clientConfig, err := ms.KubeletExecutorServer.CreateAPIServerClientConfig()
	if err != nil {
		// required for k8sm since we need to send api.Binding information
		// back to the apiserver
		log.Fatalf("No API client: %v", err)
	}
	ms.clientConfig = clientConfig

	// derive the executor cgroup and use it as:
	// - pod container cgroup root (e.g. docker cgroup-parent, optionally; see comments below)
	// - parent of kubelet container
	// - parent of kube-proxy container
	ms.mesosCgroup = findMesosCgroup(ms.cgroupPrefix)
	log.Infof("discovered mesos cgroup at %q", ms.mesosCgroup)

	// hack alert, this helps to work around systemd+docker+mesos integration problems
	// when docker's cgroup-parent flag is used (!containPodResources = don't use the docker flag)
	if ms.containPodResources {
		ms.cgroupRoot = ms.mesosCgroup
	}

	cgroupLogger := log.Infof
	if ms.cgroupRoot == "" {
		cgroupLogger = log.Warningf
	}

	cgroupLogger("using cgroup-root %q", ms.cgroupRoot)

	// run subprocesses until ms.done is closed on return of this function
	if ms.runProxy {
		ms.launchProxyServer()
	}

	// abort closes when the kubelet-executor dies
	abort := ms.launchExecutorServer()
	shouldQuit := termSignalListener(abort)
	te := tasks.MergeOutput(ms.tasks, shouldQuit)

	// TODO(jdef) do something fun here, such as reporting task completion to the apiserver

	<-te.Close().Done() // we don't listen for any specific events yet; wait for all tasks to finish
	return nil
}

// termSignalListener returns a signal chan that closes when either (a) the process receives a termination
// signal: SIGTERM, SIGINT, or SIGHUP; or (b) the abort chan closes.
func termSignalListener(abort <-chan struct{}) <-chan struct{} {
	shouldQuit := make(chan struct{})
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh)

	go func() {
		defer close(shouldQuit)
		for {
			select {
			case <-abort:
				log.Infof("executor died, aborting")
				return
			case s, ok := <-sigCh:
				if !ok {
					return
				}
				switch s {
				case os.Interrupt, os.Signal(syscall.SIGTERM), os.Signal(syscall.SIGINT), os.Signal(syscall.SIGHUP):
					log.Infof("received signal %q, aborting", s)
					return
				case os.Signal(syscall.SIGCHLD): // who cares?
				default:
					log.Errorf("unexpected signal: %T %#v", s, s)
				}

			}
		}
	}()
	return shouldQuit
}

func (ms *MinionServer) AddExecutorFlags(fs *pflag.FlagSet) {
	ms.KubeletExecutorServer.AddFlags(fs)

	// hack to forward log verbosity flag to the executor
	fs.Int32Var(&ms.logVerbosity, "v", ms.logVerbosity, "log level for V logs")
}

func (ms *MinionServer) AddMinionFlags(fs *pflag.FlagSet) {
	// general minion flags
	fs.StringVar(&ms.cgroupPrefix, "mesos-cgroup-prefix", ms.cgroupPrefix, "The cgroup prefix concatenated with MESOS_DIRECTORY must give the executor cgroup set by Mesos")
	fs.BoolVar(&ms.privateMountNS, "private-mountns", ms.privateMountNS, "Enter a private mount NS before spawning procs (linux only). Experimental, not yet compatible with k8s volumes.")
	fs.StringVar(&ms.pathOverride, "path-override", ms.pathOverride, "Override the PATH in the environment of the sub-processes.")
	fs.BoolVar(&ms.containPodResources, "contain-pod-resources", ms.containPodResources, "Allocate pod CPU and memory resources from offers and reparent pod containers into mesos cgroups; disable if you're having strange mesos/docker/systemd interactions.")

	// log file flags
	fs.Var(resource.NewQuantityFlagValue(&ms.logMaxSize), "max-log-size", "Maximum log file size for the executor and proxy before rotation")
	fs.IntVar(&ms.logMaxAgeInDays, "max-log-age", ms.logMaxAgeInDays, "Maximum log file age of the executor and proxy in days")
	fs.IntVar(&ms.logMaxBackups, "max-log-backups", ms.logMaxBackups, "Maximum log file backups of the executor and proxy to keep after rotation")

	// proxy flags
	fs.BoolVar(&ms.runProxy, "run-proxy", ms.runProxy, "Maintain a running kube-proxy instance as a child proc of this kubelet-executor.")
	fs.IntVar(&ms.proxyLogV, "proxy-logv", ms.proxyLogV, "Log verbosity of the child kube-proxy.")
	fs.BoolVar(&ms.proxyBindall, "proxy-bindall", ms.proxyBindall, "When true will cause kube-proxy to bind to 0.0.0.0.")
}
