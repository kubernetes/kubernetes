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
	"os/exec"
	"strings"
	"time"

	exservice "k8s.io/kubernetes/contrib/mesos/pkg/executor/service"
	"k8s.io/kubernetes/contrib/mesos/pkg/hyperkube"
	"k8s.io/kubernetes/contrib/mesos/pkg/minion/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	log "github.com/golang/glog"
	"github.com/kardianos/osext"
	"github.com/spf13/pflag"
	"gopkg.in/natefinch/lumberjack.v2"
)

type MinionServer struct {
	// embed the executor server to be able to use its flags
	// TODO(sttts): get rid of this mixing of the minion and the executor server with a multiflags implementation for km
	KubeletExecutorServer *exservice.KubeletExecutorServer

	privateMountNS bool
	hks            hyperkube.Interface
	clientConfig   *client.Config
	kmBinary       string
	done           chan struct{} // closed when shutting down
	exit           chan error    // to signal fatal errors

	pathOverride string // the PATH environment for the sub-processes

	logMaxSize      resource.Quantity
	logMaxBackups   int
	logMaxAgeInDays int

	runProxy     bool
	proxyLogV    int
	proxyBindall bool
}

// NewMinionServer creates the MinionServer struct with default values to be used by hyperkube
func NewMinionServer() *MinionServer {
	s := &MinionServer{
		KubeletExecutorServer: exservice.NewKubeletExecutorServer(),
		privateMountNS:        false, // disabled until Docker supports customization of the parent mount namespace
		done:                  make(chan struct{}),
		exit:                  make(chan error),

		logMaxSize:      config.DefaultLogMaxSize(),
		logMaxBackups:   config.DefaultLogMaxBackups,
		logMaxAgeInDays: config.DefaultLogMaxAgeInDays,

		runProxy: true,
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

func (ms *MinionServer) launchProxyServer() {
	bindAddress := "0.0.0.0"
	if !ms.proxyBindall {
		bindAddress = ms.KubeletExecutorServer.Address.String()
	}
	args := []string{
		fmt.Sprintf("--bind-address=%s", bindAddress),
		fmt.Sprintf("--v=%d", ms.proxyLogV),
		"--logtostderr=true",
	}

	if ms.clientConfig.Host != "" {
		args = append(args, fmt.Sprintf("--master=%s", ms.clientConfig.Host))
	}
	if ms.KubeletExecutorServer.HostnameOverride != "" {
		args = append(args, fmt.Sprintf("--hostname-override=%s", ms.KubeletExecutorServer.HostnameOverride))
	}

	ms.launchHyperkubeServer(hyperkube.CommandProxy, &args, "proxy.log")
}

func (ms *MinionServer) launchExecutorServer() {
	allArgs := os.Args[1:]

	// filter out minion flags, leaving those for the executor
	executorFlags := pflag.NewFlagSet("executor", pflag.ContinueOnError)
	executorFlags.SetOutput(ioutil.Discard)
	ms.AddExecutorFlags(executorFlags)
	executorArgs, _ := filterArgsByFlagSet(allArgs, executorFlags)

	// run executor and quit minion server when this exits cleanly
	err := ms.launchHyperkubeServer(hyperkube.CommandExecutor, &executorArgs, "executor.log")
	if err != nil {
		// just return, executor will be restarted on error
		log.Error(err)
		return
	}

	log.Info("Executor exited cleanly, stopping the minion")
	ms.exit <- nil
}

func (ms *MinionServer) launchHyperkubeServer(server string, args *[]string, logFileName string) error {
	log.V(2).Infof("Spawning hyperkube %v with args '%+v'", server, args)

	// prepare parameters
	kmArgs := []string{server}
	for _, arg := range *args {
		kmArgs = append(kmArgs, arg)
	}

	// create command
	cmd := exec.Command(ms.kmBinary, kmArgs...)
	if _, err := cmd.StdoutPipe(); err != nil {
		// fatal error => terminate minion
		err = fmt.Errorf("error getting stdout of %v: %v", server, err)
		ms.exit <- err
		return err
	}
	stderrLogs, err := cmd.StderrPipe()
	if err != nil {
		// fatal error => terminate minion
		err = fmt.Errorf("error getting stderr of %v: %v", server, err)
		ms.exit <- err
		return err
	}

	ch := make(chan struct{})
	go func() {
		defer func() {
			select {
			case <-ch:
				log.Infof("killing %v process...", server)
				if err = cmd.Process.Kill(); err != nil {
					log.Errorf("failed to kill %v process: %v", server, err)
				}
			default:
			}
		}()

		maxSize := ms.logMaxSize.Value()
		if maxSize > 0 {
			// convert to MB
			maxSize = maxSize / 1024 / 1024
			if maxSize == 0 {
				log.Warning("maximal log file size is rounded to 1 MB")
				maxSize = 1
			}
		}
		writer := &lumberjack.Logger{
			Filename:   logFileName,
			MaxSize:    int(maxSize),
			MaxBackups: ms.logMaxBackups,
			MaxAge:     ms.logMaxAgeInDays,
		}
		defer writer.Close()

		log.V(2).Infof("Starting logging for %v: max log file size %d MB, keeping %d backups, for %d days", server, maxSize, ms.logMaxBackups, ms.logMaxAgeInDays)

		<-ch
		written, err := io.Copy(writer, stderrLogs)
		if err != nil {
			log.Errorf("error writing data to %v: %v", logFileName, err)
		}

		log.Infof("wrote %d bytes to %v", written, logFileName)
	}()

	// use given environment, but add /usr/sbin to the path for the iptables binary used in kube-proxy
	if ms.pathOverride != "" {
		env := os.Environ()
		cmd.Env = make([]string, 0, len(env))
		for _, e := range env {
			if !strings.HasPrefix(e, "PATH=") {
				cmd.Env = append(cmd.Env, e)
			}
		}
		cmd.Env = append(cmd.Env, "PATH="+ms.pathOverride)
	}

	// if the server fails to start then we exit the executor, otherwise
	// wait for the proxy process to end (and release resources after).
	if err := cmd.Start(); err != nil {
		// fatal error => terminate minion
		err = fmt.Errorf("error starting %v: %v", server, err)
		ms.exit <- err
		return err
	}
	close(ch)
	if err := cmd.Wait(); err != nil {
		log.Errorf("%v exited with error: %v", server, err)
		err = fmt.Errorf("%v exited with error: %v", server, err)
		return err
	}

	return nil
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

	// run subprocesses until ms.done is closed on return of this function
	defer close(ms.done)
	if ms.runProxy {
		go runtime.Until(ms.launchProxyServer, 5*time.Second, ms.done)
	}
	go runtime.Until(ms.launchExecutorServer, 5*time.Second, ms.done)

	// wait until minion exit is requested
	// don't close ms.exit here to avoid panics of go routines writing an error to it
	return <-ms.exit
}

func (ms *MinionServer) AddExecutorFlags(fs *pflag.FlagSet) {
	ms.KubeletExecutorServer.AddFlags(fs)
}

func (ms *MinionServer) AddMinionFlags(fs *pflag.FlagSet) {
	// general minion flags
	fs.BoolVar(&ms.privateMountNS, "private-mountns", ms.privateMountNS, "Enter a private mount NS before spawning procs (linux only). Experimental, not yet compatible with k8s volumes.")
	fs.StringVar(&ms.pathOverride, "path-override", ms.pathOverride, "Override the PATH in the environment of the sub-processes.")

	// log file flags
	fs.Var(resource.NewQuantityFlagValue(&ms.logMaxSize), "max-log-size", "Maximum log file size for the executor and proxy before rotation")
	fs.IntVar(&ms.logMaxAgeInDays, "max-log-age", ms.logMaxAgeInDays, "Maximum log file age of the executor and proxy in days")
	fs.IntVar(&ms.logMaxBackups, "max-log-backups", ms.logMaxBackups, "Maximum log file backups of the executor and proxy to keep after rotation")

	// proxy flags
	fs.BoolVar(&ms.runProxy, "run-proxy", ms.runProxy, "Maintain a running kube-proxy instance as a child proc of this kubelet-executor.")
	fs.IntVar(&ms.proxyLogV, "proxy-logv", ms.proxyLogV, "Log verbosity of the child kube-proxy.")
	fs.BoolVar(&ms.proxyBindall, "proxy-bindall", ms.proxyBindall, "When true will cause kube-proxy to bind to 0.0.0.0.")
}
