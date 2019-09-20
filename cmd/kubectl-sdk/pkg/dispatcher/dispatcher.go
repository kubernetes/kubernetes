/*
Copyright 2019 The Kubernetes Authors.

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

package dispatcher

import (
	"fmt"
	"os"
	"syscall"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	utilflag "k8s.io/component-base/cli/flag"
	"k8s.io/kubernetes/cmd/kubectl-sdk/pkg/client"
	"k8s.io/kubernetes/cmd/kubectl-sdk/pkg/filepath"
	"k8s.io/kubernetes/cmd/kubectl-sdk/pkg/util"

	// klog calls in this file assume it has been initialized beforehand
	"k8s.io/klog/v2"
)

const (
	requestTimeout = "5s"        // Timeout for server version query
	cacheMaxAge    = 2 * 60 * 60 // 2 hours in seconds
)

var HelpFlags = []string{"-h", "--help"}

type Dispatcher struct {
	args            []string
	env             []string
	clientVersion   version.Info
	filepathBuilder *filepath.FilepathBuilder
}

// NewDispatcher returns a new pointer to a Dispatcher struct.
func NewDispatcher(args []string, env []string,
	clientVersion version.Info,
	filepathBuilder *filepath.FilepathBuilder) *Dispatcher {

	return &Dispatcher{
		args:            args,
		env:             env,
		clientVersion:   clientVersion,
		filepathBuilder: filepathBuilder,
	}
}

// GetArgs returns a copy of the slice of strings representing the command line arguments.
func (d *Dispatcher) GetArgs() []string {
	return util.CopyStrSlice(d.args)
}

// GetEnv returns a copy of the slice of environment variables.
func (d *Dispatcher) GetEnv() []string {
	return util.CopyStrSlice(d.env)
}

func (d *Dispatcher) GetClientVersion() version.Info {
	return d.clientVersion
}

const kubeConfigFlagSetName = "dispatcher-kube-config"

// InitKubeConfigFlags returns the ConfigFlags struct filled in with
// kube config values parsed from command line arguments. These flag values can
// affect the server version query. Therefore, the set of kubeConfigFlags MUST
// match the set used in the regular kubectl binary.
func (d *Dispatcher) InitKubeConfigFlags() (*genericclioptions.ConfigFlags, error) {

	// IMPORTANT: If there is an error parsing flags--continue.
	kubeConfigFlagSet := pflag.NewFlagSet("dispatcher-kube-config", pflag.ContinueOnError)
	kubeConfigFlagSet.ParseErrorsWhitelist.UnknownFlags = true
	kubeConfigFlagSet.SetNormalizeFunc(utilflag.WordSepNormalizeFunc)

	unusedParameter := true // Could be either true or false
	kubeConfigFlags := genericclioptions.NewConfigFlags(unusedParameter)
	kubeConfigFlags.AddFlags(kubeConfigFlagSet)

	// Remove help flags, since these are special-cased in pflag.Parse,
	// and handled in the dispatcher instead of passed to versioned binary.
	args := util.FilterList(d.GetArgs(), HelpFlags)
	if err := kubeConfigFlagSet.Parse(args[1:]); err != nil {
		return nil, err
	}
	kubeConfigFlagSet.VisitAll(func(flag *pflag.Flag) {
		klog.V(4).Infof("KubeConfig Flag: --%s=%q", flag.Name, flag.Value)
	})

	return kubeConfigFlags, nil
}

// Dispatch attempts to execute a matching version of kubectl based on the
// version of the APIServer. If successful, this method will not return, since
// current process will be overwritten (see execve(2)). Otherwise, this method
// returns an error describing why the execution could not happen.
func (d *Dispatcher) Dispatch() error {
	// Fetch the server version and generate the kubectl binary full file path
	// from this version.
	// Example:
	//   serverVersion=1.11 -> /home/seans/go/bin/kubectl.1.11
	kubeConfigFlags, err := d.InitKubeConfigFlags()
	if err != nil {
		return err
	}
	svclient := client.NewServerVersionClient(kubeConfigFlags)
	svclient.SetRequestTimeout(requestTimeout)
	svclient.SetCacheMaxAge(cacheMaxAge)
	serverVersion, err := svclient.ServerVersion()
	if err != nil {
		return err
	}
	klog.V(4).Infof("Server Version: %s", serverVersion.GitVersion)
	klog.V(4).Infof("Client Version: %s", d.GetClientVersion().GitVersion)
	if util.VersionMatch(d.GetClientVersion(), *serverVersion) {
		// TODO(seans): Consider changing to return a bool as well as error, since
		// this isn't really an error.
		return fmt.Errorf("Client/Server version match--fall through to default")
	}

	kubectlFilepath, err := d.filepathBuilder.VersionedFilePath(*serverVersion)
	if err != nil {
		return err
	}
	if err := d.filepathBuilder.ValidateFilepath(kubectlFilepath); err != nil {
		return err
	}

	// Delegate to the versioned kubectl binary. This overwrites the current process
	// (by calling execve(2) system call), and it does not return on success.
	klog.V(3).Infof("kubectl dispatching: %s\n", kubectlFilepath)
	return syscall.Exec(kubectlFilepath, d.GetArgs(), d.GetEnv())
}

// Execute is the entry point to the dispatcher. It passes in the current client
// version, which is used to determine if a delegation is necessary. If this function
// successfully delegates, then it will NOT return, since the current process will be
// overwritten (see execve(2)). If this function does not delegate, it merely falls
// through. This function assumes logging has been initialized before it is run;
// otherwise, log statements will not work.
func Execute(clientVersion version.Info) {
	klog.V(4).Info("Starting dispatcher")
	filepathBuilder := filepath.NewFilepathBuilder(&filepath.ExeDirGetter{}, os.Stat)
	dispatcher := NewDispatcher(os.Args, os.Environ(), clientVersion, filepathBuilder)
	if err := dispatcher.Dispatch(); err != nil {
		klog.V(3).Infof("Dispatch error: %v", err)
	}
}
