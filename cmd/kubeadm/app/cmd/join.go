/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"io"
	"io/ioutil"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubenode "k8s.io/kubernetes/cmd/kubeadm/app/node"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

var (
	joinDoneMsgf = dedent.Dedent(`
		Node join complete:
		* Certificate signing request sent to master and response
		  received.
		* Kubelet informed of new secure connection details.

		Run 'kubectl get nodes' on the master to see this machine join.
		`)
)

// NewCmdJoin returns "kubeadm join" command.
func NewCmdJoin(out io.Writer) *cobra.Command {
	versioned := &kubeadmapiext.NodeConfiguration{}
	api.Scheme.Default(versioned)
	cfg := kubeadmapi.NodeConfiguration{}
	api.Scheme.Convert(versioned, &cfg, nil)

	var skipPreFlight bool
	var cfgPath string

	cmd := &cobra.Command{
		Use:   "join <master address>",
		Short: "Run this on any machine you wish to join an existing cluster",
		Run: func(cmd *cobra.Command, args []string) {
			j, err := NewJoin(cfgPath, args, &cfg, skipPreFlight)
			kubeadmutil.CheckErr(err)
			kubeadmutil.CheckErr(j.Run(out))
		},
	}

	cmd.PersistentFlags().StringVar(
		&cfg.Secrets.GivenToken, "token", cfg.Secrets.GivenToken,
		"(required) Shared secret used to secure bootstrap. Must match the output of 'kubeadm init'",
	)

	cmd.PersistentFlags().StringVar(&cfgPath, "config", cfgPath, "Path to kubeadm config file")

	cmd.PersistentFlags().BoolVar(
		&skipPreFlight, "skip-preflight-checks", false,
		"skip preflight checks normally run before modifying the system",
	)

	cmd.PersistentFlags().Int32Var(
		&cfg.APIPort, "api-port", cfg.APIPort,
		"(optional) API server port on the master",
	)

	cmd.PersistentFlags().Int32Var(
		&cfg.DiscoveryPort, "discovery-port", cfg.DiscoveryPort,
		"(optional) Discovery port on the master",
	)

	return cmd
}

type Join struct {
	cfg *kubeadmapi.NodeConfiguration
}

func NewJoin(cfgPath string, args []string, cfg *kubeadmapi.NodeConfiguration, skipPreFlight bool) (*Join, error) {
	if cfgPath != "" {
		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), b, cfg); err != nil {
			return nil, fmt.Errorf("unable to decode config from %q [%v]", cfgPath, err)
		}
	}

	if len(args) == 0 && len(cfg.MasterAddresses) == 0 {
		return nil, fmt.Errorf("must specify master address (see --help)")
	}
	cfg.MasterAddresses = append(cfg.MasterAddresses, args...)
	if len(cfg.MasterAddresses) > 1 {
		return nil, fmt.Errorf("Must not specify more than one master address  (see --help)")
	}

	if !skipPreFlight {
		fmt.Println("Running pre-flight checks")
		err := preflight.RunJoinNodeChecks(cfg)
		if err != nil {
			return nil, &preflight.PreFlightError{Msg: err.Error()}
		}
	} else {
		fmt.Println("Skipping pre-flight checks")
	}

	ok, err := kubeadmutil.UseGivenTokenIfValid(&cfg.Secrets)
	if !ok {
		if err != nil {
			return nil, fmt.Errorf("%v (see --help)\n", err)
		}
		return nil, fmt.Errorf("Must specify --token (see --help)\n")
	}

	return &Join{cfg: cfg}, nil
}

// Run executes worked node provisioning and tries to join an existing cluster.
func (j *Join) Run(out io.Writer) error {
	clusterInfo, err := kubenode.RetrieveTrustedClusterInfo(j.cfg)
	if err != nil {
		return err
	}

	connectionDetails, err := kubenode.EstablishMasterConnection(j.cfg, clusterInfo)
	if err != nil {
		return err
	}

	kubeconfig, err := kubenode.PerformTLSBootstrap(connectionDetails)
	if err != nil {
		return err
	}

	err = kubeadmutil.WriteKubeconfigIfNotExists("kubelet", kubeconfig)
	if err != nil {
		return err
	}

	fmt.Fprintf(out, joinDoneMsgf)
	return nil
}
