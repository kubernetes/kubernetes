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
	"bytes"
	"io"
	"strings"
	"text/template"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/pkg/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const (
	// kubeadmUsageTemplateRequiredImages is appended to default usage template,
	// so that user can see information about how to get required images.
	kubeadmUsageTemplateRequiredImages = `Use "{{.CommandPath}} help required-images" for information about all images required to initialize a Kubernetes cluster.
`
	kubeadmRequiredImagesTemplate = `Required images:{{range .RequiredImages}}
  {{ . }}{{end}}`
)

func NewKubeadmCommand(f cmdutil.Factory, in io.Reader, out, err io.Writer) *cobra.Command {
	cmds := &cobra.Command{
		Use:   "kubeadm",
		Short: "kubeadm: easily bootstrap a secure Kubernetes cluster",
		Long: dedent.Dedent(`
			kubeadm: easily bootstrap a secure Kubernetes cluster.

			    ┌──────────────────────────────────────────────────────────┐
			    │ KUBEADM IS ALPHA, DO NOT USE IT FOR PRODUCTION CLUSTERS! │
			    │                                                          │
			    │ But, please try it out! Give us feedback at:             │
			    │ https://github.com/kubernetes/kubeadm/issues             │
			    │ and at-mention @kubernetes/sig-cluster-lifecycle         │
			    └──────────────────────────────────────────────────────────┘

			Example usage:

			    Create a two-machine cluster with one master (which controls the cluster),
			    and one node (where your workloads, like Pods and ReplicaSets run).

			    ┌──────────────────────────────────────────────────────────┐
			    │ On the first machine                                     │
			    ├──────────────────────────────────────────────────────────┤
			    │ master# kubeadm init                                     │
			    └──────────────────────────────────────────────────────────┘

			    ┌──────────────────────────────────────────────────────────┐
			    │ On the second machine                                    │
			    ├──────────────────────────────────────────────────────────┤
			    │ node# kubeadm join --token=<token> <ip-of-master>        │
			    └──────────────────────────────────────────────────────────┘

			    You can then repeat the second step on as many other machines as you like.

		`),
	}
	// TODO(phase2+) figure out how to avoid running as root
	//
	// TODO(phase2) detect interactive vs non-interactive use and adjust output accordingly
	// i.e. make it automation friendly
	//
	// TODO(phase2) create an abstraction that defines files and the content that needs to
	// be written to disc and write it all in one go at the end as we have a lot of
	// crappy little files written from different parts of this code; this could also
	// be useful for testing by having this model we can allow users to create some files before
	// `kubeadm init` runs, e.g. PKI assets, we would then be able to look at files users has
	// given an diff or validate if those are sane, we could also warn if any of the files had been deprecated

	cmds.ResetFlags()
	cmds.SetGlobalNormalizationFunc(flag.WarnWordSepNormalizeFunc)

	cmds.AddCommand(NewCmdCompletion(out, ""))
	cmds.AddCommand(NewCmdInit(out))
	cmds.AddCommand(NewCmdJoin(out))
	cmds.AddCommand(NewCmdReset(out))
	cmds.AddCommand(NewCmdVersion(out))
	cmds.AddCommand(NewCmdToken(out, err))

	// Wrap not yet fully supported commands in an alpha subcommand
	experimentalCmd := &cobra.Command{
		Use:   "alpha",
		Short: "Experimental sub-commands not yet fully functional.",
	}
	experimentalCmd.AddCommand(phases.NewCmdPhase(out))
	cmds.AddCommand(experimentalCmd)

	cmds.SetHelpCommand(newKubeadmHelpCmd(cmds.Name()))
	cmds.SetUsageTemplate(cmds.UsageTemplate() + kubeadmUsageTemplateRequiredImages)
	return cmds
}

// newKubeadmHelpCmd returns a help command for kubeadm root command.
// see cobra.Command#initHelpCmd() (command.go)
func newKubeadmHelpCmd(cmdName string) *cobra.Command {
	return &cobra.Command{
		Use:   "help [command]",
		Short: "Help about any command",
		Long: `Help provides help for any command in the application.
    Simply type ` + cmdName + ` help [path to command] for full details.`,
		PersistentPreRun:  func(cmd *cobra.Command, args []string) {},
		PersistentPostRun: func(cmd *cobra.Command, args []string) {},

		Run: func(c *cobra.Command, args []string) {
			if len(args) == 1 && strings.TrimSpace(args[0]) == "required-images" {
				if s, err := getKubeadmRequiredImages(); err != nil {
					c.Println("failed to get kubeadm required images : %v", err)
				} else {
					c.Println(s)
				}
			} else {
				cmd, _, e := c.Root().Find(args)
				if cmd == nil || e != nil {
					c.Printf("Unknown help topic %#q.", args)
					c.Root().Usage()
				} else {
					cmd.Help()
				}
			}
		},
	}
}
func getKubeadmRequiredImages() (string, error) {
	versioned := &kubeadmapiext.MasterConfiguration{}
	api.Scheme.Default(versioned)
	cfg := kubeadmapi.MasterConfiguration{}
	api.Scheme.Convert(versioned, &cfg, nil)

	requiredImages := []string{
		images.GetCoreImage(images.KubeAPIServerImage, &cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
		images.GetCoreImage(images.KubeControllerManagerImage, &cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
		images.GetCoreImage(images.KubeSchedulerImage, &cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
		images.GetCoreImage(images.KubeProxyImage, &cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
		images.GetCoreImage(images.KubeEtcdImage, &cfg, kubeadmapi.GlobalEnvParams.EtcdImage),
	}
	var buf bytes.Buffer
	t := template.New("")
	template.Must(t.Parse(kubeadmRequiredImagesTemplate))
	if err := t.Execute(&buf, map[string][]string{"RequiredImages": requiredImages}); err != nil {
		return "", err
	}
	return buf.String(), nil

}
