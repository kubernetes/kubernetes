/*
Copyright 2022 The Kubernetes Authors.

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

package phases

import (
	"io"
	"text/template"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
)

var (
	initDoneTempl = template.Must(template.New("init").Parse(dedent.Dedent(`
		Your Kubernetes control-plane has initialized successfully!

		To start using your cluster, you need to run the following as a regular user:

		  mkdir -p $HOME/.kube
		  sudo cp -i {{.KubeConfigPath}} $HOME/.kube/config
		  sudo chown $(id -u):$(id -g) $HOME/.kube/config

		Alternatively, if you are the root user, you can run:

		  export KUBECONFIG=/etc/kubernetes/admin.conf

		You should now deploy a pod network to the cluster.
		Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
		  https://kubernetes.io/docs/concepts/cluster-administration/addons/

		{{if .ControlPlaneEndpoint -}}
		{{if .UploadCerts -}}
		You can now join any number of the control-plane node running the following command on each as root:

		  {{.joinControlPlaneCommand}}

		Please note that the certificate-key gives access to cluster sensitive data, keep it secret!
		As a safeguard, uploaded-certs will be deleted in two hours; If necessary, you can use
		"kubeadm init phase upload-certs --upload-certs" to reload certs afterward.

		{{else -}}
		You can now join any number of control-plane nodes by copying certificate authorities
		and service account keys on each node and then running the following as root:

		  {{.joinControlPlaneCommand}}

		{{end}}{{end}}Then you can join any number of worker nodes by running the following on each as root:

		{{.joinWorkerCommand}}
		`)))
)

// NewShowJoinCommandPhase creates a kubeadm workflow phase that implements showing the join command.
func NewShowJoinCommandPhase() workflow.Phase {
	return workflow.Phase{
		Name:         "show-join-command",
		Short:        "Show the join command for control-plane and worker node",
		Run:          showJoinCommand,
		Dependencies: []string{"bootstrap-token", "upload-certs"},
	}
}

// showJoinCommand prints the join command after all the phases in init have finished
func showJoinCommand(c workflow.RunData) error {
	data, ok := c.(InitData)
	if !ok {
		return errors.New("show-join-command phase invoked with an invalid data struct")
	}

	adminKubeConfigPath := data.KubeConfigPath()

	// Prints the join command, multiple times in case the user has multiple tokens
	for _, token := range data.Tokens() {
		if err := printJoinCommand(data.OutputWriter(), adminKubeConfigPath, token, data); err != nil {
			return errors.Wrap(err, "failed to print join command")
		}
	}

	return nil
}

func printJoinCommand(out io.Writer, adminKubeConfigPath, token string, i InitData) error {
	joinControlPlaneCommand, err := cmdutil.GetJoinControlPlaneCommand(adminKubeConfigPath, token, i.CertificateKey(), i.SkipTokenPrint(), i.SkipCertificateKeyPrint())
	if err != nil {
		return err
	}

	joinWorkerCommand, err := cmdutil.GetJoinWorkerCommand(adminKubeConfigPath, token, i.SkipTokenPrint())
	if err != nil {
		return err
	}

	ctx := map[string]interface{}{
		"KubeConfigPath":          adminKubeConfigPath,
		"ControlPlaneEndpoint":    i.Cfg().ControlPlaneEndpoint,
		"UploadCerts":             i.UploadCerts(),
		"joinControlPlaneCommand": joinControlPlaneCommand,
		"joinWorkerCommand":       joinWorkerCommand,
	}

	return initDoneTempl.Execute(out, ctx)
}
