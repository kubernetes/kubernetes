/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"text/template"

	"gopkg.in/yaml.v2"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
)

// exportOptions contains the assignable options from the args.
type exportOptions struct {
	KubeConfig     string
	NameSpace      string
	ServiceAccount string

	genericclioptions.IOStreams

	Client kubernetes.Interface
}

// profileMetaData contains the service account related data.
type profileMetaData struct {
	CA_CRT         string
	ENDPOINT       string
	CLUSTER        string
	NAMESPACE      string
	SERVICEACCOUNT string
	USER_TOKEN     string
}

// kubConf is a struct mapping to KUBCONFIG yaml file.
type kubConf struct {
	APIVersion string `yaml:"apiVersion"`
	Clusters   []struct {
		Cluster struct {
			CertificateAuthorityData string `yaml:"certificate-authority-data"`
			Server                   string `yaml:"server"`
		} `yaml:"cluster"`
		Name string `yaml:"name"`
	} `yaml:"clusters"`
	Contexts []struct {
		Context struct {
			Cluster   string `yaml:"cluster"`
			Namespace string `yaml:"namespace"`
			User      string `yaml:"user"`
		} `yaml:"context"`
		Name string `yaml:"name"`
	} `yaml:"contexts"`
	CurrentContext string `yaml:"current-context"`
	Kind           string `yaml:"kind"`
	Preferences    struct {
	} `yaml:"preferences"`
	Users []struct {
		Name string `yaml:"name"`
		User struct {
			Token string `yaml:"token"`
		} `yaml:"user"`
	} `yaml:"users"`
}

var (
	export_long = templates.LongDesc(`
		Export kubeconfig file for a serviceaccount in a namespace.

		`)

	export_example = templates.Examples(`
		# Show and Export kubeconfig for a service account in a namespace.
		kubectl config export --namespace default --serviceaccount test
    `)
)

const kubConfTPL = `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: {{.CA_CRT}}
    server: {{.ENDPOINT}}
  name: {{.CLUSTER}}
contexts:
- context:
    cluster: {{.CLUSTER}}
    namespace: {{.NAMESPACE}}
    user: {{.SERVICEACCOUNT}}-{{.CLUSTER}}
  name: {{.SERVICEACCOUNT}}-{{.CLUSTER}}
current-context: {{.SERVICEACCOUNT}}-{{.CLUSTER}}
kind: Config
preferences: {}
users:
- name: {{.SERVICEACCOUNT}}-{{.CLUSTER}}
  user:
    token: {{.USER_TOKEN}}`

// NewCmdKubeConfigExport creates a command object for the "KUBECONFIG export" action.
func NewCmdKubeConfigExport(client kubernetes.Interface, streams genericclioptions.IOStreams) *cobra.Command {
	o := &exportOptions{
		IOStreams: streams,
		Client:    client,
	}

	cmd := &cobra.Command{
		Use:     "export",
		Short:   i18n.T("Display and export kubeconfig for a serviceaccount in a namespace"),
		Long:    export_long,
		Example: export_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.complete(cmd))
			cmdutil.CheckErr(o.validate())
			cmdutil.CheckErr(o.run())
		},
	}

	cmd.Flags().StringVar(&o.NameSpace, "namespace", o.NameSpace, "Namespace for service account")
	cmd.Flags().StringVar(&o.ServiceAccount, "serviceaccount", o.ServiceAccount, "Service account name")
	cmd.Flags().StringVar(&o.KubeConfig, "kubeconf", o.KubeConfig, "The path of the cluster admin kubeconfig")

	return cmd
}

func (o *exportOptions) complete(cmd *cobra.Command) error {
	if o.KubeConfig == "" {
		o.KubeConfig = filepath.Join(os.Getenv("HOME"), ".kube", "config")
	}

	o.NameSpace = cmdutil.GetFlagString(cmd, "namespace")
	o.ServiceAccount = cmdutil.GetFlagString(cmd, "serviceaccount")

	return nil
}

func (o exportOptions) validate() error {
	if o.NameSpace == "" || o.ServiceAccount == "" {
		return errors.New("namespace and serviceaccount need to be specified")
	}

	return nil
}

func (o exportOptions) run() error {
	config, err := loadConfig(o.KubeConfig)
	if err != nil {
		return err
	}

	if o.Client == nil {
		o.Client, err = kubernetes.NewForConfig(config)
		if err != nil {
			return err
		}
	}

	profileMetaDatas, err := o.serviceaccountData(config.Host)
	if err != nil {
		return err
	}

	t := template.New("KubeConfig")
	t.Parse(kubConfTPL)

	f, err := os.Create("k8s-" + o.ServiceAccount + "-" + o.NameSpace + "-conf")
	if err != nil {
		return err
	}

	for _, metaData := range profileMetaDatas {
		t.Execute(o.IOStreams.Out, metaData)
		t.Execute(f, metaData)
	}
	f.Close()

	return nil
}

func (o exportOptions) serviceaccountData(endpoint string) ([]profileMetaData, error) {
	serviceaccount, err := o.Client.CoreV1().ServiceAccounts(o.NameSpace).Get(o.ServiceAccount, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	if serviceaccount.Secrets == nil {
		fmt.Fprintf(os.Stdout, "No secret record found for: %v\n", serviceaccount)
		return nil, nil
	}

	secrets, err := o.Client.CoreV1().Secrets(o.NameSpace).Get(serviceaccount.Secrets[0].Name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	cluster, err := clusterName(endpoint, o.KubeConfig)
	if err != nil {
		return nil, err
	}

	profileMetaDatas := []profileMetaData{
		{base64.StdEncoding.EncodeToString(secrets.Data["ca.crt"]),
			endpoint,
			cluster,
			secrets.Namespace,
			o.ServiceAccount,
			string(secrets.Data["token"])},
	}

	return profileMetaDatas, nil

}

func loadConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to load client config: %v\n", err)
			return rest.InClusterConfig()
		} else {
			return config, err
		}
	}

	return rest.InClusterConfig()
}

func clusterName(host string, confPath string) (string, error) {
	data, err := ioutil.ReadFile(confPath)
	if err != nil {
		return "", err
	}

	v := kubConf{}
	err = yaml.Unmarshal(data, &v)
	if err != nil {
		return "", err
	}

	cluster := v.Clusters[0].Name
	for _, cl := range v.Clusters {
		if cl.Cluster.Server == host {
			cluster = cl.Name
		}
	}

	return cluster, nil
}
