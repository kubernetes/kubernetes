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

package testing

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	fedclient "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

type fakeAdminConfig struct {
	pathOptions          *clientcmd.PathOptions
	hostFactory          cmdutil.Factory
	targetClusterFactory cmdutil.Factory
	targetClusterContext string
}

func NewFakeAdminConfig(hostFactory cmdutil.Factory, targetFactory cmdutil.Factory, targetClusterContext, kubeconfigGlobal string) (util.AdminConfig, error) {
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = kubeconfigGlobal
	pathOptions.EnvVar = ""

	return &fakeAdminConfig{
		pathOptions:          pathOptions,
		hostFactory:          hostFactory,
		targetClusterFactory: targetFactory,
		targetClusterContext: targetClusterContext,
	}, nil
}

func (f *fakeAdminConfig) PathOptions() *clientcmd.PathOptions {
	return f.pathOptions
}

func (f *fakeAdminConfig) FederationClientset(context, kubeconfigPath string) (*fedclient.Clientset, error) {
	fakeRestClient, err := f.hostFactory.RESTClient()
	if err != nil {
		return nil, err
	}

	// we ignore the function params and use the client from
	// the same fakefactory to create a federation clientset
	// our fake factory exposes only the healthz api for this client
	return fedclient.New(fakeRestClient), nil
}

func (f *fakeAdminConfig) ClusterFactory(context, kubeconfigPath string) cmdutil.Factory {
	if f.targetClusterContext != "" && f.targetClusterContext == context {
		return f.targetClusterFactory
	}
	return f.hostFactory
}

func FakeKubeconfigFiles() ([]string, error) {
	kubeconfigs := []clientcmdapi.Config{
		{
			Clusters: map[string]*clientcmdapi.Cluster{
				"syndicate": {
					Server: "https://10.20.30.40",
				},
			},
			AuthInfos: map[string]*clientcmdapi.AuthInfo{
				"syndicate": {
					Token: "badge",
				},
			},
			Contexts: map[string]*clientcmdapi.Context{
				"syndicate": {
					Cluster:  "syndicate",
					AuthInfo: "syndicate",
				},
			},
			CurrentContext: "syndicate",
		},
		{
			Clusters: map[string]*clientcmdapi.Cluster{
				"ally": {
					Server: "ally256.example.com:80",
				},
			},
			AuthInfos: map[string]*clientcmdapi.AuthInfo{
				"ally": {
					Token: "souvenir",
				},
			},
			Contexts: map[string]*clientcmdapi.Context{
				"ally": {
					Cluster:  "ally",
					AuthInfo: "ally",
				},
			},
			CurrentContext: "ally",
		},
		{
			Clusters: map[string]*clientcmdapi.Cluster{
				"ally": {
					Server: "https://ally64.example.com",
				},
				"confederate": {
					Server: "10.8.8.8",
				},
			},
			AuthInfos: map[string]*clientcmdapi.AuthInfo{
				"ally": {
					Token: "souvenir",
				},
				"confederate": {
					Token: "totem",
				},
			},
			Contexts: map[string]*clientcmdapi.Context{
				"ally": {
					Cluster:  "ally",
					AuthInfo: "ally",
				},
				"confederate": {
					Cluster:  "confederate",
					AuthInfo: "confederate",
				},
			},
			CurrentContext: "confederate",
		},
	}
	kubefiles := []string{}
	for _, cfg := range kubeconfigs {
		fakeKubeFile, _ := ioutil.TempFile("", "")
		err := clientcmd.WriteToFile(cfg, fakeKubeFile.Name())
		if err != nil {
			return nil, err
		}

		kubefiles = append(kubefiles, fakeKubeFile.Name())
	}
	return kubefiles, nil
}

func RemoveFakeKubeconfigFiles(kubefiles []string) {
	for _, file := range kubefiles {
		os.Remove(file)
	}
}

func DefaultHeader() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}

func ObjBody(codec runtime.Codec, obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj))))
}

func DefaultClientConfig() *restclient.Config {
	return &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: api.Codecs,
			ContentType:          runtime.ContentTypeJSON,
			GroupVersion:         &api.Registry.GroupOrDie(api.GroupName).GroupVersion,
		},
	}
}
