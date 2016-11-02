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

package kubefed

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"testing"

	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/restclient/fake"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestJoinFederation(t *testing.T) {
	cmdErrMsg := ""
	cmdutil.BehaviorOnFatal(func(str string, code int) {
		cmdErrMsg = str
	})

	fakeKubeFiles, err := fakeKubeconfigFiles()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rmFakeKubeconfigFiles(fakeKubeFiles)

	testCases := []struct {
		cluster            string
		server             string
		token              string
		kubeconfigGlobal   string
		kubeconfigExplicit string
		expectedServer     string
		expectedErr        string
	}{
		{
			cluster:            "syndicate",
			server:             "https://10.20.30.40",
			token:              "badge",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			expectedServer:     "https://10.20.30.40",
			expectedErr:        "",
		},
		{
			cluster:            "ally",
			server:             "ally256.example.com:80",
			token:              "souvenir",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: fakeKubeFiles[1],
			expectedServer:     "https://ally256.example.com:80",
			expectedErr:        "",
		},
		{
			cluster:            "confederate",
			server:             "10.8.8.8",
			token:              "totem",
			kubeconfigGlobal:   fakeKubeFiles[1],
			kubeconfigExplicit: fakeKubeFiles[2],
			expectedServer:     "https://10.8.8.8",
			expectedErr:        "",
		},
		{
			cluster:            "affiliate",
			server:             "https://10.20.30.40",
			token:              "badge",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			expectedServer:     "https://10.20.30.40",
			expectedErr:        fmt.Sprintf("error: cluster context %q not found", "affiliate"),
		},
	}

	for i, tc := range testCases {
		cmdErrMsg = ""
		f := testJoinFederationFactory(tc.cluster, tc.expectedServer)
		buf := bytes.NewBuffer([]byte{})

		hostFactory, err := fakeJoinHostFactory(tc.cluster, tc.server, tc.token)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		joinConfig, err := newFakeJoinFederationConfig(hostFactory, tc.kubeconfigGlobal)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		cmd := NewCmdJoin(f, buf, joinConfig)

		cmd.Flags().Set("kubeconfig", tc.kubeconfigExplicit)
		cmd.Flags().Set("host", "substrate")
		cmd.Run(cmd, []string{tc.cluster})

		if tc.expectedErr == "" {
			// uses the name from the cluster, not the response
			// Actual data passed are tested in the fake secret and cluster
			// REST clients.
			if msg := buf.String(); msg != fmt.Sprintf("cluster %q created\n", tc.cluster) {
				t.Errorf("[%d] unexpected output: %s", i, msg)
				if cmdErrMsg != "" {
					t.Errorf("[%d] unexpected error message: %s", i, cmdErrMsg)
				}
			}
		} else {
			if cmdErrMsg != tc.expectedErr {
				t.Errorf("[%d] expected error: %s, got: %s, output: %s", i, tc.expectedErr, cmdErrMsg, buf.String())
			}
		}
	}
}

func testJoinFederationFactory(name, server string) cmdutil.Factory {
	want := fakeCluster(name, server)
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	codec := testapi.Federation.Codec()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/clusters" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got federationapi.Cluster
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !api.Semantic.DeepEqual(got, want) {
					return nil, fmt.Errorf("unexpected cluster object\n\tgot: %#v\n\twant: %#v", got, want)
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, &want)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
	tf.Namespace = "test"
	return f
}

type fakeJoinFederationConfig struct {
	pathOptions *clientcmd.PathOptions
	hostFactory cmdutil.Factory
}

func newFakeJoinFederationConfig(f cmdutil.Factory, kubeconfigGlobal string) (JoinFederationConfig, error) {
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = kubeconfigGlobal
	pathOptions.EnvVar = ""

	return &fakeJoinFederationConfig{
		pathOptions: pathOptions,
		hostFactory: f,
	}, nil
}

func (r *fakeJoinFederationConfig) PathOptions() *clientcmd.PathOptions {
	return r.pathOptions
}

func (r *fakeJoinFederationConfig) HostFactory(host, kubeconfigPath string) cmdutil.Factory {
	return r.hostFactory
}

func fakeJoinHostFactory(name, server, token string) (cmdutil.Factory, error) {
	kubeconfig := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			name: {
				Server: server,
			},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			name: {
				Token: token,
			},
		},
		Contexts: map[string]*clientcmdapi.Context{
			name: {
				Cluster:  name,
				AuthInfo: name,
			},
		},
		CurrentContext: name,
	}
	configBytes, err := clientcmd.Write(kubeconfig)
	if err != nil {
		return nil, err
	}
	secretObject := v1.Secret{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Secret",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: "federation-system",
		},
		Data: map[string][]byte{
			"kubeconfig": configBytes,
		},
	}

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.ClientConfig = defaultClientConfig()
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/federation-system/secrets" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got v1.Secret
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !api.Semantic.DeepEqual(got, secretObject) {
					return nil, fmt.Errorf("Unexpected secret object\n\tgot: %#v\n\twant: %#v", got, secretObject)
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, &secretObject)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
	return f, nil
}

func fakeCluster(name, server string) federationapi.Cluster {
	return federationapi.Cluster{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: federationapi.ClusterSpec{
			ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
				{
					ClientCIDR:    defaultClientCIDR,
					ServerAddress: server,
				},
			},
			SecretRef: &v1.LocalObjectReference{
				Name: name,
			},
		},
	}
}

func fakeKubeconfigFiles() ([]string, error) {
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

func rmFakeKubeconfigFiles(kubefiles []string) {
	for _, file := range kubefiles {
		os.Remove(file)
	}
}

func defaultHeader() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}

func objBody(codec runtime.Codec, obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj))))
}

func defaultClientConfig() *restclient.Config {
	return &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: api.Codecs,
			ContentType:          runtime.ContentTypeJSON,
			GroupVersion:         &registered.GroupOrDie(api.GroupName).GroupVersion,
		},
	}
}
