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
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"testing"

	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
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
		expectErr          string
	}{
		{
			cluster:            "syndicate",
			server:             "https://10.20.30.40",
			token:              "badge",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			expectedServer:     "https://10.20.30.40",
			expectErr:          "",
		},
		{
			cluster:            "ally",
			server:             "ally256.example.com:80",
			token:              "souvenir",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: fakeKubeFiles[1],
			expectedServer:     "http://ally256.example.com:80",
			expectErr:          "",
		},
		{
			cluster:            "confederate",
			server:             "10.8.8.8",
			token:              "totem",
			kubeconfigGlobal:   fakeKubeFiles[1],
			kubeconfigExplicit: fakeKubeFiles[2],
			expectedServer:     "https://10.8.8.8",
			expectErr:          "",
		},
		{
			cluster:            "affiliate",
			server:             "https://10.20.30.40",
			token:              "badge",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			expectedServer:     "https://10.20.30.40",
			expectErr:          fmt.Sprintf("error: cluster context %q not found", "affiliate"),
		},
	}

	for i, tc := range testCases {
		cmdErrMsg = ""
		f := testJoinFederationFactory(tc.cluster, tc.expectedServer)
		buf := bytes.NewBuffer([]byte{})

		joinConfig, err := newFakeJoinFederationConfig(tc.cluster, tc.server, tc.token, tc.kubeconfigGlobal, tc.kubeconfigExplicit)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		cmd := NewCmdJoinFederation(f, buf, joinConfig)

		// This is a smell that something is wrong here and it is unfortunate
		// that we need to do this. `--kubeconfig` flag is defined in the
		// factory function `pkg/kubectl/cmd/util.DefaultClientConfig()`.
		// We don't call that function in our tests because we use a fake
		// factory, not a real one. However, we read the flag in the code
		// this is testing and the flag must be defined before reading it.
		// So we define it here to enable tests.
		cmd.Flags().String("kubeconfig", "", "Path to the kubeconfig file to use for CLI requests.")

		cmd.Flags().Set("host", "substrate")
		cmd.Run(cmd, []string{tc.cluster})

		if len(tc.expectErr) == 0 {
			// uses the name from the cluster, not the response
			// Actual data passed are tested in the fake secret and cluster
			// REST clients.
			if buf.String() != fmt.Sprintf("cluster %q created\n", tc.cluster) {
				t.Errorf("[%d] unexpected output: %s", i, buf.String())
			}
		} else {
			if cmdErrMsg != tc.expectErr {
				t.Errorf("[%d] expected error: %s, got: %s, output: %s", i, tc.expectErr, cmdErrMsg, buf.String())
			}
		}
	}
}

func testJoinFederationFactory(name, server string) cmdutil.Factory {
	want := fakeCluster(name, server)
	f, tf, _, _ := NewAPIFactory()
	codec := testapi.Federation.Codec()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.Printer = &testPrinter{}
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
					return nil, fmt.Errorf("Unexpected cluster object\n\tgot: %#v\n\twant: %#v", got, want)
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

func newFakeJoinFederationConfig(name, server, token, kubeconfigGlobal, kubeconfigExplicit string) (JoinFederationConfig, error) {
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = kubeconfigGlobal
	pathOptions.EnvVar = ""
	pathOptions.LoadingRules.ExplicitPath = kubeconfigExplicit

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
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: "federation-system",
		},
		Data: map[string][]byte{
			"kubeconfig": configBytes,
		},
	}

	f, tf, codec, _ := NewAPIFactory()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/federation-system/secrets" && m == http.MethodPost:
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
					return nil, fmt.Errorf("Unexpected cluster object\n\tgot: %#v\n\twant: %#v", got, secretObject)
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: defaultHeader(), Body: objBody(codec, &secretObject)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
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

func fakeCluster(name, server string) federationapi.Cluster {
	return federationapi.Cluster{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: federationapi.ClusterSpec{
			ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
				{
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
