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
	"io/ioutil"
	"net/http"
	"testing"

	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	kubefedtesting "k8s.io/kubernetes/federation/pkg/kubefed/testing"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/restclient/fake"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func TestJoinFederation(t *testing.T) {
	cmdErrMsg := ""
	cmdutil.BehaviorOnFatal(func(str string, code int) {
		cmdErrMsg = str
	})

	fakeKubeFiles, err := kubefedtesting.FakeKubeconfigFiles()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer kubefedtesting.RemoveFakeKubeconfigFiles(fakeKubeFiles)

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

		adminConfig, err := kubefedtesting.NewFakeAdminConfig(hostFactory, tc.kubeconfigGlobal)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		cmd := NewCmdJoin(f, buf, adminConfig)

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
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &want)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
	tf.Namespace = "test"
	return f
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
			Namespace: util.DefaultFederationSystemNamespace,
		},
		Data: map[string][]byte{
			"kubeconfig": configBytes,
		},
	}

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.ClientConfig = kubefedtesting.DefaultClientConfig()
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
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &secretObject)}, nil
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
