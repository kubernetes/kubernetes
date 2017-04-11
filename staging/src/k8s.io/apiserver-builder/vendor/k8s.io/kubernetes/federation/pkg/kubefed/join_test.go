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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	kubefedtesting "k8s.io/kubernetes/federation/pkg/kubefed/testing"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
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
		clusterCtx         string
		secret             string
		server             string
		token              string
		kubeconfigGlobal   string
		kubeconfigExplicit string
		expectedServer     string
		expectedErr        string
	}{
		{
			cluster:            "syndicate",
			clusterCtx:         "",
			secret:             "",
			server:             "https://10.20.30.40",
			token:              "badge",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			expectedServer:     "https://10.20.30.40",
			expectedErr:        "",
		},
		{
			cluster:            "ally",
			clusterCtx:         "",
			secret:             "",
			server:             "ally256.example.com:80",
			token:              "souvenir",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: fakeKubeFiles[1],
			expectedServer:     "https://ally256.example.com:80",
			expectedErr:        "",
		},
		{
			cluster:            "confederate",
			clusterCtx:         "",
			secret:             "",
			server:             "10.8.8.8",
			token:              "totem",
			kubeconfigGlobal:   fakeKubeFiles[1],
			kubeconfigExplicit: fakeKubeFiles[2],
			expectedServer:     "https://10.8.8.8",
			expectedErr:        "",
		},
		{
			cluster:            "associate",
			clusterCtx:         "confederate",
			secret:             "confidential",
			server:             "10.8.8.8",
			token:              "totem",
			kubeconfigGlobal:   fakeKubeFiles[1],
			kubeconfigExplicit: fakeKubeFiles[2],
			expectedServer:     "https://10.8.8.8",
			expectedErr:        "",
		},
		{
			cluster:            "affiliate",
			clusterCtx:         "",
			secret:             "",
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
		f := testJoinFederationFactory(tc.cluster, tc.secret, tc.expectedServer)
		buf := bytes.NewBuffer([]byte{})

		hostFactory, err := fakeJoinHostFactory(tc.cluster, tc.clusterCtx, tc.secret, tc.server, tc.token)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		targetClusterFactory, err := fakeJoinTargetClusterFactory(tc.cluster, tc.clusterCtx)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		targetClusterContext := tc.clusterCtx
		if targetClusterContext == "" {
			targetClusterContext = tc.cluster
		}
		adminConfig, err := kubefedtesting.NewFakeAdminConfig(hostFactory, targetClusterFactory, targetClusterContext, tc.kubeconfigGlobal)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		cmd := NewCmdJoin(f, buf, adminConfig)

		cmd.Flags().Set("kubeconfig", tc.kubeconfigExplicit)
		cmd.Flags().Set("host-cluster-context", "substrate")
		if tc.clusterCtx != "" {
			cmd.Flags().Set("cluster-context", tc.clusterCtx)
		}
		if tc.secret != "" {
			cmd.Flags().Set("secret-name", tc.secret)
		}

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

func testJoinFederationFactory(clusterName, secretName, server string) cmdutil.Factory {
	if secretName == "" {
		secretName = clusterName
	}

	want := fakeCluster(clusterName, secretName, server)
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	codec := testapi.Federation.Codec()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.Client = &fake.RESTClient{
		APIRegistry:          api.Registry,
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
				if !apiequality.Semantic.DeepEqual(got, want) {
					return nil, fmt.Errorf("Unexpected cluster object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, want))
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

func fakeJoinHostFactory(clusterName, clusterCtx, secretName, server, token string) (cmdutil.Factory, error) {
	if clusterCtx == "" {
		clusterCtx = clusterName
	}
	if secretName == "" {
		secretName = clusterName
	}

	kubeconfig := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			clusterCtx: {
				Server: server,
			},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			clusterCtx: {
				Token: token,
			},
		},
		Contexts: map[string]*clientcmdapi.Context{
			clusterCtx: {
				Cluster:  clusterCtx,
				AuthInfo: clusterCtx,
			},
		},
		CurrentContext: clusterCtx,
	}
	configBytes, err := clientcmd.Write(kubeconfig)
	if err != nil {
		return nil, err
	}

	secretObject := v1.Secret{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Secret",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: util.DefaultFederationSystemNamespace,
		},
		Data: map[string][]byte{
			"kubeconfig": configBytes,
		},
	}

	cmName := "controller-manager"
	deploymentList := v1beta1.DeploymentList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "DeploymentList",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		Items: []v1beta1.Deployment{
			{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Deployment",
					APIVersion: testapi.Extensions.GroupVersion().String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      cmName,
					Namespace: util.DefaultFederationSystemNamespace,
					Annotations: map[string]string{
						util.FedDomainMapKey: fmt.Sprintf("%s=%s", clusterCtx, "test-dns-zone"),
					},
				},
			},
		},
	}

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	extensionCodec := testapi.Extensions.Codec()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.ClientConfig = kubefedtesting.DefaultClientConfig()
	tf.Client = &fake.RESTClient{
		APIRegistry:          api.Registry,
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
				if !apiequality.Semantic.DeepEqual(got, secretObject) {
					return nil, fmt.Errorf("Unexpected secret object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, secretObject))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &secretObject)}, nil
			case p == "/apis/extensions/v1beta1/namespaces/federation-system/deployments" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(extensionCodec, &deploymentList)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
	return f, nil
}

func fakeJoinTargetClusterFactory(clusterName, clusterCtx string) (cmdutil.Factory, error) {
	if clusterCtx == "" {
		clusterCtx = clusterName
	}

	configmapObject := v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      util.KubeDnsConfigmapName,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			util.FedDomainMapKey: fmt.Sprintf("%s=%s", clusterCtx, "test-dns-zone"),
		},
	}

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.ClientConfig = kubefedtesting.DefaultClientConfig()
	tf.Client = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/kube-system/configmaps/" && m == http.MethodPost:
				body, err := ioutil.ReadAll(req.Body)
				if err != nil {
					return nil, err
				}
				var got v1.ConfigMap
				_, _, err = codec.Decode(body, nil, &got)
				if err != nil {
					return nil, err
				}
				if !apiequality.Semantic.DeepEqual(got, configmapObject) {
					return nil, fmt.Errorf("Unexpected configmap object\n\tDiff: %s", diff.ObjectGoPrintDiff(got, configmapObject))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &configmapObject)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
	return f, nil
}

func fakeCluster(clusterName, secretName, server string) federationapi.Cluster {
	return federationapi.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: clusterName,
		},
		Spec: federationapi.ClusterSpec{
			ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
				{
					ClientCIDR:    defaultClientCIDR,
					ServerAddress: server,
				},
			},
			SecretRef: &v1.LocalObjectReference{
				Name: secretName,
			},
		},
	}
}
