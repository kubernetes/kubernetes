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
	"os"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/federation/apis/federation"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	kubefedtesting "k8s.io/kubernetes/federation/pkg/kubefed/testing"
	"k8s.io/kubernetes/federation/pkg/kubefed/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	k8srbacv1beta1 "k8s.io/kubernetes/pkg/apis/rbac/v1beta1"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const (
	// testFederationName is a name to use for the federation in tests. Since the federation
	// name is recovered from the federation itself, this constant is an appropriate
	// functional replica.
	testFederationName = "test-federation"

	zoneName      = "test-dns-zone"
	coreDNSServer = "11.22.33.44:53"
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
		dnsProvider        string
		isRBACAPIAvailable bool
	}{
		{
			cluster:            "syndicate",
			clusterCtx:         "",
			server:             "https://10.20.30.40",
			token:              "badge",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			expectedServer:     "https://10.20.30.40",
			expectedErr:        "",
			dnsProvider:        util.FedDNSProviderCoreDNS,
			isRBACAPIAvailable: true,
		},
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
			isRBACAPIAvailable: false,
		},
		{
			cluster:            "ally",
			clusterCtx:         "",
			server:             "ally256.example.com:80",
			token:              "souvenir",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: fakeKubeFiles[1],
			expectedServer:     "https://ally256.example.com:80",
			expectedErr:        "",
			isRBACAPIAvailable: true,
		},
		{
			cluster:            "confederate",
			clusterCtx:         "",
			server:             "10.8.8.8",
			token:              "totem",
			kubeconfigGlobal:   fakeKubeFiles[1],
			kubeconfigExplicit: fakeKubeFiles[2],
			expectedServer:     "https://10.8.8.8",
			expectedErr:        "",
			isRBACAPIAvailable: true,
		},
		{
			cluster:            "associate",
			clusterCtx:         "confederate",
			server:             "10.8.8.8",
			token:              "totem",
			kubeconfigGlobal:   fakeKubeFiles[1],
			kubeconfigExplicit: fakeKubeFiles[2],
			expectedServer:     "https://10.8.8.8",
			expectedErr:        "",
			isRBACAPIAvailable: true,
		},
		{
			cluster:            "affiliate",
			clusterCtx:         "",
			server:             "https://10.20.30.40",
			token:              "badge",
			kubeconfigGlobal:   fakeKubeFiles[0],
			kubeconfigExplicit: "",
			expectedServer:     "https://10.20.30.40",
			expectedErr:        fmt.Sprintf("error: cluster context %q not found", "affiliate"),
			isRBACAPIAvailable: true,
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
	}

	for i, tc := range testCases {
		cmdErrMsg = ""
		f := testJoinFederationFactory(tc.cluster, tc.secret, tc.expectedServer, tc.isRBACAPIAvailable)
		buf := bytes.NewBuffer([]byte{})

		hostFactory, err := fakeJoinHostFactory(tc.cluster, tc.clusterCtx, tc.secret, tc.server, tc.token, tc.dnsProvider, tc.isRBACAPIAvailable)
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}

		// The fake discovery client caches results by default, so invalidate it by modifying the temporary directory.
		// Refer to pkg/kubectl/cmd/testing/fake (fakeAPIFactory.DiscoveryClient()) for details of tmpDir
		tmpDirPath, err := ioutil.TempDir("", "")
		if err != nil {
			t.Fatalf("[%d] unexpected error: %v", i, err)
		}
		defer os.Remove(tmpDirPath)

		targetClusterFactory, err := fakeJoinTargetClusterFactory(tc.cluster, tc.clusterCtx, tc.dnsProvider, tmpDirPath, tc.isRBACAPIAvailable)
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

func testJoinFederationFactory(clusterName, secretName, server string, isRBACAPIAvailable bool) cmdutil.Factory {

	want := fakeCluster(clusterName, secretName, server, isRBACAPIAvailable)
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
				// If the secret name was generated, test it separately.
				if secretName == "" {
					if got.Spec.SecretRef.Name == "" {
						return nil, fmt.Errorf("expected a generated secret name, got \"\"")
					}
					got.Spec.SecretRef.Name = ""
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

func fakeJoinHostFactory(clusterName, clusterCtx, secretName, server, token, dnsProvider string, isRBACAPIAvailable bool) (cmdutil.Factory, error) {
	if clusterCtx == "" {
		clusterCtx = clusterName
	}

	placeholderSecretName := secretName
	if placeholderSecretName == "" {
		placeholderSecretName = "secretName"
	}
	var secretObject v1.Secret
	if isRBACAPIAvailable {
		secretObject = v1.Secret{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Secret",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      placeholderSecretName,
				Namespace: util.DefaultFederationSystemNamespace,
				Annotations: map[string]string{
					federation.FederationNameAnnotation: testFederationName,
					federation.ClusterNameAnnotation:    clusterName,
				},
			},
			Data: map[string][]byte{
				"ca.crt": []byte("cert"),
				"token":  []byte("token"),
			},
		}
	} else {
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

		secretObject = v1.Secret{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Secret",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      placeholderSecretName,
				Namespace: util.DefaultFederationSystemNamespace,
				Annotations: map[string]string{
					federation.FederationNameAnnotation: testFederationName,
					federation.ClusterNameAnnotation:    clusterName,
				},
			},
			Data: map[string][]byte{
				"kubeconfig": configBytes,
			},
		}
	}

	cmName := "controller-manager"
	deployment := v1beta1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: testapi.Extensions.GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      cmName,
			Namespace: util.DefaultFederationSystemNamespace,
			Annotations: map[string]string{
				util.FedDomainMapKey:                fmt.Sprintf("%s=%s", clusterCtx, zoneName),
				federation.FederationNameAnnotation: testFederationName,
			},
		},
	}
	if dnsProvider == util.FedDNSProviderCoreDNS {
		deployment.Annotations[util.FedDNSZoneName] = zoneName
		deployment.Annotations[util.FedNameServer] = coreDNSServer
		deployment.Annotations[util.FedDNSProvider] = util.FedDNSProviderCoreDNS
	}
	deploymentList := v1beta1.DeploymentList{Items: []v1beta1.Deployment{deployment}}

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

				// If the secret name was generated, test it separately.
				if secretName == "" {
					if got.Name == "" {
						return nil, fmt.Errorf("expected a generated secret name, got \"\"")
					}
					got.Name = placeholderSecretName
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

func serviceAccountName(clusterName string) string {
	return fmt.Sprintf("%s-substrate", clusterName)
}

func fakeJoinTargetClusterFactory(clusterName, clusterCtx, dnsProvider, tmpDirPath string, isRBACAPIAvailable bool) (cmdutil.Factory, error) {
	if clusterCtx == "" {
		clusterCtx = clusterName
	}

	configmapObject := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      util.KubeDnsConfigmapName,
			Namespace: metav1.NamespaceSystem,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: testFederationName,
				federation.ClusterNameAnnotation:    clusterName,
			},
		},
		Data: map[string]string{
			util.FedDomainMapKey: fmt.Sprintf("%s=%s", clusterCtx, zoneName),
		},
	}

	saSecretName := "serviceaccountsecret"
	saSecret := v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saSecretName,
			Namespace: util.DefaultFederationSystemNamespace,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: testFederationName,
				federation.ClusterNameAnnotation:    clusterName,
			},
		},
		Data: map[string][]byte{
			"ca.crt": []byte("cert"),
			"token":  []byte("token"),
		},
		Type: v1.SecretTypeServiceAccountToken,
	}

	saName := serviceAccountName(clusterName)

	serviceAccount := v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name: saName,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: testFederationName,
				federation.ClusterNameAnnotation:    clusterName,
			},
		},
		Secrets: []v1.ObjectReference{
			{Name: saSecretName},
		},
	}
	if dnsProvider == util.FedDNSProviderCoreDNS {
		annotations := map[string]string{
			util.FedDNSProvider: util.FedDNSProviderCoreDNS,
			util.FedDNSZoneName: zoneName,
			util.FedNameServer:  coreDNSServer,
		}
		configmapObject = populateStubDomainsIfRequiredTest(configmapObject, annotations)
	}

	namespace := v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "federation-system",
			Annotations: map[string]string{
				federation.FederationNameAnnotation: testFederationName,
				federation.ClusterNameAnnotation:    clusterName,
			},
		},
	}

	roleName := util.ClusterRoleName(testFederationName, saName)
	clusterRole := rbacv1beta1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name:      roleName,
			Namespace: util.DefaultFederationSystemNamespace,
			Annotations: map[string]string{
				federation.FederationNameAnnotation: testFederationName,
				federation.ClusterNameAnnotation:    clusterName,
			},
		},
		Rules: []rbacv1beta1.PolicyRule{
			k8srbacv1beta1.NewRule(rbacv1beta1.VerbAll).Groups(rbacv1beta1.APIGroupAll).Resources(rbacv1beta1.ResourceAll).RuleOrDie(),
		},
	}

	clusterRoleBinding, err := k8srbacv1beta1.NewClusterBinding(roleName).SAs(util.DefaultFederationSystemNamespace, saName).Binding()
	if err != nil {
		return nil, err
	}

	testGroup := metav1.APIGroup{
		Name: "testAPIGroup",
		Versions: []metav1.GroupVersionForDiscovery{
			{
				GroupVersion: "testAPIGroup/testAPIVersion",
				Version:      "testAPIVersion",
			},
		},
	}
	apiGroupList := &metav1.APIGroupList{}
	apiGroupList.Groups = append(apiGroupList.Groups, testGroup)
	if isRBACAPIAvailable {
		rbacGroup := metav1.APIGroup{
			Name: rbacv1beta1.GroupName,
			Versions: []metav1.GroupVersionForDiscovery{
				{
					GroupVersion: rbacv1beta1.GroupName + "/v1beta1",
					Version:      "v1beta1",
				},
			},
		}
		apiGroupList.Groups = append(apiGroupList.Groups, rbacGroup)
	}

	f, tf, codec, _ := cmdtesting.NewAPIFactory()
	defaultCodec := testapi.Default.Codec()
	rbacCodec := testapi.Rbac.Codec()
	ns := dynamic.ContentConfig().NegotiatedSerializer
	tf.TmpDir = tmpDirPath
	tf.ClientConfig = kubefedtesting.DefaultClientConfig()
	tf.Client = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m, r := req.URL.Path, req.Method, isRBACAPIAvailable; {
			case p == "/api/v1/namespaces" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(defaultCodec, &namespace)}, nil

			case p == "/api" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, &metav1.APIVersions{})}, nil
			case p == "/apis" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, apiGroupList)}, nil

			case p == fmt.Sprintf("/api/v1/namespaces/federation-system/serviceaccounts/%s", saName) && m == http.MethodGet && r:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(defaultCodec, &serviceAccount)}, nil
			case p == "/api/v1/namespaces/federation-system/serviceaccounts" && m == http.MethodPost && r:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(defaultCodec, &serviceAccount)}, nil

			case p == "/apis/rbac.authorization.k8s.io/v1beta1/clusterroles" && m == http.MethodPost && r:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(rbacCodec, &clusterRole)}, nil
			case p == "/apis/rbac.authorization.k8s.io/v1beta1/clusterrolebindings" && m == http.MethodPost && r:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(rbacCodec, &clusterRoleBinding)}, nil

			case p == "/api/v1/namespaces/federation-system/secrets/serviceaccountsecret" && m == http.MethodGet && r:
				return &http.Response{StatusCode: http.StatusOK, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(defaultCodec, &saSecret)}, nil

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
				if !apiequality.Semantic.DeepEqual(&got, configmapObject) {
					return nil, fmt.Errorf("Unexpected configmap object\n\tDiff: %s", diff.ObjectGoPrintDiff(&got, configmapObject))
				}
				return &http.Response{StatusCode: http.StatusCreated, Header: kubefedtesting.DefaultHeader(), Body: kubefedtesting.ObjBody(codec, configmapObject)}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}

	return f, nil
}

func fakeCluster(clusterName, secretName, server string, isRBACAPIAvailable bool) federationapi.Cluster {
	cluster := federationapi.Cluster{
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
	if isRBACAPIAvailable {
		saName := serviceAccountName(clusterName)
		annotations := map[string]string{
			kubectl.ServiceAccountNameAnnotation: saName,
			kubectl.ClusterRoleNameAnnotation:    util.ClusterRoleName(testFederationName, saName),
		}
		cluster.ObjectMeta.SetAnnotations(annotations)
	}
	return cluster
}

// TODO: Reuse the function populateStubDomainsIfRequired once that function is converted to use versioned objects.
func populateStubDomainsIfRequiredTest(configMap *v1.ConfigMap, annotations map[string]string) *v1.ConfigMap {
	dnsProvider := annotations[util.FedDNSProvider]
	dnsZoneName := annotations[util.FedDNSZoneName]
	nameServer := annotations[util.FedNameServer]

	if dnsProvider != util.FedDNSProviderCoreDNS || dnsZoneName == "" || nameServer == "" {
		return configMap
	}
	configMap.Data[util.KubeDnsStubDomains] = fmt.Sprintf(`{"%s":["%s"]}`, dnsZoneName, nameServer)
	return configMap
}
