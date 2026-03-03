/*
Copyright 2025 The Kubernetes Authors.

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

package conditionalauthorization

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
	"time"

	authorizationv1 "k8s.io/api/authorization/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilcompatibility "k8s.io/apiserver/pkg/util/compatibility"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/util/cert"
	basecompatibility "k8s.io/component-base/compatibility"
	"k8s.io/component-base/featuregate"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	aggregatorclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	kastesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	wardlev1alpha1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	"k8s.io/sample-apiserver/pkg/apiserver"
	sampleserver "k8s.io/sample-apiserver/pkg/cmd/server"
	wardleclient "k8s.io/sample-apiserver/pkg/generated/clientset/versioned"
	netutils "k8s.io/utils/net"
)

// TestAggregatedConditionalAuthorization tests that conditional authorization
// works end-to-end through an aggregated API server. The flow is:
//
//  1. Client creates Flunder → kube-apiserver aggregator → wardle
//  2. Wardle delegates auth via SAR → kube-apiserver → webhook → conditional webhook
//  3. Conditional decision flows back to wardle
//  4. Wardle's AuthorizationConditionsEnforcer evaluates conditions via ACR → conditional webhook
//  5. CEL conditions are evaluated against the actual Flunder object
func TestAggregatedConditionalAuthorization(t *testing.T) {
	// makes the kube-apiserver very responsive
	dynamiccertificates.FileRefreshDuration = 1 * time.Second

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)

	dir := t.TempDir()

	// Start a webhook server that handles both SAR and ACR
	webhookServer := newWebhookServer(t)
	defer webhookServer.server.Close()

	// Set a default sarHandler that allows everything during setup.
	// Individual test cases override this with user-specific handlers.
	webhookServer.handler.sarHandler = func(sar *authorizationv1.SubjectAccessReview) {
		sar.Status.Allowed = true
		sar.Status.Reason = "default allow during setup"
	}

	// Write a kubeconfig for the webhook server with two contexts:
	// - "default" context for SAR on /authorize
	// - "conditions" context for ACR on /conditionsreview
	webhookKubeconfigPath := filepath.Join(dir, "webhook-kubeconfig.yaml")
	if err := os.WriteFile(webhookKubeconfigPath, []byte(fmt.Sprintf(`
apiVersion: v1
kind: Config
clusters:
- name: authorize
  cluster:
    server: %q
    insecure-skip-tls-verify: true
- name: conditions
  cluster:
    server: %q
    insecure-skip-tls-verify: true
contexts:
- name: default
  context:
    cluster: authorize
    user: test
- name: conditions
  context:
    cluster: conditions
    user: test
current-context: default
users:
- name: test
`, webhookServer.server.URL+"/authorize", webhookServer.server.URL+"/conditionsreview")), 0644); err != nil {
		t.Fatal(err)
	}

	// Write an AuthorizationConfiguration file for kube-apiserver
	authzConfigPath := filepath.Join(dir, "authz-config.yaml")
	if err := os.WriteFile(authzConfigPath, []byte(fmt.Sprintf(`
apiVersion: apiserver.config.k8s.io/v1beta1
kind: AuthorizationConfiguration
authorizers:
- type: Webhook
  name: conditional-webhook
  webhook:
    timeout: 10s
    subjectAccessReviewVersion: v1
    matchConditionSubjectAccessReviewVersion: v1
    failurePolicy: NoOpinion
    authorizedTTL: 1ms
    unauthorizedTTL: 1ms
    connectionInfo:
      type: KubeConfigFile
      kubeConfigFile: %q
    conditionsReview:
      kubeConfigContextName: conditions
      version: v1alpha1
- type: RBAC
  name: rbac
`, webhookKubeconfigPath)), 0644); err != nil {
		t.Fatal(err)
	}

	// Create a listener for the wardle server
	listener, wardlePort, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0", net.ListenConfig{})
	if err != nil {
		t.Fatal(err)
	}

	// Override the service resolver to point to our local wardle server
	namespace := "wardle-conditional-auth-ns"
	t.Cleanup(app.SetServiceResolverForTests(staticURLServiceResolver(fmt.Sprintf("https://127.0.0.1:%d", wardlePort))))

	// Start kube-apiserver with AuthorizationConfiguration + conditional authorization
	kasFlags := []string{
		"--feature-gates=ConditionalAuthorization=true",
		"--runtime-config=authorization.k8s.io/v1alpha1=true",
		"--authorization-config=" + authzConfigPath,
		"--enable-admission-plugins=AuthorizationConditionsEnforcer",
	}
	testServer := kastesting.StartTestServerOrDie(t,
		&kastesting.TestServerInstanceOptions{
			EnableCertAuth: true,
		},
		kasFlags,
		framework.SharedEtcd(),
	)
	t.Cleanup(testServer.TearDownFn)

	kubeConfig := getKubeConfig(testServer)
	kubeClient := clientset.NewForConfigOrDie(kubeConfig)

	// Create the namespace and service required for the APIService
	_, err = kubeClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: namespace},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = kubeClient.CoreV1().Services(namespace).Create(ctx, &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "api"},
		Spec: corev1.ServiceSpec{
			ExternalName: "needs-to-be-non-empty",
			Type:         corev1.ServiceTypeExternalName,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Create wardle server options (following the pattern from apiserver_test.go)
	componentGlobalsRegistry := basecompatibility.NewComponentGlobalsRegistry()
	_, _ = componentGlobalsRegistry.ComponentGlobalsOrRegister(
		basecompatibility.DefaultKubeComponent,
		utilcompatibility.DefaultKubeEffectiveVersionForTest(),
		utilfeature.DefaultFeatureGate.DeepCopy(),
	)
	wardleBinaryVersion := "1.2.0"
	_, _ = componentGlobalsRegistry.ComponentGlobalsOrRegister(
		apiserver.WardleComponentName,
		basecompatibility.NewEffectiveVersionFromString(wardleBinaryVersion, "", ""),
		featuregate.NewVersionedFeatureGate(version.MustParse(wardleBinaryVersion)),
	)

	wardleOptions := sampleserver.NewWardleServerOptions(os.Stdout, os.Stderr)
	wardleOptions.ComponentGlobalsRegistry = componentGlobalsRegistry
	wardleOptions.AlternateDNS = []string{
		fmt.Sprintf("api.%s.svc", namespace),
	}
	wardleOptions.RecommendedOptions.SecureServing.Listener = listener
	wardleOptions.RecommendedOptions.SecureServing.BindAddress = netutils.ParseIPSloppy("127.0.0.1")

	certDir := filepath.Join(dir, "wardle-certs")

	// Write kubeconfig for wardle → kube-apiserver connection (for authentication + kubeconfig)
	wardleToKASKubeConfigFile := writeKubeConfigForConnection(t, rest.CopyConfig(kubeConfig))
	t.Cleanup(func() { os.Remove(wardleToKASKubeConfigFile) })

	wardleAuthzKubeConfigFile := writeKubeConfigForConnectionWithConditionsContext(t,
		rest.CopyConfig(kubeConfig))
	t.Cleanup(func() { os.Remove(wardleAuthzKubeConfigFile) })

	// Start wardle with conditional authorization support
	go func() {
		args := []string{
			"--authentication-kubeconfig", wardleToKASKubeConfigFile,
			"--authorization-kubeconfig", wardleAuthzKubeConfigFile,
			"--authorization-webhook-cache-authorized-ttl", "1ms",
			"--authorization-webhook-cache-unauthorized-ttl", "1ms",
			"--enable-admission-plugins", "AuthorizationConditionsEnforcer",
			"--feature-gates", "ConditionalAuthorization=true",
			"--etcd-servers", framework.GetEtcdURL(),
			"--cert-dir", certDir,
			"--kubeconfig", wardleToKASKubeConfigFile,
			"--emulated-version", fmt.Sprintf("wardle=%s", wardleBinaryVersion),
		}
		wardleCmd := sampleserver.NewCommandStartWardleServer(ctx, wardleOptions, false)
		wardleCmd.SetArgs(args)
		if err := wardleCmd.Execute(); err != nil {
			t.Error(err)
		}
	}()

	// Wait for wardle to be running
	wardleCAFile := filepath.Join(certDir, "apiserver.crt")
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (done bool, err error) {
		if _, err := os.Stat(wardleCAFile); os.IsNotExist(err) {
			return false, nil
		}
		directConfig := rest.AnonymousClientConfig(rest.CopyConfig(kubeConfig))
		directConfig.CAFile = wardleCAFile
		directConfig.CAData = nil
		directConfig.ServerName = ""
		directConfig.BearerToken = kubeConfig.BearerToken
		directConfig.Host = fmt.Sprintf("https://127.0.0.1:%d", wardlePort)
		directClient, err := clientset.NewForConfig(directConfig)
		if err != nil {
			return false, nil
		}
		healthStatus := 0
		directClient.Discovery().RESTClient().Get().AbsPath("/healthz").Do(ctx).StatusCode(&healthStatus)
		return healthStatus == http.StatusOK, nil
	})
	if err != nil {
		t.Fatalf("wardle server did not become healthy: %v", err)
	}

	// Register the APIService for wardle
	wardleCA, err := os.ReadFile(wardleCAFile)
	if err != nil {
		t.Fatal(err)
	}
	aggClient := aggregatorclient.NewForConfigOrDie(kubeConfig)
	_, err = aggClient.ApiregistrationV1().APIServices().Create(ctx, &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1alpha1.wardle.example.com"},
		Spec: apiregistrationv1.APIServiceSpec{
			Service: &apiregistrationv1.ServiceReference{
				Namespace: namespace,
				Name:      "api",
			},
			Group:                "wardle.example.com",
			Version:              "v1alpha1",
			CABundle:             wardleCA,
			GroupPriorityMinimum: 200,
			VersionPriority:      200,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Wait for the APIService to be available
	err = wait.Poll(time.Second, 60*time.Second, func() (done bool, err error) {
		apiService, err := aggClient.ApiregistrationV1().APIServices().Get(ctx, "v1alpha1.wardle.example.com", metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		for _, condition := range apiService.Status.Conditions {
			if condition.Type == apiregistrationv1.Available && condition.Status == apiregistrationv1.ConditionTrue {
				return true, nil
			}
		}
		t.Logf("APIService not yet available: %v", apiService.Status.Conditions)
		return false, nil
	})
	if err != nil {
		t.Fatalf("APIService did not become available: %v", err)
	}

	// Verify wardle resources are in discovery
	err = wait.Poll(time.Second, 30*time.Second, func() (done bool, err error) {
		apiResources, err := kubeClient.Discovery().ServerResourcesForGroupVersion("wardle.example.com/v1alpha1")
		if err != nil {
			return false, nil
		}
		resources := make([]string, 0, len(apiResources.APIResources))
		for _, r := range apiResources.APIResources {
			resources = append(resources, r.Name)
		}
		sort.Strings(resources)
		if !reflect.DeepEqual([]string{"fischers", "flunders"}, resources) {
			t.Logf("unexpected resources: %v", resources)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("wardle resources not found in discovery: %v", err)
	}

	// Create test namespace in wardle (flunders are namespaced)
	wardleNS := "test-ns"
	_, err = kubeClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: wardleNS},
	}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatal(err)
	}

	// userSARHandler wraps a per-user sarHandler so that only SARs for the
	// specified test user get the test-specific behavior. All other users
	// (system:kube-aggregator, system:anonymous, etc.) are unconditionally allowed
	// so that aggregator probes, discovery, and other system requests succeed.
	userSARHandler := func(testUser string, handler func(sar *authorizationv1.SubjectAccessReview)) func(sar *authorizationv1.SubjectAccessReview) {
		return func(sar *authorizationv1.SubjectAccessReview) {
			if sar.Spec.User != testUser {
				sar.Status.Allowed = true
				sar.Status.Reason = "system user allowed"
				return
			}
			handler(sar)
		}
	}

	testCases := []struct {
		name            string
		user            string
		webhookBehavior func(ws *webhookServerHandler)
		flunder         *wardlev1alpha1.Flunder
		expectAllowed   bool
	}{
		{
			name: "aggregated unconditional allow",
			user: "agg-allow-user",
			webhookBehavior: func(ws *webhookServerHandler) {
				ws.sarHandler = userSARHandler("agg-allow-user", func(sar *authorizationv1.SubjectAccessReview) {
					sar.Status.Allowed = true
					sar.Status.Reason = "unconditionally allowed"
				})
			},
			flunder: &wardlev1alpha1.Flunder{
				ObjectMeta: metav1.ObjectMeta{Name: "allowed-flunder"},
			},
			expectAllowed: true,
		},
		{
			name: "aggregated unconditional deny",
			user: "agg-deny-user",
			webhookBehavior: func(ws *webhookServerHandler) {
				ws.sarHandler = userSARHandler("agg-deny-user", func(sar *authorizationv1.SubjectAccessReview) {
					sar.Status.Allowed = false
					sar.Status.Denied = true
					sar.Status.Reason = "unconditionally denied"
				})
			},
			flunder: &wardlev1alpha1.Flunder{
				ObjectMeta: metav1.ObjectMeta{Name: "denied-flunder"},
			},
			expectAllowed: false,
		},
		{
			name: "aggregated conditional allow by name",
			user: "agg-cond-allow-user",
			webhookBehavior: func(ws *webhookServerHandler) {
				ws.sarHandler = userSARHandler("agg-cond-allow-user", func(sar *authorizationv1.SubjectAccessReview) {
					sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
						{
							ConditionsType: "k8s.io/authorization-cel",
							Conditions: []authorizationv1.SubjectAccessReviewCondition{
								{
									ID:          "allow-safe-prefix",
									Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
									Condition:   `object.metadata.name.startsWith("safe-")`,
									Description: "only allow flunders with safe- prefix",
								},
							},
						},
					}
				})
				ws.acrHandler = acrEvaluateCEL(ws.t, "k8s.io/authorization-cel")
			},
			flunder: &wardlev1alpha1.Flunder{
				ObjectMeta: metav1.ObjectMeta{Name: "safe-flunder"},
			},
			expectAllowed: true,
		},
		{
			name: "aggregated conditional deny by name mismatch",
			user: "agg-cond-deny-user",
			webhookBehavior: func(ws *webhookServerHandler) {
				ws.sarHandler = userSARHandler("agg-cond-deny-user", func(sar *authorizationv1.SubjectAccessReview) {
					sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
						{
							ConditionsType: "k8s.io/authorization-cel",
							Conditions: []authorizationv1.SubjectAccessReviewCondition{
								{
									ID:          "allow-safe-prefix",
									Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
									Condition:   `object.metadata.name.startsWith("safe-")`,
									Description: "only allow flunders with safe- prefix",
								},
							},
						},
					}
				})
				ws.acrHandler = acrEvaluateCEL(ws.t, "k8s.io/authorization-cel")
			},
			flunder: &wardlev1alpha1.Flunder{
				ObjectMeta: metav1.ObjectMeta{Name: "unsafe-flunder"},
			},
			expectAllowed: false,
		},
		{
			name: "aggregated conditional deny by label",
			user: "agg-cond-label-deny-user",
			webhookBehavior: func(ws *webhookServerHandler) {
				ws.sarHandler = userSARHandler("agg-cond-label-deny-user", func(sar *authorizationv1.SubjectAccessReview) {
					sar.Status.ConditionalDecisionChain = []authorizationv1.SubjectAccessReviewAuthorizationDecision{
						{
							ConditionsType: "k8s.io/authorization-cel",
							Conditions: []authorizationv1.SubjectAccessReviewCondition{
								{
									ID:          "allow-all",
									Effect:      authorizationv1.SubjectAccessReviewConditionEffectAllow,
									Condition:   "true",
									Description: "base allow",
								},
								{
									ID:     "deny-restricted-label",
									Effect: authorizationv1.SubjectAccessReviewConditionEffectDeny,
									Condition: `has(object.metadata.labels) && ` +
										`has(object.metadata.labels.restricted) && ` +
										`object.metadata.labels.restricted == "true"`,
									Description: "deny restricted labels",
								},
							},
						},
					}
				})
				ws.acrHandler = acrEvaluateCEL(ws.t, "k8s.io/authorization-cel")
			},
			flunder: &wardlev1alpha1.Flunder{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "restricted-flunder",
					Labels: map[string]string{"restricted": "true"},
				},
			},
			expectAllowed: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Configure the webhook for this test case
			tc.webhookBehavior(webhookServer.handler)

			// Create an impersonated client for the test user via kube-apiserver
			impersonationConfig := rest.CopyConfig(kubeConfig)
			impersonationConfig.Impersonate.UserName = tc.user
			wardleClient := wardleclient.NewForConfigOrDie(impersonationConfig)

			// Create the Flunder via the aggregated path
			tc.flunder.Namespace = wardleNS
			_, err := wardleClient.WardleV1alpha1().Flunders(wardleNS).Create(ctx, tc.flunder, metav1.CreateOptions{})

			if tc.expectAllowed {
				if err != nil {
					t.Fatalf("expected request to be allowed, got error: %v", err)
				}
			} else {
				if err == nil {
					t.Fatalf("expected request to be denied, got success")
				}
				if !apierrors.IsForbidden(err) && !apierrors.IsUnauthorized(err) {
					t.Fatalf("expected Forbidden or Unauthorized error, got: %v", err)
				}
			}
		})
	}
}

// staticURLServiceResolver returns a ServiceResolver that always returns the given URL.
type staticURLServiceResolver string

func (u staticURLServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return url.Parse(string(u))
}

func getKubeConfig(testServer *kastesting.TestServer) *rest.Config {
	kubeClientConfig := rest.CopyConfig(testServer.ClientConfig)
	kubeClientConfig.ContentType = ""
	kubeClientConfig.AcceptContentTypes = ""
	return kubeClientConfig
}

func writeKubeConfigForConnection(t *testing.T, kubeClientConfig *rest.Config) string {
	t.Helper()

	// Get the real serving cert from the server
	servingCerts, _, err := cert.GetServingCertificatesForURL(kubeClientConfig.Host, "")
	if err != nil {
		t.Fatal(err)
	}
	encodedServing, err := cert.EncodeCertificates(servingCerts...)
	if err != nil {
		t.Fatal(err)
	}
	kubeClientConfig.CAData = encodedServing

	config := clientcmdapi.NewConfig()

	credentials := clientcmdapi.NewAuthInfo()
	credentials.Token = kubeClientConfig.BearerToken
	credentials.ClientCertificate = kubeClientConfig.TLSClientConfig.CertFile
	if len(credentials.ClientCertificate) == 0 {
		credentials.ClientCertificateData = kubeClientConfig.TLSClientConfig.CertData
	}
	credentials.ClientKey = kubeClientConfig.TLSClientConfig.KeyFile
	if len(credentials.ClientKey) == 0 {
		credentials.ClientKeyData = kubeClientConfig.TLSClientConfig.KeyData
	}
	config.AuthInfos["user"] = credentials

	cluster := clientcmdapi.NewCluster()
	cluster.Server = kubeClientConfig.Host
	cluster.CertificateAuthority = kubeClientConfig.CAFile
	if len(cluster.CertificateAuthority) == 0 {
		cluster.CertificateAuthorityData = kubeClientConfig.CAData
	}
	cluster.InsecureSkipTLSVerify = kubeClientConfig.Insecure
	config.Clusters["cluster"] = cluster

	ctx := clientcmdapi.NewContext()
	ctx.Cluster = "cluster"
	ctx.AuthInfo = "user"
	config.Contexts["context"] = ctx
	config.CurrentContext = "context"

	f, _ := os.CreateTemp("", "")
	if err := clientcmd.WriteToFile(*config, f.Name()); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return f.Name()
}

// TODO: Find a better way to write the kubeconfig for wardle -> kas comms
func writeKubeConfigForConnectionWithConditionsContext(t *testing.T, kubeClientConfig *rest.Config) string {
	t.Helper()

	// Get the real serving cert from the server
	servingCerts, _, err := cert.GetServingCertificatesForURL(kubeClientConfig.Host, "")
	if err != nil {
		t.Fatal(err)
	}
	encodedServing, err := cert.EncodeCertificates(servingCerts...)
	if err != nil {
		t.Fatal(err)
	}
	kubeClientConfig.CAData = encodedServing

	config := clientcmdapi.NewConfig()

	credentials := clientcmdapi.NewAuthInfo()
	credentials.Token = kubeClientConfig.BearerToken
	credentials.ClientCertificate = kubeClientConfig.TLSClientConfig.CertFile
	if len(credentials.ClientCertificate) == 0 {
		credentials.ClientCertificateData = kubeClientConfig.TLSClientConfig.CertData
	}
	credentials.ClientKey = kubeClientConfig.TLSClientConfig.KeyFile
	if len(credentials.ClientKey) == 0 {
		credentials.ClientKeyData = kubeClientConfig.TLSClientConfig.KeyData
	}
	config.AuthInfos["user"] = credentials

	// Default cluster: kube-apiserver
	cluster := clientcmdapi.NewCluster()
	cluster.Server = kubeClientConfig.Host
	cluster.CertificateAuthority = kubeClientConfig.CAFile
	if len(cluster.CertificateAuthority) == 0 {
		cluster.CertificateAuthorityData = kubeClientConfig.CAData
	}
	cluster.InsecureSkipTLSVerify = kubeClientConfig.Insecure
	config.Clusters["cluster"] = cluster

	// Default context
	ctx := clientcmdapi.NewContext()
	ctx.Cluster = "cluster"
	ctx.AuthInfo = "user"
	config.Contexts["context"] = ctx
	config.CurrentContext = "context"

	f, _ := os.CreateTemp("", "")
	if err := clientcmd.WriteToFile(*config, f.Name()); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return f.Name()
}
