/*
Copyright 2024 The Kubernetes Authors.

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

package storageversionmigrator

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"

	corev1 "k8s.io/api/core/v1"
	svmv1alpha1 "k8s.io/api/storagemigration/v1alpha1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	crdintegration "k8s.io/apiextensions-apiserver/test/integration"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	endpointsdiscovery "k8s.io/apiserver/pkg/endpoints/discovery"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	utiltesting "k8s.io/client-go/util/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubecontrollermanagertesting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	"k8s.io/kubernetes/pkg/controller/storageversionmigrator"
	"k8s.io/kubernetes/test/images/agnhost/crd-conversion-webhook/converter"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/kubeconfig"
	utilnet "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

const (
	secretKey                = "api_key"
	secretVal                = "086a7ffc-0225-11e8-ba89-0ed5f89f718b" // Fake value for testing.
	secretName               = "test-secret"
	triggerSecretName        = "trigger-for-svm"
	svmName                  = "test-svm"
	secondSVMName            = "second-test-svm"
	auditPolicyFileName      = "audit-policy.yaml"
	auditLogFileName         = "audit.log"
	encryptionConfigFileName = "encryption.conf"
	metricPrefix             = "apiserver_encryption_config_controller_automatic_reload_success_total"
	defaultNamespace         = "default"
	crdName                  = "testcrd"
	crdGroup                 = "stable.example.com"
	servicePort              = int32(9443)
	webhookHandler           = "crdconvert"
)

var (
	resources = map[string]string{
		"auditPolicy": `
apiVersion: audit.k8s.io/v1
kind: Policy
omitStages:
  - "RequestReceived"
rules:
  - level: Metadata
    resources:
    - group: ""
      resources: ["secrets"]
    verbs: ["patch"]
  - level: Metadata
    resources:
    - group: "stable.example.com"
      resources: ["testcrds"]
    users: ["system:serviceaccount:kube-system:storage-version-migrator-controller"]
`,
		"initialEncryptionConfig": `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`,
		"updatedEncryptionConfig": `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key2
          secret: c2VjcmV0IGlzIHNlY3VyZSwgaXMgaXQ/
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`,
	}

	v1CRDVersion = []apiextensionsv1.CustomResourceDefinitionVersion{
		{
			Name:    "v1",
			Served:  true,
			Storage: true,
			Schema: &apiextensionsv1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"hostPort": {Type: "string"},
					},
				},
			},
		},
	}
	v2CRDVersion = []apiextensionsv1.CustomResourceDefinitionVersion{
		{
			Name:    "v2",
			Served:  true,
			Storage: false,
			Schema: &apiextensionsv1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"host": {Type: "string"},
						"port": {Type: "string"},
					},
				},
			},
		},
		{
			Name:    "v1",
			Served:  true,
			Storage: true,
			Schema: &apiextensionsv1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"hostPort": {Type: "string"},
					},
				},
			},
		},
	}
	v2StorageCRDVersion = []apiextensionsv1.CustomResourceDefinitionVersion{
		{
			Name:    "v1",
			Served:  true,
			Storage: false,
			Schema: &apiextensionsv1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"hostPort": {Type: "string"},
					},
				},
			},
		},
		{
			Name:    "v2",
			Served:  true,
			Storage: true,
			Schema: &apiextensionsv1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"host": {Type: "string"},
						"port": {Type: "string"},
					},
				},
			},
		},
	}
	v1NotServingCRDVersion = []apiextensionsv1.CustomResourceDefinitionVersion{
		{
			Name:    "v1",
			Served:  false,
			Storage: false,
			Schema: &apiextensionsv1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"hostPort": {Type: "string"},
					},
				},
			},
		},
		{
			Name:    "v2",
			Served:  true,
			Storage: true,
			Schema: &apiextensionsv1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensionsv1.JSONSchemaProps{
						"host": {Type: "string"},
						"port": {Type: "string"},
					},
				},
			},
		},
	}
)

type svmTest struct {
	policyFile                  *os.File
	logFile                     *os.File
	client                      clientset.Interface
	clientConfig                *rest.Config
	dynamicClient               *dynamic.DynamicClient
	discoveryClient             *discovery.DiscoveryClient
	storageConfig               *storagebackend.Config
	server                      *kubeapiservertesting.TestServer
	apiextensionsclient         *apiextensionsclientset.Clientset
	filePathForEncryptionConfig string
}

func svmSetup(ctx context.Context, t *testing.T) *svmTest {
	t.Helper()

	filePathForEncryptionConfig, err := createEncryptionConfig(t, resources["initialEncryptionConfig"])
	if err != nil {
		t.Fatalf("failed to create encryption config: %v", err)
	}

	policyFile, logFile := setupAudit(t)
	apiServerFlags := []string{
		"--encryption-provider-config", filepath.Join(filePathForEncryptionConfig, encryptionConfigFileName),
		"--encryption-provider-config-automatic-reload=true",
		"--disable-admission-plugins", "ServiceAccount",
		"--audit-policy-file", policyFile.Name(),
		"--audit-log-version", "audit.k8s.io/v1",
		"--audit-log-mode", "blocking",
		"--audit-log-path", logFile.Name(),
		"--authorization-mode=RBAC",
		fmt.Sprintf("--runtime-config=%s=true", svmv1alpha1.SchemeGroupVersion),
	}
	storageConfig := framework.SharedEtcd()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, apiServerFlags, storageConfig)

	kubeConfigFile := createKubeConfigFileForRestConfig(t, server.ClientConfig)

	kcm := kubecontrollermanagertesting.StartTestServerOrDie(ctx, []string{
		"--kubeconfig=" + kubeConfigFile,
		"--controllers=garbagecollector,svm",     // these are the only controllers needed for this test
		"--use-service-account-credentials=true", // exercise RBAC of SVM controller
		"--leader-elect=false",                   // KCM leader election calls os.Exit when it ends, so it is easier to just turn it off altogether
	})

	clientSet, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("error in create clientset: %v", err)
	}
	rvDiscoveryClient, err := discovery.NewDiscoveryClientForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create discovery client: %v", err)
	}
	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("error in create dynamic client: %v", err)
	}

	svmTest := &svmTest{
		storageConfig:               storageConfig,
		server:                      server,
		client:                      clientSet,
		clientConfig:                server.ClientConfig,
		dynamicClient:               dynamicClient,
		discoveryClient:             rvDiscoveryClient,
		policyFile:                  policyFile,
		logFile:                     logFile,
		filePathForEncryptionConfig: filePathForEncryptionConfig,
	}

	t.Cleanup(func() {
		var validCodes = sets.New[int32](http.StatusOK, http.StatusConflict) // make sure SVM controller never creates
		_ = svmTest.countMatchingAuditEvents(t, func(event utils.AuditEvent) bool {
			if event.User != "system:serviceaccount:kube-system:storage-version-migrator-controller" {
				return false
			}
			if !validCodes.Has(event.Code) {
				t.Errorf("svm controller had invalid response code for event: %#v", event)
				return true
			}
			return false
		})

		kcm.TearDownFn()
		server.TearDownFn()
		utiltesting.CloseAndRemove(t, svmTest.logFile)
		utiltesting.CloseAndRemove(t, svmTest.policyFile)
		err = os.RemoveAll(svmTest.filePathForEncryptionConfig)
		if err != nil {
			t.Errorf("error while removing temp directory: %v", err)
		}
	})

	return svmTest
}

func createKubeConfigFileForRestConfig(t *testing.T, restConfig *rest.Config) string {
	t.Helper()

	clientConfig := kubeconfig.CreateKubeConfig(restConfig)

	kubeConfigFile := filepath.Join(t.TempDir(), "kubeconfig.yaml")
	if err := clientcmd.WriteToFile(*clientConfig, kubeConfigFile); err != nil {
		t.Fatal(err)
	}
	return kubeConfigFile
}

func createEncryptionConfig(t *testing.T, encryptionConfig string) (
	filePathForEncryptionConfig string,
	err error,
) {
	t.Helper()
	tempDir, err := os.MkdirTemp("", svmName)
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %w", err)
	}

	if err = os.WriteFile(filepath.Join(tempDir, encryptionConfigFileName), []byte(encryptionConfig), 0644); err != nil {
		err = os.RemoveAll(tempDir)
		if err != nil {
			t.Errorf("error while removing temp directory: %v", err)
		}
		return tempDir, fmt.Errorf("error while writing encryption config: %w", err)
	}

	return tempDir, nil
}

func (svm *svmTest) createSecret(ctx context.Context, t *testing.T, name, namespace string) (*corev1.Secret, error) {
	t.Helper()
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string][]byte{
			secretKey: []byte(secretVal),
		},
	}

	return svm.client.CoreV1().Secrets(secret.Namespace).Create(ctx, secret, metav1.CreateOptions{})
}

func (svm *svmTest) getRawSecretFromETCD(t *testing.T, name, namespace string) ([]byte, error) {
	t.Helper()
	secretETCDPath := svm.getETCDPathForResource(t, svm.storageConfig.Prefix, "", "secrets", name, namespace)
	etcdResponse, err := svm.readRawRecordFromETCD(t, secretETCDPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s from etcd: %w", secretETCDPath, err)
	}
	return etcdResponse.Kvs[0].Value, nil
}

func (svm *svmTest) getETCDPathForResource(t *testing.T, storagePrefix, group, resource, name, namespaceName string) string {
	t.Helper()
	groupResource := resource
	if group != "" {
		groupResource = fmt.Sprintf("%s/%s", group, resource)
	}
	if namespaceName == "" {
		return fmt.Sprintf("/%s/%s/%s", storagePrefix, groupResource, name)
	}
	return fmt.Sprintf("/%s/%s/%s/%s", storagePrefix, groupResource, namespaceName, name)
}

func (svm *svmTest) readRawRecordFromETCD(t *testing.T, path string) (*clientv3.GetResponse, error) {
	t.Helper()
	rawClient, etcdClient, err := integration.GetEtcdClients(svm.server.ServerOpts.Etcd.StorageConfig.Transport)
	if err != nil {
		return nil, fmt.Errorf("failed to create etcd client: %w", err)
	}
	// kvClient is a wrapper around rawClient and to avoid leaking goroutines we need to
	// close the client (which we can do by closing rawClient).
	defer func() {
		if err := rawClient.Close(); err != nil {
			t.Errorf("error closing rawClient: %v", err)
		}
	}()

	response, err := etcdClient.Get(context.Background(), path, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve secret from etcd %w", err)
	}

	return response, nil
}

func (svm *svmTest) getRawCRFromETCD(t *testing.T, name, namespace, crdGroup, crdName string) ([]byte, error) {
	t.Helper()
	crdETCDPath := svm.getETCDPathForResource(t, svm.storageConfig.Prefix, crdGroup, crdName, name, namespace)
	etcdResponse, err := svm.readRawRecordFromETCD(t, crdETCDPath)
	if err != nil {
		t.Fatalf("failed to read %s from etcd: %v", crdETCDPath, err)
	}
	return etcdResponse.Kvs[0].Value, nil
}

func (svm *svmTest) updateFile(t *testing.T, configDir, filename string, newContent []byte) {
	t.Helper()
	// Create a temporary file
	tempFile, err := os.CreateTemp(configDir, "tempfile")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := tempFile.Close(); err != nil {
			t.Errorf("error closing tempFile: %v", err)
		}
	}()

	// Write the new content to the temporary file
	_, err = tempFile.Write(newContent)
	if err != nil {
		t.Fatal(err)
	}

	// Atomically replace the original file with the temporary file
	err = os.Rename(tempFile.Name(), filepath.Join(configDir, filename))
	if err != nil {
		t.Fatal(err)
	}
}

func (svm *svmTest) createSVMResource(ctx context.Context, t *testing.T, name string, gvr svmv1alpha1.GroupVersionResource) (
	*svmv1alpha1.StorageVersionMigration,
	error,
) {
	t.Helper()
	svmResource := &svmv1alpha1.StorageVersionMigration{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: svmv1alpha1.StorageVersionMigrationSpec{
			Resource: svmv1alpha1.GroupVersionResource{
				Group:    gvr.Group,
				Version:  gvr.Version,
				Resource: gvr.Resource,
			},
		},
	}

	return svm.client.StoragemigrationV1alpha1().
		StorageVersionMigrations().
		Create(ctx, svmResource, metav1.CreateOptions{})
}

func (svm *svmTest) getSVM(ctx context.Context, t *testing.T, name string) (
	*svmv1alpha1.StorageVersionMigration,
	error,
) {
	t.Helper()
	return svm.client.StoragemigrationV1alpha1().
		StorageVersionMigrations().
		Get(ctx, name, metav1.GetOptions{})
}

func setupAudit(t *testing.T) (
	policyFile *os.File,
	logFile *os.File,
) {
	t.Helper()
	// prepare audit policy file
	policyFile, err := os.CreateTemp("", auditPolicyFileName)
	if err != nil {
		t.Fatalf("Failed to create audit policy file: %v", err)
	}
	if _, err := policyFile.Write([]byte(resources["auditPolicy"])); err != nil {
		t.Fatalf("Failed to write audit policy file: %v", err)
	}

	// prepare audit log file
	logFile, err = os.CreateTemp("", auditLogFileName)
	if err != nil {
		t.Fatalf("Failed to create audit log file: %v", err)
	}

	return policyFile, logFile
}

func (svm *svmTest) getAutomaticReloadSuccessTotal(ctx context.Context, t *testing.T) int {
	t.Helper()

	copyConfig := rest.CopyConfig(svm.server.ClientConfig)
	copyConfig.GroupVersion = &schema.GroupVersion{}
	copyConfig.NegotiatedSerializer = unstructuredscheme.NewUnstructuredNegotiatedSerializer()
	rc, err := rest.RESTClientFor(copyConfig)
	if err != nil {
		t.Fatalf("Failed to create REST client: %v", err)
	}

	body, err := rc.Get().AbsPath("/metrics").DoRaw(ctx)
	if err != nil {
		t.Fatal(err)
	}

	metricRegex := regexp.MustCompile(fmt.Sprintf(`%s{.*} (\d+)`, metricPrefix))
	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, metricPrefix) {
			matches := metricRegex.FindStringSubmatch(line)
			if len(matches) == 2 {
				metricValue, err := strconv.Atoi(matches[1])
				if err != nil {
					t.Fatalf("Failed to convert metric value to integer: %v", err)
				}
				return metricValue
			}
		}
	}

	return 0
}

func (svm *svmTest) isEncryptionConfigFileUpdated(ctx context.Context, t *testing.T, metricBeforeUpdate int) bool {
	t.Helper()

	err := wait.PollUntilContextTimeout(
		ctx,
		500*time.Millisecond,
		wait.ForeverTestTimeout,
		true,
		func(ctx context.Context) (bool, error) {
			metric := svm.getAutomaticReloadSuccessTotal(ctx, t)
			return metric == (metricBeforeUpdate + 1), nil
		},
	)

	return err == nil
}

// waitForResourceMigration checks following conditions:
// 1. The svm resource has SuccessfullyMigrated condition.
// 2. The audit log contains patch events for the given secret.
func (svm *svmTest) waitForResourceMigration(
	ctx context.Context,
	t *testing.T,
	svmName, name string,
	expectedEvents int,
) bool {
	t.Helper()

	var triggerOnce sync.Once

	err := wait.PollUntilContextTimeout(
		ctx,
		500*time.Millisecond,
		5*time.Minute,
		true,
		func(ctx context.Context) (bool, error) {
			svmResource, err := svm.getSVM(ctx, t, svmName)
			if err != nil {
				t.Fatalf("Failed to get SVM resource: %v", err)
			}

			if storageversionmigrator.IsConditionTrue(svmResource, svmv1alpha1.MigrationFailed) {
				t.Logf("%q SVM has failed migration, %#v", svmName, svmResource.Status.Conditions)
				return false, fmt.Errorf("SVM has failed migration")
			}

			if svmResource.Status.ResourceVersion == "" {
				t.Logf("%q SVM has no resourceVersion", svmName)
				return false, nil
			}

			if storageversionmigrator.IsConditionTrue(svmResource, svmv1alpha1.MigrationSucceeded) {
				t.Logf("%q SVM has completed migration", svmName)
				return true, nil
			}

			if storageversionmigrator.IsConditionTrue(svmResource, svmv1alpha1.MigrationRunning) {
				t.Logf("%q SVM migration is running, %#v", svmName, svmResource.Status.Conditions)
				return false, nil
			}

			t.Logf("%q SVM has not started migration, %#v", svmName, svmResource.Status.Conditions)

			// We utilize the LastSyncResourceVersion of the Garbage Collector (GC) to ensure that the cache is up-to-date before proceeding with the migration.
			// However, in a quiet cluster, the GC may not be updated unless there is some activity or the watch receives a bookmark event after every 10 minutes.
			// To expedite the update of the GC cache, we create a dummy secret and then promptly delete it.
			// This action forces the GC to refresh its cache, enabling us to proceed with the migration.
			// At this point we know that the RV has been set on the SVM resource, so the trigger will always have a higher RV.
			// We only need to do this once.
			triggerOnce.Do(func() {
				_, err = svm.createSecret(ctx, t, triggerSecretName, defaultNamespace)
				if err != nil {
					t.Fatalf("Failed to create secret: %v", err)
				}
				err = svm.client.CoreV1().Secrets(defaultNamespace).Delete(ctx, triggerSecretName, metav1.DeleteOptions{})
				if err != nil {
					t.Fatalf("Failed to delete secret: %v", err)
				}
			})

			return false, nil
		},
	)
	if err != nil {
		t.Logf("Failed to wait for resource migration for SVM %q with secret %q: %v", svmName, name, err)
		return false
	}

	err = wait.PollUntilContextTimeout(
		ctx,
		500*time.Millisecond,
		wait.ForeverTestTimeout,
		true,
		func(_ context.Context) (bool, error) {
			want := utils.AuditEvent{
				Level:             auditinternal.LevelMetadata,
				Stage:             auditinternal.StageResponseComplete,
				RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets/%s?fieldManager=storage-version-migrator-controller", defaultNamespace, name),
				Verb:              "patch",
				Code:              http.StatusOK,
				User:              "system:serviceaccount:kube-system:storage-version-migrator-controller",
				Resource:          "secrets",
				Namespace:         "default",
				AuthorizeDecision: "allow",
				RequestObject:     false,
				ResponseObject:    false,
			}

			if seen := svm.countMatchingAuditEvents(t, func(event utils.AuditEvent) bool { return reflect.DeepEqual(event, want) }); expectedEvents > seen {
				t.Logf("audit log did not contain %d expected audit events, only has %d", expectedEvents, seen)
				return false, nil
			}

			return true, nil
		},
	)
	if err != nil {
		t.Logf("Failed to wait for audit logs events for SVM %q with secret %q: %v", svmName, name, err)
		return false
	}

	return true
}

func (svm *svmTest) countMatchingAuditEvents(t *testing.T, f func(utils.AuditEvent) bool) int {
	t.Helper()

	var seen int
	for _, event := range svm.getAuditEvents(t) {
		if f(event) {
			seen++
		}
	}
	return seen
}

func (svm *svmTest) getAuditEvents(t *testing.T) []utils.AuditEvent {
	t.Helper()

	stream, err := os.Open(svm.logFile.Name())
	if err != nil {
		t.Fatalf("Failed to open audit log file: %v", err)
	}
	defer func() {
		if err := stream.Close(); err != nil {
			t.Errorf("error while closing audit log file: %v", err)
		}
	}()

	missingReport, err := utils.CheckAuditLines(stream, nil, auditv1.SchemeGroupVersion)
	if err != nil {
		t.Fatalf("Failed to check audit log: %v", err)
	}

	return missingReport.AllEvents
}

func (svm *svmTest) createCRD(
	t *testing.T,
	name, group string,
	certCtx *certContext,
	crdVersions []apiextensionsv1.CustomResourceDefinitionVersion,
) *apiextensionsv1.CustomResourceDefinition {
	t.Helper()
	pluralName := name + "s"
	listKind := name + "List"

	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: pluralName + "." + group,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: group,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Kind:     name,
				ListKind: listKind,
				Plural:   pluralName,
				Singular: name,
			},
			Scope:                 apiextensionsv1.NamespaceScoped,
			Versions:              crdVersions,
			PreserveUnknownFields: false,
		},
	}

	if certCtx != nil {
		crd.Spec.Conversion = &apiextensionsv1.CustomResourceConversion{
			Strategy: apiextensionsv1.WebhookConverter,
			Webhook: &apiextensionsv1.WebhookConversion{
				ClientConfig: &apiextensionsv1.WebhookClientConfig{
					CABundle: certCtx.signingCert,
					URL: ptr.To(
						fmt.Sprintf("https://127.0.0.1:%d/%s", servicePort, webhookHandler),
					),
				},
				ConversionReviewVersions: []string{"v1", "v2"},
			},
		}
	}

	apiextensionsclient, err := apiextensionsclientset.NewForConfig(svm.clientConfig)
	if err != nil {
		t.Fatalf("Failed to create apiextensions client: %v", err)
	}
	svm.apiextensionsclient = apiextensionsclient

	etcd.CreateTestCRDs(t, apiextensionsclient, false, crd)
	return crd
}

func (svm *svmTest) updateCRD(
	ctx context.Context,
	t *testing.T,
	crdName string,
	updatesCRDVersions []apiextensionsv1.CustomResourceDefinitionVersion,
	expectedServingVersions []string,
	expectedStorageVersion string,
) {
	t.Helper()

	var err error
	crd, err := crdintegration.UpdateV1CustomResourceDefinitionWithRetry(svm.apiextensionsclient, crdName, func(c *apiextensionsv1.CustomResourceDefinition) {
		c.Spec.Versions = updatesCRDVersions
	})
	if err != nil {
		t.Fatalf("Failed to update CRD: %v", err)
	}

	svm.waitForCRDUpdate(ctx, t, crd.Spec.Names.Kind, expectedServingVersions, expectedStorageVersion)
}

func (svm *svmTest) waitForCRDUpdate(
	ctx context.Context,
	t *testing.T,
	crdKind string,
	expectedServingVersions []string,
	expectedStorageVersion string,
) {
	t.Helper()

	err := wait.PollUntilContextTimeout(
		ctx,
		500*time.Millisecond,
		time.Second*60,
		true,
		func(ctx context.Context) (bool, error) {
			apiGroups, _, err := svm.discoveryClient.ServerGroupsAndResources()
			if err != nil {
				return false, fmt.Errorf("failed to get server groups and resources: %w", err)
			}
			for _, api := range apiGroups {
				if api.Name == crdGroup {
					var servingVersions []string
					for _, apiVersion := range api.Versions {
						servingVersions = append(servingVersions, apiVersion.Version)
					}
					sort.Strings(servingVersions)

					// Check if the serving versions are as expected
					if reflect.DeepEqual(expectedServingVersions, servingVersions) {
						expectedHash := endpointsdiscovery.StorageVersionHash(crdGroup, expectedStorageVersion, crdKind)
						resourceList, err := svm.discoveryClient.ServerResourcesForGroupVersion(crdGroup + "/" + api.PreferredVersion.Version)
						if err != nil {
							return false, fmt.Errorf("failed to get server resources for group version: %w", err)
						}

						// Check if the storage version is as expected
						for _, resource := range resourceList.APIResources {
							if resource.Kind == crdKind {
								if resource.StorageVersionHash == expectedHash {
									return true, nil
								}
							}
						}
					}
				}
			}
			return false, nil
		},
	)
	if err != nil {
		t.Fatalf("Failed to update a CRD: Name: %s, Err: %v", crdName, err)
	}
}

type testingT interface {
	Helper()
	Fatalf(format string, args ...any)
}

func (svm *svmTest) createCR(ctx context.Context, t testingT, crName, version string) *unstructured.Unstructured {
	t.Helper()

	crdResource := schema.GroupVersionResource{
		Group:    crdGroup,
		Version:  version,
		Resource: crdName + "s",
	}

	crdUnstructured := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": crdResource.GroupVersion().String(),
			"kind":       crdName,
			"metadata": map[string]interface{}{
				"name":      crName,
				"namespace": defaultNamespace,
			},
		},
	}

	crdUnstructured, err := svm.dynamicClient.Resource(crdResource).Namespace(defaultNamespace).Create(ctx, crdUnstructured, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create CR: %v", err)
	}

	return crdUnstructured
}

func (svm *svmTest) getCR(ctx context.Context, t *testing.T, crName, version string) *unstructured.Unstructured {
	t.Helper()

	crdResource := schema.GroupVersionResource{
		Group:    crdGroup,
		Version:  version,
		Resource: crdName + "s",
	}

	cr, err := svm.dynamicClient.Resource(crdResource).Namespace(defaultNamespace).Get(ctx, crName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get CR: %v", err)
	}

	return cr
}

func (svm *svmTest) listCR(ctx context.Context, t *testing.T, version string) error {
	t.Helper()

	crdResource := schema.GroupVersionResource{
		Group:    crdGroup,
		Version:  version,
		Resource: crdName + "s",
	}

	_, err := svm.dynamicClient.Resource(crdResource).Namespace(defaultNamespace).List(ctx, metav1.ListOptions{})

	return err
}

func (svm *svmTest) deleteCR(ctx context.Context, t testingT, name, version string) {
	t.Helper()
	crdResource := schema.GroupVersionResource{
		Group:    crdGroup,
		Version:  version,
		Resource: crdName + "s",
	}
	err := svm.dynamicClient.Resource(crdResource).Namespace(defaultNamespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete CR: %v", err)
	}
}

func (svm *svmTest) createConversionWebhook(ctx context.Context, t *testing.T, certCtx *certContext) context.CancelFunc {
	t.Helper()

	mux := http.NewServeMux()
	mux.HandleFunc(fmt.Sprintf("/%s", webhookHandler), converter.ServeExampleConvert)

	block, _ := pem.Decode(certCtx.key)
	if block == nil {
		panic("failed to parse PEM block containing the key")
	}
	key, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		t.Fatalf("Failed to parse private key: %v", err)
	}

	blockCer, _ := pem.Decode(certCtx.cert)
	if blockCer == nil {
		panic("failed to parse PEM block containing the key")
	}
	webhookCert, err := x509.ParseCertificate(blockCer.Bytes)
	if err != nil {
		t.Fatalf("Failed to parse certificate: %v", err)
	}

	server := &http.Server{
		Addr:    fmt.Sprintf("127.0.0.1:%d", servicePort),
		Handler: mux,
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{
				{
					Certificate: [][]byte{webhookCert.Raw},
					PrivateKey:  key,
				},
			},
		},
	}

	go func() {
		// skipping error handling here because this always returns a non-nil error.
		// after Server.Shutdown, the returned error is ErrServerClosed.
		_ = server.ListenAndServeTLS("", "")

	}()

	serverCtx, cancel := context.WithCancel(ctx)
	go func(ctx context.Context, t *testing.T) {
		<-ctx.Done()
		// Context was cancelled, shutdown the server
		if err := server.Shutdown(context.Background()); err != nil {
			t.Logf("Failed to shutdown server: %v", err)
		}
	}(serverCtx, t)

	return cancel
}

type certContext struct {
	cert        []byte
	key         []byte
	signingCert []byte
}

func (svm *svmTest) setupServerCert(t *testing.T) *certContext {
	t.Helper()
	certDir, err := os.MkdirTemp("", "test-e2e-server-cert")
	if err != nil {
		t.Fatalf("Failed to create a temp dir for cert generation %v", err)
	}
	defer func(path string) {
		err := os.RemoveAll(path)
		if err != nil {
			t.Fatalf("Failed to remove temp dir %v", err)
		}
	}(certDir)
	signingKey, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatalf("Failed to create CA private key %v", err)
	}
	signingCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "e2e-server-cert-ca"}, signingKey)
	if err != nil {
		t.Fatalf("Failed to create CA cert for apiserver %v", err)
	}
	caCertFile, err := os.CreateTemp(certDir, "ca.crt")
	if err != nil {
		t.Fatalf("Failed to create a temp file for ca cert generation %v", err)
	}
	defer utiltesting.CloseAndRemove(&testing.T{}, caCertFile)
	if err := os.WriteFile(caCertFile.Name(), utils.EncodeCertPEM(signingCert), 0644); err != nil {
		t.Fatalf("Failed to write CA cert %v", err)
	}
	key, err := utils.NewPrivateKey()
	if err != nil {
		t.Fatalf("Failed to create private key for %v", err)
	}
	signedCert, err := utils.NewSignedCert(
		&cert.Config{
			CommonName: "127.0.0.1",
			AltNames: cert.AltNames{
				IPs: []net.IP{utilnet.ParseIPSloppy("127.0.0.1")},
			},
			Usages: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		},
		key, signingCert, signingKey,
	)
	if err != nil {
		t.Fatalf("Failed to create cert%v", err)
	}
	certFile, err := os.CreateTemp(certDir, "server.crt")
	if err != nil {
		t.Fatalf("Failed to create a temp file for cert generation %v", err)
	}
	defer utiltesting.CloseAndRemove(&testing.T{}, certFile)
	keyFile, err := os.CreateTemp(certDir, "server.key")
	if err != nil {
		t.Fatalf("Failed to create a temp file for key generation %v", err)
	}
	if err = os.WriteFile(certFile.Name(), utils.EncodeCertPEM(signedCert), 0600); err != nil {
		t.Fatalf("Failed to write cert file %v", err)
	}
	privateKeyPEM, err := keyutil.MarshalPrivateKeyToPEM(key)
	if err != nil {
		t.Fatalf("Failed to marshal key %v", err)
	}
	if err = os.WriteFile(keyFile.Name(), privateKeyPEM, 0644); err != nil {
		t.Fatalf("Failed to write key file %v", err)
	}
	defer utiltesting.CloseAndRemove(&testing.T{}, keyFile)
	return &certContext{
		cert:        utils.EncodeCertPEM(signedCert),
		key:         privateKeyPEM,
		signingCert: utils.EncodeCertPEM(signingCert),
	}
}

func (svm *svmTest) isCRStoredAtVersion(t *testing.T, version, crName string) bool {
	t.Helper()

	data, err := svm.getRawCRFromETCD(t, crName, defaultNamespace, crdGroup, crdName+"s")
	if err != nil {
		t.Fatalf("Failed to get CR from etcd: %v", err)
	}

	// parse data to unstructured.Unstructured
	obj := &unstructured.Unstructured{}
	err = obj.UnmarshalJSON(data)
	if err != nil {
		t.Fatalf("Failed to unmarshal data to unstructured: %v", err)
	}

	return obj.GetAPIVersion() == fmt.Sprintf("%s/%s", crdGroup, version)
}

func (svm *svmTest) isCRDMigrated(ctx context.Context, t *testing.T, crdSVMName, triggerCRName string) bool {
	t.Helper()

	var triggerOnce sync.Once

	err := wait.PollUntilContextTimeout(
		ctx,
		500*time.Millisecond,
		5*time.Minute,
		true,
		func(ctx context.Context) (bool, error) {
			svmResource, err := svm.getSVM(ctx, t, crdSVMName)
			if err != nil {
				t.Fatalf("Failed to get SVM resource: %v", err)
			}

			if storageversionmigrator.IsConditionTrue(svmResource, svmv1alpha1.MigrationFailed) {
				t.Logf("%q SVM has failed migration, %#v", crdSVMName, svmResource.Status.Conditions)
				return false, fmt.Errorf("SVM has failed migration")
			}

			if svmResource.Status.ResourceVersion == "" {
				t.Logf("%q SVM has no resourceVersion", crdSVMName)
				return false, nil
			}

			if storageversionmigrator.IsConditionTrue(svmResource, svmv1alpha1.MigrationSucceeded) {
				t.Logf("%q SVM has completed migration", crdSVMName)
				return true, nil
			}

			if storageversionmigrator.IsConditionTrue(svmResource, svmv1alpha1.MigrationRunning) {
				t.Logf("%q SVM migration is running, %#v", crdSVMName, svmResource.Status.Conditions)
				return false, nil
			}

			t.Logf("%q SVM has not started migration, %#v", crdSVMName, svmResource.Status.Conditions)

			// at this point we know that the RV has been set on the SVM resource,
			// and we need to make sure that the GC list RV has caught up to that without waiting for a watch bookmark.
			// we cannot trigger this any earlier as the rest mapper of the RV controller can be delayed
			// and thus may not have observed the new CRD yet.  we only need to do this once.
			triggerOnce.Do(func() {
				triggerCR := svm.createCR(ctx, t, triggerCRName, "v1")
				svm.deleteCR(ctx, t, triggerCR.GetName(), "v1")
			})

			return false, nil
		},
	)
	return err == nil
}

type versions struct {
	generation  int64
	rv          string
	isRVUpdated bool
}

func (svm *svmTest) validateRVAndGeneration(ctx context.Context, t *testing.T, crVersions map[string]versions, getCRVersion string) {
	t.Helper()

	for crName, version := range crVersions {
		// get CR from etcd
		data, err := svm.getRawCRFromETCD(t, crName, defaultNamespace, crdGroup, crdName+"s")
		if err != nil {
			t.Fatalf("Failed to get CR from etcd: %v", err)
		}

		// parse data to unstructured.Unstructured
		obj := &unstructured.Unstructured{}
		err = obj.UnmarshalJSON(data)
		if err != nil {
			t.Fatalf("Failed to unmarshal data to unstructured: %v", err)
		}

		// validate resourceVersion and generation
		crVersion := svm.getCR(ctx, t, crName, getCRVersion).GetResourceVersion()
		isRVUnchanged := crVersion == version.rv
		if version.isRVUpdated && isRVUnchanged {
			t.Fatalf("ResourceVersion of CR %s should not be equal. Expected: %s, Got: %s", crName, version.rv, crVersion)
		}
		if !version.isRVUpdated && !isRVUnchanged {
			t.Fatalf("ResourceVersion of CR %s should be equal. Expected: %s, Got: %s", crName, version.rv, crVersion)
		}
		if obj.GetGeneration() != version.generation {
			t.Fatalf("Generation of CR %s should be equal. Expected: %d, Got: %d", crName, version.generation, obj.GetGeneration())
		}
	}
}

func (svm *svmTest) createChaos(ctx context.Context, t *testing.T) {
	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(ctx)

	noFailT := ignoreFailures{} // these create and delete requests are not coordinated with the rest of the test and can fail

	const workers = 10
	wg.Add(workers)
	for i := range workers {
		i := i
		go func() {
			defer wg.Done()

			for {
				select {
				case <-ctx.Done():
					return
				default:
				}

				_ = svm.createCR(ctx, noFailT, "chaos-cr-"+strconv.Itoa(i), "v1")
				svm.deleteCR(ctx, noFailT, "chaos-cr-"+strconv.Itoa(i), "v1")
			}
		}()
	}

	t.Cleanup(func() {
		cancel()
		wg.Wait()
	})
}

type ignoreFailures struct{}

func (ignoreFailures) Helper()                           {}
func (ignoreFailures) Fatalf(format string, args ...any) {}
