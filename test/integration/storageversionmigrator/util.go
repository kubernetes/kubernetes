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
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/controller-manager/pkg/informerfactory"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/controller/storageversionmigrator"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"

	clientv3 "go.etcd.io/etcd/client/v3"
	corev1 "k8s.io/api/core/v1"
	svmv1alpha1 "k8s.io/api/storagemigration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	clientset "k8s.io/client-go/kubernetes"
	utiltesting "k8s.io/client-go/util/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
)

const (
	secretKey                = "api_key"
	secretVal                = "086a7ffc-0225-11e8-ba89-0ed5f89f718b" // Fake value for testing.
	secretName               = "test-secret"
	secretNamespace          = "default"
	triggerSecretName        = "trigger-for-svm"
	svmName                  = "test-svm"
	secondSVMName            = "second-test-svm"
	auditPolicyFileName      = "audit-policy.yaml"
	auditLogFileName         = "audit.log"
	encryptionConfigFileName = "encryption.conf"
	metricPrefix             = "apiserver_encryption_config_controller_automatic_reload_success_total"
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
)

type svmTest struct {
	policyFile                  *os.File
	logFile                     *os.File
	client                      clientset.Interface
	storageConfig               *storagebackend.Config
	server                      *kubeapiservertesting.TestServer
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
	}
	storageConfig := framework.SharedEtcd()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, apiServerFlags, storageConfig)

	clientSet, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("error in create clientset: %v", err)
	}

	discoveryClient := cacheddiscovery.NewMemCacheClient(clientSet.Discovery())
	rvDiscoveryClient, err := discovery.NewDiscoveryClientForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create discovery client: %v", err)
	}
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	restMapper.Reset()
	metadataClient, err := metadata.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create metadataClient: %v", err)
	}
	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("error in create dynamic client: %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(clientSet, 0)
	metadataInformers := metadatainformer.NewSharedInformerFactory(metadataClient, 0)
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)

	dependencyGraphBuilder := garbagecollector.NewDependencyGraphBuilder(
		ctx,
		metadataClient,
		alwaysStarted,
		restMapper,
		informerfactory.NewInformerFactory(sharedInformers, metadataInformers),
	)
	attemptToDelete, attemptToOrphan, absentOwnerCache, eventBroadcaster := dependencyGraphBuilder.GetGraphResources()
	gc, err := garbagecollector.NewGarbageCollector(
		ctx,
		clientSet,
		metadataClient,
		restMapper,
		garbagecollector.DefaultIgnoredResources(),
		sharedInformers,
		dependencyGraphBuilder,
		attemptToDelete,
		attemptToOrphan,
		absentOwnerCache,
		eventBroadcaster,
	)
	if err != nil {
		t.Fatalf("error while creating garbage collector: %v", err)

	}
	startGC := func() {
		syncPeriod := 5 * time.Second
		go wait.Until(func() {
			restMapper.Reset()
		}, syncPeriod, ctx.Done())
		go gc.Run(ctx, 1)
		go gc.Sync(ctx, clientSet.Discovery(), syncPeriod)
	}

	svmController := storageversionmigrator.NewSVMController(
		ctx,
		clientSet,
		dynamicClient,
		sharedInformers.Storagemigration().V1alpha1().StorageVersionMigrations(),
		names.StorageVersionMigratorController,
		restMapper,
		dependencyGraphBuilder,
	)

	rvController := storageversionmigrator.NewResourceVersionController(
		ctx,
		clientSet,
		rvDiscoveryClient,
		metadataClient,
		sharedInformers.Storagemigration().V1alpha1().StorageVersionMigrations(),
		restMapper,
	)

	// Start informer and controllers
	sharedInformers.Start(ctx.Done())
	startGC()
	go svmController.Run(ctx)
	go rvController.Run(ctx)

	svmTest := &svmTest{
		storageConfig:               storageConfig,
		server:                      server,
		client:                      clientSet,
		policyFile:                  policyFile,
		logFile:                     logFile,
		filePathForEncryptionConfig: filePathForEncryptionConfig,
	}

	t.Cleanup(func() {
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

func (svm *svmTest) createSVMResource(ctx context.Context, t *testing.T, name string) (
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
				Group:    "",
				Version:  "v1",
				Resource: "secrets",
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
	svmName, name, namespace string,
	expectedEvents int,
) bool {
	t.Helper()

	var isMigrated bool
	err := wait.PollUntilContextTimeout(
		ctx,
		500*time.Millisecond,
		wait.ForeverTestTimeout,
		true,
		func(ctx context.Context) (bool, error) {
			svmResource, err := svm.getSVM(ctx, t, svmName)
			if err != nil {
				t.Fatalf("Failed to get SVM resource: %v", err)
			}
			if svmResource.Status.ResourceVersion == "" {
				return false, nil
			}

			if storageversionmigrator.IsConditionTrue(svmResource, svmv1alpha1.MigrationSucceeded) {
				isMigrated = true
			}

			// We utilize the LastSyncResourceVersion of the Garbage Collector (GC) to ensure that the cache is up-to-date before proceeding with the migration.
			// However, in a quiet cluster, the GC may not be updated unless there is some activity or the watch receives a bookmark event after every 10 minutes.
			// To expedite the update of the GC cache, we create a dummy secret and then promptly delete it.
			// This action forces the GC to refresh its cache, enabling us to proceed with the migration.
			_, err = svm.createSecret(ctx, t, triggerSecretName, secretNamespace)
			if err != nil {
				t.Fatalf("Failed to create secret: %v", err)
			}
			err = svm.client.CoreV1().Secrets(secretNamespace).Delete(ctx, triggerSecretName, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("Failed to delete secret: %v", err)
			}

			stream, err := os.Open(svm.logFile.Name())
			if err != nil {
				t.Fatalf("Failed to open audit log file: %v", err)
			}
			defer func() {
				if err := stream.Close(); err != nil {
					t.Errorf("error	while closing audit log file: %v", err)
				}
			}()

			missingReport, err := utils.CheckAuditLines(
				stream,
				[]utils.AuditEvent{
					{
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets/%s?fieldManager=storage-version-migrator-controller", namespace, name),
						Verb:              "patch",
						Code:              200,
						User:              "system:apiserver",
						Resource:          "secrets",
						Namespace:         "default",
						AuthorizeDecision: "allow",
						RequestObject:     false,
						ResponseObject:    false,
					},
				},
				auditv1.SchemeGroupVersion,
			)
			if err != nil {
				t.Fatalf("Failed to check audit log: %v", err)
			}
			if (len(missingReport.MissingEvents) != 0) && (expectedEvents < missingReport.NumEventsChecked) {
				isMigrated = false
			}

			return isMigrated, nil
		},
	)
	if err != nil {
		return false
	}

	return isMigrated
}
