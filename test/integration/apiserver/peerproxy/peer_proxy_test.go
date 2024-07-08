/*
Copyright 2023 The Kubernetes Authors.

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

package peerproxy

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/transport"
	"k8s.io/client-go/util/cert"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	kastesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/storageversiongc"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	kubefeatures "k8s.io/kubernetes/pkg/features"

	"k8s.io/kubernetes/test/integration/framework"
	testutil "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestPeerProxiedRequest(t *testing.T) {

	ktesting.SetDefaultVerbosity(1)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)

	// ensure to stop cert reloading after shutdown
	transport.DialerStopCh = ctx.Done()

	// enable feature flags
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.UnknownVersionInteroperabilityProxy, true)

	// create sharedetcd
	etcd := framework.SharedEtcd()

	// create certificates for aggregation and client-cert auth
	proxyCA, err := createProxyCertContent()
	require.NoError(t, err)

	// start test server with all APIs enabled
	// override hostname to ensure unique ips
	server.SetHostnameFuncForTests("test-server-a")
	serverA := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{
		EnableCertAuth: true,
		ProxyCA:        &proxyCA},
		[]string{}, etcd)
	defer serverA.TearDownFn()

	// start another test server with some api disabled
	// override hostname to ensure unique ips
	server.SetHostnameFuncForTests("test-server-b")
	serverB := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{
		EnableCertAuth: true,
		ProxyCA:        &proxyCA},
		[]string{fmt.Sprintf("--runtime-config=%s", "batch/v1=false")}, etcd)
	defer serverB.TearDownFn()

	kubeClientSetA, err := kubernetes.NewForConfig(serverA.ClientConfig)
	require.NoError(t, err)

	kubeClientSetB, err := kubernetes.NewForConfig(serverB.ClientConfig)
	require.NoError(t, err)

	// create jobs resource using serverA
	job := createJobResource()
	_, err = kubeClientSetA.BatchV1().Jobs("default").Create(context.Background(), job, metav1.CreateOptions{})
	require.NoError(t, err)

	klog.Infof("\nServerA has created jobs\n")

	// List jobs using ServerB
	// This request should be proxied to ServerA since ServerB does not have batch API enabled
	jobsB, err := kubeClientSetB.BatchV1().Jobs("default").List(context.Background(), metav1.ListOptions{})
	klog.Infof("\nServerB has retrieved jobs list of length %v \n\n", len(jobsB.Items))
	require.NoError(t, err)
	assert.NotEmpty(t, jobsB)
	assert.Equal(t, job.Name, jobsB.Items[0].Name)
}

func TestPeerProxiedRequestToThirdServerAfterFirstDies(t *testing.T) {

	ktesting.SetDefaultVerbosity(1)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)

	// ensure to stop cert reloading after shutdown
	transport.DialerStopCh = ctx.Done()

	// enable feature flags
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.UnknownVersionInteroperabilityProxy, true)

	// create sharedetcd
	etcd := framework.SharedEtcd()

	// create certificates for aggregation and client-cert auth
	proxyCA, err := createProxyCertContent()
	require.NoError(t, err)

	// set lease duration to 1s for serverA to ensure that storageversions for serverA are updated
	// once it is shutdown
	controlplaneapiserver.IdentityLeaseDurationSeconds = 10
	controlplaneapiserver.IdentityLeaseGCPeriod = time.Second
	controlplaneapiserver.IdentityLeaseRenewIntervalPeriod = 10 * time.Second

	// start serverA with all APIs enabled
	// override hostname to ensure unique ips
	server.SetHostnameFuncForTests("test-server-a")
	serverA := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{EnableCertAuth: true, ProxyCA: &proxyCA}, []string{}, etcd)
	kubeClientSetA, err := kubernetes.NewForConfig(serverA.ClientConfig)
	require.NoError(t, err)
	// ensure storageversion garbage collector ctlr is set up
	informersA := informers.NewSharedInformerFactory(kubeClientSetA, time.Second)
	setupStorageVersionGC(ctx, kubeClientSetA, informersA)
	// reset lease duration to default value for serverB and serverC since we will not be
	// shutting these down
	controlplaneapiserver.IdentityLeaseDurationSeconds = 20

	// start serverB with some api disabled
	// override hostname to ensure unique ips
	server.SetHostnameFuncForTests("test-server-b")
	serverB := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{EnableCertAuth: true, ProxyCA: &proxyCA}, []string{
		fmt.Sprintf("--runtime-config=%v", "batch/v1=false")}, etcd)
	defer serverB.TearDownFn()
	kubeClientSetB, err := kubernetes.NewForConfig(serverB.ClientConfig)
	require.NoError(t, err)
	// ensure storageversion garbage collector ctlr is set up
	informersB := informers.NewSharedInformerFactory(kubeClientSetB, time.Second)
	setupStorageVersionGC(ctx, kubeClientSetB, informersB)

	// start serverC with all APIs enabled
	// override hostname to ensure unique ips
	server.SetHostnameFuncForTests("test-server-c")
	serverC := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{EnableCertAuth: true, ProxyCA: &proxyCA}, []string{}, etcd)
	defer serverC.TearDownFn()

	// create jobs resource using serverA
	job := createJobResource()
	_, err = kubeClientSetA.BatchV1().Jobs("default").Create(context.Background(), job, metav1.CreateOptions{})
	require.NoError(t, err)
	klog.Infof("\nServerA has created jobs\n")

	// shutdown serverA
	serverA.TearDownFn()

	var jobsB *v1.JobList
	// list jobs using ServerB which it should proxy to ServerC and get back valid response
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		jobsB, err = kubeClientSetB.BatchV1().Jobs("default").List(context.Background(), metav1.ListOptions{})
		if err != nil {
			return false, nil
		}
		if jobsB != nil {
			return true, nil
		}
		return false, nil
	})
	klog.Infof("\nServerB has retrieved jobs list of length %v \n\n", len(jobsB.Items))
	require.NoError(t, err)
	assert.NotEmpty(t, jobsB)
	assert.Equal(t, job.Name, jobsB.Items[0].Name)
}

func setupStorageVersionGC(ctx context.Context, kubeClientSet *kubernetes.Clientset, informers informers.SharedInformerFactory) {
	leaseInformer := informers.Coordination().V1().Leases()
	storageVersionInformer := informers.Internal().V1alpha1().StorageVersions()
	go leaseInformer.Informer().Run(ctx.Done())
	go storageVersionInformer.Informer().Run(ctx.Done())

	controller := storageversiongc.NewStorageVersionGC(ctx, kubeClientSet, leaseInformer, storageVersionInformer)
	go controller.Run(ctx)
}

func createProxyCertContent() (kastesting.ProxyCA, error) {
	result := kastesting.ProxyCA{}
	proxySigningKey, err := testutil.NewPrivateKey()
	if err != nil {
		return result, err
	}
	proxySigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "front-proxy-ca"}, proxySigningKey)
	if err != nil {
		return result, err
	}

	result = kastesting.ProxyCA{
		ProxySigningCert: proxySigningCert,
		ProxySigningKey:  proxySigningKey,
	}
	return result, nil
}

func createJobResource() *v1.Job {
	return &v1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-job",
			Namespace: "default",
		},
		Spec: v1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "test",
							Image: "test",
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
		},
	}
}
