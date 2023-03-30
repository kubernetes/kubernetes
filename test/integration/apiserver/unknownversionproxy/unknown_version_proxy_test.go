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

package unknownversionproxy

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/cert"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kastesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"

	"k8s.io/kubernetes/test/integration/framework"
	testutil "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestUnknownVersionProxiedRequest(t *testing.T) {

	ktesting.SetDefaultVerbosity(1)
	// enable feature flags
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionAPI, true)()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UnknownVersionInteroperabilityProxy, true)()

	// create sharedetcd
	etcd := framework.SharedEtcd()

	// create certificates for aggregation and client-cert auth
	proxyCA, err := createProxyCertContent()
	require.NoError(t, err)

	// start test server with all APIs enabled
	serverA := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{EnableCertAuth: true, ProxyCA: &proxyCA}, []string{}, etcd)
	//serverA := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, etcd)
	defer serverA.TearDownFn()

	// start another test server with some api disabled
	serverB := kastesting.StartTestServerOrDie(t, &kastesting.TestServerInstanceOptions{EnableCertAuth: true, ProxyCA: &proxyCA}, []string{
		fmt.Sprintf("--runtime-config=%v", "batch/v1=false")}, etcd)
	defer serverB.TearDownFn()

	kubeClientSetA, err := kubernetes.NewForConfig(serverA.ClientConfig)
	require.NoError(t, err)

	kubeClientSetB, err := kubernetes.NewForConfig(serverB.ClientConfig)
	require.NoError(t, err)

	//time.Sleep(20 * time.Minute)
	// create jobs resource using serverA
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-job",
			Namespace: "default",
		},
		Spec: batchv1.JobSpec{
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
	_, err = kubeClientSetA.BatchV1().Jobs("default").Create(context.Background(), job, metav1.CreateOptions{})
	require.NoError(t, err)

	fmt.Printf("\nServerA has created jobs\n")

	// list jobs using ServerB
	jobsB, err := kubeClientSetB.BatchV1().Jobs("default").List(context.Background(), metav1.ListOptions{})
	require.NoError(t, err)
	fmt.Printf("\nServerB has retrieved jobs list of length %v \n\n", len(jobsB.Items))
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
