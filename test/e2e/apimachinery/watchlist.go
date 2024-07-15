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

package apimachinery

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/consistencydetector"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("API Streaming (aka. WatchList)", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("watchlist")
	ginkgo.It("should be requested by informers when WatchListClient is enabled", func(ctx context.Context) {
		featuregatetesting.SetFeatureGateDuringTest(ginkgo.GinkgoTB(), utilfeature.DefaultFeatureGate, featuregate.Feature(clientfeatures.WatchListClient), true)
		stopCh := make(chan struct{})
		defer close(stopCh)
		secretInformer := cache.NewSharedIndexInformer(
			&cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					return nil, fmt.Errorf("unexpected list call")
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Watch(context.TODO(), options)
				},
			},
			&v1.Secret{},
			time.Duration(0),
			nil,
		)

		ginkgo.By(fmt.Sprintf("Adding 5 secrets to %s namespace", f.Namespace.Name))
		var expectedSecrets []v1.Secret
		for i := 1; i <= 5; i++ {
			secret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, newSecret(fmt.Sprintf("secret-%d", i)), metav1.CreateOptions{})
			framework.ExpectNoError(err)
			expectedSecrets = append(expectedSecrets, *secret)
		}

		ginkgo.By("Starting the secret informer")
		go secretInformer.Run(stopCh)

		ginkgo.By("Waiting until the secret informer is fully synchronised")
		err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, false, func(context.Context) (done bool, err error) {
			return secretInformer.HasSynced(), nil
		})
		framework.ExpectNoError(err, "Failed waiting for the secret informer in %s namespace to be synced", f.Namespace.Namespace)

		ginkgo.By("Verifying if the secret informer was properly synchronised")
		verifyStore(ctx, expectedSecrets, secretInformer.GetStore())

		ginkgo.By("Modifying a secret and checking if the update was picked up by the secret informer")
		secret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Get(ctx, "secret-1", metav1.GetOptions{})
		framework.ExpectNoError(err)
		secret.StringData = map[string]string{"foo": "bar"}
		secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Update(ctx, secret, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		expectedSecrets[0] = *secret
		verifyStore(ctx, expectedSecrets, secretInformer.GetStore())
	})
	ginkgo.It("should be requested by client-go's List method when WatchListClient is enabled", func(ctx context.Context) {
		featuregatetesting.SetFeatureGateDuringTest(ginkgo.GinkgoTB(), utilfeature.DefaultFeatureGate, featuregate.Feature(clientfeatures.WatchListClient), true)

		ginkgo.By(fmt.Sprintf("Adding 5 secrets to %s namespace", f.Namespace.Name))
		var expectedSecrets []v1.Secret
		for i := 1; i <= 5; i++ {
			secret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, newSecret(fmt.Sprintf("secret-%d", i)), metav1.CreateOptions{})
			framework.ExpectNoError(err)
			expectedSecrets = append(expectedSecrets, *secret)
		}

		rt, clientConfig := clientConfigWithRoundTripper(f)
		wrappedKubeClient, err := kubernetes.NewForConfig(clientConfig)
		framework.ExpectNoError(err)

		ginkgo.By("Streaming secrets from the server")
		secretList, err := wrappedKubeClient.CoreV1().Secrets(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying if the secret list was properly streamed")
		streamedSecrets := secretList.Items
		gomega.Expect(cmp.Equal(expectedSecrets, streamedSecrets)).To(gomega.BeTrueBecause("data received via watchlist must match the added data"))

		ginkgo.By("Verifying if expected requests were sent to the server")
		expectedRequestMadeByKubeClient := getExpectedRequestMadeByClientFor(secretList.ResourceVersion)
		gomega.Expect(rt.actualRequests).To(gomega.Equal(expectedRequestMadeByKubeClient))
	})
})

type roundTripper struct {
	actualRequests []string
	delegate       http.RoundTripper
}

func (r *roundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	r.actualRequests = append(r.actualRequests, req.URL.RawQuery)
	return r.delegate.RoundTrip(req)
}

func (r *roundTripper) Wrap(delegate http.RoundTripper) http.RoundTripper {
	r.delegate = delegate
	return r
}

func clientConfigWithRoundTripper(f *framework.Framework) (*roundTripper, *rest.Config) {
	clientConfig := f.ClientConfig()
	rt := &roundTripper{}
	clientConfig.Wrap(rt.Wrap)

	return rt, clientConfig
}

func verifyStore(ctx context.Context, expectedSecrets []v1.Secret, store cache.Store) {
	err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (done bool, err error) {
		ginkgo.By("Comparing secrets retrieved directly from the server with the ones that have been streamed to the secret informer")
		rawStreamedSecrets := store.List()
		streamedSecrets := make([]v1.Secret, 0, len(rawStreamedSecrets))
		for _, rawSecret := range rawStreamedSecrets {
			streamedSecrets = append(streamedSecrets, *rawSecret.(*v1.Secret))
		}
		sort.Sort(byName(expectedSecrets))
		sort.Sort(byName(streamedSecrets))
		return cmp.Equal(expectedSecrets, streamedSecrets), nil
	})
	framework.ExpectNoError(err)
}

func getExpectedRequestMadeByClientFor(rv string) []string {
	expectedRequestMadeByClient := []string{
		// corresponds to a streaming request made by the client to stream the secrets
		"allowWatchBookmarks=true&resourceVersionMatch=NotOlderThan&sendInitialEvents=true&watch=true",
	}
	if consistencydetector.IsDataConsistencyDetectionForWatchListEnabled() {
		// corresponds to a standard list request made by the consistency detector build in into the client
		expectedRequestMadeByClient = append(expectedRequestMadeByClient, fmt.Sprintf("resourceVersion=%s&resourceVersionMatch=Exact", rv))
	}
	return expectedRequestMadeByClient
}

type byName []v1.Secret

func (a byName) Len() int           { return len(a) }
func (a byName) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a byName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func newSecret(name string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}
