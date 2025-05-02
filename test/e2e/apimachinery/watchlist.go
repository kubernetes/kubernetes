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
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/consistencydetector"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("API Streaming (aka. WatchList)", framework.WithFeatureGate(features.WatchList), framework.WithSerial(), func() {
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
					options.LabelSelector = "watchlist=true"
					return f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Watch(context.TODO(), options)
				},
			},
			&v1.Secret{},
			time.Duration(0),
			nil,
		)

		expectedSecrets := addWellKnownSecrets(ctx, f)

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

		expectedSecrets := addWellKnownSecrets(ctx, f)

		rt, clientConfig := clientConfigWithRoundTripper(f)
		wrappedKubeClient, err := kubernetes.NewForConfig(clientConfig)
		framework.ExpectNoError(err)

		ginkgo.By("Streaming secrets from the server")
		secretList, err := wrappedKubeClient.CoreV1().Secrets(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: "watchlist=true"})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying if the secret list was properly streamed")
		streamedSecrets := secretList.Items
		gomega.Expect(cmp.Equal(expectedSecrets, streamedSecrets)).To(gomega.BeTrueBecause("data received via watchlist must match the added data"))

		ginkgo.By("Verifying if expected requests were sent to the server")
		expectedRequestsMadeByKubeClient := getExpectedRequestsMadeByClientFor(secretList.ResourceVersion)
		gomega.Expect(rt.actualRequests).To(gomega.Equal(expectedRequestsMadeByKubeClient))
	})
	ginkgo.It("should be requested by dynamic client's List method when WatchListClient is enabled", func(ctx context.Context) {
		featuregatetesting.SetFeatureGateDuringTest(ginkgo.GinkgoTB(), utilfeature.DefaultFeatureGate, featuregate.Feature(clientfeatures.WatchListClient), true)

		ginkgo.By(fmt.Sprintf("Adding 5 secrets to %s namespace", f.Namespace.Name))
		expectedSecrets := addWellKnownUnstructuredSecrets(ctx, f)

		rt, clientConfig := clientConfigWithRoundTripper(f)
		wrappedDynamicClient, err := dynamic.NewForConfig(clientConfig)
		framework.ExpectNoError(err)

		ginkgo.By("Streaming secrets from the server")
		secretList, err := wrappedDynamicClient.Resource(v1.SchemeGroupVersion.WithResource("secrets")).Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: "watchlist=true"})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying if the secret list was properly streamed")
		streamedSecrets := secretList.Items
		gomega.Expect(cmp.Equal(expectedSecrets, streamedSecrets)).To(gomega.BeTrueBecause("data received via watchlist must match the added data"))
		gomega.Expect(secretList.GetObjectKind().GroupVersionKind()).To(gomega.Equal(v1.SchemeGroupVersion.WithKind("SecretList")))

		ginkgo.By("Verifying if expected requests were sent to the server")
		expectedRequestsMadeByDynamicClient := getExpectedRequestsMadeByClientFor(secretList.GetResourceVersion())
		gomega.Expect(rt.actualRequests).To(gomega.Equal(expectedRequestsMadeByDynamicClient))
	})
	ginkgo.It("should be requested by metadata client's List method when WatchListClient is enabled", func(ctx context.Context) {
		featuregatetesting.SetFeatureGateDuringTest(ginkgo.GinkgoTB(), utilfeature.DefaultFeatureGate, featuregate.Feature(clientfeatures.WatchListClient), true)

		metaClient, err := metadata.NewForConfig(f.ClientConfig())
		framework.ExpectNoError(err)
		expectedMetaSecrets := []metav1.PartialObjectMetadata{}
		for _, addedSecret := range addWellKnownSecrets(ctx, f) {
			addedSecretMeta, err := metaClient.Resource(v1.SchemeGroupVersion.WithResource("secrets")).Namespace(f.Namespace.Name).Get(ctx, addedSecret.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			expectedMetaSecrets = append(expectedMetaSecrets, *addedSecretMeta)
		}

		rt, clientConfig := clientConfigWithRoundTripper(f)
		wrappedMetaClient, err := metadata.NewForConfig(clientConfig)
		framework.ExpectNoError(err)

		ginkgo.By("Streaming secrets metadata from the server")
		secretMetaList, err := wrappedMetaClient.Resource(v1.SchemeGroupVersion.WithResource("secrets")).Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: "watchlist=true"})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying if the secret meta list was properly streamed")
		streamedMetaSecrets := secretMetaList.Items
		gomega.Expect(cmp.Equal(expectedMetaSecrets, streamedMetaSecrets)).To(gomega.BeTrueBecause("data received via watchlist must match the added data"))

		ginkgo.By("Verifying if expected requests were sent to the server")
		expectedRequestsMadeByMetaClient := getExpectedRequestsMadeByClientFor(secretMetaList.GetResourceVersion())
		gomega.Expect(rt.actualRequests).To(gomega.Equal(expectedRequestsMadeByMetaClient))
	})

	// Validates unsupported Accept headers in WatchList.
	// Sets AcceptContentType to "application/json;as=Table", which the API doesn't support, returning a 406 error.
	// After the 406, the client falls back to a regular list request.
	ginkgo.It("doesn't support receiving resources as Tables", func(ctx context.Context) {
		featuregatetesting.SetFeatureGateDuringTest(ginkgo.GinkgoTB(), utilfeature.DefaultFeatureGate, featuregate.Feature(clientfeatures.WatchListClient), true)

		ginkgo.By(fmt.Sprintf("Adding 5 secrets to %s namespace", f.Namespace.Name))
		_ = addWellKnownUnstructuredSecrets(ctx, f)

		rt, clientConfig := clientConfigWithRoundTripper(f)
		modifiedClientConfig := dynamic.ConfigFor(clientConfig)
		modifiedClientConfig.AcceptContentTypes = strings.Join([]string{
			fmt.Sprintf("application/json;as=Table;v=%s;g=%s", metav1.SchemeGroupVersion.Version, metav1.GroupName),
		}, ",")
		modifiedClientConfig.GroupVersion = &v1.SchemeGroupVersion
		restClient, err := rest.RESTClientFor(modifiedClientConfig)
		framework.ExpectNoError(err)
		wrappedDynamicClient := dynamic.New(restClient)

		// note that the client in case of an error (406) will fall back
		// to a standard list request thus the overall call passes
		ginkgo.By("Streaming secrets as Table from the server")
		secretTable, err := wrappedDynamicClient.Resource(v1.SchemeGroupVersion.WithResource("secrets")).Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: "watchlist=true"})
		framework.ExpectNoError(err)
		gomega.Expect(secretTable.GetObjectKind().GroupVersionKind()).To(gomega.Equal(metav1.SchemeGroupVersion.WithKind("Table")))

		ginkgo.By("Verifying if expected response was sent by the server")
		gomega.Expect(rt.actualResponseStatuses[0]).To(gomega.Equal("406 Not Acceptable"))
		expectedRequestsMadeByDynamicClient := getExpectedRequestsMadeByClientWhenFallbackToListFor(secretTable.GetResourceVersion())
		gomega.Expect(rt.actualRequests).To(gomega.Equal(expectedRequestsMadeByDynamicClient))

	})

	// Sets AcceptContentType to both "application/json;as=Table" and "application/json".
	// Unlike the previous test, no 406 error occurs, as the API falls back to "application/json" and returns a valid response.
	ginkgo.It("falls backs to supported content type when when receiving resources as Tables was requested", func(ctx context.Context) {
		featuregatetesting.SetFeatureGateDuringTest(ginkgo.GinkgoTB(), utilfeature.DefaultFeatureGate, featuregate.Feature(clientfeatures.WatchListClient), true)

		ginkgo.By(fmt.Sprintf("Adding 5 secrets to %s namespace", f.Namespace.Name))
		expectedSecrets := addWellKnownUnstructuredSecrets(ctx, f)

		rt, clientConfig := clientConfigWithRoundTripper(f)
		modifiedClientConfig := dynamic.ConfigFor(clientConfig)
		modifiedClientConfig.AcceptContentTypes = strings.Join([]string{
			fmt.Sprintf("application/json;as=Table;v=%s;g=%s", metav1.SchemeGroupVersion.Version, metav1.GroupName),
			"application/json",
		}, ",")
		modifiedClientConfig.GroupVersion = &v1.SchemeGroupVersion
		restClient, err := rest.RESTClientFor(modifiedClientConfig)
		framework.ExpectNoError(err)
		wrappedDynamicClient := dynamic.New(restClient)

		ginkgo.By("Streaming secrets from the server")
		secretList, err := wrappedDynamicClient.Resource(v1.SchemeGroupVersion.WithResource("secrets")).Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: "watchlist=true"})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying if the secret list was properly streamed")
		streamedSecrets := secretList.Items
		gomega.Expect(cmp.Equal(expectedSecrets, streamedSecrets)).To(gomega.BeTrueBecause("data received via watchlist must match the added data"))

		ginkgo.By("Verifying if expected requests were sent to the server")
		expectedRequestsMadeByDynamicClient := getExpectedRequestsMadeByClientFor(secretList.GetResourceVersion())
		gomega.Expect(rt.actualRequests).To(gomega.Equal(expectedRequestsMadeByDynamicClient))
	})
})

type roundTripper struct {
	actualRequests         []string
	actualResponseStatuses []string
	delegate               http.RoundTripper
}

func (r *roundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	r.actualRequests = append(r.actualRequests, req.URL.RawQuery)
	rsp, err := r.delegate.RoundTrip(req)
	if rsp != nil {
		r.actualResponseStatuses = append(r.actualResponseStatuses, rsp.Status)
	}
	return rsp, err
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

// corresponds to a streaming request made by the client to stream the secrets
var expectedStreamingRequestMadeByClient = func() string {
	params := url.Values{}
	params.Add("allowWatchBookmarks", "true")
	params.Add("labelSelector", "watchlist=true")
	params.Add("resourceVersionMatch", "NotOlderThan")
	params.Add("sendInitialEvents", "true")
	params.Add("watch", "true")
	return params.Encode()
}()

func getExpectedListRequestMadeByConsistencyDetectorFor(rv string) string {
	params := url.Values{}
	params.Add("labelSelector", "watchlist=true")
	params.Add("resourceVersion", rv)
	params.Add("resourceVersionMatch", "Exact")
	return params.Encode()
}

func getExpectedRequestsMadeByClientFor(rv string) []string {
	expectedRequestMadeByClient := []string{
		expectedStreamingRequestMadeByClient,
	}
	if consistencydetector.IsDataConsistencyDetectionForWatchListEnabled() {
		// corresponds to a standard list request made by the consistency detector build in into the client
		expectedRequestMadeByClient = append(expectedRequestMadeByClient, getExpectedListRequestMadeByConsistencyDetectorFor(rv))
	}
	return expectedRequestMadeByClient
}

func getExpectedRequestsMadeByClientWhenFallbackToListFor(rv string) []string {
	expectedRequestMadeByClient := []string{
		expectedStreamingRequestMadeByClient,
		// corresponds to a list request made by the client
		func() string {
			params := url.Values{}
			params.Add("labelSelector", "watchlist=true")
			return params.Encode()
		}(),
	}
	if consistencydetector.IsDataConsistencyDetectionForListEnabled() {
		// corresponds to a standard list request made by the consistency detector build in into the client
		expectedRequestMadeByClient = append(expectedRequestMadeByClient, getExpectedListRequestMadeByConsistencyDetectorFor(rv))
	}
	return expectedRequestMadeByClient
}

func addWellKnownSecrets(ctx context.Context, f *framework.Framework) []v1.Secret {
	ginkgo.By(fmt.Sprintf("Adding 5 secrets to %s namespace", f.Namespace.Name))
	var secrets []v1.Secret
	for i := 1; i <= 5; i++ {
		secret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, newSecret(fmt.Sprintf("secret-%d", i)), metav1.CreateOptions{})
		framework.ExpectNoError(err)
		secrets = append(secrets, *secret)
	}
	return secrets
}

// addWellKnownUnstructuredSecrets exists because secrets from addWellKnownSecrets
// don't have type info and cannot be converted.
func addWellKnownUnstructuredSecrets(ctx context.Context, f *framework.Framework) []unstructured.Unstructured {
	var secrets []unstructured.Unstructured
	for i := 1; i <= 5; i++ {
		unstructuredSecret, err := runtime.DefaultUnstructuredConverter.ToUnstructured(newSecret(fmt.Sprintf("secret-%d", i)))
		framework.ExpectNoError(err)
		secret, err := f.DynamicClient.Resource(v1.SchemeGroupVersion.WithResource("secrets")).Namespace(f.Namespace.Name).Create(ctx, &unstructured.Unstructured{Object: unstructuredSecret}, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		secrets = append(secrets, *secret)
	}
	return secrets
}

type byName []v1.Secret

func (a byName) Len() int           { return len(a) }
func (a byName) Less(i, j int) bool { return a[i].Name < a[j].Name }
func (a byName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func newSecret(name string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"watchlist": "true"},
		},
	}
}
