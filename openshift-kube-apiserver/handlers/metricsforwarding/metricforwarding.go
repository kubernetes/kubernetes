// used to provide metric access via check-endpoints in kube-apiserver static pods
package metricsforwarding

import (
	"fmt"
	"io/ioutil"
	"net/http"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
)

type metricForwarding struct {
	client *http.Client
}

func newMetricForwardingHandler() (*metricForwarding, error) {
	clientConfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: "/etc/kubernetes/static-pod-certs/configmaps/control-plane-node-kubeconfig/kubeconfig"}, nil).
		ClientConfig()
	if err != nil {
		return nil, err
	}
	// these are required to create the RESTClient. We want the REST client to get the correct authn/authz behavior
	clientConfig.GroupVersion = &schema.GroupVersion{Version: "if-this-is-used-find-the-bug"}
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	clientConfig.NegotiatedSerializer = codecs
	rest.SetKubernetesDefaults(clientConfig)

	// we don't need to ensure the identity because it's just localhost and we only allow certificate based authentication.
	clientConfig.Host = "localhost:17697"
	clientConfig.Insecure = true
	clientConfig.CAData = nil
	clientConfig.CAFile = ""
	if len(clientConfig.BearerToken) > 0 {
		return nil, fmt.Errorf("unexpected token in kubeconfig, this should be a cert based identity; if done manually, delete the file")
	}

	restClient, err := rest.RESTClientFor(clientConfig)
	if err != nil {
		return nil, err
	}

	return &metricForwarding{
		client: restClient.Client,
	}, nil
}

// these metrics must never show personally identifying information
const url = "https://localhost:17697/metrics"

func (h *metricForwarding) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	resp, err := h.client.Get(url)
	if err != nil {
		http.Error(w, "couldn't contact check-endpoints", http.StatusInternalServerError)
		klog.Warningf("Failed to get %q: %v", url, err)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "failed to read response from kube-apiserver", http.StatusInternalServerError)
		klog.Warningf("Failed to read the response body: %v", err)
		return
	}

	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}

func WithCheckEndpointsMetricsForwarding(handler http.Handler) http.Handler {
	metricForwarding, err := newMetricForwardingHandler()
	if err != nil {
		// don't fail, just log a message. This may happen on the bootstrap node for instance.
		// we will know to check this because we will see the ServiceMonitor fail.
		utilruntime.HandleError(err)
		return handler
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/debug/openshift/check-endpoints-metrics" || req.Method != http.MethodGet {
			handler.ServeHTTP(w, req)
			return
		}

		metricForwarding.ServeHTTP(w, req)
		return
	})
}
