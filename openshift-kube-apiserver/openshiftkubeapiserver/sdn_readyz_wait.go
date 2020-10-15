package openshiftkubeapiserver

import (
	gocontext "context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
)

func newOpenshiftAPIServiceReachabilityCheck() *aggregatedAPIServiceAvailabilityCheck {
	return newAggregatedAPIServiceReachabilityCheck("openshift-apiserver", "api")
}

func newOAuthPIServiceReachabilityCheck() *aggregatedAPIServiceAvailabilityCheck {
	return newAggregatedAPIServiceReachabilityCheck("openshift-oauth-apiserver", "api")
}

// if the API service is not found, then this check returns quickly.
// if the endpoint is not accessible within 60 seconds, we report ready no matter what
// otherwise, wait for up to 60 seconds to be able to reach the apiserver
func newAggregatedAPIServiceReachabilityCheck(namespace, service string) *aggregatedAPIServiceAvailabilityCheck {
	return &aggregatedAPIServiceAvailabilityCheck{
		done:        make(chan struct{}),
		namespace:   namespace,
		serviceName: service,
	}
}

type aggregatedAPIServiceAvailabilityCheck struct {
	// done indicates that this check is complete (success or failure) and the check should return true
	done chan struct{}

	// namespace is the namespace hosting the service for the aggregated api
	namespace string
	// serviceName is used to get a list of endpoints to directly dial
	serviceName string
}

func (c *aggregatedAPIServiceAvailabilityCheck) Name() string {
	return fmt.Sprintf("%s-%s-available", c.serviceName, c.namespace)
}

func (c *aggregatedAPIServiceAvailabilityCheck) Check(req *http.Request) error {
	select {
	case <-c.done:
		return nil
	default:
		return fmt.Errorf("check is not yet complete")
	}
}

func (c *aggregatedAPIServiceAvailabilityCheck) checkForConnection(context genericapiserver.PostStartHookContext) {
	defer utilruntime.HandleCrash()

	reachedAggregatedAPIServer := make(chan struct{})
	noAggregatedAPIServer := make(chan struct{})
	waitUntilCh := make(chan struct{})
	defer func() {
		close(waitUntilCh) // this stops the endpoint check
		close(c.done)      // once this method is done, the ready check should return true
	}()
	start := time.Now()

	kubeClient, err := kubernetes.NewForConfig(context.LoopbackClientConfig)
	if err != nil {
		// shouldn't happen.  this means the loopback config didn't work.
		panic(err)
	}

	// Start a thread which repeatedly tries to connect to any aggregated apiserver endpoint.
	//  1. if the aggregated apiserver endpoint doesn't exist, logs a warning and reports ready
	//  2. if a connection cannot be made, after 60 seconds logs an error and reports ready -- this avoids a rebootstrapping cycle
	//  3. as soon as a connection can be made, logs a time to be ready and reports ready.
	go func() {
		defer utilruntime.HandleCrash()

		client := http.Client{
			Transport: &http.Transport{
				// since any http return code satisfies us, we don't bother to send credentials.
				// we don't care about someone faking a response and we aren't sending credentials, so we don't check the server CA
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			},
			Timeout: 1 * time.Second, // these should all be very fast.  if none work, we continue anyway.
		}

		wait.PollImmediateUntil(1*time.Second, func() (bool, error) {
			ctx := gocontext.TODO()
			openshiftEndpoints, err := kubeClient.CoreV1().Endpoints(c.namespace).Get(ctx, c.serviceName, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				// if we have no aggregated apiserver endpoint, we have no reason to wait
				klog.Warningf("%s.%s.svc endpoints were not found", c.serviceName, c.namespace)
				close(noAggregatedAPIServer)
				return true, nil
			}
			if err != nil {
				utilruntime.HandleError(err)
				return false, nil
			}
			for _, subset := range openshiftEndpoints.Subsets {
				for _, address := range subset.Addresses {
					url := fmt.Sprintf("https://%v", net.JoinHostPort(address.IP, "8443"))
					resp, err := client.Get(url)
					if err == nil { // any http response is fine.  it means that we made contact
						response, dumpErr := httputil.DumpResponse(resp, true)
						klog.V(4).Infof("reached to connect to %q: %v\n%v", url, dumpErr, string(response))
						close(reachedAggregatedAPIServer)
						resp.Body.Close()
						return true, nil
					}
					klog.V(2).Infof("failed to connect to %q: %v", url, err)
				}
			}

			return false, nil
		}, waitUntilCh)
	}()

	select {
	case <-time.After(60 * time.Second):
		// if we timeout, always return ok so that we can start from a case where all kube-apiservers are down and the SDN isn't coming up
		utilruntime.HandleError(fmt.Errorf("%s never reached apiserver", c.Name()))
		return
	case <-context.StopCh:
		utilruntime.HandleError(fmt.Errorf("%s interrupted", c.Name()))
		return
	case <-noAggregatedAPIServer:
		utilruntime.HandleError(fmt.Errorf("%s did not find an %s endpoint", c.Name(), c.namespace))
		return

	case <-reachedAggregatedAPIServer:
		end := time.Now()
		klog.Infof("reached %s via SDN after %v milliseconds", c.namespace, end.Sub(start).Milliseconds())
		return
	}
}
