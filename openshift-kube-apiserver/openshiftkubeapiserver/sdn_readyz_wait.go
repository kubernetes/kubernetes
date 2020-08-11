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

func newOpenshiftAPIServiceReachabilityCheck() *openshiftAPIServiceAvailabilityCheck {
	return &openshiftAPIServiceAvailabilityCheck{done: make(chan struct{})}
}

type openshiftAPIServiceAvailabilityCheck struct {
	// done indicates that this check is complete (success or failure) and the check should return true
	done chan struct{}
}

func (c *openshiftAPIServiceAvailabilityCheck) Name() string {
	return "openshift-apiservices-available"
}

func (c *openshiftAPIServiceAvailabilityCheck) Check(req *http.Request) error {
	select {
	case <-c.done:
		return nil
	default:
		return fmt.Errorf("check is not yet complete")
	}
}

func (c *openshiftAPIServiceAvailabilityCheck) checkForConnection(context genericapiserver.PostStartHookContext) {
	defer utilruntime.HandleCrash()

	reachedOpenshiftAPIServer := make(chan struct{})
	noOpenshiftAPIServer := make(chan struct{})
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

	// Start a thread which repeatedly tries to connect to any openshift-apiserver endpoint.
	//  1. if the openshift-apiserver endpoint doesn't exist, logs a warning and reports ready
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
			openshiftEndpoints, err := kubeClient.CoreV1().Endpoints("openshift-apiserver").Get(ctx, "api", metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				// if we have no openshift apiserver endpoint, we have no reason to wait
				klog.Warning("api.openshift-apiserver.svc endpoints were not found")
				close(noOpenshiftAPIServer)
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
						close(reachedOpenshiftAPIServer)
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
		utilruntime.HandleError(fmt.Errorf("openshift.io-openshift-apiserver-reachable never reached openshift apiservice"))
		return
	case <-context.StopCh:
		utilruntime.HandleError(fmt.Errorf("openshift.io-openshift-apiserver-reachable interrupted"))
		return
	case <-noOpenshiftAPIServer:
		utilruntime.HandleError(fmt.Errorf("openshift.io-openshift-apiserver-reachable did not find an openshift-apiserver endpoint"))
		return

	case <-reachedOpenshiftAPIServer:
		end := time.Now()
		klog.Infof("reached openshift apiserver via SDN after %v milliseconds", end.Sub(start).Milliseconds())
		return
	}
}
