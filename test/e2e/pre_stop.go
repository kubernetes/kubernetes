package e2e

import (
	"fmt"
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func testPreStop(c *client.Client, ns string) {
	// This is the server that will receive the preStop notification
	podDescr := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "server",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				api.Container{
					Name: "server",
					Image: "gcr.io/google_containers/nettest:1.3",
					Ports: []api.ContainerPort{{ContainerPort: 8080}},
				},
			},
		},
	}
	By(fmt.Sprintf("Creating server pod %s in namespace %s", podDescr.Name, ns))
	podOut, err := c.Pods(ns).Create(podDescr)
	expectNoError(err, fmt.Sprintf("creating pod %s", podDescr.Name))

	// At the end of the test, clean up by removing the pod.
	defer func() {
		By("deleting the server pod")
		c.Pods(ns).Delete(podDescr.Name, nil)
	}()

	val := fmt.Sprintf("{\"foo\": \"epoch%d\"}", time.Now().Unix())

	preStopDescr := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "tester",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				api.Container{
					Name: "tester",
					Image: "busybox",
					Lifecycle: &api.Lifecycle{
						PreStop: &api.Handler{
							Exec: &api.ExecAction{
								Command: []string{
									"wget", "-O-", "--post-data=" + val, fmt.Sprintf("http://%s/write", podOut.Status.HostIP),
								},
							},
						},
					},
				},
			},
		},
	}

	By(fmt.Sprintf("Creating tester pod %s in namespace %s", podDescr.Name, ns))
	_, err = c.Pods(ns).Create(preStopDescr)
	expectNoError(err, fmt.Sprintf("creating pod %s", preStopDescr.Name))
	deletePreStop := true

	// At the end of the test, clean up by removing the pod.
	defer func() {
		if deletePreStop {
			By("deleting the tester pod")
			c.Pods(ns).Delete(preStopDescr.Name, nil)
		}
	}()

	By("Waiting for pods to come up.")
	err = waitForPodRunningInNamespace(c, podDescr.Name, ns)
	expectNoError(err, fmt.Sprintf("waiting for server pod to start"))

	err = waitForPodRunningInNamespace(c, preStopDescr.Name, ns)
	expectNoError(err, fmt.Sprintf("waiting for tester pod to start"))

	// Delete the pod with the preStop handler.
	By("Deleting pre-stop pod")
	if err := c.Pods(ns).Delete(preStopDescr.Name, nil); err == nil {
		deletePreStop = false
	}
	expectNoError(err, fmt.Sprintf("deleting pod: %s", preStopDescr.Name))

	// Validate that the server received the web poke.
	wait.Poll(time.Second*5, time.Second*60, func() (bool, error) {
		if body, err := c.Get().
			Namespace(ns).Prefix("proxy").
			Resource("pods").
			Name(podDescr.Name).
			Suffix("read").
			DoRaw(); err != nil {
			By(fmt.Sprintf("error validating prestop: %v", err))
		} else if string(body) == "{\"foo\": \"bar\"}" {
			return true, nil
		}
		return false, nil
	})
}

var _ = Describe("PreStop", func() {
	//This namespace is modified throughout the course of the test.
	var namespace *api.Namespace
	var c *client.Client = nil

	BeforeEach(func() {
		//Assert basic external connectivity.
		//Since this is not really a test of kubernetes in any way, we
		//leave it as a pre-test assertion, rather than a Ginko test.
		By("Executing a successful http request from the external internet")
		resp, err := http.Get("http://google.com")
		if err != nil {
			Failf("Unable to connect/talk to the internet: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			Failf("Unexpected error code, expected 200, got, %v (%v)", resp.StatusCode, resp)
		}

		By("Creating a kubernetes client")
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())

		By("Building a namespace api object")
		namespace, err = createTestingNS("prestoptest", c)
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		By(fmt.Sprintf("Destroying namespace for this suite %v", namespace.Name))
		if err := c.Namespaces().Delete(namespace.Name); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	It("should call prestop when killing a pod", func() {
		testPreStop(c, namespace.Name)
	})
})
