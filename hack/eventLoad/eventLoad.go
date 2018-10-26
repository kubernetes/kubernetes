/*
Copyright 2018 The Kubernetes Authors.

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

// The event load system is responsible for running load against an Kubernetes cluster
// then playing some nasty games to see how well it functions when we do things like
// kill master components.

package main

import (
	"github.com/spf13/cobra"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	appsv1 "k8s.io/api/apps/v1"
	extv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/apimachinery/pkg/util/intstr"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/wait"
	"path/filepath"
	"os/exec"
	"bytes"
	"io"
	"crypto/rand"
	"math/big"
)

var (
	eventLoadCmd = &cobra.Command{
		Short: "Run a simple event related load test against Kubernetes",
		Long: `TBD`,
		Run: func(cmd *cobra.Command, args []string) {
			runEventLoad()
		},
	}
	opts = eventLoadOpts{}
)

type eventLoadOpts struct {
	numSites int
	concurrentSites int
	startInterval int64
	kubeConfig string
	image string
	killList []string
	killInterval int64
	tidy bool
	inCluster bool
}

type resultMessage struct {
	success bool
	err error
}

type command string

const (
	Test     command = "Test"
	Shutdown command = "Shutdown"
)

type site struct {
	namespace string
	site string
	secret string
	deployment string
	service string
	volume string
}

type testSite struct {
	resultChannel chan resultMessage
	commandChannel chan command
	id int
	running bool
}

type processInfo struct {
	owner string
	pid string
	ppid string
	cpu string
	start string
	terminal string
	time string
	command []string
}

func main() {
	flags := eventLoadCmd.Flags()
	flags.IntVar(&opts.numSites, "numSites", 10, "number of sites to create during the test")
	flags.IntVar(&opts.concurrentSites, "concurrentSites", 3, "number of concurrent sites allowed in the test")
	flags.Int64Var(&opts.startInterval, "startInterval", 30, "seconds between starting customer sites")
	flags.StringVar(&opts.kubeConfig, "kubeconfig", "", "absolute path to the kubeconfig file")
	flags.StringVar(&opts.image, "image", "gcr.io/wfender-test/tomcat-amd64:1522278020", "primary docker image")
	flags.StringSliceVar(&opts.killList, "killList", []string {}, "List of the services which the test should periodically kill")
	flags.Int64Var(&opts.killInterval, "killInterval", 0, "seconds between killing services")
	flags.BoolVar(&opts.tidy, "tidy", true, "should we clean up all the sites")
	flags.BoolVar(&opts.inCluster, "inCluster", false, "should be assume the test is running in cluster")
	eventLoadCmd.Execute()
}

// runEventLoad will actually run the test.
func runEventLoad() {
	err := validateOptions()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	clientset, err := initClientSet()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	err = verifyNamespaces(clientset)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	stop := make(chan struct{})
	if opts.inCluster && len(opts.killList) > 0 && opts.killInterval > 0 {
		go killServices(stop)
	} else {
		fmt.Printf("Skipping the chaos killer function! (%t,%t,%t)\r\n", opts.inCluster, len(opts.killList) > 0, opts.killInterval > 0)
	}

	testSites := make([]testSite, opts.concurrentSites)
	for id := 0; id < opts.numSites; id++ {
		index := id % opts.concurrentSites
		if testSites[index].running {
			fmt.Printf("Shutting down test site %d\r\n", testSites[index].id)
			testSites[index].commandChannel <- Shutdown
			fmt.Printf("Getting results for test site %d\r\n", testSites[index].id)
			result := <- testSites[index].resultChannel
			if result.success {
				fmt.Printf("Test site %d succeeded!\r\n", testSites[index].id)
			} else {
				fmt.Printf("Test site %d failed with error %v!\r\n", testSites[index].id, result.err)
			}
			close(testSites[index].resultChannel)
			close(testSites[index].commandChannel)
		}
		testSites[index].resultChannel = make(chan resultMessage, 1)
		testSites[index].commandChannel = make(chan command, 1)
		testSites[index].id = id
		testSites[index].running = true
		fmt.Printf("Starting test site %d\r\n", testSites[index].id)
		go runTestSite(testSites[index].commandChannel, testSites[index].resultChannel, clientset, testSites[index].id, opts.tidy)
		time.Sleep(time.Duration(opts.startInterval) * time.Second)
	}

	for index := 0; index < opts.concurrentSites; index++ {
		testSites[index].commandChannel <- Shutdown
		result := <-testSites[index].resultChannel
		if result.success {
			fmt.Printf("Test site %d succeeded!\r\n ", testSites[index].id)
		} else {
			fmt.Printf("Test site %d failed with error %v!\r\n", testSites[index].id, result.err)
		}
		testSites[index].running = false
	}
	stop <- struct{}{}

	fmt.Println("Test finished")
}

func killServices(stop chan struct{}) {
	for {
		select {
		case <-stop:
			return
		default:
		}
		time.Sleep(time.Duration(opts.killInterval) * time.Second)
		select {
		case <-stop:
			return
		default:
		}
		index, err := rand.Int(rand.Reader, big.NewInt(int64(len(opts.killList))))
		if err != nil {
			fmt.Printf("Failed to pick a service to kill, %v!\r\n", err)
		}
		service := opts.killList[int(index.Int64())]
		fmt.Printf("Going to kill %s at time %v!\r\n", service, time.Now())
		pinfo, err := getProcessInfo(service)
		if err != nil {
			continue
		}
		cmd := exec.Command("sudo", "kill", pinfo.pid)
		err = cmd.Run()
		if err != nil {
			fmt.Printf("Failed to kill %s(%s) error %v\r\n", service, pinfo.pid, err)
			continue
		}
		fmt.Printf("Killed service %s(%s) at time %v.\r\n", service, pinfo.pid, time.Now())
	}
}

func getProcessInfo(service string) (*processInfo, error) {
	cmd1 := exec.Command("ps", "-ef")
	cmd2 := exec.Command("egrep", service)
	cmd3 := exec.Command("egrep", "-v", "grep|eventLoad")
	r1, w1 := io.Pipe()
	r2, w2 := io.Pipe()
	cmd1.Stdout = w1
	cmd2.Stdin = r1
	cmd2.Stdout = w2
	cmd3.Stdin = r2
	var out bytes.Buffer
	cmd3.Stdout = &out
	err := cmd1.Start()
	if err != nil {
		fmt.Printf("Failed to kill %s as we could not ps! %v\r\n", service, err)
		return nil, err
	}
	err = cmd2.Start()
	if err != nil {
		fmt.Printf("Failed to kill %s as we could not egrep! %v\r\n", service, err)
		return nil, err
	}
	err = cmd3.Start()
	if err != nil {
		fmt.Printf("Failed to kill %s as we could not exclude grep! %v\r\n", service, err)
		return nil, err
	}
	err = cmd1.Wait()
	if err != nil {
		fmt.Printf("Failed to kill %s as ps failed! %v\r\n", service, err)
		return nil, err
	}
	err = w1.Close()
	if err != nil {
		fmt.Printf("Failed to kill %s as pipe close failed! %v\r\n", service, err)
		return nil, err
	}
	err = cmd2.Wait()
	if err != nil {
		fmt.Printf("Failed to kill %s as egrep failed! %v\r\n", service, err)
		return nil, err
	}
	err = w2.Close()
	if err != nil {
		fmt.Printf("Failed to kill %s as second pipe close failed! %v\r\n", service, err)
		return nil, err
	}
	err = cmd3.Wait()
	if err != nil {
		fmt.Printf("Failed to kill %s as exclude grep failed! %v\r\n", service, err)
		return nil, err
	}
	results := strings.Split(out.String(), "\n")
	if len(results) != 2 {
		fmt.Printf("Failed to kill %s as we got too many results! %v\r\n", service, results)
		return nil, errors.New(fmt.Sprintf("Got %d results looking for %s!\r\n", len(results) - 1, service))
	}
	r := strings.NewReplacer("    ", " ", "   ", " ", "  ", " ")
	params := strings.Split(r.Replace(r.Replace(results[0])), " ")
	result := &processInfo{
		owner: params[0],
		pid: params[1],
		ppid: params[2],
		cpu: params[3],
		start: params[4],
		terminal: params[5],
		time: params[6],
		command: params[7:],
	}
	return result, nil
}

func runTestSite(requestChannel chan command, resultChannel chan resultMessage, clientset *kubernetes.Clientset, index int, tidy bool) {
	namespace := fmt.Sprintf("site-ns-%d", index)
	siteName := fmt.Sprintf("site-%d", index)
	secret := fmt.Sprintf("%s-secret", siteName)
	deployment := fmt.Sprintf("cust-%s", siteName)
	service := fmt.Sprintf("%s-%s-tomcat", namespace, siteName)
	volume := fmt.Sprintf("%s-%s.www", deployment, namespace)

	result := resultMessage{success: false, err: nil}
	siteInfo := site {
		namespace: namespace,
		site: siteName,
		secret: secret,
		deployment: deployment,
		service: service,
		volume: volume,
	}
	defer func() {
		if tidy {
			fmt.Printf("Tearing down site %s!\r\n", siteInfo.namespace)
			err := tearDownSite(clientset, siteInfo)
			if err != nil && result.success {
				result.success = false
				result.err = err
			}
		}
		resultChannel <- result
	}()

	err := createTestSite(clientset, siteInfo)
	if err != nil {
		result.err = err
		return
	}

	cmd := Test
	for cmd != Shutdown {
		ip, err := getExternalIP(clientset, siteInfo.namespace, siteInfo.deployment, siteInfo.service)
		if err != nil {
			result.err = err
			return
		}
		tomcaturl := fmt.Sprintf("http://%s/sample/", ip)
		err = verifyHttpEndpoint(tomcaturl)
		if err != nil {
			result.err = err
			return
		}
		fmt.Printf("Request to %s succeeded!!\r\n", tomcaturl)
		select {
			case cmd = <-requestChannel:
				fmt.Printf("Received command %v for test site %d.\r\n", cmd, index)
			default:
				time.Sleep(time.Duration(1) * time.Second)
		}
	}

	result.success = true
}

func createTestSite(clientset *kubernetes.Clientset, siteInfo site) error {
	err := initNamespace(clientset, siteInfo.namespace)
	if err != nil {
		return err
	}
	err = initSecret(clientset, siteInfo.namespace, siteInfo.secret)
	if err != nil {
		return err
	}
	err = initNetworkPolicy(clientset, siteInfo.namespace, "inbound")
	if err != nil {
		return err
	}
	err = initPersistentVolume(clientset, siteInfo.volume)
	if err != nil {
		return err
	}
	err = initPersistentVolumeClaim(clientset, siteInfo.namespace, siteInfo.volume)
	if err != nil {
		return err
	}
	err = initDeployment(clientset, siteInfo.namespace, siteInfo.deployment, siteInfo.site, opts.image, siteInfo.secret)
	if err != nil {
		return err
	}
	err = initService(clientset, siteInfo.namespace, siteInfo.service, siteInfo.site)
	if err != nil {
		return err
	}
	if ! opts.inCluster {
		// Skip ingress if running "in cluster"
		err = initIngress(clientset, siteInfo.namespace, siteInfo.deployment, siteInfo.service)
		if err != nil {
			return err
		}
	}

	// Cluster state has been state, verify/wait till its now in a working state.

	ok, err := verifyPods(clientset, siteInfo.namespace, 2)
	if err != nil {
		return err
	}
	if !ok {
		return errors.New("unable to find pods for your site")
	}
	ok, err = verifyEndpoints(clientset, siteInfo.namespace, siteInfo.service, 2)
	if err != nil {
		return err
	}
	if !ok {
		return errors.New("unable to find endpoints for your site")
	}
	if ! opts.inCluster {
		ok, err = verifyIngress(clientset, siteInfo.namespace, siteInfo.deployment, 1)
		if err != nil {
			return err
		}
		if !ok {
			return errors.New("unable to find ingress for your site")
		}
	}
	return nil
}

func tearDownSite(clientset *kubernetes.Clientset, siteInfo site) error {
	if ! opts.inCluster {
		err := deleteIngress(clientset, siteInfo.namespace, siteInfo.deployment, siteInfo.site)
		if err != nil {
			return err
		}
	}
	err := deleteService(clientset, siteInfo.namespace, siteInfo.service, siteInfo.site)
	if err != nil {
		return err
	}
	err = deleteDeployment(clientset, siteInfo.namespace, siteInfo.deployment, siteInfo.site, opts.image, siteInfo.secret)
	if err != nil {
		return err
	}
	err = deletePersistentVolumeClaim(clientset, siteInfo.namespace, siteInfo.volume)
	if err != nil {
		return err
	}
	err = deletePersistentVolume(clientset, siteInfo.volume)
	if err != nil {
		return err
	}
	err = deleteNetworkPolicy(clientset, siteInfo.namespace, "inbound")
	if err != nil {
		return err
	}
	err = deleteSecret(clientset, siteInfo.namespace, siteInfo.secret)
	if err != nil {
		return err
	}
	err = deleteNamespace(clientset, siteInfo.namespace)
	if err != nil {
		return err
	}

	return nil
}

// validateOptions will actually validate the inputs.
func validateOptions() error {
	if opts.numSites < 1 || opts.numSites > 10000 {
		errMsg := fmt.Sprintf("Number of sites should be between 0 and 10000, not %d.", opts.numSites)
		return errors.New(errMsg)
	}
	if opts.kubeConfig == "" {
		if ! opts.inCluster {
			home := homeDir()
			fmt.Printf("Got a home of %s.\r\n", home)
			if home == "" {
				errMsg := fmt.Sprintf("Need with kubeconfig set or a HOME env variable set")
				return errors.New(errMsg)
			}
			kubeconfig := filepath.Join(home, ".kube", "config")
			fmt.Printf("Got a kubeconfig of %s.\r\n", kubeconfig)
			opts.kubeConfig = kubeconfig
		}
	}

	if opts.concurrentSites < 1 || opts.concurrentSites > opts.numSites {
		errMsg := fmt.Sprintf("Number of concurrent sites should be between 1 and # sites, not %d.", opts.concurrentSites)
		return errors.New(errMsg)
	}
	if opts.startInterval < 1 || opts.startInterval > 1800 {
		errMsg := fmt.Sprintf("Number of concurrent sites should be between 1 and 1800, not %d.", opts.startInterval)
		return errors.New(errMsg)
	}
	if opts.image == "" {
		errMsg := fmt.Sprintf("Need an actual container image with which to run the test.")
		return errors.New(errMsg)
	}
	return nil
}

func homeDir() string {
	if h := os.Getenv("HOME"); h != "" {
		return h
	}
	return os.Getenv("USERPROFILE") // windows
}

func initClientSet() (*kubernetes.Clientset, error) {
	// use the current context in kubeconfig
	var config *rest.Config
	var err error
	if opts.kubeConfig != "" {
		fmt.Printf("Building config with %s.\r\n", opts.kubeConfig)
		config, err = clientcmd.BuildConfigFromFlags("", opts.kubeConfig)
	} else {
		fmt.Printf("Building config using clientcmd.DefaultClientConfig.\r\n")
		config, err = clientcmd.DefaultClientConfig.ClientConfig()
	}
	if err != nil {
		return nil, err
	}

	// create the clientset
	// fmt.Printf("Building client with config %v.\r\n", config)
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	return clientset, nil
}

// requests the namespaces list
func verifyNamespaces(clientset *kubernetes.Clientset) error {
	// get the namespaces
	missingDefault := true
	missingPublic := true
	missingSystem := true
	nslist, err := clientset.CoreV1().Namespaces().List(metav1.ListOptions{})
	if err != nil {
		return err
	}
	for _, ns := range nslist.Items {
		if ns.Status.Phase != corev1.NamespaceActive {
			continue
		}
		switch (ns.Name) {
		case "default":
			missingDefault = false
		case "kube-public":
			missingPublic = false
		case "kube-system":
			missingSystem = false

		}
	}
	if missingDefault || missingPublic || missingSystem {
		return errors.New("Missing kubernetes namespace")
	}

	return nil
}

// verifyHttpEndpoint hits the relevant url and ensures it is working.
func verifyHttpEndpoint(endpoint string) error {
	var body []byte
	var termErr error
	wait.Poll(5*time.Second, 300*time.Second, func() (bool, error) {
		termErr = nil
		resp, err := http.Get(endpoint)
		if err != nil {
			termErr = err
			return false, nil
		}
		defer resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode > 299 {
			errMsg := fmt.Sprintf("received response %d calling GET on %s.", resp.StatusCode, endpoint)
			termErr = errors.New(errMsg)
			return false, nil
		}
		body, err = ioutil.ReadAll(resp.Body)
		if err != nil {
			fmt.Printf("error on ReadAll %T: %v.\r\n", err, err)
			termErr = err
			return false, nil
		}
		return true, nil
	})
	if termErr != nil {
		fmt.Printf("error on wait.Poll(Get) %T: %v.\r\n", termErr, termErr)
		return termErr
	}
	r := strings.NewReplacer("\r", "", "\n", "")
	html := r.Replace(fmt.Sprintf("%s", body))
	re := regexp.MustCompile("^<html>.*</html>$")
	if ! re.MatchString(html) {
		errMsg := fmt.Sprintf("received response %s for %s was bad.", html, endpoint)
		return errors.New(errMsg)
	}
	return nil
}

func initNamespace(clientset *kubernetes.Clientset, namespace string) error {
	_, err := clientset.CoreV1().Namespaces().Get(namespace, metav1.GetOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
	} else {
		return nil
	}

	_, err = clientset.CoreV1().Namespaces().Create(&corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespace,
		},
	})
	if err != nil {
		return err
	}

	_, err = clientset.CoreV1().Namespaces().Get(namespace, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func deleteNamespace(clientset *kubernetes.Clientset, namespace string) error {
	err := clientset.CoreV1().Namespaces().Delete(namespace, &metav1.DeleteOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
		// Apparently the Ingress was already deleted, ignoring.
		return nil
	}

	ns, err := clientset.CoreV1().Namespaces().Get(namespace, metav1.GetOptions{})
	if err == nil {
		errors.New(fmt.Sprintf("failed to delete namespace %s, namespace has status %v", namespace, ns.Status))
	}
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		return err
	}
	if se.Status().Reason != metav1.StatusReasonNotFound {
		return err
	}
	return nil
}

func initSecret(clientset *kubernetes.Clientset, namespace string, name string) error {
	_, err := clientset.CoreV1().Secrets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
	} else {
		return nil
	}

	_, err = clientset.CoreV1().Secrets(namespace).Create(&corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Type: corev1.SecretTypeOpaque,
		Data: map[string][]byte {
			"credentials.json": []byte("MIIFbTCCA1WgAwIBAgIJAN338vEmMtLsMA0GCSqGSIb3DQEBCwUAME0xCzAJBgNVBAYTAlVLMRMwEQYDVQQIDApUZXN0LVN0YXRlMRUwEwYDVQQKDAxHb2xhbmcgVGVzdHMxEjAQBgNVBAMMCXRlc3QtZmlsZTAeFw0xNzAyMDEyMzUyMDhaFw0yNzAxMzAyMzUyMDhaME0xCzAJBgNVBAYTAlVLMRMwEQYDVQQIDApUZXN0LVN0YXRlMRUwEwYDVQQKDAxHb2xhbmcgVGVzdHMxEjAQBgNVBAMMCXRlc3QtZmlsZTCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBAPMGiLjdiffQo3Xc8oUe7wsDhSaAJFOhO6Qsi0xYrYl7jmCuz9rGD2fdgk5cLqGazKuQ6fIFzHXFU2BKs4CWXt9KO0KFEhfvZeuWjG5d7C1ZUiuKOrPqjKVu8SZtFPc7y7Ke7msXzY+Z2LLyiJJ93LCMq4+cTSGNXVlIKqUxhxeoD5/QkUPyQy/ilu3GMYfx/YORhDP6Edcuskfj8wRh1UxBejP8YPMvI6StcE2GkxoEGqDWnQ/61F18te6WI3MD29tnKXOkXVhnSC+yvRLljotW2/tAhHKBG4tjiQWT5Ri4Wrw2tXxPKRLsVWc7e1/hdxhnuvYpXkWNhKsm002jzkFXlzfEwPd8nZdw5aT6gPUBN2AAzdoqZI7E200i0orEF7WaSoMfjU1tbHvExp3vyAPOfJ5PS2MQ6W03Zsy5dTVH+OBH++rkRzQCFcnIv/OIhya5XZ9KX9nFPgBEP7Xq2A+IjH7B6VN/S/bv8lhp2V+SQvlew9GttKC4hKuPsl5o7+CMbcqcNUdxm9gGkN8epGEKCuix97bpNlxNfHZxHE5+8GMzPXMkCD56y5TNKR6ut7JGHMPtGl5lPCLqzG/HzYyFgxsDfDUu2B0AGKj0lGpnLfGqwhs2/s3jpY7+pcvVQxEpvVTId5byDxu1ujP4HjO/VTQ2P72rE8FtC6J2Av0tAgMBAAGjUDBOMB0GA1UdDgQWBBTLT/RbyfBB/Pa07oBnaM+QSJPO9TAfBgNVHSMEGDAWgBTLT/RbyfBB/Pa07oBnaM+QSJPO9TAMBgNVHRMEBTADAQH/MA0GCSqGSIb3DQEBCwUAA4ICAQB3sCntCcQwhMgRPPyvOCMyTcQ/Iv+cpfxz2Ck14nlxAkEAH2CH0ov5GWTt07/ur3aa5x+SAKi0J3wTD1cdiw4U/6Uin6jWGKKxvoo4IaeKSbM8w/6eKx6UbmHx7PA/eRABY9tTlpdPCVgw7/o3WDr03QM+IAtatzvaCPPczakepbdLwmBZB/v8V+6jUajy6jOgdSH0PyffGnt7MWgDETmNC6p/Xigp5eh+C8Fb4NGTxgHES5PBC+sruWp4u22bJGDKTvYNdZHsnw/CaKQWNsQqwisxa3/8N5v+PCff/pxlr05pE3PdHn9JrCl4iWdVlgtiI9BoPtQyDfa/OEFaScE8KYR8LxaAgdgp3zYncWlsBpwQ6Y/A2wIkhlD9eEp5Ib2hz7isXOs9UwjdriKqrBXqcIAE5M+YIk3+KAQKxAtd4YsK3CSJ010uphr12YKqlScj4vuKFjuOtd5RyyMIxUG3lrrhAu2AzCeKCLdVgA8+75FrYMApUdvcjp4uzbBoED4XRQlx9kdFHVbYgmE/+yddBYJM8u4YlgAL0hW2/D8pz9JWIfxVmjJnBnXaKGBuiUyZ864A3PJndP6EMMo7TzS2CDnfCYuJjvI0KvDjFNmcrQA04+qfMSEz3nmKhbbZu4eYLzlADhfH8tT4GMtXf71WLA5AUHGf2Y4+HIHTsmHGvQ=="),
		},
	})
	if err != nil {
		return err
	}

	_, err = clientset.CoreV1().Secrets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func deleteSecret(clientset *kubernetes.Clientset, namespace string, name string) error {
	err := clientset.CoreV1().Secrets(namespace).Delete(name, &metav1.DeleteOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
		// Apparently the Secret was already deleted, ignoring.
		return nil
	}

	s, err := clientset.CoreV1().Secrets(namespace).Get(name, metav1.GetOptions{})
	if err == nil {
		errors.New(fmt.Sprintf("failed to delete secret %s, secret has metadata %v", name, s.GetObjectMeta()))
	}
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		return err
	}
	if se.Status().Reason != metav1.StatusReasonNotFound {
		return err
	}
	return nil
}

func initNetworkPolicy(clientset *kubernetes.Clientset, namespace string, name string) error {
	_, err := clientset.NetworkingV1().NetworkPolicies(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
	} else {
		return nil
	}

	port := intstr.FromInt(80)
	_, err = clientset.NetworkingV1().NetworkPolicies(namespace).Create(&networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1.NetworkPolicySpec{
			Ingress: []networkingv1.NetworkPolicyIngressRule{
				{
					Ports: []networkingv1.NetworkPolicyPort{
						{
							Port: &port,
						},
					},
				},
			},
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string {
					"component": "tomcat",
					"instance": namespace,
				},
			},
			PolicyTypes: []networkingv1.PolicyType {
				networkingv1.PolicyTypeIngress,
			},
		},
	})
	if err != nil {
		return err
	}

	_, err = clientset.NetworkingV1().NetworkPolicies(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func deleteNetworkPolicy(clientset *kubernetes.Clientset, namespace string, name string) error {
	err := clientset.NetworkingV1().NetworkPolicies(namespace).Delete(name, &metav1.DeleteOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
		// Apparently the Network Policy was already deleted, ignoring.
		return nil
	}

	np, err := clientset.NetworkingV1().NetworkPolicies(namespace).Get(name, metav1.GetOptions{})
	if err == nil {
		errors.New(fmt.Sprintf("failed to delete network policy %s, network policy has metadata %v", name, np.GetObjectMeta()))
	}
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		return err
	}
	if se.Status().Reason != metav1.StatusReasonNotFound {
		return err
	}
	return nil
}

func initPersistentVolume(clientset *kubernetes.Clientset, name string) error {
	_, err := clientset.CoreV1().PersistentVolumes().Get(name, metav1.GetOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
	} else {
		return nil
	}

	_, err = clientset.CoreV1().PersistentVolumes().Create(&corev1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string {
				"pv.kubernetes.io/bound-by-controller": "yes",
			},
			Labels: map[string]string {
				"name": name,
			},
		},
		Spec: corev1.PersistentVolumeSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode {
				corev1.ReadWriteMany,
			},
			Capacity: corev1.ResourceList{
				corev1.ResourceStorage: resource.MustParse("64Mi"),
			},
			PersistentVolumeSource: corev1.PersistentVolumeSource {
				NFS: &corev1.NFSVolumeSource{
					Path: "/gce/wrf/wrf/www",
					Server: "127.0.0.1",
				},
			},
			PersistentVolumeReclaimPolicy: corev1.PersistentVolumeReclaimRetain,
			StorageClassName: "slow",
		},
	})
	if err != nil {
		return err
	}

	_, err = clientset.CoreV1().PersistentVolumes().Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func deletePersistentVolume(clientset *kubernetes.Clientset, name string) error {
	err := clientset.CoreV1().PersistentVolumes().Delete(name, &metav1.DeleteOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
		// Apparently the PersistentVolume was already deleted, ignoring.
		return nil
	}

	pv, err := clientset.CoreV1().PersistentVolumes().Get(name, metav1.GetOptions{})
	if err == nil {
		errors.New(fmt.Sprintf("failed to delete pv %s, pv has status %v", name, pv.Status))
	}
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		return err
	}
	if se.Status().Reason != metav1.StatusReasonNotFound {
		return err
	}
	return nil
}

func initPersistentVolumeClaim(clientset *kubernetes.Clientset, namespace string, name string) error {
	_, err := clientset.CoreV1().PersistentVolumeClaims(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
	} else {
		return nil
	}

	slow := "slow"
	_, err = clientset.CoreV1().PersistentVolumeClaims(namespace).Create(&corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode {
				corev1.ReadWriteMany,
			},
			Resources: corev1.ResourceRequirements {
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: resource.MustParse("64Mi"),
				},
			},
			Selector: &metav1.LabelSelector {
				MatchLabels: map[string]string {
					"name": name,
				},
			},
			StorageClassName: &slow,
			VolumeName: name,
		},
	})
	if err != nil {
		return err
	}

	_, err = clientset.CoreV1().PersistentVolumeClaims(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func deletePersistentVolumeClaim(clientset *kubernetes.Clientset, namespace string, name string) error {
	err := clientset.CoreV1().PersistentVolumeClaims(namespace).Delete(name, &metav1.DeleteOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
		// Apparently the PersistentVolumeClaim was already deleted, ignoring.
		return nil
	}

	pvc, err := clientset.CoreV1().PersistentVolumeClaims(namespace).Get(name, metav1.GetOptions{})
	if err == nil {
		errors.New(fmt.Sprintf("failed to delete pvc %s, pvc has status %v", name, pvc.Status))
	}
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		return err
	}
	if se.Status().Reason != metav1.StatusReasonNotFound {
		return err
	}
	return nil
}

func initDeployment(clientset *kubernetes.Clientset, namespace string, name string, site string, image string, secret string) error {
	_, err := clientset.AppsV1().Deployments(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
	} else {
		return nil
	}

	replicas := int32(2)
	limit := int32(2)
	max := intstr.FromInt(1)
	gracePeriod := int64(30)
	mode := int32(0640) // Leading 0 indicates this is an octal.
	_, err = clientset.AppsV1().Deployments(namespace).Create(&appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string {
				"deployment.kubernetes.io/revision": "1",
			},
			Labels: map[string]string {
				"site": site,
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			RevisionHistoryLimit: &limit,
			Selector: &metav1.LabelSelector {
				MatchLabels: map[string]string {
					"site": site,
				},
			},
			Strategy: appsv1.DeploymentStrategy {
				RollingUpdate: &appsv1.RollingUpdateDeployment {
					MaxSurge: &max,
					MaxUnavailable: &max,
				},
				Type: appsv1.RollingUpdateDeploymentStrategyType,
			},
			Template: corev1.PodTemplateSpec {
				ObjectMeta: metav1.ObjectMeta{
					CreationTimestamp: metav1.Now(),
					Labels: map[string]string {
						"site": site,
					},
				},
				Spec: corev1.PodSpec {
					Affinity: &corev1.Affinity {
						PodAntiAffinity: &corev1.PodAntiAffinity {
							PreferredDuringSchedulingIgnoredDuringExecution: []corev1.WeightedPodAffinityTerm {
								{
									PodAffinityTerm: corev1.PodAffinityTerm {
										LabelSelector: &metav1.LabelSelector {
											MatchExpressions: []metav1.LabelSelectorRequirement {
												{
													Key: "site",
													Operator: metav1.LabelSelectorOpIn,
													Values: []string {
														site,
													},
												},
											},
										},
										TopologyKey: "kubernetes.io/hostname",
									},
									Weight: 50,
								},
							},


						},
					},
					Containers: []corev1.Container {
						{
							Name: "tomcat",
							Env: []corev1.EnvVar {
								{
									Name: "SITE_ID",
									Value: site,
								},
							},
							Image: image,
							ImagePullPolicy: corev1.PullIfNotPresent,
							Ports: []corev1.ContainerPort {
								{
									ContainerPort: 8080,
									Protocol: corev1.ProtocolTCP,
								},
							},
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("100m"),
									corev1.ResourceMemory: resource.MustParse("300M"),
								},
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("50m"),
									corev1.ResourceMemory: resource.MustParse("150M"),
								},
							},
							TerminationMessagePath: "/dev/termination-log",
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
							VolumeMounts: []corev1.VolumeMount {
								{
									Name: secret,
									MountPath: "/secret",
									ReadOnly: true,
								},
							},
						},
					},
					DNSPolicy: corev1.DNSClusterFirst,
					RestartPolicy: corev1.RestartPolicyAlways,
					SchedulerName: "default-scheduler",
					SecurityContext: &corev1.PodSecurityContext {},
					TerminationGracePeriodSeconds: &gracePeriod,
					Volumes: []corev1.Volume {
						{
							Name: secret,
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource {
									DefaultMode: &mode,
									SecretName: secret,

								},
							},
						},
						{
							Name: "upgrade",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource {},
							},
						},
						{
							Name: "opcache",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource {},
							},
						},
						{
							Name: "logs",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource {},
							},
						},
					},
				},
			},
		},
	})
	if err != nil {
		return err
	}

	_, err = clientset.AppsV1().Deployments(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func deleteDeployment(clientset *kubernetes.Clientset, namespace string, name string, site string, image string, secret string) error {
	err := clientset.AppsV1().Deployments(namespace).Delete(name, &metav1.DeleteOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
		// Apparently the Deployment was already deleted, ignoring.
		return nil
	}

	d, err := clientset.AppsV1().Deployments(namespace).Get(name, metav1.GetOptions{})
	if err == nil {
		errors.New(fmt.Sprintf("failed to delete deployment %s, deployment has status %v", name, d.Status))
	}
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		return err
	}
	if se.Status().Reason != metav1.StatusReasonNotFound {
		return err
	}
	return nil
}

func initService(clientset *kubernetes.Clientset, namespace string, name string, site string) error {
	_, err := clientset.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
	} else {
		return nil
	}

	var serviceType corev1.ServiceType
	if opts.inCluster {
		serviceType = corev1.ServiceTypeNodePort
	} else {
		serviceType = corev1.ServiceTypeClusterIP
	}
	_, err = clientset.CoreV1().Services(namespace).Create(&corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Port:       80,
					Protocol:   corev1.ProtocolTCP,
					TargetPort: intstr.FromInt(8080),
				},
			},
			Selector: map[string]string{
				"site": site,
			},
			SessionAffinity: corev1.ServiceAffinityNone,
			Type:            serviceType,
		},
	})
	if err != nil {
		return err
	}

	_, err = clientset.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func deleteService(clientset *kubernetes.Clientset, namespace string, name string, site string) error {
	err := clientset.CoreV1().Services(namespace).Delete(name, &metav1.DeleteOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
		// Apparently the Service was already deleted, ignoring.
		return nil
	}

	s, err := clientset.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
	if err == nil {
		errors.New(fmt.Sprintf("failed to delete service %s, service has status %v", name, s.Status))
	}
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		return err
	}
	if se.Status().Reason != metav1.StatusReasonNotFound {
		return err
	}
	return nil
}

func initIngress(clientset *kubernetes.Clientset, namespace string, name string, service string) error {
	_, err := clientset.ExtensionsV1beta1().Ingresses(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
	} else {
		return nil
	}

	_, err = clientset.ExtensionsV1beta1().Ingresses(namespace).Create(&extv1beta1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: extv1beta1.IngressSpec{
			Backend: &extv1beta1.IngressBackend{
				ServiceName: service,
				ServicePort: intstr.FromInt(80),
			},
		},
	})
	if err != nil {
		return err
	}

	_, err = clientset.ExtensionsV1beta1().Ingresses(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	return nil
}

func deleteIngress(clientset *kubernetes.Clientset, namespace string, name string, site string) error {
	err := clientset.ExtensionsV1beta1().Ingresses(namespace).Delete(name, &metav1.DeleteOptions{})
	if err != nil {
		se, ok := err.(*apierrors.StatusError)
		if !ok {
			return err
		}
		if se.Status().Reason != metav1.StatusReasonNotFound {
			return err
		}
		// Apparently the Ingress was already deleted, ignoring.
		return nil
	}

	i, err := clientset.ExtensionsV1beta1().Ingresses(namespace).Get(name, metav1.GetOptions{})
	if err == nil {
		errors.New(fmt.Sprintf("failed to delete ingress %s, ingress has status %v", name, i.Status))
	}
	se, ok := err.(*apierrors.StatusError)
	if !ok {
		return err
	}
	if se.Status().Reason != metav1.StatusReasonNotFound {
		return err
	}
	return nil
}

func verifyEndpoints(clientset *kubernetes.Clientset, namespace string, name string, expected int) (bool, error) {
	result := false
	var reqErr error
	wait.Poll(1*time.Second, 120*time.Second, func() (bool, error) {
		endpoints, err := clientset.CoreV1().Endpoints(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			reqErr = err
		}
		if len(endpoints.Subsets) != 1 {
			return false, nil
		}
		if len(endpoints.Subsets[0].Addresses) != expected {
			return false, nil
		}
		result = true
		return true, nil
	})
	if result {
		reqErr = nil
	}
	return result, reqErr
}

func verifyIngress(clientset *kubernetes.Clientset, namespace string, name string, expected int) (bool, error) {
	result := false
	var reqErr error
	wait.Poll(250*time.Millisecond, 180*time.Second, func() (bool, error) {
		ingress, err := clientset.ExtensionsV1beta1().Ingresses(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			reqErr = err
		}
		if len(ingress.Status.LoadBalancer.Ingress) != expected {
			return false, nil
		}
		result = true
		return true, nil
	})
	if result {
		reqErr = nil
	}
	return result, reqErr
}

func verifyPods(clientset *kubernetes.Clientset, namespace string, expected int) (bool, error) {
	result := false
	var reqErr error
	wait.Poll(250*time.Millisecond, 30*time.Second, func() (bool, error) {
		count := 0
		pods, err := clientset.CoreV1().Pods(namespace).List(metav1.ListOptions{})
		if err != nil {
			reqErr = err
		}
		if len(pods.Items) != expected {
			return false, nil
		}
		for _, pod := range pods.Items {
			status := pod.Status
			if status.Phase == corev1.PodRunning {
				count++
			}
		}
		if count < expected {
			return false, nil
		}
		result = true
		return true, nil
	})
	if result {
		reqErr = nil
	}
	return result, reqErr
}

func getExternalIP(clientset *kubernetes.Clientset, namespace string, name string, service string) (string, error) {
	if ! opts.inCluster {
		ingress, err := clientset.ExtensionsV1beta1().Ingresses(namespace).Get(name, metav1.GetOptions{})
		if err == nil {
			status := ingress.Status
			lb := status.LoadBalancer
			in := lb.Ingress
			if len(in) != 1 {
				return "", errors.New(fmt.Sprintf("got %d addresses on ingress %s:%s", len(in), namespace, name))
			}
			return in[0].IP, nil
		}
	}
	srvc, err := clientset.CoreV1().Services(namespace).Get(service, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	spec := srvc.Spec
	ip := spec.ClusterIP
	return ip, nil
}
