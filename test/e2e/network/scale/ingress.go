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

package scale

import (
	"fmt"
	"io/ioutil"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	clientset "k8s.io/client-go/kubernetes"

	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	numIngressesSmall      = 5
	numIngressesMedium     = 20
	numIngressesLarge      = 50
	numIngressesExtraLarge = 99

	scaleTestIngressNamePrefix = "ing-scale"
	scaleTestBackendName       = "echoheaders-scale"
	scaleTestSecretName        = "tls-secret-scale"
	scaleTestHostname          = "scale.ingress.com"
	scaleTestNumBackends       = 10
	scaleTestPollInterval      = 15 * time.Second

	// We don't expect waitForIngress to take longer
	// than waitForIngressMaxTimeout.
	waitForIngressMaxTimeout = 80 * time.Minute
	ingressesCleanupTimeout  = 80 * time.Minute
)

var (
	scaleTestLabels = map[string]string{
		"app": scaleTestBackendName,
	}
)

// IngressScaleFramework defines the framework for ingress scale testing.
type IngressScaleFramework struct {
	Clientset     clientset.Interface
	Jig           *framework.IngressTestJig
	GCEController *framework.GCEIngressController
	CloudConfig   framework.CloudConfig
	Logger        framework.TestLogger

	Namespace        string
	EnableTLS        bool
	NumIngressesTest []int
	OutputFile       string

	ScaleTestDeploy *extensions.Deployment
	ScaleTestSvcs   []*v1.Service
	ScaleTestIngs   []*extensions.Ingress

	// BatchCreateLatencies stores all ingress creation latencies, in different
	// batches.
	BatchCreateLatencies [][]time.Duration
	// BatchDurations stores the total duration for each ingress batch creation.
	BatchDurations []time.Duration
	// StepCreateLatencies stores the single ingress creation latency, which happens
	// after each ingress batch creation is complete.
	StepCreateLatencies []time.Duration
	// StepCreateLatencies stores the single ingress update latency, which happens
	// after each ingress batch creation is complete.
	StepUpdateLatencies []time.Duration
}

// NewIngressScaleFramework returns a new framework for ingress scale testing.
func NewIngressScaleFramework(cs clientset.Interface, ns string, cloudConfig framework.CloudConfig) *IngressScaleFramework {
	return &IngressScaleFramework{
		Namespace:   ns,
		Clientset:   cs,
		CloudConfig: cloudConfig,
		Logger:      &framework.E2ELogger{},
		EnableTLS:   true,
		NumIngressesTest: []int{
			numIngressesSmall,
			numIngressesMedium,
			numIngressesLarge,
			numIngressesExtraLarge,
		},
	}
}

// PrepareScaleTest prepares framework for ingress scale testing.
func (f *IngressScaleFramework) PrepareScaleTest() error {
	f.Logger.Infof("Initializing ingress test suite and gce controller...")
	f.Jig = framework.NewIngressTestJig(f.Clientset)
	f.Jig.Logger = f.Logger
	f.Jig.PollInterval = scaleTestPollInterval
	f.GCEController = &framework.GCEIngressController{
		Client: f.Clientset,
		Cloud:  f.CloudConfig,
	}
	if err := f.GCEController.Init(); err != nil {
		return fmt.Errorf("Failed to initialize GCE controller: %v", err)
	}

	f.ScaleTestSvcs = []*v1.Service{}
	f.ScaleTestIngs = []*extensions.Ingress{}

	return nil
}

// CleanupScaleTest cleans up framework for ingress scale testing.
func (f *IngressScaleFramework) CleanupScaleTest() []error {
	var errs []error

	f.Logger.Infof("Cleaning up ingresses...")
	for _, ing := range f.ScaleTestIngs {
		if ing != nil {
			if err := f.Clientset.ExtensionsV1beta1().Ingresses(ing.Namespace).Delete(ing.Name, nil); err != nil {
				errs = append(errs, fmt.Errorf("Error while deleting ingress %s/%s: %v", ing.Namespace, ing.Name, err))
			}
		}
	}
	f.Logger.Infof("Cleaning up services...")
	for _, svc := range f.ScaleTestSvcs {
		if svc != nil {
			if err := f.Clientset.CoreV1().Services(svc.Namespace).Delete(svc.Name, nil); err != nil {
				errs = append(errs, fmt.Errorf("Error while deleting service %s/%s: %v", svc.Namespace, svc.Name, err))
			}
		}
	}
	if f.ScaleTestDeploy != nil {
		f.Logger.Infof("Cleaning up deployment %s...", f.ScaleTestDeploy.Name)
		if err := f.Clientset.ExtensionsV1beta1().Deployments(f.ScaleTestDeploy.Namespace).Delete(f.ScaleTestDeploy.Name, nil); err != nil {
			errs = append(errs, fmt.Errorf("Error while delting deployment %s/%s: %v", f.ScaleTestDeploy.Namespace, f.ScaleTestDeploy.Name, err))
		}
	}

	f.Logger.Infof("Cleaning up cloud resources...")
	if err := f.GCEController.CleanupGCEIngressControllerWithTimeout(ingressesCleanupTimeout); err != nil {
		errs = append(errs, err)
	}

	return errs
}

// RunScaleTest runs ingress scale testing.
func (f *IngressScaleFramework) RunScaleTest() []error {
	var errs []error

	testDeploy := generateScaleTestBackendDeploymentSpec(scaleTestNumBackends)
	f.Logger.Infof("Creating deployment %s...", testDeploy.Name)
	testDeploy, err := f.Jig.Client.ExtensionsV1beta1().Deployments(f.Namespace).Create(testDeploy)
	if err != nil {
		errs = append(errs, fmt.Errorf("Failed to create deployment %s: %v", testDeploy.Name, err))
		return errs
	}
	f.ScaleTestDeploy = testDeploy

	if f.EnableTLS {
		f.Logger.Infof("Ensuring TLS secret %s...", scaleTestSecretName)
		if err := f.Jig.PrepareTLSSecret(f.Namespace, scaleTestSecretName, scaleTestHostname); err != nil {
			errs = append(errs, fmt.Errorf("Failed to prepare TLS secret %s: %v", scaleTestSecretName, err))
			return errs
		}
	}

	// numIngsCreated keeps track of how many ingresses have been created.
	numIngsCreated := 0

	prepareIngsFunc := func(numIngsNeeded int) {
		var ingWg sync.WaitGroup
		numIngsToCreate := numIngsNeeded - numIngsCreated
		ingWg.Add(numIngsToCreate)
		svcQueue := make(chan *v1.Service, numIngsToCreate)
		ingQueue := make(chan *extensions.Ingress, numIngsToCreate)
		errQueue := make(chan error, numIngsToCreate)
		latencyQueue := make(chan time.Duration, numIngsToCreate)
		start := time.Now()
		for ; numIngsCreated < numIngsNeeded; numIngsCreated++ {
			suffix := fmt.Sprintf("%d", numIngsCreated)
			go func() {
				defer ingWg.Done()

				start := time.Now()
				svcCreated, ingCreated, err := f.createScaleTestServiceIngress(suffix, f.EnableTLS)
				svcQueue <- svcCreated
				ingQueue <- ingCreated
				if err != nil {
					errQueue <- err
					return
				}
				f.Logger.Infof("Waiting for ingress %s to come up...", ingCreated.Name)
				if err := f.Jig.WaitForGivenIngressWithTimeout(ingCreated, false, waitForIngressMaxTimeout); err != nil {
					errQueue <- err
					return
				}
				elapsed := time.Since(start)
				f.Logger.Infof("Spent %s for ingress %s to come up", elapsed, ingCreated.Name)
				latencyQueue <- elapsed
			}()
		}

		// Wait until all ingress creations are complete.
		f.Logger.Infof("Waiting for %d ingresses to come up...", numIngsToCreate)
		ingWg.Wait()
		close(svcQueue)
		close(ingQueue)
		close(errQueue)
		close(latencyQueue)
		elapsed := time.Since(start)
		for svc := range svcQueue {
			f.ScaleTestSvcs = append(f.ScaleTestSvcs, svc)
		}
		for ing := range ingQueue {
			f.ScaleTestIngs = append(f.ScaleTestIngs, ing)
		}
		var createLatencies []time.Duration
		for latency := range latencyQueue {
			createLatencies = append(createLatencies, latency)
		}
		f.BatchCreateLatencies = append(f.BatchCreateLatencies, createLatencies)
		if len(errQueue) != 0 {
			f.Logger.Errorf("Failed while creating services and ingresses, spent %v", elapsed)
			for err := range errQueue {
				errs = append(errs, err)
			}
			return
		}
		f.Logger.Infof("Spent %s for %d ingresses to come up", elapsed, numIngsToCreate)
		f.BatchDurations = append(f.BatchDurations, elapsed)
	}

	measureCreateUpdateFunc := func() {
		f.Logger.Infof("Create one more ingress and wait for it to come up")
		start := time.Now()
		svcCreated, ingCreated, err := f.createScaleTestServiceIngress(fmt.Sprintf("%d", numIngsCreated), f.EnableTLS)
		numIngsCreated = numIngsCreated + 1
		f.ScaleTestSvcs = append(f.ScaleTestSvcs, svcCreated)
		f.ScaleTestIngs = append(f.ScaleTestIngs, ingCreated)
		if err != nil {
			errs = append(errs, err)
			return
		}

		f.Logger.Infof("Waiting for ingress %s to come up...", ingCreated.Name)
		if err := f.Jig.WaitForGivenIngressWithTimeout(ingCreated, false, waitForIngressMaxTimeout); err != nil {
			errs = append(errs, err)
			return
		}
		elapsed := time.Since(start)
		f.Logger.Infof("Spent %s for ingress %s to come up", elapsed, ingCreated.Name)
		f.StepCreateLatencies = append(f.StepCreateLatencies, elapsed)

		f.Logger.Infof("Updating ingress and wait for change to take effect")
		ingToUpdate, err := f.Clientset.ExtensionsV1beta1().Ingresses(f.Namespace).Get(ingCreated.Name, metav1.GetOptions{})
		if err != nil {
			errs = append(errs, err)
			return
		}
		addTestPathToIngress(ingToUpdate)
		start = time.Now()
		ingToUpdate, err = f.Clientset.ExtensionsV1beta1().Ingresses(f.Namespace).Update(ingToUpdate)
		if err != nil {
			errs = append(errs, err)
			return
		}

		if err := f.Jig.WaitForGivenIngressWithTimeout(ingToUpdate, false, waitForIngressMaxTimeout); err != nil {
			errs = append(errs, err)
			return
		}
		elapsed = time.Since(start)
		f.Logger.Infof("Spent %s for updating ingress %s", elapsed, ingToUpdate.Name)
		f.StepUpdateLatencies = append(f.StepUpdateLatencies, elapsed)
	}

	defer f.dumpLatencies()

	for _, num := range f.NumIngressesTest {
		f.Logger.Infof("Create more ingresses until we reach %d ingresses", num)
		prepareIngsFunc(num)
		f.Logger.Infof("Measure create and update latency with %d ingresses", num)
		measureCreateUpdateFunc()

		if len(errs) != 0 {
			return errs
		}
	}

	return errs
}

func (f *IngressScaleFramework) dumpLatencies() {
	f.Logger.Infof("Dumping scale test latencies...")
	formattedData := f.GetFormattedLatencies()
	if f.OutputFile != "" {
		f.Logger.Infof("Dumping scale test latencies to file %s...", f.OutputFile)
		ioutil.WriteFile(f.OutputFile, []byte(formattedData), 0644)
		return
	}
	f.Logger.Infof("\n%v", formattedData)
}

// GetFormattedLatencies returns the formatted latencies output.
// TODO: Need a better way/format for data output.
func (f *IngressScaleFramework) GetFormattedLatencies() string {
	if len(f.NumIngressesTest) == 0 ||
		len(f.NumIngressesTest) != len(f.BatchCreateLatencies) ||
		len(f.NumIngressesTest) != len(f.BatchDurations) ||
		len(f.NumIngressesTest) != len(f.StepCreateLatencies) ||
		len(f.NumIngressesTest) != len(f.StepUpdateLatencies) {
		return "Failed to construct latencies output."
	}

	res := "--- Procedure logs ---\n"
	for i, latencies := range f.BatchCreateLatencies {
		res += fmt.Sprintf("Create %d ingresses parallelly, each of them takes below amount of time before starts serving traffic:\n", len(latencies))
		for _, latency := range latencies {
			res = res + fmt.Sprintf("- %v\n", latency)
		}
		res += fmt.Sprintf("Total duration for completing %d ingress creations: %v\n", len(latencies), f.BatchDurations[i])
		res += fmt.Sprintf("Duration to create one more ingress with %d ingresses existing: %v\n", f.NumIngressesTest[i], f.StepCreateLatencies[i])
		res += fmt.Sprintf("Duration to update one ingress with %d ingresses existing: %v\n", f.NumIngressesTest[i]+1, f.StepUpdateLatencies[i])
	}
	res = res + "--- Summary ---\n"
	var batchTotalStr, batchAvgStr, singleCreateStr, singleUpdateStr string
	for i, latencies := range f.BatchCreateLatencies {
		batchTotalStr += fmt.Sprintf("Batch creation total latency for %d ingresses with %d ingresses existing: %v\n", len(latencies), f.NumIngressesTest[i]-len(latencies), f.BatchDurations[i])
		var avgLatency time.Duration
		for _, latency := range latencies {
			avgLatency = avgLatency + latency
		}
		avgLatency /= time.Duration(len(latencies))
		batchAvgStr += fmt.Sprintf("Batch creation average latency for %d ingresses with %d ingresses existing: %v\n", len(latencies), f.NumIngressesTest[i]-len(latencies), avgLatency)
		singleCreateStr += fmt.Sprintf("Single ingress creation latency with %d ingresses existing: %v\n", f.NumIngressesTest[i], f.StepCreateLatencies[i])
		singleUpdateStr += fmt.Sprintf("Single ingress update latency with %d ingresses existing: %v\n", f.NumIngressesTest[i]+1, f.StepUpdateLatencies[i])
	}
	res += batchTotalStr + batchAvgStr + singleCreateStr + singleUpdateStr
	return res
}

func addTestPathToIngress(ing *extensions.Ingress) {
	ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths = append(
		ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths,
		extensions.HTTPIngressPath{
			Path:    "/test",
			Backend: ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].Backend,
		})
}

func (f *IngressScaleFramework) createScaleTestServiceIngress(suffix string, enableTLS bool) (*v1.Service, *extensions.Ingress, error) {
	svcCreated, err := f.Clientset.CoreV1().Services(f.Namespace).Create(generateScaleTestServiceSpec(suffix))
	if err != nil {
		return nil, nil, err
	}
	ingCreated, err := f.Clientset.ExtensionsV1beta1().Ingresses(f.Namespace).Create(generateScaleTestIngressSpec(suffix, enableTLS))
	if err != nil {
		return nil, nil, err
	}
	return svcCreated, ingCreated, nil
}

func generateScaleTestIngressSpec(suffix string, enableTLS bool) *extensions.Ingress {
	ing := &extensions.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("%s-%s", scaleTestIngressNamePrefix, suffix),
		},
		Spec: extensions.IngressSpec{
			TLS: []extensions.IngressTLS{
				{SecretName: scaleTestSecretName},
			},
			Rules: []extensions.IngressRule{
				{
					Host: scaleTestHostname,
					IngressRuleValue: extensions.IngressRuleValue{
						HTTP: &extensions.HTTPIngressRuleValue{
							Paths: []extensions.HTTPIngressPath{
								{
									Path: "/scale",
									Backend: extensions.IngressBackend{
										ServiceName: fmt.Sprintf("%s-%s", scaleTestBackendName, suffix),
										ServicePort: intstr.IntOrString{
											Type:   intstr.Int,
											IntVal: 80,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if enableTLS {
		ing.Spec.TLS = []extensions.IngressTLS{
			{SecretName: scaleTestSecretName},
		}
	}
	return ing
}

func generateScaleTestServiceSpec(suffix string) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:   fmt.Sprintf("%s-%s", scaleTestBackendName, suffix),
			Labels: scaleTestLabels,
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Name:       "http",
				Protocol:   v1.ProtocolTCP,
				Port:       80,
				TargetPort: intstr.FromInt(8080),
			}},
			Selector: scaleTestLabels,
			Type:     v1.ServiceTypeNodePort,
		},
	}
}

func generateScaleTestBackendDeploymentSpec(numReplicas int32) *extensions.Deployment {
	return &extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: scaleTestBackendName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: &numReplicas,
			Selector: &metav1.LabelSelector{MatchLabels: scaleTestLabels},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: scaleTestLabels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  scaleTestBackendName,
							Image: "k8s.gcr.io/echoserver:1.10",
							Ports: []v1.ContainerPort{{ContainerPort: 8080}},
							ReadinessProbe: &v1.Probe{
								Handler: v1.Handler{
									HTTPGet: &v1.HTTPGetAction{
										Port: intstr.FromInt(8080),
										Path: "/healthz",
									},
								},
								FailureThreshold: 10,
								PeriodSeconds:    1,
								SuccessThreshold: 1,
								TimeoutSeconds:   1,
							},
						},
					},
				},
			},
		},
	}
}
