/*
Copyright 2016 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	batchinternal "k8s.io/kubernetes/pkg/apis/batch"
	batch "k8s.io/kubernetes/pkg/apis/batch/v1"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"github.com/golang/glog"
)

const (
	// String used to mark pod deletion
	nonExist = "NonExist"
)

func removePtr(replicas *int32) int32 {
	if replicas == nil {
		return 0
	}
	return *replicas
}

func WaitUntilPodIsScheduled(c clientset.Interface, name, namespace string, timeout time.Duration) (*v1.Pod, error) {
	// Wait until it's scheduled
	p, err := c.Core().Pods(namespace).Get(name, metav1.GetOptions{ResourceVersion: "0"})
	if err == nil && p.Spec.NodeName != "" {
		return p, nil
	}
	pollingPeriod := 200 * time.Millisecond
	startTime := time.Now()
	for startTime.Add(timeout).After(time.Now()) {
		time.Sleep(pollingPeriod)
		p, err := c.Core().Pods(namespace).Get(name, metav1.GetOptions{ResourceVersion: "0"})
		if err == nil && p.Spec.NodeName != "" {
			return p, nil
		}
	}
	return nil, fmt.Errorf("Timed out after %v when waiting for pod %v/%v to start.", timeout, namespace, name)
}

func RunPodAndGetNodeName(c clientset.Interface, pod *v1.Pod, timeout time.Duration) (string, error) {
	name := pod.Name
	namespace := pod.Namespace
	var err error
	// Create a Pod
	for i := 0; i < retries; i++ {
		_, err = c.Core().Pods(namespace).Create(pod)
		if err == nil || apierrs.IsAlreadyExists(err) {
			break
		}
	}
	if err != nil && !apierrs.IsAlreadyExists(err) {
		return "", err
	}
	p, err := WaitUntilPodIsScheduled(c, name, namespace, timeout)
	if err != nil {
		return "", err
	}
	return p.Spec.NodeName, nil
}

type RunObjectConfig interface {
	Run() error
	GetName() string
	GetNamespace() string
	GetKind() schema.GroupKind
	GetClient() clientset.Interface
	GetInternalClient() internalclientset.Interface
	SetClient(clientset.Interface)
	SetInternalClient(internalclientset.Interface)
	GetReplicas() int
}

type RCConfig struct {
	Client         clientset.Interface
	InternalClient internalclientset.Interface
	Image          string
	Command        []string
	Name           string
	Namespace      string
	PollInterval   time.Duration
	Timeout        time.Duration
	PodStatusFile  *os.File
	Replicas       int
	CpuRequest     int64 // millicores
	CpuLimit       int64 // millicores
	MemRequest     int64 // bytes
	MemLimit       int64 // bytes
	ReadinessProbe *v1.Probe
	DNSPolicy      *v1.DNSPolicy

	// Env vars, set the same for every pod.
	Env map[string]string

	// Extra labels added to every pod.
	Labels map[string]string

	// Node selector for pods in the RC.
	NodeSelector map[string]string

	// Ports to declare in the container (map of name to containerPort).
	Ports map[string]int
	// Ports to declare in the container as host and container ports.
	HostPorts map[string]int

	Volumes      []v1.Volume
	VolumeMounts []v1.VolumeMount

	// Pointer to a list of pods; if non-nil, will be set to a list of pods
	// created by this RC by RunRC.
	CreatedPods *[]*v1.Pod

	// Maximum allowable container failures. If exceeded, RunRC returns an error.
	// Defaults to replicas*0.1 if unspecified.
	MaxContainerFailures *int

	// If set to false starting RC will print progress, otherwise only errors will be printed.
	Silent bool

	// If set this function will be used to print log lines instead of glog.
	LogFunc func(fmt string, args ...interface{})
	// If set those functions will be used to gather data from Nodes - in integration tests where no
	// kubelets are running those variables should be nil.
	NodeDumpFunc      func(c clientset.Interface, nodeNames []string, logFunc func(fmt string, args ...interface{}))
	ContainerDumpFunc func(c clientset.Interface, ns string, logFunc func(ftm string, args ...interface{}))

	// Names of the secrets to mount
	SecretNames []string
}

func (rc *RCConfig) RCConfigLog(fmt string, args ...interface{}) {
	if rc.LogFunc != nil {
		rc.LogFunc(fmt, args...)
	}
	glog.Infof(fmt, args...)
}

type DeploymentConfig struct {
	RCConfig
}

type ReplicaSetConfig struct {
	RCConfig
}

type JobConfig struct {
	RCConfig
}

// podInfo contains pod information useful for debugging e2e tests.
type podInfo struct {
	oldHostname string
	oldPhase    string
	hostname    string
	phase       string
}

// PodDiff is a map of pod name to podInfos
type PodDiff map[string]*podInfo

// Print formats and prints the give PodDiff.
func (p PodDiff) String(ignorePhases sets.String) string {
	ret := ""
	for name, info := range p {
		if ignorePhases.Has(info.phase) {
			continue
		}
		if info.phase == nonExist {
			ret += fmt.Sprintf("Pod %v was deleted, had phase %v and host %v\n", name, info.oldPhase, info.oldHostname)
			continue
		}
		phaseChange, hostChange := false, false
		msg := fmt.Sprintf("Pod %v ", name)
		if info.oldPhase != info.phase {
			phaseChange = true
			if info.oldPhase == nonExist {
				msg += fmt.Sprintf("in phase %v ", info.phase)
			} else {
				msg += fmt.Sprintf("went from phase: %v -> %v ", info.oldPhase, info.phase)
			}
		}
		if info.oldHostname != info.hostname {
			hostChange = true
			if info.oldHostname == nonExist || info.oldHostname == "" {
				msg += fmt.Sprintf("assigned host %v ", info.hostname)
			} else {
				msg += fmt.Sprintf("went from host: %v -> %v ", info.oldHostname, info.hostname)
			}
		}
		if phaseChange || hostChange {
			ret += msg + "\n"
		}
	}
	return ret
}

// Diff computes a PodDiff given 2 lists of pods.
func Diff(oldPods []*v1.Pod, curPods []*v1.Pod) PodDiff {
	podInfoMap := PodDiff{}

	// New pods will show up in the curPods list but not in oldPods. They have oldhostname/phase == nonexist.
	for _, pod := range curPods {
		podInfoMap[pod.Name] = &podInfo{hostname: pod.Spec.NodeName, phase: string(pod.Status.Phase), oldHostname: nonExist, oldPhase: nonExist}
	}

	// Deleted pods will show up in the oldPods list but not in curPods. They have a hostname/phase == nonexist.
	for _, pod := range oldPods {
		if info, ok := podInfoMap[pod.Name]; ok {
			info.oldHostname, info.oldPhase = pod.Spec.NodeName, string(pod.Status.Phase)
		} else {
			podInfoMap[pod.Name] = &podInfo{hostname: nonExist, phase: nonExist, oldHostname: pod.Spec.NodeName, oldPhase: string(pod.Status.Phase)}
		}
	}
	return podInfoMap
}

// RunDeployment Launches (and verifies correctness) of a Deployment
// and will wait for all pods it spawns to become "Running".
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling Cleanup).
func RunDeployment(config DeploymentConfig) error {
	err := config.create()
	if err != nil {
		return err
	}
	return config.start()
}

func (config *DeploymentConfig) Run() error {
	return RunDeployment(*config)
}

func (config *DeploymentConfig) GetKind() schema.GroupKind {
	return extensionsinternal.Kind("Deployment")
}

func (config *DeploymentConfig) create() error {
	deployment := &extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: func(i int) *int32 { x := int32(i); return &x }(config.Replicas),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"name": config.Name,
				},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": config.Name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    config.Name,
							Image:   config.Image,
							Command: config.Command,
							Ports:   []v1.ContainerPort{{ContainerPort: 80}},
						},
					},
				},
			},
		},
	}

	if len(config.SecretNames) > 0 {
		attachSecrets(&deployment.Spec.Template, config.SecretNames)
	}

	config.applyTo(&deployment.Spec.Template)

	_, err := config.Client.Extensions().Deployments(config.Namespace).Create(deployment)
	if err != nil {
		return fmt.Errorf("Error creating deployment: %v", err)
	}
	config.RCConfigLog("Created deployment with name: %v, namespace: %v, replica count: %v", deployment.Name, config.Namespace, removePtr(deployment.Spec.Replicas))
	return nil
}

// RunReplicaSet launches (and verifies correctness) of a ReplicaSet
// and waits until all the pods it launches to reach the "Running" state.
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling Cleanup).
func RunReplicaSet(config ReplicaSetConfig) error {
	err := config.create()
	if err != nil {
		return err
	}
	return config.start()
}

func (config *ReplicaSetConfig) Run() error {
	return RunReplicaSet(*config)
}

func (config *ReplicaSetConfig) GetKind() schema.GroupKind {
	return extensionsinternal.Kind("ReplicaSet")
}

func (config *ReplicaSetConfig) create() error {
	rs := &extensions.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func(i int) *int32 { x := int32(i); return &x }(config.Replicas),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"name": config.Name,
				},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": config.Name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    config.Name,
							Image:   config.Image,
							Command: config.Command,
							Ports:   []v1.ContainerPort{{ContainerPort: 80}},
						},
					},
				},
			},
		},
	}

	if len(config.SecretNames) > 0 {
		attachSecrets(&rs.Spec.Template, config.SecretNames)
	}

	config.applyTo(&rs.Spec.Template)

	_, err := config.Client.Extensions().ReplicaSets(config.Namespace).Create(rs)
	if err != nil {
		return fmt.Errorf("Error creating replica set: %v", err)
	}
	config.RCConfigLog("Created replica set with name: %v, namespace: %v, replica count: %v", rs.Name, config.Namespace, removePtr(rs.Spec.Replicas))
	return nil
}

// RunJob baunches (and verifies correctness) of a Job
// and will wait for all pods it spawns to become "Running".
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling Cleanup).
func RunJob(config JobConfig) error {
	err := config.create()
	if err != nil {
		return err
	}
	return config.start()
}

func (config *JobConfig) Run() error {
	return RunJob(*config)
}

func (config *JobConfig) GetKind() schema.GroupKind {
	return batchinternal.Kind("Job")
}

func (config *JobConfig) create() error {
	job := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Spec: batch.JobSpec{
			Parallelism: func(i int) *int32 { x := int32(i); return &x }(config.Replicas),
			Completions: func(i int) *int32 { x := int32(i); return &x }(config.Replicas),
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": config.Name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    config.Name,
							Image:   config.Image,
							Command: config.Command,
						},
					},
					RestartPolicy: v1.RestartPolicyOnFailure,
				},
			},
		},
	}

	if len(config.SecretNames) > 0 {
		attachSecrets(&job.Spec.Template, config.SecretNames)
	}

	config.applyTo(&job.Spec.Template)

	_, err := config.Client.Batch().Jobs(config.Namespace).Create(job)
	if err != nil {
		return fmt.Errorf("Error creating job: %v", err)
	}
	config.RCConfigLog("Created job with name: %v, namespace: %v, parallelism/completions: %v", job.Name, config.Namespace, job.Spec.Parallelism)
	return nil
}

// RunRC Launches (and verifies correctness) of a Replication Controller
// and will wait for all pods it spawns to become "Running".
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling Cleanup).
func RunRC(config RCConfig) error {
	err := config.create()
	if err != nil {
		return err
	}
	return config.start()
}

func (config *RCConfig) Run() error {
	return RunRC(*config)
}

func (config *RCConfig) GetName() string {
	return config.Name
}

func (config *RCConfig) GetNamespace() string {
	return config.Namespace
}

func (config *RCConfig) GetKind() schema.GroupKind {
	return api.Kind("ReplicationController")
}

func (config *RCConfig) GetClient() clientset.Interface {
	return config.Client
}

func (config *RCConfig) GetInternalClient() internalclientset.Interface {
	return config.InternalClient
}

func (config *RCConfig) SetClient(c clientset.Interface) {
	config.Client = c
}

func (config *RCConfig) SetInternalClient(c internalclientset.Interface) {
	config.InternalClient = c
}

func (config *RCConfig) GetReplicas() int {
	return config.Replicas
}

func (config *RCConfig) create() error {
	dnsDefault := v1.DNSDefault
	if config.DNSPolicy == nil {
		config.DNSPolicy = &dnsDefault
	}
	rc := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int) *int32 { x := int32(i); return &x }(config.Replicas),
			Selector: map[string]string{
				"name": config.Name,
			},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": config.Name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:           config.Name,
							Image:          config.Image,
							Command:        config.Command,
							Ports:          []v1.ContainerPort{{ContainerPort: 80}},
							ReadinessProbe: config.ReadinessProbe,
						},
					},
					DNSPolicy:    *config.DNSPolicy,
					NodeSelector: config.NodeSelector,
				},
			},
		},
	}

	if len(config.SecretNames) > 0 {
		attachSecrets(rc.Spec.Template, config.SecretNames)
	}

	config.applyTo(rc.Spec.Template)

	_, err := config.Client.Core().ReplicationControllers(config.Namespace).Create(rc)
	if err != nil {
		return fmt.Errorf("Error creating replication controller: %v", err)
	}
	config.RCConfigLog("Created replication controller with name: %v, namespace: %v, replica count: %v", rc.Name, config.Namespace, removePtr(rc.Spec.Replicas))
	return nil
}

func (config *RCConfig) applyTo(template *v1.PodTemplateSpec) {
	if config.Env != nil {
		for k, v := range config.Env {
			c := &template.Spec.Containers[0]
			c.Env = append(c.Env, v1.EnvVar{Name: k, Value: v})
		}
	}
	if config.Labels != nil {
		for k, v := range config.Labels {
			template.ObjectMeta.Labels[k] = v
		}
	}
	if config.NodeSelector != nil {
		template.Spec.NodeSelector = make(map[string]string)
		for k, v := range config.NodeSelector {
			template.Spec.NodeSelector[k] = v
		}
	}
	if config.Ports != nil {
		for k, v := range config.Ports {
			c := &template.Spec.Containers[0]
			c.Ports = append(c.Ports, v1.ContainerPort{Name: k, ContainerPort: int32(v)})
		}
	}
	if config.HostPorts != nil {
		for k, v := range config.HostPorts {
			c := &template.Spec.Containers[0]
			c.Ports = append(c.Ports, v1.ContainerPort{Name: k, ContainerPort: int32(v), HostPort: int32(v)})
		}
	}
	if config.CpuLimit > 0 || config.MemLimit > 0 {
		template.Spec.Containers[0].Resources.Limits = v1.ResourceList{}
	}
	if config.CpuLimit > 0 {
		template.Spec.Containers[0].Resources.Limits[v1.ResourceCPU] = *resource.NewMilliQuantity(config.CpuLimit, resource.DecimalSI)
	}
	if config.MemLimit > 0 {
		template.Spec.Containers[0].Resources.Limits[v1.ResourceMemory] = *resource.NewQuantity(config.MemLimit, resource.DecimalSI)
	}
	if config.CpuRequest > 0 || config.MemRequest > 0 {
		template.Spec.Containers[0].Resources.Requests = v1.ResourceList{}
	}
	if config.CpuRequest > 0 {
		template.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(config.CpuRequest, resource.DecimalSI)
	}
	if config.MemRequest > 0 {
		template.Spec.Containers[0].Resources.Requests[v1.ResourceMemory] = *resource.NewQuantity(config.MemRequest, resource.DecimalSI)
	}
	if len(config.Volumes) > 0 {
		template.Spec.Volumes = config.Volumes
	}
	if len(config.VolumeMounts) > 0 {
		template.Spec.Containers[0].VolumeMounts = config.VolumeMounts
	}
}

type RCStartupStatus struct {
	Expected              int
	Terminating           int
	Running               int
	RunningButNotReady    int
	Waiting               int
	Pending               int
	Unknown               int
	Inactive              int
	FailedContainers      int
	Created               []*v1.Pod
	ContainerRestartNodes sets.String
}

func (s *RCStartupStatus) String(name string) string {
	return fmt.Sprintf("%v Pods: %d out of %d created, %d running, %d pending, %d waiting, %d inactive, %d terminating, %d unknown, %d runningButNotReady ",
		name, len(s.Created), s.Expected, s.Running, s.Pending, s.Waiting, s.Inactive, s.Terminating, s.Unknown, s.RunningButNotReady)
}

func ComputeRCStartupStatus(pods []*v1.Pod, expected int) RCStartupStatus {
	startupStatus := RCStartupStatus{
		Expected:              expected,
		Created:               make([]*v1.Pod, 0, expected),
		ContainerRestartNodes: sets.NewString(),
	}
	for _, p := range pods {
		if p.DeletionTimestamp != nil {
			startupStatus.Terminating++
			continue
		}
		startupStatus.Created = append(startupStatus.Created, p)
		if p.Status.Phase == v1.PodRunning {
			ready := false
			for _, c := range p.Status.Conditions {
				if c.Type == v1.PodReady && c.Status == v1.ConditionTrue {
					ready = true
					break
				}
			}
			if ready {
				// Only count a pod is running when it is also ready.
				startupStatus.Running++
			} else {
				startupStatus.RunningButNotReady++
			}
			for _, v := range FailedContainers(p) {
				startupStatus.FailedContainers = startupStatus.FailedContainers + v.Restarts
				startupStatus.ContainerRestartNodes.Insert(p.Spec.NodeName)
			}
		} else if p.Status.Phase == v1.PodPending {
			if p.Spec.NodeName == "" {
				startupStatus.Waiting++
			} else {
				startupStatus.Pending++
			}
		} else if p.Status.Phase == v1.PodSucceeded || p.Status.Phase == v1.PodFailed {
			startupStatus.Inactive++
		} else if p.Status.Phase == v1.PodUnknown {
			startupStatus.Unknown++
		}
	}
	return startupStatus
}

func (config *RCConfig) start() error {
	// Don't force tests to fail if they don't care about containers restarting.
	var maxContainerFailures int
	if config.MaxContainerFailures == nil {
		maxContainerFailures = int(math.Max(1.0, float64(config.Replicas)*.01))
	} else {
		maxContainerFailures = *config.MaxContainerFailures
	}

	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": config.Name}))

	PodStore := NewPodStore(config.Client, config.Namespace, label, fields.Everything())
	defer PodStore.Stop()

	interval := config.PollInterval
	if interval <= 0 {
		interval = 10 * time.Second
	}
	timeout := config.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	oldPods := make([]*v1.Pod, 0)
	oldRunning := 0
	lastChange := time.Now()
	for oldRunning != config.Replicas {
		time.Sleep(interval)

		pods := PodStore.List()
		startupStatus := ComputeRCStartupStatus(pods, config.Replicas)

		pods = startupStatus.Created
		if config.CreatedPods != nil {
			*config.CreatedPods = pods
		}
		if !config.Silent {
			config.RCConfigLog(startupStatus.String(config.Name))
		}

		if config.PodStatusFile != nil {
			fmt.Fprintf(config.PodStatusFile, "%d, running, %d, pending, %d, waiting, %d, inactive, %d, unknown, %d, runningButNotReady\n", startupStatus.Running, startupStatus.Pending, startupStatus.Waiting, startupStatus.Inactive, startupStatus.Unknown, startupStatus.RunningButNotReady)
		}

		if startupStatus.FailedContainers > maxContainerFailures {
			if config.NodeDumpFunc != nil {
				config.NodeDumpFunc(config.Client, startupStatus.ContainerRestartNodes.List(), config.RCConfigLog)
			}
			if config.ContainerDumpFunc != nil {
				// Get the logs from the failed containers to help diagnose what caused them to fail
				config.ContainerDumpFunc(config.Client, config.Namespace, config.RCConfigLog)
			}
			return fmt.Errorf("%d containers failed which is more than allowed %d", startupStatus.FailedContainers, maxContainerFailures)
		}
		if len(pods) < len(oldPods) || len(pods) > config.Replicas {
			// This failure mode includes:
			// kubelet is dead, so node controller deleted pods and rc creates more
			//	- diagnose by noting the pod diff below.
			// pod is unhealthy, so replication controller creates another to take its place
			//	- diagnose by comparing the previous "2 Pod states" lines for inactive pods
			errorStr := fmt.Sprintf("Number of reported pods for %s changed: %d vs %d", config.Name, len(pods), len(oldPods))
			config.RCConfigLog("%v, pods that changed since the last iteration:", errorStr)
			config.RCConfigLog(Diff(oldPods, pods).String(sets.NewString()))
			return fmt.Errorf(errorStr)
		}

		if len(pods) > len(oldPods) || startupStatus.Running > oldRunning {
			lastChange = time.Now()
		}
		oldPods = pods
		oldRunning = startupStatus.Running

		if time.Since(lastChange) > timeout {
			break
		}
	}

	if oldRunning != config.Replicas {
		// List only pods from a given replication controller.
		options := metav1.ListOptions{LabelSelector: label.String()}
		if pods, err := config.Client.Core().Pods(metav1.NamespaceAll).List(options); err == nil {

			for _, pod := range pods.Items {
				config.RCConfigLog("Pod %s\t%s\t%s\t%s", pod.Name, pod.Spec.NodeName, pod.Status.Phase, pod.DeletionTimestamp)
			}
		} else {
			config.RCConfigLog("Can't list pod debug info: %v", err)
		}
		return fmt.Errorf("Only %d pods started out of %d", oldRunning, config.Replicas)
	}
	return nil
}

// Simplified version of RunRC, that does not create RC, but creates plain Pods.
// Optionally waits for pods to start running (if waitForRunning == true).
// The number of replicas must be non-zero.
func StartPods(c clientset.Interface, replicas int, namespace string, podNamePrefix string,
	pod v1.Pod, waitForRunning bool, logFunc func(fmt string, args ...interface{})) error {
	// no pod to start
	if replicas < 1 {
		panic("StartPods: number of replicas must be non-zero")
	}
	startPodsID := string(uuid.NewUUID()) // So that we can label and find them
	for i := 0; i < replicas; i++ {
		podName := fmt.Sprintf("%v-%v", podNamePrefix, i)
		pod.ObjectMeta.Name = podName
		pod.ObjectMeta.Labels["name"] = podName
		pod.ObjectMeta.Labels["startPodsID"] = startPodsID
		pod.Spec.Containers[0].Name = podName
		_, err := c.Core().Pods(namespace).Create(&pod)
		if err != nil {
			return err
		}
	}
	logFunc("Waiting for running...")
	if waitForRunning {
		label := labels.SelectorFromSet(labels.Set(map[string]string{"startPodsID": startPodsID}))
		err := WaitForPodsWithLabelRunning(c, namespace, label)
		if err != nil {
			return fmt.Errorf("Error waiting for %d pods to be running - probably a timeout: %v", replicas, err)
		}
	}
	return nil
}

// Wait up to 10 minutes for all matching pods to become Running and at least one
// matching pod exists.
func WaitForPodsWithLabelRunning(c clientset.Interface, ns string, label labels.Selector) error {
	running := false
	PodStore := NewPodStore(c, ns, label, fields.Everything())
	defer PodStore.Stop()
waitLoop:
	for start := time.Now(); time.Since(start) < 10*time.Minute; time.Sleep(5 * time.Second) {
		pods := PodStore.List()
		if len(pods) == 0 {
			continue waitLoop
		}
		for _, p := range pods {
			if p.Status.Phase != v1.PodRunning {
				continue waitLoop
			}
		}
		running = true
		break
	}
	if !running {
		return fmt.Errorf("Timeout while waiting for pods with labels %q to be running", label.String())
	}
	return nil
}

type CountToStrategy struct {
	Count    int
	Strategy PrepareNodeStrategy
}

type TestNodePreparer interface {
	PrepareNodes() error
	CleanupNodes() error
}

type PrepareNodeStrategy interface {
	PreparePatch(node *v1.Node) []byte
	CleanupNode(node *v1.Node) *v1.Node
}

type TrivialNodePrepareStrategy struct{}

func (*TrivialNodePrepareStrategy) PreparePatch(*v1.Node) []byte {
	return []byte{}
}

func (*TrivialNodePrepareStrategy) CleanupNode(node *v1.Node) *v1.Node {
	nodeCopy := *node
	return &nodeCopy
}

type LabelNodePrepareStrategy struct {
	labelKey   string
	labelValue string
}

func NewLabelNodePrepareStrategy(labelKey string, labelValue string) *LabelNodePrepareStrategy {
	return &LabelNodePrepareStrategy{
		labelKey:   labelKey,
		labelValue: labelValue,
	}
}

func (s *LabelNodePrepareStrategy) PreparePatch(*v1.Node) []byte {
	labelString := fmt.Sprintf("{\"%v\":\"%v\"}", s.labelKey, s.labelValue)
	patch := fmt.Sprintf(`{"metadata":{"labels":%v}}`, labelString)
	return []byte(patch)
}

func (s *LabelNodePrepareStrategy) CleanupNode(node *v1.Node) *v1.Node {
	objCopy, err := api.Scheme.Copy(node)
	if err != nil {
		return &v1.Node{}
	}
	nodeCopy, ok := (objCopy).(*v1.Node)
	if !ok {
		return &v1.Node{}
	}
	if node.Labels != nil && len(node.Labels[s.labelKey]) != 0 {
		delete(nodeCopy.Labels, s.labelKey)
	}
	return nodeCopy
}

func DoPrepareNode(client clientset.Interface, node *v1.Node, strategy PrepareNodeStrategy) error {
	var err error
	patch := strategy.PreparePatch(node)
	if len(patch) == 0 {
		return nil
	}
	for attempt := 0; attempt < retries; attempt++ {
		if _, err = client.Core().Nodes().Patch(node.Name, types.MergePatchType, []byte(patch)); err == nil {
			return nil
		}
		if !apierrs.IsConflict(err) {
			return fmt.Errorf("Error while applying patch %v to Node %v: %v", string(patch), node.Name, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("To many conflicts when applying patch %v to Node %v", string(patch), node.Name)
}

func DoCleanupNode(client clientset.Interface, nodeName string, strategy PrepareNodeStrategy) error {
	for attempt := 0; attempt < retries; attempt++ {
		node, err := client.Core().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("Skipping cleanup of Node: failed to get Node %v: %v", nodeName, err)
		}
		updatedNode := strategy.CleanupNode(node)
		if apiequality.Semantic.DeepEqual(node, updatedNode) {
			return nil
		}
		if _, err = client.Core().Nodes().Update(updatedNode); err == nil {
			return nil
		}
		if !apierrs.IsConflict(err) {
			return fmt.Errorf("Error when updating Node %v: %v", nodeName, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("To many conflicts when trying to cleanup Node %v", nodeName)
}

type TestPodCreateStrategy func(client clientset.Interface, namespace string, podCount int) error

type CountToPodStrategy struct {
	Count    int
	Strategy TestPodCreateStrategy
}

type TestPodCreatorConfig map[string][]CountToPodStrategy

func NewTestPodCreatorConfig() *TestPodCreatorConfig {
	config := make(TestPodCreatorConfig)
	return &config
}

func (c *TestPodCreatorConfig) AddStrategy(
	namespace string, podCount int, strategy TestPodCreateStrategy) {
	(*c)[namespace] = append((*c)[namespace], CountToPodStrategy{Count: podCount, Strategy: strategy})
}

type TestPodCreator struct {
	Client clientset.Interface
	// namespace -> count -> strategy
	Config *TestPodCreatorConfig
}

func NewTestPodCreator(client clientset.Interface, config *TestPodCreatorConfig) *TestPodCreator {
	return &TestPodCreator{
		Client: client,
		Config: config,
	}
}

func (c *TestPodCreator) CreatePods() error {
	for ns, v := range *(c.Config) {
		for _, countToStrategy := range v {
			if err := countToStrategy.Strategy(c.Client, ns, countToStrategy.Count); err != nil {
				return err
			}
		}
	}
	return nil
}

func MakePodSpec() v1.PodSpec {
	return v1.PodSpec{
		Containers: []v1.Container{{
			Name:  "pause",
			Image: "kubernetes/pause",
			Ports: []v1.ContainerPort{{ContainerPort: 80}},
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("500Mi"),
				},
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("500Mi"),
				},
			},
		}},
	}
}

func makeCreatePod(client clientset.Interface, namespace string, podTemplate *v1.Pod) error {
	var err error
	for attempt := 0; attempt < retries; attempt++ {
		if _, err := client.Core().Pods(namespace).Create(podTemplate); err == nil {
			return nil
		}
		glog.Errorf("Error while creating pod, maybe retry: %v", err)
	}
	return fmt.Errorf("Terminal error while creating pod, won't retry: %v", err)
}

func createPod(client clientset.Interface, namespace string, podCount int, podTemplate *v1.Pod) error {
	var createError error
	lock := sync.Mutex{}
	createPodFunc := func(i int) {
		if err := makeCreatePod(client, namespace, podTemplate); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = err
		}
	}

	if podCount < 30 {
		workqueue.Parallelize(podCount, podCount, createPodFunc)
	} else {
		workqueue.Parallelize(30, podCount, createPodFunc)
	}
	return createError
}

func createController(client clientset.Interface, controllerName, namespace string, podCount int, podTemplate *v1.Pod) error {
	rc := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name: controllerName,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int) *int32 { x := int32(i); return &x }(podCount),
			Selector: map[string]string{"name": controllerName},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": controllerName},
				},
				Spec: podTemplate.Spec,
			},
		},
	}
	var err error
	for attempt := 0; attempt < retries; attempt++ {
		if _, err := client.Core().ReplicationControllers(namespace).Create(rc); err == nil {
			return nil
		}
		glog.Errorf("Error while creating rc, maybe retry: %v", err)
	}
	return fmt.Errorf("Terminal error while creating rc, won't retry: %v", err)
}

func NewCustomCreatePodStrategy(podTemplate *v1.Pod) TestPodCreateStrategy {
	return func(client clientset.Interface, namespace string, podCount int) error {
		return createPod(client, namespace, podCount, podTemplate)
	}
}

func NewSimpleCreatePodStrategy() TestPodCreateStrategy {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "simple-pod-",
		},
		Spec: MakePodSpec(),
	}
	return NewCustomCreatePodStrategy(basePod)
}

func NewSimpleWithControllerCreatePodStrategy(controllerName string) TestPodCreateStrategy {
	return func(client clientset.Interface, namespace string, podCount int) error {
		basePod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: controllerName + "-pod-",
				Labels:       map[string]string{"name": controllerName},
			},
			Spec: MakePodSpec(),
		}
		if err := createController(client, controllerName, namespace, podCount, basePod); err != nil {
			return err
		}
		return createPod(client, namespace, podCount, basePod)
	}
}

type SecretConfig struct {
	Content   map[string]string
	Client    clientset.Interface
	Name      string
	Namespace string
	// If set this function will be used to print log lines instead of glog.
	LogFunc func(fmt string, args ...interface{})
}

func (config *SecretConfig) Run() error {
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		StringData: map[string]string{},
	}
	for k, v := range config.Content {
		secret.StringData[k] = v
	}

	_, err := config.Client.Core().Secrets(config.Namespace).Create(secret)
	if err != nil {
		return fmt.Errorf("Error creating secret: %v", err)
	}
	config.LogFunc("Created secret %v/%v", config.Namespace, config.Name)
	return nil
}

func (config *SecretConfig) Stop() error {
	if err := config.Client.Core().Secrets(config.Namespace).Delete(config.Name, &metav1.DeleteOptions{}); err != nil {
		return fmt.Errorf("Error deleting secret: %v", err)
	}
	config.LogFunc("Deleted secret %v/%v", config.Namespace, config.Name)
	return nil
}

// TODO: attach secrets using different possibilities: env vars, image pull secrets.
func attachSecrets(template *v1.PodTemplateSpec, secretNames []string) {
	volumes := make([]v1.Volume, 0, len(secretNames))
	mounts := make([]v1.VolumeMount, 0, len(secretNames))
	for _, name := range secretNames {
		volumes = append(volumes, v1.Volume{
			Name: name,
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{
					SecretName: name,
				},
			},
		})
		mounts = append(mounts, v1.VolumeMount{
			Name:      name,
			MountPath: fmt.Sprintf("/%v", name),
		})
	}

	template.Spec.Volumes = volumes
	template.Spec.Containers[0].VolumeMounts = mounts
}

type DaemonConfig struct {
	Client    clientset.Interface
	Name      string
	Namespace string
	Image     string
	// If set this function will be used to print log lines instead of glog.
	LogFunc func(fmt string, args ...interface{})
	// How long we wait for DaemonSet to become running.
	Timeout time.Duration
}

func (config *DaemonConfig) Run() error {
	if config.Image == "" {
		config.Image = "kubernetes/pause"
	}
	nameLabel := map[string]string{
		"name": config.Name + "-daemon",
	}
	daemon := &extensions.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Spec: extensions.DaemonSetSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: nameLabel,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  config.Name,
							Image: config.Image,
						},
					},
				},
			},
		},
	}

	_, err := config.Client.Extensions().DaemonSets(config.Namespace).Create(daemon)
	if err != nil {
		return fmt.Errorf("Error creating DaemonSet %v: %v", config.Name, err)
	}

	var nodes *v1.NodeList
	for i := 0; i < retries; i++ {
		// Wait for all daemons to be running
		nodes, err = config.Client.Core().Nodes().List(metav1.ListOptions{ResourceVersion: "0"})
		if err == nil {
			break
		} else if i+1 == retries {
			return fmt.Errorf("Error listing Nodes while waiting for DaemonSet %v: %v", config.Name, err)
		}
	}

	timeout := config.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}

	podStore := NewPodStore(config.Client, config.Namespace, labels.SelectorFromSet(nameLabel), fields.Everything())
	defer podStore.Stop()

	err = wait.Poll(time.Second, timeout, func() (bool, error) {
		pods := podStore.List()

		nodeHasDaemon := sets.NewString()
		for _, pod := range pods {
			podReady, _ := PodRunningReady(pod)
			if pod.Spec.NodeName != "" && podReady {
				nodeHasDaemon.Insert(pod.Spec.NodeName)
			}
		}

		running := len(nodeHasDaemon)
		config.LogFunc("Found %v/%v Daemons %v running", running, config.Name, len(nodes.Items))
		return running == len(nodes.Items), nil
	})
	if err != nil {
		config.LogFunc("Timed out while waiting for DaemonsSet %v/%v to be running.", config.Namespace, config.Name)
	} else {
		config.LogFunc("Created Daemon %v/%v", config.Namespace, config.Name)
	}

	return err
}
