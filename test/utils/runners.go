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
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	apps "k8s.io/api/apps/v1"
	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/client-go/util/workqueue"
	batchinternal "k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"

	"k8s.io/klog"
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
	p, err := c.CoreV1().Pods(namespace).Get(context.TODO(), name, metav1.GetOptions{ResourceVersion: "0"})
	if err == nil && p.Spec.NodeName != "" {
		return p, nil
	}
	pollingPeriod := 200 * time.Millisecond
	startTime := time.Now()
	for startTime.Add(timeout).After(time.Now()) {
		time.Sleep(pollingPeriod)
		p, err := c.CoreV1().Pods(namespace).Get(context.TODO(), name, metav1.GetOptions{ResourceVersion: "0"})
		if err == nil && p.Spec.NodeName != "" {
			return p, nil
		}
	}
	return nil, fmt.Errorf("Timed out after %v when waiting for pod %v/%v to start.", timeout, namespace, name)
}

func RunPodAndGetNodeName(c clientset.Interface, pod *v1.Pod, timeout time.Duration) (string, error) {
	name := pod.Name
	namespace := pod.Namespace
	if err := CreatePodWithRetries(c, namespace, pod); err != nil {
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
	GetScalesGetter() scaleclient.ScalesGetter
	SetClient(clientset.Interface)
	SetScalesClient(scaleclient.ScalesGetter)
	GetReplicas() int
	GetLabelValue(string) (string, bool)
	GetGroupResource() schema.GroupResource
	GetGroupVersionResource() schema.GroupVersionResource
}

type RCConfig struct {
	Affinity                      *v1.Affinity
	Client                        clientset.Interface
	ScalesGetter                  scaleclient.ScalesGetter
	Image                         string
	Command                       []string
	Name                          string
	Namespace                     string
	PollInterval                  time.Duration
	Timeout                       time.Duration
	PodStatusFile                 *os.File
	Replicas                      int
	CpuRequest                    int64 // millicores
	CpuLimit                      int64 // millicores
	MemRequest                    int64 // bytes
	MemLimit                      int64 // bytes
	GpuLimit                      int64 // count
	ReadinessProbe                *v1.Probe
	DNSPolicy                     *v1.DNSPolicy
	PriorityClassName             string
	TerminationGracePeriodSeconds *int64
	Lifecycle                     *v1.Lifecycle

	// Env vars, set the same for every pod.
	Env map[string]string

	// Extra labels and annotations added to every pod.
	Labels      map[string]string
	Annotations map[string]string

	// Node selector for pods in the RC.
	NodeSelector map[string]string

	// Tolerations for pods in the RC.
	Tolerations []v1.Toleration

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
	// Maximum allowed pod deletions count. If exceeded, RunRC returns an error.
	// Defaults to 0.
	MaxAllowedPodDeletions int

	// If set to false starting RC will print progress, otherwise only errors will be printed.
	Silent bool

	// If set this function will be used to print log lines instead of klog.
	LogFunc func(fmt string, args ...interface{})
	// If set those functions will be used to gather data from Nodes - in integration tests where no
	// kubelets are running those variables should be nil.
	NodeDumpFunc      func(c clientset.Interface, nodeNames []string, logFunc func(fmt string, args ...interface{}))
	ContainerDumpFunc func(c clientset.Interface, ns string, logFunc func(ftm string, args ...interface{}))

	// Names of the secrets and configmaps to mount.
	SecretNames    []string
	ConfigMapNames []string

	ServiceAccountTokenProjections int
}

func (rc *RCConfig) RCConfigLog(fmt string, args ...interface{}) {
	if rc.LogFunc != nil {
		rc.LogFunc(fmt, args...)
	}
	klog.Infof(fmt, args...)
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

// DeletedPods returns a slice of pods that were present at the beginning
// and then disappeared.
func (p PodDiff) DeletedPods() []string {
	var deletedPods []string
	for podName, podInfo := range p {
		if podInfo.hostname == nonExist {
			deletedPods = append(deletedPods, podName)
		}
	}
	return deletedPods
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

func (config *DeploymentConfig) GetGroupResource() schema.GroupResource {
	return extensionsinternal.Resource("deployments")
}

func (config *DeploymentConfig) GetGroupVersionResource() schema.GroupVersionResource {
	return extensionsinternal.SchemeGroupVersion.WithResource("deployments")
}

func (config *DeploymentConfig) create() error {
	deployment := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Spec: apps.DeploymentSpec{
			Replicas: func(i int) *int32 { x := int32(i); return &x }(config.Replicas),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"name": config.Name,
				},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      map[string]string{"name": config.Name},
					Annotations: config.Annotations,
				},
				Spec: v1.PodSpec{
					Affinity:                      config.Affinity,
					TerminationGracePeriodSeconds: config.getTerminationGracePeriodSeconds(nil),
					Containers: []v1.Container{
						{
							Name:      config.Name,
							Image:     config.Image,
							Command:   config.Command,
							Ports:     []v1.ContainerPort{{ContainerPort: 80}},
							Lifecycle: config.Lifecycle,
						},
					},
				},
			},
		},
	}

	if len(config.SecretNames) > 0 {
		attachSecrets(&deployment.Spec.Template, config.SecretNames)
	}
	if len(config.ConfigMapNames) > 0 {
		attachConfigMaps(&deployment.Spec.Template, config.ConfigMapNames)
	}

	for i := 0; i < config.ServiceAccountTokenProjections; i++ {
		attachServiceAccountTokenProjection(&deployment.Spec.Template, fmt.Sprintf("tok-%d", i))
	}

	config.applyTo(&deployment.Spec.Template)

	if err := CreateDeploymentWithRetries(config.Client, config.Namespace, deployment); err != nil {
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

func (config *ReplicaSetConfig) GetGroupResource() schema.GroupResource {
	return extensionsinternal.Resource("replicasets")
}

func (config *ReplicaSetConfig) GetGroupVersionResource() schema.GroupVersionResource {
	return extensionsinternal.SchemeGroupVersion.WithResource("replicasets")
}

func (config *ReplicaSetConfig) create() error {
	rs := &apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Spec: apps.ReplicaSetSpec{
			Replicas: func(i int) *int32 { x := int32(i); return &x }(config.Replicas),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"name": config.Name,
				},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      map[string]string{"name": config.Name},
					Annotations: config.Annotations,
				},
				Spec: v1.PodSpec{
					Affinity:                      config.Affinity,
					TerminationGracePeriodSeconds: config.getTerminationGracePeriodSeconds(nil),
					Containers: []v1.Container{
						{
							Name:      config.Name,
							Image:     config.Image,
							Command:   config.Command,
							Ports:     []v1.ContainerPort{{ContainerPort: 80}},
							Lifecycle: config.Lifecycle,
						},
					},
				},
			},
		},
	}

	if len(config.SecretNames) > 0 {
		attachSecrets(&rs.Spec.Template, config.SecretNames)
	}
	if len(config.ConfigMapNames) > 0 {
		attachConfigMaps(&rs.Spec.Template, config.ConfigMapNames)
	}

	config.applyTo(&rs.Spec.Template)

	if err := CreateReplicaSetWithRetries(config.Client, config.Namespace, rs); err != nil {
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

func (config *JobConfig) GetGroupResource() schema.GroupResource {
	return batchinternal.Resource("jobs")
}

func (config *JobConfig) GetGroupVersionResource() schema.GroupVersionResource {
	return batchinternal.SchemeGroupVersion.WithResource("jobs")
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
					Labels:      map[string]string{"name": config.Name},
					Annotations: config.Annotations,
				},
				Spec: v1.PodSpec{
					Affinity:                      config.Affinity,
					TerminationGracePeriodSeconds: config.getTerminationGracePeriodSeconds(nil),
					Containers: []v1.Container{
						{
							Name:      config.Name,
							Image:     config.Image,
							Command:   config.Command,
							Lifecycle: config.Lifecycle,
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
	if len(config.ConfigMapNames) > 0 {
		attachConfigMaps(&job.Spec.Template, config.ConfigMapNames)
	}

	config.applyTo(&job.Spec.Template)

	if err := CreateJobWithRetries(config.Client, config.Namespace, job); err != nil {
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

func (config *RCConfig) GetGroupResource() schema.GroupResource {
	return api.Resource("replicationcontrollers")
}

func (config *RCConfig) GetGroupVersionResource() schema.GroupVersionResource {
	return api.SchemeGroupVersion.WithResource("replicationcontrollers")
}

func (config *RCConfig) GetClient() clientset.Interface {
	return config.Client
}

func (config *RCConfig) GetScalesGetter() scaleclient.ScalesGetter {
	return config.ScalesGetter
}

func (config *RCConfig) SetClient(c clientset.Interface) {
	config.Client = c
}

func (config *RCConfig) SetScalesClient(getter scaleclient.ScalesGetter) {
	config.ScalesGetter = getter
}

func (config *RCConfig) GetReplicas() int {
	return config.Replicas
}

func (config *RCConfig) GetLabelValue(key string) (string, bool) {
	value, found := config.Labels[key]
	return value, found
}

func (config *RCConfig) create() error {
	dnsDefault := v1.DNSDefault
	if config.DNSPolicy == nil {
		config.DNSPolicy = &dnsDefault
	}
	one := int64(1)
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
					Labels:      map[string]string{"name": config.Name},
					Annotations: config.Annotations,
				},
				Spec: v1.PodSpec{
					Affinity: config.Affinity,
					Containers: []v1.Container{
						{
							Name:           config.Name,
							Image:          config.Image,
							Command:        config.Command,
							Ports:          []v1.ContainerPort{{ContainerPort: 80}},
							ReadinessProbe: config.ReadinessProbe,
							Lifecycle:      config.Lifecycle,
						},
					},
					DNSPolicy:                     *config.DNSPolicy,
					NodeSelector:                  config.NodeSelector,
					Tolerations:                   config.Tolerations,
					TerminationGracePeriodSeconds: config.getTerminationGracePeriodSeconds(&one),
					PriorityClassName:             config.PriorityClassName,
				},
			},
		},
	}

	if len(config.SecretNames) > 0 {
		attachSecrets(rc.Spec.Template, config.SecretNames)
	}
	if len(config.ConfigMapNames) > 0 {
		attachConfigMaps(rc.Spec.Template, config.ConfigMapNames)
	}

	config.applyTo(rc.Spec.Template)

	if err := CreateRCWithRetries(config.Client, config.Namespace, rc); err != nil {
		return fmt.Errorf("Error creating replication controller: %v", err)
	}
	config.RCConfigLog("Created replication controller with name: %v, namespace: %v, replica count: %v", rc.Name, config.Namespace, removePtr(rc.Spec.Replicas))
	return nil
}

func (config *RCConfig) applyTo(template *v1.PodTemplateSpec) {
	for k, v := range config.Env {
		c := &template.Spec.Containers[0]
		c.Env = append(c.Env, v1.EnvVar{Name: k, Value: v})
	}
	for k, v := range config.Labels {
		template.ObjectMeta.Labels[k] = v
	}
	template.Spec.NodeSelector = make(map[string]string)
	for k, v := range config.NodeSelector {
		template.Spec.NodeSelector[k] = v
	}
	if config.Tolerations != nil {
		template.Spec.Tolerations = append([]v1.Toleration{}, config.Tolerations...)
	}
	for k, v := range config.Ports {
		c := &template.Spec.Containers[0]
		c.Ports = append(c.Ports, v1.ContainerPort{Name: k, ContainerPort: int32(v)})
	}
	for k, v := range config.HostPorts {
		c := &template.Spec.Containers[0]
		c.Ports = append(c.Ports, v1.ContainerPort{Name: k, ContainerPort: int32(v), HostPort: int32(v)})
	}
	if config.CpuLimit > 0 || config.MemLimit > 0 || config.GpuLimit > 0 {
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
	if config.GpuLimit > 0 {
		template.Spec.Containers[0].Resources.Limits["nvidia.com/gpu"] = *resource.NewQuantity(config.GpuLimit, resource.DecimalSI)
	}
	if config.Lifecycle != nil {
		template.Spec.Containers[0].Lifecycle = config.Lifecycle
	}
	if len(config.Volumes) > 0 {
		template.Spec.Volumes = config.Volumes
	}
	if len(config.VolumeMounts) > 0 {
		template.Spec.Containers[0].VolumeMounts = config.VolumeMounts
	}
	if config.PriorityClassName != "" {
		template.Spec.PriorityClassName = config.PriorityClassName
	}
}

type RCStartupStatus struct {
	Expected              int
	Terminating           int
	Running               int
	RunningButNotReady    int
	Waiting               int
	Pending               int
	Scheduled             int
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
		// Record count of scheduled pods (useful for computing scheduler throughput).
		if p.Spec.NodeName != "" {
			startupStatus.Scheduled++
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

	ps, err := NewPodStore(config.Client, config.Namespace, label, fields.Everything())
	if err != nil {
		return err
	}
	defer ps.Stop()

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
	podDeletionsCount := 0
	for oldRunning != config.Replicas {
		time.Sleep(interval)

		pods := ps.List()
		startupStatus := ComputeRCStartupStatus(pods, config.Replicas)

		if config.CreatedPods != nil {
			*config.CreatedPods = startupStatus.Created
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

		diff := Diff(oldPods, pods)
		deletedPods := diff.DeletedPods()
		podDeletionsCount += len(deletedPods)
		if podDeletionsCount > config.MaxAllowedPodDeletions {
			// Number of pods which disappeared is over threshold
			err := fmt.Errorf("%d pods disappeared for %s: %v", podDeletionsCount, config.Name, strings.Join(deletedPods, ", "))
			config.RCConfigLog(err.Error())
			config.RCConfigLog(diff.String(sets.NewString()))
			return err
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
		if pods, err := config.Client.CoreV1().Pods(config.Namespace).List(context.TODO(), options); err == nil {
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
		if err := CreatePodWithRetries(c, namespace, &pod); err != nil {
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
	return WaitForEnoughPodsWithLabelRunning(c, ns, label, -1)
}

// Wait up to 10 minutes for at least 'replicas' many pods to be Running and at least
// one matching pod exists. If 'replicas' is < 0, wait for all matching pods running.
func WaitForEnoughPodsWithLabelRunning(c clientset.Interface, ns string, label labels.Selector, replicas int) error {
	running := false
	ps, err := NewPodStore(c, ns, label, fields.Everything())
	if err != nil {
		return err
	}
	defer ps.Stop()

	for start := time.Now(); time.Since(start) < 10*time.Minute; time.Sleep(5 * time.Second) {
		pods := ps.List()
		if len(pods) == 0 {
			continue
		}
		runningPodsCount := 0
		for _, p := range pods {
			if p.Status.Phase == v1.PodRunning {
				runningPodsCount++
			}
		}
		if (replicas < 0 && runningPodsCount < len(pods)) || (runningPodsCount < replicas) {
			continue
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
	// Modify pre-created Node objects before the test starts.
	PreparePatch(node *v1.Node) []byte
	// Create or modify any objects that depend on the node before the test starts.
	// Caller will re-try when http.StatusConflict error is returned.
	PrepareDependentObjects(node *v1.Node, client clientset.Interface) error
	// Clean up any node modifications after the test finishes.
	CleanupNode(node *v1.Node) *v1.Node
	// Clean up any objects that depend on the node after the test finishes.
	// Caller will re-try when http.StatusConflict error is returned.
	CleanupDependentObjects(nodeName string, client clientset.Interface) error
}

type TrivialNodePrepareStrategy struct{}

var _ PrepareNodeStrategy = &TrivialNodePrepareStrategy{}

func (*TrivialNodePrepareStrategy) PreparePatch(*v1.Node) []byte {
	return []byte{}
}

func (*TrivialNodePrepareStrategy) CleanupNode(node *v1.Node) *v1.Node {
	nodeCopy := *node
	return &nodeCopy
}

func (*TrivialNodePrepareStrategy) PrepareDependentObjects(node *v1.Node, client clientset.Interface) error {
	return nil
}

func (*TrivialNodePrepareStrategy) CleanupDependentObjects(nodeName string, client clientset.Interface) error {
	return nil
}

type LabelNodePrepareStrategy struct {
	LabelKey      string
	LabelValues   []string
	roundRobinIdx int
}

var _ PrepareNodeStrategy = &LabelNodePrepareStrategy{}

func NewLabelNodePrepareStrategy(labelKey string, labelValues ...string) *LabelNodePrepareStrategy {
	return &LabelNodePrepareStrategy{
		LabelKey:    labelKey,
		LabelValues: labelValues,
	}
}

func (s *LabelNodePrepareStrategy) PreparePatch(*v1.Node) []byte {
	labelString := fmt.Sprintf("{\"%v\":\"%v\"}", s.LabelKey, s.LabelValues[s.roundRobinIdx])
	patch := fmt.Sprintf(`{"metadata":{"labels":%v}}`, labelString)
	s.roundRobinIdx++
	if s.roundRobinIdx == len(s.LabelValues) {
		s.roundRobinIdx = 0
	}
	return []byte(patch)
}

func (s *LabelNodePrepareStrategy) CleanupNode(node *v1.Node) *v1.Node {
	nodeCopy := node.DeepCopy()
	if node.Labels != nil && len(node.Labels[s.LabelKey]) != 0 {
		delete(nodeCopy.Labels, s.LabelKey)
	}
	return nodeCopy
}

func (*LabelNodePrepareStrategy) PrepareDependentObjects(node *v1.Node, client clientset.Interface) error {
	return nil
}

func (*LabelNodePrepareStrategy) CleanupDependentObjects(nodeName string, client clientset.Interface) error {
	return nil
}

// NodeAllocatableStrategy fills node.status.allocatable and csiNode.spec.drivers[*].allocatable.
// csiNode is created if it does not exist. On cleanup, any csiNode.spec.drivers[*].allocatable is
// set to nil.
type NodeAllocatableStrategy struct {
	// Node.status.allocatable to fill to all nodes.
	NodeAllocatable map[v1.ResourceName]string
	// Map <driver_name> -> VolumeNodeResources to fill into csiNode.spec.drivers[<driver_name>].
	CsiNodeAllocatable map[string]*storagev1beta1.VolumeNodeResources
	// List of in-tree volume plugins migrated to CSI.
	MigratedPlugins []string
}

var _ PrepareNodeStrategy = &NodeAllocatableStrategy{}

func NewNodeAllocatableStrategy(nodeAllocatable map[v1.ResourceName]string, csiNodeAllocatable map[string]*storagev1beta1.VolumeNodeResources, migratedPlugins []string) *NodeAllocatableStrategy {
	return &NodeAllocatableStrategy{
		NodeAllocatable:    nodeAllocatable,
		CsiNodeAllocatable: csiNodeAllocatable,
		MigratedPlugins:    migratedPlugins,
	}
}

func (s *NodeAllocatableStrategy) PreparePatch(node *v1.Node) []byte {
	newNode := node.DeepCopy()
	for name, value := range s.NodeAllocatable {
		newNode.Status.Allocatable[name] = resource.MustParse(value)
	}

	oldJSON, err := json.Marshal(node)
	if err != nil {
		panic(err)
	}
	newJSON, err := json.Marshal(newNode)
	if err != nil {
		panic(err)
	}

	patch, err := strategicpatch.CreateTwoWayMergePatch(oldJSON, newJSON, v1.Node{})
	if err != nil {
		panic(err)
	}
	return patch
}

func (s *NodeAllocatableStrategy) CleanupNode(node *v1.Node) *v1.Node {
	nodeCopy := node.DeepCopy()
	for name := range s.NodeAllocatable {
		delete(nodeCopy.Status.Allocatable, name)
	}
	return nodeCopy
}

func (s *NodeAllocatableStrategy) createCSINode(nodeName string, client clientset.Interface) error {
	csiNode := &storagev1beta1.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
			Annotations: map[string]string{
				v1.MigratedPluginsAnnotationKey: strings.Join(s.MigratedPlugins, ","),
			},
		},
		Spec: storagev1beta1.CSINodeSpec{
			Drivers: []storagev1beta1.CSINodeDriver{},
		},
	}

	for driver, allocatable := range s.CsiNodeAllocatable {
		d := storagev1beta1.CSINodeDriver{
			Name:        driver,
			Allocatable: allocatable,
			NodeID:      nodeName,
		}
		csiNode.Spec.Drivers = append(csiNode.Spec.Drivers, d)
	}

	_, err := client.StorageV1beta1().CSINodes().Create(context.TODO(), csiNode, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		// Something created CSINode instance after we checked it did not exist.
		// Make the caller to re-try PrepareDependentObjects by returning Conflict error
		err = apierrors.NewConflict(storagev1beta1.Resource("csinodes"), nodeName, err)
	}
	return err
}

func (s *NodeAllocatableStrategy) updateCSINode(csiNode *storagev1beta1.CSINode, client clientset.Interface) error {
	for driverName, allocatable := range s.CsiNodeAllocatable {
		found := false
		for i, driver := range csiNode.Spec.Drivers {
			if driver.Name == driverName {
				found = true
				csiNode.Spec.Drivers[i].Allocatable = allocatable
				break
			}
		}
		if !found {
			d := storagev1beta1.CSINodeDriver{
				Name:        driverName,
				Allocatable: allocatable,
			}

			csiNode.Spec.Drivers = append(csiNode.Spec.Drivers, d)
		}
	}
	csiNode.Annotations[v1.MigratedPluginsAnnotationKey] = strings.Join(s.MigratedPlugins, ",")

	_, err := client.StorageV1beta1().CSINodes().Update(context.TODO(), csiNode, metav1.UpdateOptions{})
	return err
}

func (s *NodeAllocatableStrategy) PrepareDependentObjects(node *v1.Node, client clientset.Interface) error {
	csiNode, err := client.StorageV1beta1().CSINodes().Get(context.TODO(), node.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return s.createCSINode(node.Name, client)
		}
		return err
	}
	return s.updateCSINode(csiNode, client)
}

func (s *NodeAllocatableStrategy) CleanupDependentObjects(nodeName string, client clientset.Interface) error {
	csiNode, err := client.StorageV1beta1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}

	for driverName := range s.CsiNodeAllocatable {
		for i, driver := range csiNode.Spec.Drivers {
			if driver.Name == driverName {
				csiNode.Spec.Drivers[i].Allocatable = nil
			}
		}
	}
	return s.updateCSINode(csiNode, client)
}

// UniqueNodeLabelStrategy sets a unique label for each node.
type UniqueNodeLabelStrategy struct {
	LabelKey string
}

var _ PrepareNodeStrategy = &UniqueNodeLabelStrategy{}

func NewUniqueNodeLabelStrategy(labelKey string) *UniqueNodeLabelStrategy {
	return &UniqueNodeLabelStrategy{
		LabelKey: labelKey,
	}
}

func (s *UniqueNodeLabelStrategy) PreparePatch(*v1.Node) []byte {
	labelString := fmt.Sprintf("{\"%v\":\"%v\"}", s.LabelKey, string(uuid.NewUUID()))
	patch := fmt.Sprintf(`{"metadata":{"labels":%v}}`, labelString)
	return []byte(patch)
}

func (s *UniqueNodeLabelStrategy) CleanupNode(node *v1.Node) *v1.Node {
	nodeCopy := node.DeepCopy()
	if node.Labels != nil && len(node.Labels[s.LabelKey]) != 0 {
		delete(nodeCopy.Labels, s.LabelKey)
	}
	return nodeCopy
}

func (*UniqueNodeLabelStrategy) PrepareDependentObjects(node *v1.Node, client clientset.Interface) error {
	return nil
}

func (*UniqueNodeLabelStrategy) CleanupDependentObjects(nodeName string, client clientset.Interface) error {
	return nil
}

func DoPrepareNode(client clientset.Interface, node *v1.Node, strategy PrepareNodeStrategy) error {
	var err error
	patch := strategy.PreparePatch(node)
	if len(patch) == 0 {
		return nil
	}
	for attempt := 0; attempt < retries; attempt++ {
		if _, err = client.CoreV1().Nodes().Patch(context.TODO(), node.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{}); err == nil {
			break
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("Error while applying patch %v to Node %v: %v", string(patch), node.Name, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		return fmt.Errorf("Too many conflicts when applying patch %v to Node %v: %s", string(patch), node.Name, err)
	}

	for attempt := 0; attempt < retries; attempt++ {
		if err = strategy.PrepareDependentObjects(node, client); err == nil {
			break
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("Error while preparing objects for node %s: %s", node.Name, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		return fmt.Errorf("Too many conflicts when creating objects for node %s: %s", node.Name, err)
	}
	return nil
}

func DoCleanupNode(client clientset.Interface, nodeName string, strategy PrepareNodeStrategy) error {
	var err error
	for attempt := 0; attempt < retries; attempt++ {
		node, err := client.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("Skipping cleanup of Node: failed to get Node %v: %v", nodeName, err)
		}
		updatedNode := strategy.CleanupNode(node)
		if apiequality.Semantic.DeepEqual(node, updatedNode) {
			return nil
		}
		if _, err = client.CoreV1().Nodes().Update(context.TODO(), updatedNode, metav1.UpdateOptions{}); err == nil {
			break
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("Error when updating Node %v: %v", nodeName, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		return fmt.Errorf("Too many conflicts when trying to cleanup Node %v: %s", nodeName, err)
	}

	for attempt := 0; attempt < retries; attempt++ {
		err = strategy.CleanupDependentObjects(nodeName, client)
		if err == nil {
			break
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("Error when cleaning up Node %v objects: %v", nodeName, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		return fmt.Errorf("Too many conflicts when trying to cleanup Node %v objects: %s", nodeName, err)
	}
	return nil
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
			Image: "k8s.gcr.io/pause:3.2",
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
	if err := CreatePodWithRetries(client, namespace, podTemplate); err != nil {
		return fmt.Errorf("Error creating pod: %v", err)
	}
	return nil
}

func CreatePod(client clientset.Interface, namespace string, podCount int, podTemplate *v1.Pod) error {
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
		workqueue.ParallelizeUntil(context.TODO(), podCount, podCount, createPodFunc)
	} else {
		workqueue.ParallelizeUntil(context.TODO(), 30, podCount, createPodFunc)
	}
	return createError
}

func CreatePodWithPersistentVolume(client clientset.Interface, namespace string, claimTemplate *v1.PersistentVolumeClaim, factory volumeFactory, podTemplate *v1.Pod, count int, bindVolume bool) error {
	var createError error
	lock := sync.Mutex{}
	createPodFunc := func(i int) {
		pvcName := fmt.Sprintf("pvc-%d", i)
		// pvc
		pvc := claimTemplate.DeepCopy()
		pvc.Name = pvcName
		// pv
		pv := factory(i)
		// PVs are cluster-wide resources.
		// Prepend a namespace to make the name globally unique.
		pv.Name = fmt.Sprintf("%s-%s", namespace, pv.Name)
		if bindVolume {
			// bind pv to "pvc-$i"
			pv.Spec.ClaimRef = &v1.ObjectReference{
				Kind:       "PersistentVolumeClaim",
				Namespace:  namespace,
				Name:       pvcName,
				APIVersion: "v1",
			}
			pv.Status.Phase = v1.VolumeBound

			// bind pvc to "pv-$i"
			// pvc.Spec.VolumeName = pv.Name
			pvc.Status.Phase = v1.ClaimBound
		} else {
			pv.Status.Phase = v1.VolumeAvailable
		}
		if err := CreatePersistentVolumeWithRetries(client, pv); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error creating PV: %s", err)
			return
		}
		// We need to update statuses separately, as creating pv/pvc resets status to the default one.
		if _, err := client.CoreV1().PersistentVolumes().UpdateStatus(context.TODO(), pv, metav1.UpdateOptions{}); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error updating PV status: %s", err)
			return
		}

		if err := CreatePersistentVolumeClaimWithRetries(client, namespace, pvc); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error creating PVC: %s", err)
			return
		}
		if _, err := client.CoreV1().PersistentVolumeClaims(namespace).UpdateStatus(context.TODO(), pvc, metav1.UpdateOptions{}); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = fmt.Errorf("error updating PVC status: %s", err)
			return
		}

		// pod
		pod := podTemplate.DeepCopy()
		pod.Spec.Volumes = []v1.Volume{
			{
				Name: "vol",
				VolumeSource: v1.VolumeSource{
					PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
						ClaimName: pvcName,
					},
				},
			},
		}
		if err := makeCreatePod(client, namespace, pod); err != nil {
			lock.Lock()
			defer lock.Unlock()
			createError = err
			return
		}
	}

	if count < 30 {
		workqueue.ParallelizeUntil(context.TODO(), count, count, createPodFunc)
	} else {
		workqueue.ParallelizeUntil(context.TODO(), 30, count, createPodFunc)
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
	if err := CreateRCWithRetries(client, namespace, rc); err != nil {
		return fmt.Errorf("Error creating replication controller: %v", err)
	}
	return nil
}

func NewCustomCreatePodStrategy(podTemplate *v1.Pod) TestPodCreateStrategy {
	return func(client clientset.Interface, namespace string, podCount int) error {
		return CreatePod(client, namespace, podCount, podTemplate)
	}
}

// volumeFactory creates an unique PersistentVolume for given integer.
type volumeFactory func(uniqueID int) *v1.PersistentVolume

func NewCreatePodWithPersistentVolumeStrategy(claimTemplate *v1.PersistentVolumeClaim, factory volumeFactory, podTemplate *v1.Pod) TestPodCreateStrategy {
	return func(client clientset.Interface, namespace string, podCount int) error {
		return CreatePodWithPersistentVolume(client, namespace, claimTemplate, factory, podTemplate, podCount, true /* bindVolume */)
	}
}

func makeUnboundPersistentVolumeClaim(storageClass string) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany},
			StorageClassName: &storageClass,
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}
}

func NewCreatePodWithPersistentVolumeWithFirstConsumerStrategy(factory volumeFactory, podTemplate *v1.Pod) TestPodCreateStrategy {
	return func(client clientset.Interface, namespace string, podCount int) error {
		volumeBindingMode := storage.VolumeBindingWaitForFirstConsumer
		storageClass := &storage.StorageClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: "storage-class-1",
			},
			Provisioner:       "kubernetes.io/gce-pd",
			VolumeBindingMode: &volumeBindingMode,
		}
		claimTemplate := makeUnboundPersistentVolumeClaim(storageClass.Name)

		if err := CreateStorageClassWithRetries(client, storageClass); err != nil {
			return fmt.Errorf("failed to create storage class: %v", err)
		}

		factoryWithStorageClass := func(i int) *v1.PersistentVolume {
			pv := factory(i)
			pv.Spec.StorageClassName = storageClass.Name
			return pv
		}

		return CreatePodWithPersistentVolume(client, namespace, claimTemplate, factoryWithStorageClass, podTemplate, podCount, false /* bindVolume */)
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
		return CreatePod(client, namespace, podCount, basePod)
	}
}

type SecretConfig struct {
	Content   map[string]string
	Client    clientset.Interface
	Name      string
	Namespace string
	// If set this function will be used to print log lines instead of klog.
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

	if err := CreateSecretWithRetries(config.Client, config.Namespace, secret); err != nil {
		return fmt.Errorf("Error creating secret: %v", err)
	}
	config.LogFunc("Created secret %v/%v", config.Namespace, config.Name)
	return nil
}

func (config *SecretConfig) Stop() error {
	if err := DeleteResourceWithRetries(config.Client, api.Kind("Secret"), config.Namespace, config.Name, metav1.DeleteOptions{}); err != nil {
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

type ConfigMapConfig struct {
	Content   map[string]string
	Client    clientset.Interface
	Name      string
	Namespace string
	// If set this function will be used to print log lines instead of klog.
	LogFunc func(fmt string, args ...interface{})
}

func (config *ConfigMapConfig) Run() error {
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Data: map[string]string{},
	}
	for k, v := range config.Content {
		configMap.Data[k] = v
	}

	if err := CreateConfigMapWithRetries(config.Client, config.Namespace, configMap); err != nil {
		return fmt.Errorf("Error creating configmap: %v", err)
	}
	config.LogFunc("Created configmap %v/%v", config.Namespace, config.Name)
	return nil
}

func (config *ConfigMapConfig) Stop() error {
	if err := DeleteResourceWithRetries(config.Client, api.Kind("ConfigMap"), config.Namespace, config.Name, metav1.DeleteOptions{}); err != nil {
		return fmt.Errorf("Error deleting configmap: %v", err)
	}
	config.LogFunc("Deleted configmap %v/%v", config.Namespace, config.Name)
	return nil
}

// TODO: attach configmaps using different possibilities: env vars.
func attachConfigMaps(template *v1.PodTemplateSpec, configMapNames []string) {
	volumes := make([]v1.Volume, 0, len(configMapNames))
	mounts := make([]v1.VolumeMount, 0, len(configMapNames))
	for _, name := range configMapNames {
		volumes = append(volumes, v1.Volume{
			Name: name,
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: name,
					},
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

func (config *RCConfig) getTerminationGracePeriodSeconds(defaultGrace *int64) *int64 {
	if config.TerminationGracePeriodSeconds == nil || *config.TerminationGracePeriodSeconds < 0 {
		return defaultGrace
	}
	return config.TerminationGracePeriodSeconds
}

func attachServiceAccountTokenProjection(template *v1.PodTemplateSpec, name string) {
	template.Spec.Containers[0].VolumeMounts = append(template.Spec.Containers[0].VolumeMounts,
		v1.VolumeMount{
			Name:      name,
			MountPath: "/var/service-account-tokens/" + name,
		})

	template.Spec.Volumes = append(template.Spec.Volumes,
		v1.Volume{
			Name: name,
			VolumeSource: v1.VolumeSource{
				Projected: &v1.ProjectedVolumeSource{
					Sources: []v1.VolumeProjection{
						{
							ServiceAccountToken: &v1.ServiceAccountTokenProjection{
								Path:     "token",
								Audience: name,
							},
						},
						{
							ConfigMap: &v1.ConfigMapProjection{
								LocalObjectReference: v1.LocalObjectReference{
									Name: "kube-root-ca-crt",
								},
								Items: []v1.KeyToPath{
									{
										Key:  "ca.crt",
										Path: "ca.crt",
									},
								},
							},
						},
						{
							DownwardAPI: &v1.DownwardAPIProjection{
								Items: []v1.DownwardAPIVolumeFile{
									{
										Path: "namespace",
										FieldRef: &v1.ObjectFieldSelector{
											APIVersion: "v1",
											FieldPath:  "metadata.namespace",
										},
									},
								},
							},
						},
					},
				},
			},
		})
}

type DaemonConfig struct {
	Client    clientset.Interface
	Name      string
	Namespace string
	Image     string
	// If set this function will be used to print log lines instead of klog.
	LogFunc func(fmt string, args ...interface{})
	// How long we wait for DaemonSet to become running.
	Timeout time.Duration
}

func (config *DaemonConfig) Run() error {
	if config.Image == "" {
		config.Image = "k8s.gcr.io/pause:3.2"
	}
	nameLabel := map[string]string{
		"name": config.Name + "-daemon",
	}
	daemon := &apps.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Name,
		},
		Spec: apps.DaemonSetSpec{
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

	if err := CreateDaemonSetWithRetries(config.Client, config.Namespace, daemon); err != nil {
		return fmt.Errorf("Error creating daemonset: %v", err)
	}

	var nodes *v1.NodeList
	var err error
	for i := 0; i < retries; i++ {
		// Wait for all daemons to be running
		nodes, err = config.Client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{ResourceVersion: "0"})
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

	ps, err := NewPodStore(config.Client, config.Namespace, labels.SelectorFromSet(nameLabel), fields.Everything())
	if err != nil {
		return err
	}
	defer ps.Stop()

	err = wait.Poll(time.Second, timeout, func() (bool, error) {
		pods := ps.List()

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
		config.LogFunc("Timed out while waiting for DaemonSet %v/%v to be running.", config.Namespace, config.Name)
	} else {
		config.LogFunc("Created Daemon %v/%v", config.Namespace, config.Name)
	}

	return err
}
