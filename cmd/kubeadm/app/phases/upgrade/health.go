/*
Copyright 2017 The Kubernetes Authors.

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

package upgrade

import (
	"fmt"
	"os"
	"time"

	"github.com/pkg/errors"

	apps "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	utilpointer "k8s.io/utils/pointer"
)

// healthCheck is a helper struct for easily performing healthchecks against the cluster and printing the output
type healthCheck struct {
	name   string
	client clientset.Interface
	cfg    *kubeadmapi.ClusterConfiguration
	// f is invoked with a k8s client and a kubeadm ClusterConfiguration passed to it. Should return an optional error
	f func(clientset.Interface, *kubeadmapi.ClusterConfiguration) error
}

// Check is part of the preflight.Checker interface
func (c *healthCheck) Check() (warnings, errors []error) {
	if err := c.f(c.client, c.cfg); err != nil {
		return nil, []error{err}
	}
	return nil, nil
}

// Name is part of the preflight.Checker interface
func (c *healthCheck) Name() string {
	return c.name
}

// CheckClusterHealth makes sure:
// - the API /healthz endpoint is healthy
// - all control-plane Nodes are Ready
// - (if self-hosted) that there are DaemonSets with at least one Pod for all control plane components
// - (if static pod-hosted) that all required Static Pod manifests exist on disk
func CheckClusterHealth(client clientset.Interface, cfg *kubeadmapi.ClusterConfiguration, ignoreChecksErrors sets.String) error {
	fmt.Println("[upgrade] Running cluster health checks")

	healthChecks := []preflight.Checker{
		&healthCheck{
			name:   "CreateJob",
			client: client,
			cfg:    cfg,
			f:      createJob,
		},
		&healthCheck{
			name:   "ControlPlaneNodesReady",
			client: client,
			f:      controlPlaneNodesReady,
		},
		&healthCheck{
			name:   "StaticPodManifest",
			client: client,
			cfg:    cfg,
			f:      staticPodManifestHealth,
		},
	}

	return preflight.RunChecks(healthChecks, os.Stderr, ignoreChecksErrors)
}

// CreateJob is a check that verifies that a Job can be created in the cluster
func createJob(client clientset.Interface, cfg *kubeadmapi.ClusterConfiguration) (lastError error) {
	const (
		jobName = "upgrade-health-check"
		ns      = metav1.NamespaceSystem
		timeout = 15 * time.Second
	)

	// If client.Discovery().RESTClient() is nil, the fake client is used.
	// Return early because the kubeadm dryrun dynamic client only handles the core/v1 GroupVersion.
	if client.Discovery().RESTClient() == nil {
		fmt.Printf("[dryrun] Would create the Job %q in namespace %q and wait until it completes\n", jobName, ns)
		return nil
	}

	// Prepare Job
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: ns,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: utilpointer.Int32Ptr(0),
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					SecurityContext: &v1.PodSecurityContext{
						RunAsUser:    utilpointer.Int64Ptr(999),
						RunAsGroup:   utilpointer.Int64Ptr(999),
						RunAsNonRoot: utilpointer.BoolPtr(true),
					},
					Tolerations: []v1.Toleration{
						{
							Key:    "node-role.kubernetes.io/master",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					Containers: []v1.Container{
						{
							Name:  jobName,
							Image: images.GetPauseImage(cfg),
							Args:  []string{"-v"},
						},
					},
				},
			},
		},
	}

	// Check if the Job already exists and delete it
	if _, err := client.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{}); err == nil {
		if err = deleteHealthCheckJob(client, ns, jobName); err != nil {
			return err
		}
	}

	// Cleanup the Job on exit
	defer func() {
		lastError = deleteHealthCheckJob(client, ns, jobName)
	}()

	// Create the Job, but retry in case it is being currently deleted
	klog.V(2).Infof("Creating Job %q in the namespace %q", jobName, ns)
	err := wait.PollImmediate(time.Second*1, timeout, func() (bool, error) {
		if _, err := client.BatchV1().Jobs(ns).Create(job); err != nil {
			klog.V(2).Infof("Could not create Job %q in the namespace %q, retrying: %v", jobName, ns, err)
			lastError = err
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return errors.Wrapf(lastError, "could not create Job %q in the namespace %q", jobName, ns)
	}

	// Waiting and manually deleteing the Job is a workaround to not enabling the TTL controller.
	// TODO: refactor this if the TTL controller is enabled in kubeadm once it goes Beta.

	// Wait for the Job to complete
	err = wait.PollImmediate(time.Second*1, timeout, func() (bool, error) {
		job, err := client.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
		if err != nil {
			lastError = err
			klog.V(2).Infof("could not get Job %q in the namespace %q, retrying: %v", jobName, ns, err)
			return false, nil
		}
		for _, cond := range job.Status.Conditions {
			if cond.Type == batchv1.JobComplete {
				return true, nil
			}
		}
		lastError = errors.Errorf("no condition of type %v", batchv1.JobComplete)
		klog.V(2).Infof("Job %q in the namespace %q is not yet complete, retrying", jobName, ns)
		return false, nil
	})
	if err != nil {
		return errors.Wrapf(lastError, "Job %q in the namespace %q did not complete in %v", jobName, ns, timeout)
	}

	klog.V(2).Infof("Job %q in the namespace %q completed", jobName, ns)

	return nil
}

func deleteHealthCheckJob(client clientset.Interface, ns, jobName string) error {
	klog.V(2).Infof("Deleting Job %q in the namespace %q", jobName, ns)
	propagation := metav1.DeletePropagationForeground
	deleteOptions := &metav1.DeleteOptions{
		PropagationPolicy: &propagation,
	}
	if err := client.BatchV1().Jobs(ns).Delete(jobName, deleteOptions); err != nil {
		return errors.Wrapf(err, "could not delete Job %q in the namespace %q", jobName, ns)
	}
	return nil
}

// controlPlaneNodesReady checks whether all control-plane Nodes in the cluster are in the Running state
func controlPlaneNodesReady(client clientset.Interface, _ *kubeadmapi.ClusterConfiguration) error {
	selector := labels.SelectorFromSet(labels.Set(map[string]string{
		constants.LabelNodeRoleMaster: "",
	}))
	controlPlanes, err := client.CoreV1().Nodes().List(metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	if err != nil {
		return errors.Wrap(err, "couldn't list control-planes in cluster")
	}

	if len(controlPlanes.Items) == 0 {
		return errors.New("failed to find any nodes with a control-plane role")
	}

	notReadyControlPlanes := getNotReadyNodes(controlPlanes.Items)
	if len(notReadyControlPlanes) != 0 {
		return errors.Errorf("there are NotReady control-planes in the cluster: %v", notReadyControlPlanes)
	}
	return nil
}

// staticPodManifestHealth makes sure the required static pods are presents
func staticPodManifestHealth(_ clientset.Interface, _ *kubeadmapi.ClusterConfiguration) error {
	nonExistentManifests := []string{}
	for _, component := range constants.ControlPlaneComponents {
		manifestFile := constants.GetStaticPodFilepath(component, constants.GetStaticPodDirectory())
		if _, err := os.Stat(manifestFile); os.IsNotExist(err) {
			nonExistentManifests = append(nonExistentManifests, manifestFile)
		}
	}
	if len(nonExistentManifests) == 0 {
		return nil
	}
	return errors.Errorf("The control plane seems to be Static Pod-hosted, but some of the manifests don't seem to exist on disk. This probably means you're running 'kubeadm upgrade' on a remote machine, which is not supported for a Static Pod-hosted cluster. Manifest files not found: %v", nonExistentManifests)
}

// IsControlPlaneSelfHosted returns whether the control plane is self hosted or not
func IsControlPlaneSelfHosted(client clientset.Interface) bool {
	notReadyDaemonSets, err := getNotReadyDaemonSets(client)
	if err != nil {
		return false
	}

	// If there are no NotReady DaemonSets, we are using selfhosting
	return len(notReadyDaemonSets) == 0
}

// getNotReadyDaemonSets gets the amount of Ready control plane DaemonSets
func getNotReadyDaemonSets(client clientset.Interface) ([]error, error) {
	notReadyDaemonSets := []error{}
	for _, component := range constants.ControlPlaneComponents {
		dsName := constants.AddSelfHostedPrefix(component)
		ds, err := client.AppsV1().DaemonSets(metav1.NamespaceSystem).Get(dsName, metav1.GetOptions{})
		if err != nil {
			return nil, errors.Errorf("couldn't get daemonset %q in the %s namespace", dsName, metav1.NamespaceSystem)
		}

		if err := daemonSetHealth(&ds.Status); err != nil {
			notReadyDaemonSets = append(notReadyDaemonSets, errors.Wrapf(err, "DaemonSet %q not healthy", dsName))
		}
	}
	return notReadyDaemonSets, nil
}

// daemonSetHealth is a helper function for getting the health of a DaemonSet's status
func daemonSetHealth(dsStatus *apps.DaemonSetStatus) error {
	if dsStatus.CurrentNumberScheduled != dsStatus.DesiredNumberScheduled {
		return errors.Errorf("current number of scheduled Pods ('%d') doesn't match the amount of desired Pods ('%d')",
			dsStatus.CurrentNumberScheduled, dsStatus.DesiredNumberScheduled)
	}
	if dsStatus.NumberAvailable == 0 {
		return errors.New("no available Pods for DaemonSet")
	}
	if dsStatus.NumberReady == 0 {
		return errors.New("no ready Pods for DaemonSet")
	}
	return nil
}

// getNotReadyNodes returns a string slice of nodes in the cluster that are NotReady
func getNotReadyNodes(nodes []v1.Node) []string {
	notReadyNodes := []string{}
	for _, node := range nodes {
		for _, condition := range node.Status.Conditions {
			if condition.Type == v1.NodeReady && condition.Status != v1.ConditionTrue {
				notReadyNodes = append(notReadyNodes, node.ObjectMeta.Name)
			}
		}
	}
	return notReadyNodes
}
