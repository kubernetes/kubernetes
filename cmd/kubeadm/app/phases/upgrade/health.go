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
	"context"
	"fmt"
	"os"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

// createJobHealthCheckPrefix is name prefix for the Job health check.
const createJobHealthCheckPrefix = "upgrade-health-check"

// healthCheck is a helper struct for easily performing healthchecks against the cluster and printing the output
type healthCheck struct {
	name    string
	client  clientset.Interface
	cfg     *kubeadmapi.ClusterConfiguration
	dryRun  bool
	printer output.Printer
	// f is invoked with a k8s client and a kubeadm ClusterConfiguration passed to it. Should return an optional error
	f func(clientset.Interface, *kubeadmapi.ClusterConfiguration, bool, output.Printer) error
}

// Check is part of the preflight.Checker interface
func (c *healthCheck) Check() (warnings, errors []error) {
	if err := c.f(c.client, c.cfg, c.dryRun, c.printer); err != nil {
		return nil, []error{err}
	}
	return nil, nil
}

// Name is part of the preflight.Checker interface
func (c *healthCheck) Name() string {
	return c.name
}

// CheckClusterHealth makes sure:
// - the cluster can accept a workload
// - all control-plane Nodes are Ready
// - (if static pod-hosted) that all required Static Pod manifests exist on disk
func CheckClusterHealth(client clientset.Interface, cfg *kubeadmapi.ClusterConfiguration, ignoreChecksErrors sets.Set[string], dryRun bool, printer output.Printer) error {
	_, _ = printer.Println("[upgrade] Running cluster health checks")

	healthChecks := []preflight.Checker{
		&healthCheck{
			name:    "CreateJob",
			client:  client,
			cfg:     cfg,
			f:       createJob,
			dryRun:  dryRun,
			printer: printer,
		},
		&healthCheck{
			name:    "ControlPlaneNodesReady",
			client:  client,
			f:       controlPlaneNodesReady,
			dryRun:  dryRun,
			printer: printer,
		},
		&healthCheck{
			name:    "StaticPodManifest",
			f:       staticPodManifestHealth,
			dryRun:  dryRun,
			printer: printer,
		},
	}

	return preflight.RunChecks(healthChecks, os.Stderr, ignoreChecksErrors)
}

// createJob is a check that verifies that a Job can be created in the cluster
func createJob(client clientset.Interface, cfg *kubeadmapi.ClusterConfiguration, _ bool, _ output.Printer) error {
	const (
		fieldSelector = "spec.unschedulable=false"
		ns            = metav1.NamespaceSystem
		timeout       = 15 * time.Second
		timeoutMargin = 5 * time.Second
	)
	var (
		err, lastError error
		ctx            = context.Background()
		nodes          *v1.NodeList
		listOptions    = metav1.ListOptions{Limit: 1, FieldSelector: fieldSelector}
	)

	// Check if there is at least one Node where a Job's Pod can schedule. If not, skip this preflight check.
	err = wait.PollUntilContextTimeout(ctx, time.Second*1, timeout, true, func(_ context.Context) (bool, error) {
		nodes, err = client.CoreV1().Nodes().List(context.Background(), listOptions)
		if err != nil {
			klog.V(2).Infof("Could not list Nodes with field selector %q: %v", fieldSelector, err)
			lastError = err
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return errors.Wrap(lastError, "could not check if there is at least one Node that can schedule a test Pod")
	}

	if len(nodes.Items) == 0 {
		klog.Warning("The preflight check \"CreateJob\" was skipped because there are no schedulable Nodes in the cluster.")
		return nil
	}

	// Adding a margin of error to the polling timeout.
	timeoutWithMargin := timeout.Seconds() + timeoutMargin.Seconds()

	// Do not use ObjectMeta.GenerateName to avoid the problem where the dry-run client for upgrade cannot obtain
	// a Name for this Job during the GET call right after the CREATE call.
	jobName := fmt.Sprintf("%s-%d", createJobHealthCheckPrefix, time.Now().UTC().UnixMilli())

	// Prepare Job
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: ns,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit:            ptr.To[int32](0),
			TTLSecondsAfterFinished: ptr.To[int32](int32(timeoutWithMargin)),
			ActiveDeadlineSeconds:   ptr.To[int64](int64(timeoutWithMargin)),
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					SecurityContext: &v1.PodSecurityContext{
						RunAsUser:    ptr.To[int64](999),
						RunAsGroup:   ptr.To[int64](999),
						RunAsNonRoot: ptr.To(true),
					},
					Tolerations: []v1.Toleration{
						{
							Key:    constants.LabelNodeRoleControlPlane,
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					Containers: []v1.Container{
						{
							Name:  createJobHealthCheckPrefix,
							Image: images.GetPauseImage(cfg),
							Args:  []string{"-v"},
						},
					},
				},
			},
		},
	}

	// Create the Job, but retry if it fails
	klog.V(2).Infof("Creating a Job %q in the namespace %q", jobName, ns)
	err = wait.PollUntilContextTimeout(ctx, time.Second*1, timeout, true, func(_ context.Context) (bool, error) {
		_, err := client.BatchV1().Jobs(ns).Create(context.Background(), job, metav1.CreateOptions{})
		if err != nil {
			klog.V(2).Infof("Could not create Job %q in the namespace %q, retrying: %v", jobName, ns, err)
			lastError = err
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return errors.Wrapf(lastError, "could not create Job %q in the namespace %q", jobName, ns)
	}

	// Wait for the Job to complete
	err = wait.PollUntilContextTimeout(ctx, time.Second*1, timeout, true, func(_ context.Context) (bool, error) {
		job, err := client.BatchV1().Jobs(ns).Get(context.Background(), jobName, metav1.GetOptions{})
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

// controlPlaneNodesReady checks whether all control-plane Nodes in the cluster are in the Running state
func controlPlaneNodesReady(client clientset.Interface, _ *kubeadmapi.ClusterConfiguration, _ bool, _ output.Printer) error {
	selectorControlPlane := labels.SelectorFromSet(map[string]string{
		constants.LabelNodeRoleControlPlane: "",
	})
	nodes, err := client.CoreV1().Nodes().List(context.Background(), metav1.ListOptions{
		LabelSelector: selectorControlPlane.String(),
	})
	if err != nil {
		return errors.Wrapf(err, "could not list nodes labeled with %q", constants.LabelNodeRoleControlPlane)
	}

	notReadyControlPlanes := getNotReadyNodes(nodes.Items)
	if len(notReadyControlPlanes) != 0 {
		return errors.Errorf("there are NotReady control-planes in the cluster: %v", notReadyControlPlanes)
	}
	return nil
}

// staticPodManifestHealth makes sure the required static pods are presents
func staticPodManifestHealth(_ clientset.Interface, _ *kubeadmapi.ClusterConfiguration, dryRun bool, printer output.Printer) error {
	var nonExistentManifests []string
	for _, component := range constants.ControlPlaneComponents {
		manifestFile := constants.GetStaticPodFilepath(component, constants.GetStaticPodDirectory())
		if dryRun {
			_, _ = printer.Printf("[dryrun] would check if %s exists\n", manifestFile)
			continue
		}
		if _, err := os.Stat(manifestFile); os.IsNotExist(err) {
			nonExistentManifests = append(nonExistentManifests, manifestFile)
		}
	}
	if len(nonExistentManifests) == 0 {
		return nil
	}
	return errors.Errorf("manifest files not found: %v", nonExistentManifests)
}

// getNotReadyNodes returns a string slice of nodes in the cluster that are NotReady
func getNotReadyNodes(nodes []v1.Node) []string {
	var notReadyNodes []string
	for _, node := range nodes {
		for _, condition := range node.Status.Conditions {
			if condition.Type == v1.NodeReady && condition.Status != v1.ConditionTrue {
				notReadyNodes = append(notReadyNodes, node.ObjectMeta.Name)
			}
		}
	}
	return notReadyNodes
}
