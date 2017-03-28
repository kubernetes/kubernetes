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

package internalversion

import (
	"bytes"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/events"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/storage"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/pkg/util/node"
)

const loadBalancerWidth = 16

// NOTE: When adding a new resource type here, please update the list
// pkg/kubectl/cmd/get.go to reflect the new resource type.
var (
	podColumns                       = []string{"NAME", "READY", "STATUS", "RESTARTS", "AGE"}
	podWideColumns                   = []string{"IP", "NODE"}
	podTemplateColumns               = []string{"TEMPLATE", "CONTAINER(S)", "IMAGE(S)", "PODLABELS"}
	podDisruptionBudgetColumns       = []string{"NAME", "MIN-AVAILABLE", "ALLOWED-DISRUPTIONS", "AGE"}
	replicationControllerColumns     = []string{"NAME", "DESIRED", "CURRENT", "READY", "AGE"}
	replicationControllerWideColumns = []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
	replicaSetColumns                = []string{"NAME", "DESIRED", "CURRENT", "READY", "AGE"}
	replicaSetWideColumns            = []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
	jobColumns                       = []string{"NAME", "DESIRED", "SUCCESSFUL", "AGE"}
	cronJobColumns                   = []string{"NAME", "SCHEDULE", "SUSPEND", "ACTIVE", "LAST-SCHEDULE"}
	batchJobWideColumns              = []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
	serviceColumns                   = []string{"NAME", "CLUSTER-IP", "EXTERNAL-IP", "PORT(S)", "AGE"}
	serviceWideColumns               = []string{"SELECTOR"}
	ingressColumns                   = []string{"NAME", "HOSTS", "ADDRESS", "PORTS", "AGE"}
	statefulSetColumns               = []string{"NAME", "DESIRED", "CURRENT", "AGE"}
	endpointColumns                  = []string{"NAME", "ENDPOINTS", "AGE"}
	nodeColumns                      = []string{"NAME", "STATUS", "AGE", "VERSION"}
	nodeWideColumns                  = []string{"EXTERNAL-IP", "OS-IMAGE", "KERNEL-VERSION"}
	daemonSetColumns                 = []string{"NAME", "DESIRED", "CURRENT", "READY", "UP-TO-DATE", "AVAILABLE", "NODE-SELECTOR", "AGE"}
	daemonSetWideColumns             = []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
	eventColumns                     = []string{"LASTSEEN", "FIRSTSEEN", "COUNT", "NAME", "KIND", "SUBOBJECT", "TYPE", "REASON", "SOURCE", "MESSAGE"}
	limitRangeColumns                = []string{"NAME", "AGE"}
	resourceQuotaColumns             = []string{"NAME", "AGE"}
	namespaceColumns                 = []string{"NAME", "STATUS", "AGE"}
	secretColumns                    = []string{"NAME", "TYPE", "DATA", "AGE"}
	serviceAccountColumns            = []string{"NAME", "SECRETS", "AGE"}
	persistentVolumeColumns          = []string{"NAME", "CAPACITY", "ACCESSMODES", "RECLAIMPOLICY", "STATUS", "CLAIM", "STORAGECLASS", "REASON", "AGE"}
	persistentVolumeClaimColumns     = []string{"NAME", "STATUS", "VOLUME", "CAPACITY", "ACCESSMODES", "STORAGECLASS", "AGE"}
	componentStatusColumns           = []string{"NAME", "STATUS", "MESSAGE", "ERROR"}
	thirdPartyResourceColumns        = []string{"NAME", "DESCRIPTION", "VERSION(S)"}
	roleColumns                      = []string{"NAME", "AGE"}
	roleBindingColumns               = []string{"NAME", "AGE"}
	roleBindingWideColumns           = []string{"ROLE", "USERS", "GROUPS", "SERVICEACCOUNTS"}
	clusterRoleColumns               = []string{"NAME", "AGE"}
	clusterRoleBindingColumns        = []string{"NAME", "AGE"}
	clusterRoleBindingWideColumns    = []string{"ROLE", "USERS", "GROUPS", "SERVICEACCOUNTS"}
	storageClassColumns              = []string{"NAME", "TYPE"}
	statusColumns                    = []string{"STATUS", "REASON", "MESSAGE"}

	// TODO: consider having 'KIND' for third party resource data
	thirdPartyResourceDataColumns    = []string{"NAME", "LABELS", "DATA"}
	horizontalPodAutoscalerColumns   = []string{"NAME", "REFERENCE", "TARGETS", "MINPODS", "MAXPODS", "REPLICAS", "AGE"}
	deploymentColumns                = []string{"NAME", "DESIRED", "CURRENT", "UP-TO-DATE", "AVAILABLE", "AGE"}
	deploymentWideColumns            = []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
	configMapColumns                 = []string{"NAME", "DATA", "AGE"}
	podSecurityPolicyColumns         = []string{"NAME", "PRIV", "CAPS", "SELINUX", "RUNASUSER", "FSGROUP", "SUPGROUP", "READONLYROOTFS", "VOLUMES"}
	clusterColumns                   = []string{"NAME", "STATUS", "AGE"}
	networkPolicyColumns             = []string{"NAME", "POD-SELECTOR", "AGE"}
	certificateSigningRequestColumns = []string{"NAME", "AGE", "REQUESTOR", "CONDITION"}
)

func printPod(pod *api.Pod, w io.Writer, options printers.PrintOptions) error {
	if err := printPodBase(pod, w, options); err != nil {
		return err
	}

	return nil
}

func printPodList(podList *api.PodList, w io.Writer, options printers.PrintOptions) error {
	for _, pod := range podList.Items {
		if err := printPodBase(&pod, w, options); err != nil {
			return err
		}
	}
	return nil
}

// AddHandlers adds print handlers for default Kubernetes types dealing with internal versions.
func AddHandlers(h *printers.HumanReadablePrinter) {
	h.Handler(podColumns, podWideColumns, printPodList)
	h.Handler(podColumns, podWideColumns, printPod)
	h.Handler(podTemplateColumns, nil, printPodTemplate)
	h.Handler(podTemplateColumns, nil, printPodTemplateList)
	h.Handler(podDisruptionBudgetColumns, nil, printPodDisruptionBudget)
	h.Handler(podDisruptionBudgetColumns, nil, printPodDisruptionBudgetList)
	h.Handler(replicationControllerColumns, replicationControllerWideColumns, printReplicationController)
	h.Handler(replicationControllerColumns, replicationControllerWideColumns, printReplicationControllerList)
	h.Handler(replicaSetColumns, replicaSetWideColumns, printReplicaSet)
	h.Handler(replicaSetColumns, replicaSetWideColumns, printReplicaSetList)
	h.Handler(daemonSetColumns, daemonSetWideColumns, printDaemonSet)
	h.Handler(daemonSetColumns, daemonSetWideColumns, printDaemonSetList)
	h.Handler(jobColumns, batchJobWideColumns, printJob)
	h.Handler(jobColumns, batchJobWideColumns, printJobList)
	h.Handler(cronJobColumns, batchJobWideColumns, printCronJob)
	h.Handler(cronJobColumns, batchJobWideColumns, printCronJobList)
	h.Handler(serviceColumns, serviceWideColumns, printService)
	h.Handler(serviceColumns, serviceWideColumns, printServiceList)
	h.Handler(ingressColumns, nil, printIngress)
	h.Handler(ingressColumns, nil, printIngressList)
	h.Handler(statefulSetColumns, nil, printStatefulSet)
	h.Handler(statefulSetColumns, nil, printStatefulSetList)
	h.Handler(endpointColumns, nil, printEndpoints)
	h.Handler(endpointColumns, nil, printEndpointsList)
	h.Handler(nodeColumns, nodeWideColumns, printNode)
	h.Handler(nodeColumns, nodeWideColumns, printNodeList)
	h.Handler(eventColumns, nil, printEvent)
	h.Handler(eventColumns, nil, printEventList)
	h.Handler(limitRangeColumns, nil, printLimitRange)
	h.Handler(limitRangeColumns, nil, printLimitRangeList)
	h.Handler(resourceQuotaColumns, nil, printResourceQuota)
	h.Handler(resourceQuotaColumns, nil, printResourceQuotaList)
	h.Handler(namespaceColumns, nil, printNamespace)
	h.Handler(namespaceColumns, nil, printNamespaceList)
	h.Handler(secretColumns, nil, printSecret)
	h.Handler(secretColumns, nil, printSecretList)
	h.Handler(serviceAccountColumns, nil, printServiceAccount)
	h.Handler(serviceAccountColumns, nil, printServiceAccountList)
	h.Handler(persistentVolumeClaimColumns, nil, printPersistentVolumeClaim)
	h.Handler(persistentVolumeClaimColumns, nil, printPersistentVolumeClaimList)
	h.Handler(persistentVolumeColumns, nil, printPersistentVolume)
	h.Handler(persistentVolumeColumns, nil, printPersistentVolumeList)
	h.Handler(componentStatusColumns, nil, printComponentStatus)
	h.Handler(componentStatusColumns, nil, printComponentStatusList)
	h.Handler(thirdPartyResourceColumns, nil, printThirdPartyResource)
	h.Handler(thirdPartyResourceColumns, nil, printThirdPartyResourceList)
	h.Handler(deploymentColumns, deploymentWideColumns, printDeployment)
	h.Handler(deploymentColumns, deploymentWideColumns, printDeploymentList)
	h.Handler(horizontalPodAutoscalerColumns, nil, printHorizontalPodAutoscaler)
	h.Handler(horizontalPodAutoscalerColumns, nil, printHorizontalPodAutoscalerList)
	h.Handler(configMapColumns, nil, printConfigMap)
	h.Handler(configMapColumns, nil, printConfigMapList)
	h.Handler(podSecurityPolicyColumns, nil, printPodSecurityPolicy)
	h.Handler(podSecurityPolicyColumns, nil, printPodSecurityPolicyList)
	h.Handler(thirdPartyResourceDataColumns, nil, printThirdPartyResourceData)
	h.Handler(thirdPartyResourceDataColumns, nil, printThirdPartyResourceDataList)
	h.Handler(clusterColumns, nil, printCluster)
	h.Handler(clusterColumns, nil, printClusterList)
	h.Handler(networkPolicyColumns, nil, printNetworkPolicy)
	h.Handler(networkPolicyColumns, nil, printNetworkPolicyList)
	h.Handler(roleColumns, nil, printRole)
	h.Handler(roleColumns, nil, printRoleList)
	h.Handler(roleBindingColumns, roleBindingWideColumns, printRoleBinding)
	h.Handler(roleBindingColumns, roleBindingWideColumns, printRoleBindingList)
	h.Handler(clusterRoleColumns, nil, printClusterRole)
	h.Handler(clusterRoleColumns, nil, printClusterRoleList)
	h.Handler(clusterRoleBindingColumns, clusterRoleBindingWideColumns, printClusterRoleBinding)
	h.Handler(clusterRoleBindingColumns, clusterRoleBindingWideColumns, printClusterRoleBindingList)
	h.Handler(certificateSigningRequestColumns, nil, printCertificateSigningRequest)
	h.Handler(certificateSigningRequestColumns, nil, printCertificateSigningRequestList)
	h.Handler(storageClassColumns, nil, printStorageClass)
	h.Handler(storageClassColumns, nil, printStorageClassList)
	h.Handler(statusColumns, nil, printStatus)
}

// formatResourceName receives a resource kind, name, and boolean specifying
// whether or not to update the current name to "kind/name"
func formatResourceName(kind, name string, withKind bool) string {
	if !withKind || kind == "" {
		return name
	}

	return kind + "/" + name
}

// Pass ports=nil for all ports.
func formatEndpoints(endpoints *api.Endpoints, ports sets.String) string {
	if len(endpoints.Subsets) == 0 {
		return "<none>"
	}
	list := []string{}
	max := 3
	more := false
	count := 0
	for i := range endpoints.Subsets {
		ss := &endpoints.Subsets[i]
		for i := range ss.Ports {
			port := &ss.Ports[i]
			if ports == nil || ports.Has(port.Name) {
				for i := range ss.Addresses {
					if len(list) == max {
						more = true
					}
					addr := &ss.Addresses[i]
					if !more {
						list = append(list, fmt.Sprintf("%s:%d", addr.IP, port.Port))
					}
					count++
				}
			}
		}
	}
	ret := strings.Join(list, ",")
	if more {
		return fmt.Sprintf("%s + %d more...", ret, count-max)
	}
	return ret
}

// translateTimestamp returns the elapsed time since timestamp in
// human-readable approximation.
func translateTimestamp(timestamp metav1.Time) string {
	if timestamp.IsZero() {
		return "<unknown>"
	}
	return printers.ShortHumanDuration(time.Now().Sub(timestamp.Time))
}

func printPodBase(pod *api.Pod, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, pod.Name, options.WithKind)
	namespace := pod.Namespace

	restarts := 0
	totalContainers := len(pod.Spec.Containers)
	readyContainers := 0

	reason := string(pod.Status.Phase)
	if pod.Status.Reason != "" {
		reason = pod.Status.Reason
	}

	initializing := false
	for i := range pod.Status.InitContainerStatuses {
		container := pod.Status.InitContainerStatuses[i]
		restarts += int(container.RestartCount)
		switch {
		case container.State.Terminated != nil && container.State.Terminated.ExitCode == 0:
			continue
		case container.State.Terminated != nil:
			// initialization is failed
			if len(container.State.Terminated.Reason) == 0 {
				if container.State.Terminated.Signal != 0 {
					reason = fmt.Sprintf("Init:Signal:%d", container.State.Terminated.Signal)
				} else {
					reason = fmt.Sprintf("Init:ExitCode:%d", container.State.Terminated.ExitCode)
				}
			} else {
				reason = "Init:" + container.State.Terminated.Reason
			}
			initializing = true
		case container.State.Waiting != nil && len(container.State.Waiting.Reason) > 0 && container.State.Waiting.Reason != "PodInitializing":
			reason = "Init:" + container.State.Waiting.Reason
			initializing = true
		default:
			reason = fmt.Sprintf("Init:%d/%d", i, len(pod.Spec.InitContainers))
			initializing = true
		}
		break
	}
	if !initializing {
		restarts = 0
		for i := len(pod.Status.ContainerStatuses) - 1; i >= 0; i-- {
			container := pod.Status.ContainerStatuses[i]

			restarts += int(container.RestartCount)
			if container.State.Waiting != nil && container.State.Waiting.Reason != "" {
				reason = container.State.Waiting.Reason
			} else if container.State.Terminated != nil && container.State.Terminated.Reason != "" {
				reason = container.State.Terminated.Reason
			} else if container.State.Terminated != nil && container.State.Terminated.Reason == "" {
				if container.State.Terminated.Signal != 0 {
					reason = fmt.Sprintf("Signal:%d", container.State.Terminated.Signal)
				} else {
					reason = fmt.Sprintf("ExitCode:%d", container.State.Terminated.ExitCode)
				}
			} else if container.Ready && container.State.Running != nil {
				readyContainers++
			}
		}
	}

	if pod.DeletionTimestamp != nil && pod.Status.Reason == node.NodeUnreachablePodReason {
		reason = "Unknown"
	} else if pod.DeletionTimestamp != nil {
		reason = "Terminating"
	}

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%d/%d\t%s\t%d\t%s",
		name,
		readyContainers,
		totalContainers,
		reason,
		restarts,
		translateTimestamp(pod.CreationTimestamp),
	); err != nil {
		return err
	}

	if options.Wide {
		nodeName := pod.Spec.NodeName
		podIP := pod.Status.PodIP
		if podIP == "" {
			podIP = "<none>"
		}
		if _, err := fmt.Fprintf(w, "\t%s\t%s",
			podIP,
			nodeName,
		); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprint(w, AppendLabels(pod.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, pod.Labels)); err != nil {
		return err
	}

	return nil
}

func printPodTemplate(pod *api.PodTemplate, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, pod.Name, options.WithKind)

	namespace := pod.Namespace

	containers := pod.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s", name); err != nil {
		return err
	}
	if err := layoutContainers(containers, w); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "\t%s", labels.FormatLabels(pod.Template.Labels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(pod.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, pod.Labels)); err != nil {
		return err
	}

	return nil
}

func printPodTemplateList(podList *api.PodTemplateList, w io.Writer, options printers.PrintOptions) error {
	for _, pod := range podList.Items {
		if err := printPodTemplate(&pod, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printPodDisruptionBudget(pdb *policy.PodDisruptionBudget, w io.Writer, options printers.PrintOptions) error {
	// name, minavailable, selector
	name := formatResourceName(options.Kind, pdb.Name, options.WithKind)
	namespace := pdb.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%d\t%s\n",
		name,
		pdb.Spec.MinAvailable.String(),
		pdb.Status.PodDisruptionsAllowed,
		translateTimestamp(pdb.CreationTimestamp),
	); err != nil {
		return err
	}

	return nil
}

func printPodDisruptionBudgetList(pdbList *policy.PodDisruptionBudgetList, w io.Writer, options printers.PrintOptions) error {
	for _, pdb := range pdbList.Items {
		if err := printPodDisruptionBudget(&pdb, w, options); err != nil {
			return err
		}
	}
	return nil
}

// TODO(AdoHe): try to put wide output in a single method
func printReplicationController(controller *api.ReplicationController, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, controller.Name, options.WithKind)

	namespace := controller.Namespace
	containers := controller.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	desiredReplicas := controller.Spec.Replicas
	currentReplicas := controller.Status.Replicas
	readyReplicas := controller.Status.ReadyReplicas
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%d\t%s",
		name,
		desiredReplicas,
		currentReplicas,
		readyReplicas,
		translateTimestamp(controller.CreationTimestamp),
	); err != nil {
		return err
	}

	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", labels.FormatLabels(controller.Spec.Selector)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(controller.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, controller.Labels)); err != nil {
		return err
	}

	return nil
}

func printReplicationControllerList(list *api.ReplicationControllerList, w io.Writer, options printers.PrintOptions) error {
	for _, controller := range list.Items {
		if err := printReplicationController(&controller, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printReplicaSet(rs *extensions.ReplicaSet, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, rs.Name, options.WithKind)

	namespace := rs.Namespace
	containers := rs.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	desiredReplicas := rs.Spec.Replicas
	currentReplicas := rs.Status.Replicas
	readyReplicas := rs.Status.ReadyReplicas
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%d\t%s",
		name,
		desiredReplicas,
		currentReplicas,
		readyReplicas,
		translateTimestamp(rs.CreationTimestamp),
	); err != nil {
		return err
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", metav1.FormatLabelSelector(rs.Spec.Selector)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(rs.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, rs.Labels)); err != nil {
		return err
	}

	return nil
}

func printReplicaSetList(list *extensions.ReplicaSetList, w io.Writer, options printers.PrintOptions) error {
	for _, rs := range list.Items {
		if err := printReplicaSet(&rs, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printCluster(c *federation.Cluster, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, c.Name, options.WithKind)

	var statuses []string
	for _, condition := range c.Status.Conditions {
		if condition.Status == api.ConditionTrue {
			statuses = append(statuses, string(condition.Type))
		} else {
			statuses = append(statuses, "Not"+string(condition.Type))
		}
	}
	if len(statuses) == 0 {
		statuses = append(statuses, "Unknown")
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n",
		name,
		strings.Join(statuses, ","),
		translateTimestamp(c.CreationTimestamp),
	); err != nil {
		return err
	}
	return nil
}
func printClusterList(list *federation.ClusterList, w io.Writer, options printers.PrintOptions) error {
	for _, rs := range list.Items {
		if err := printCluster(&rs, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printJob(job *batch.Job, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, job.Name, options.WithKind)

	namespace := job.Namespace
	containers := job.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	selector, err := metav1.LabelSelectorAsSelector(job.Spec.Selector)
	if err != nil {
		// this shouldn't happen if LabelSelector passed validation
		return err
	}
	if job.Spec.Completions != nil {
		if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%s",
			name,
			*job.Spec.Completions,
			job.Status.Succeeded,
			translateTimestamp(job.CreationTimestamp),
		); err != nil {
			return err
		}
	} else {
		if _, err := fmt.Fprintf(w, "%s\t%s\t%d\t%s",
			name,
			"<none>",
			job.Status.Succeeded,
			translateTimestamp(job.CreationTimestamp),
		); err != nil {
			return err
		}
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", selector.String()); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(job.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, job.Labels)); err != nil {
		return err
	}

	return nil
}

func printJobList(list *batch.JobList, w io.Writer, options printers.PrintOptions) error {
	for _, job := range list.Items {
		if err := printJob(&job, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printCronJob(cronJob *batch.CronJob, w io.Writer, options printers.PrintOptions) error {
	name := cronJob.Name
	namespace := cronJob.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	lastScheduleTime := "<none>"
	if cronJob.Status.LastScheduleTime != nil {
		lastScheduleTime = cronJob.Status.LastScheduleTime.Time.Format(time.RFC1123Z)
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%d\t%s\n",
		name,
		cronJob.Spec.Schedule,
		printBoolPtr(cronJob.Spec.Suspend),
		len(cronJob.Status.Active),
		lastScheduleTime,
	); err != nil {
		return err
	}

	return nil
}

func printCronJobList(list *batch.CronJobList, w io.Writer, options printers.PrintOptions) error {
	for _, cronJob := range list.Items {
		if err := printCronJob(&cronJob, w, options); err != nil {
			return err
		}
	}
	return nil
}

// loadBalancerStatusStringer behaves mostly like a string interface and converts the given status to a string.
// `wide` indicates whether the returned value is meant for --o=wide output. If not, it's clipped to 16 bytes.
func loadBalancerStatusStringer(s api.LoadBalancerStatus, wide bool) string {
	ingress := s.Ingress
	result := []string{}
	for i := range ingress {
		if ingress[i].IP != "" {
			result = append(result, ingress[i].IP)
		} else if ingress[i].Hostname != "" {
			result = append(result, ingress[i].Hostname)
		}
	}
	r := strings.Join(result, ",")
	if !wide && len(r) > loadBalancerWidth {
		r = r[0:(loadBalancerWidth-3)] + "..."
	}
	return r
}

func getServiceExternalIP(svc *api.Service, wide bool) string {
	switch svc.Spec.Type {
	case api.ServiceTypeClusterIP:
		if len(svc.Spec.ExternalIPs) > 0 {
			return strings.Join(svc.Spec.ExternalIPs, ",")
		}
		return "<none>"
	case api.ServiceTypeNodePort:
		if len(svc.Spec.ExternalIPs) > 0 {
			return strings.Join(svc.Spec.ExternalIPs, ",")
		}
		return "<nodes>"
	case api.ServiceTypeLoadBalancer:
		lbIps := loadBalancerStatusStringer(svc.Status.LoadBalancer, wide)
		if len(svc.Spec.ExternalIPs) > 0 {
			result := append(strings.Split(lbIps, ","), svc.Spec.ExternalIPs...)
			return strings.Join(result, ",")
		}
		if len(lbIps) > 0 {
			return lbIps
		}
		return "<pending>"
	case api.ServiceTypeExternalName:
		return svc.Spec.ExternalName
	}
	return "<unknown>"
}

func makePortString(ports []api.ServicePort) string {
	pieces := make([]string, len(ports))
	for ix := range ports {
		port := &ports[ix]
		pieces[ix] = fmt.Sprintf("%d/%s", port.Port, port.Protocol)
		if port.NodePort > 0 {
			pieces[ix] = fmt.Sprintf("%d:%d/%s", port.Port, port.NodePort, port.Protocol)
		}
	}
	return strings.Join(pieces, ",")
}

func printService(svc *api.Service, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, svc.Name, options.WithKind)

	namespace := svc.Namespace

	internalIP := svc.Spec.ClusterIP
	externalIP := getServiceExternalIP(svc, options.Wide)

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s",
		name,
		internalIP,
		externalIP,
		makePortString(svc.Spec.Ports),
		translateTimestamp(svc.CreationTimestamp),
	); err != nil {
		return err
	}
	if options.Wide {
		if _, err := fmt.Fprintf(w, "\t%s", labels.FormatLabels(svc.Spec.Selector)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(svc.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, svc.Labels))
	return err
}

func printServiceList(list *api.ServiceList, w io.Writer, options printers.PrintOptions) error {
	for _, svc := range list.Items {
		if err := printService(&svc, w, options); err != nil {
			return err
		}
	}
	return nil
}

// backendStringer behaves just like a string interface and converts the given backend to a string.
func backendStringer(backend *extensions.IngressBackend) string {
	if backend == nil {
		return ""
	}
	return fmt.Sprintf("%v:%v", backend.ServiceName, backend.ServicePort.String())
}

func formatHosts(rules []extensions.IngressRule) string {
	list := []string{}
	max := 3
	more := false
	for _, rule := range rules {
		if len(list) == max {
			more = true
		}
		if !more && len(rule.Host) != 0 {
			list = append(list, rule.Host)
		}
	}
	if len(list) == 0 {
		return "*"
	}
	ret := strings.Join(list, ",")
	if more {
		return fmt.Sprintf("%s + %d more...", ret, len(rules)-max)
	}
	return ret
}

func formatPorts(tls []extensions.IngressTLS) string {
	if len(tls) != 0 {
		return "80, 443"
	}
	return "80"
}

func printIngress(ingress *extensions.Ingress, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, ingress.Name, options.WithKind)

	namespace := ingress.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(w, "%s\t%v\t%v\t%v\t%s",
		name,
		formatHosts(ingress.Spec.Rules),
		loadBalancerStatusStringer(ingress.Status.LoadBalancer, options.Wide),
		formatPorts(ingress.Spec.TLS),
		translateTimestamp(ingress.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(ingress.Labels, options.ColumnLabels)); err != nil {
		return err
	}

	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, ingress.Labels)); err != nil {
		return err
	}
	return nil
}

func printIngressList(ingressList *extensions.IngressList, w io.Writer, options printers.PrintOptions) error {
	for _, ingress := range ingressList.Items {
		if err := printIngress(&ingress, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printStatefulSet(ps *apps.StatefulSet, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, ps.Name, options.WithKind)

	namespace := ps.Namespace
	containers := ps.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	desiredReplicas := ps.Spec.Replicas
	currentReplicas := ps.Status.Replicas
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%s",
		name,
		desiredReplicas,
		currentReplicas,
		translateTimestamp(ps.CreationTimestamp),
	); err != nil {
		return err
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", metav1.FormatLabelSelector(ps.Spec.Selector)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(ps.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, ps.Labels)); err != nil {
		return err
	}

	return nil
}

func printStatefulSetList(statefulSetList *apps.StatefulSetList, w io.Writer, options printers.PrintOptions) error {
	for _, ps := range statefulSetList.Items {
		if err := printStatefulSet(&ps, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printDaemonSet(ds *extensions.DaemonSet, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, ds.Name, options.WithKind)

	namespace := ds.Namespace

	containers := ds.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	desiredScheduled := ds.Status.DesiredNumberScheduled
	currentScheduled := ds.Status.CurrentNumberScheduled
	numberReady := ds.Status.NumberReady
	numberUpdated := ds.Status.UpdatedNumberScheduled
	numberAvailable := ds.Status.NumberAvailable
	selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		// this shouldn't happen if LabelSelector passed validation
		return err
	}
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s",
		name,
		desiredScheduled,
		currentScheduled,
		numberReady,
		numberUpdated,
		numberAvailable,
		labels.FormatLabels(ds.Spec.Template.Spec.NodeSelector),
		translateTimestamp(ds.CreationTimestamp),
	); err != nil {
		return err
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", selector.String()); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(ds.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, ds.Labels)); err != nil {
		return err
	}

	return nil
}

func printDaemonSetList(list *extensions.DaemonSetList, w io.Writer, options printers.PrintOptions) error {
	for _, ds := range list.Items {
		if err := printDaemonSet(&ds, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printEndpoints(endpoints *api.Endpoints, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, endpoints.Name, options.WithKind)

	namespace := endpoints.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s", name, formatEndpoints(endpoints, nil), translateTimestamp(endpoints.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(endpoints.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, endpoints.Labels))
	return err
}

func printEndpointsList(list *api.EndpointsList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printEndpoints(&item, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printNamespace(item *api.Namespace, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	if options.WithNamespace {
		return fmt.Errorf("namespace is not namespaced")
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s", name, item.Status.Phase, translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(item.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, item.Labels))
	return err
}

func printNamespaceList(list *api.NamespaceList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printNamespace(&item, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printSecret(item *api.Secret, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	namespace := item.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%v\t%s", name, item.Type, len(item.Data), translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(item.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, item.Labels))
	return err
}

func printSecretList(list *api.SecretList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printSecret(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printServiceAccount(item *api.ServiceAccount, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	namespace := item.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%d\t%s", name, len(item.Secrets), translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(item.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, item.Labels))
	return err
}

func printServiceAccountList(list *api.ServiceAccountList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printServiceAccount(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printNode(node *api.Node, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, node.Name, options.WithKind)

	if options.WithNamespace {
		return fmt.Errorf("node is not namespaced")
	}
	conditionMap := make(map[api.NodeConditionType]*api.NodeCondition)
	NodeAllConditions := []api.NodeConditionType{api.NodeReady}
	for i := range node.Status.Conditions {
		cond := node.Status.Conditions[i]
		conditionMap[cond.Type] = &cond
	}
	var status []string
	for _, validCondition := range NodeAllConditions {
		if condition, ok := conditionMap[validCondition]; ok {
			if condition.Status == api.ConditionTrue {
				status = append(status, string(condition.Type))
			} else {
				status = append(status, "Not"+string(condition.Type))
			}
		}
	}
	if len(status) == 0 {
		status = append(status, "Unknown")
	}
	if node.Spec.Unschedulable {
		status = append(status, "SchedulingDisabled")
	}
	role := findNodeRole(node)
	if role != "" {
		status = append(status, role)
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s", name, strings.Join(status, ","), translateTimestamp(node.CreationTimestamp), node.Status.NodeInfo.KubeletVersion); err != nil {
		return err
	}

	if options.Wide {
		osImage, kernelVersion := node.Status.NodeInfo.OSImage, node.Status.NodeInfo.KernelVersion
		if osImage == "" {
			osImage = "<unknown>"
		}
		if kernelVersion == "" {
			kernelVersion = "<unknown>"
		}
		if _, err := fmt.Fprintf(w, "\t%s\t%s\t%s", getNodeExternalIP(node), osImage, kernelVersion); err != nil {
			return err
		}
	}
	// Display caller specify column labels first.
	if _, err := fmt.Fprint(w, AppendLabels(node.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, node.Labels))
	return err
}

// Returns first external ip of the node or "<none>" if none is found.
func getNodeExternalIP(node *api.Node) string {
	for _, address := range node.Status.Addresses {
		if address.Type == api.NodeExternalIP {
			return address.Address
		}
	}

	return "<none>"
}

// findNodeRole returns the role of a given node, or "" if none found.
// The role is determined by looking in order for:
// * a kubernetes.io/role label
// * a kubeadm.alpha.kubernetes.io/role label
// If no role is found, ("", nil) is returned
func findNodeRole(node *api.Node) string {
	if role := node.Labels[metav1.NodeLabelRole]; role != "" {
		return role
	}
	if role := node.Labels[metav1.NodeLabelKubeadmAlphaRole]; role != "" {
		return role
	}
	// No role found
	return ""
}

func printNodeList(list *api.NodeList, w io.Writer, options printers.PrintOptions) error {
	for _, node := range list.Items {
		if err := printNode(&node, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printPersistentVolume(pv *api.PersistentVolume, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, pv.Name, options.WithKind)

	if options.WithNamespace {
		return fmt.Errorf("persistentVolume is not namespaced")
	}

	claimRefUID := ""
	if pv.Spec.ClaimRef != nil {
		claimRefUID += pv.Spec.ClaimRef.Namespace
		claimRefUID += "/"
		claimRefUID += pv.Spec.ClaimRef.Name
	}

	modesStr := api.GetAccessModesAsString(pv.Spec.AccessModes)
	reclaimPolicyStr := string(pv.Spec.PersistentVolumeReclaimPolicy)

	aQty := pv.Spec.Capacity[api.ResourceStorage]
	aSize := aQty.String()

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
		name,
		aSize, modesStr, reclaimPolicyStr,
		pv.Status.Phase,
		claimRefUID,
		api.GetPersistentVolumeClass(pv),
		pv.Status.Reason,
		translateTimestamp(pv.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(pv.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, pv.Labels))
	return err
}

func printPersistentVolumeList(list *api.PersistentVolumeList, w io.Writer, options printers.PrintOptions) error {
	for _, pv := range list.Items {
		if err := printPersistentVolume(&pv, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printPersistentVolumeClaim(pvc *api.PersistentVolumeClaim, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, pvc.Name, options.WithKind)

	namespace := pvc.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	phase := pvc.Status.Phase
	storage := pvc.Spec.Resources.Requests[api.ResourceStorage]
	capacity := ""
	accessModes := ""
	if pvc.Spec.VolumeName != "" {
		accessModes = api.GetAccessModesAsString(pvc.Status.AccessModes)
		storage = pvc.Status.Capacity[api.ResourceStorage]
		capacity = storage.String()
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s", name, phase, pvc.Spec.VolumeName, capacity, accessModes, api.GetPersistentVolumeClaimClass(pvc), translateTimestamp(pvc.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(pvc.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, pvc.Labels))
	return err
}

func printPersistentVolumeClaimList(list *api.PersistentVolumeClaimList, w io.Writer, options printers.PrintOptions) error {
	for _, psd := range list.Items {
		if err := printPersistentVolumeClaim(&psd, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printEvent(event *api.Event, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, event.InvolvedObject.Name, options.WithKind)

	namespace := event.Namespace
	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	// While watching event, we should print absolute time.
	var FirstTimestamp, LastTimestamp string
	if options.AbsoluteTimestamps {
		FirstTimestamp = event.FirstTimestamp.String()
		LastTimestamp = event.LastTimestamp.String()
	} else {
		FirstTimestamp = translateTimestamp(event.FirstTimestamp)
		LastTimestamp = translateTimestamp(event.LastTimestamp)
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
		LastTimestamp,
		FirstTimestamp,
		event.Count,
		name,
		event.InvolvedObject.Kind,
		event.InvolvedObject.FieldPath,
		event.Type,
		event.Reason,
		formatEventSource(event.Source),
		event.Message,
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(event.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, event.Labels))
	return err
}

// Sorts and prints the EventList in a human-friendly format.
func printEventList(list *api.EventList, w io.Writer, options printers.PrintOptions) error {
	sort.Sort(events.SortableEvents(list.Items))
	for i := range list.Items {
		if err := printEvent(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printLimitRange(limitRange *api.LimitRange, w io.Writer, options printers.PrintOptions) error {
	return printObjectMeta(limitRange.ObjectMeta, w, options, true)
}

// Prints the LimitRangeList in a human-friendly format.
func printLimitRangeList(list *api.LimitRangeList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printLimitRange(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

// printObjectMeta prints the object metadata of a given resource.
func printObjectMeta(meta metav1.ObjectMeta, w io.Writer, options printers.PrintOptions, namespaced bool) error {
	name := formatResourceName(options.Kind, meta.Name, options.WithKind)

	if namespaced && options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", meta.Namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s",
		name,
		translateTimestamp(meta.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(meta.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, meta.Labels))
	return err
}

func printResourceQuota(resourceQuota *api.ResourceQuota, w io.Writer, options printers.PrintOptions) error {
	return printObjectMeta(resourceQuota.ObjectMeta, w, options, true)
}

// Prints the ResourceQuotaList in a human-friendly format.
func printResourceQuotaList(list *api.ResourceQuotaList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printResourceQuota(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printRole(role *rbac.Role, w io.Writer, options printers.PrintOptions) error {
	return printObjectMeta(role.ObjectMeta, w, options, true)
}

// Prints the Role in a human-friendly format.
func printRoleList(list *rbac.RoleList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printRole(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printRoleBinding(roleBinding *rbac.RoleBinding, w io.Writer, options printers.PrintOptions) error {
	meta := roleBinding.ObjectMeta
	name := formatResourceName(options.Kind, meta.Name, options.WithKind)

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", meta.Namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s",
		name,
		translateTimestamp(meta.CreationTimestamp),
	); err != nil {
		return err
	}

	if options.Wide {
		roleRef := fmt.Sprintf("%s/%s", roleBinding.RoleRef.Kind, roleBinding.RoleRef.Name)
		users, groups, sas, _ := rbac.SubjectsStrings(roleBinding.Subjects)
		if _, err := fmt.Fprintf(w, "\t%s\t%v\t%v\t%v",
			roleRef,
			strings.Join(users, ", "),
			strings.Join(groups, ", "),
			strings.Join(sas, ", "),
		); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprint(w, AppendLabels(meta.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, meta.Labels))
	return err
}

// Prints the RoleBinding in a human-friendly format.
func printRoleBindingList(list *rbac.RoleBindingList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printRoleBinding(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printClusterRole(clusterRole *rbac.ClusterRole, w io.Writer, options printers.PrintOptions) error {
	if options.WithNamespace {
		return fmt.Errorf("clusterRole is not namespaced")
	}
	return printObjectMeta(clusterRole.ObjectMeta, w, options, false)
}

// Prints the ClusterRole in a human-friendly format.
func printClusterRoleList(list *rbac.ClusterRoleList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printClusterRole(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printClusterRoleBinding(clusterRoleBinding *rbac.ClusterRoleBinding, w io.Writer, options printers.PrintOptions) error {
	meta := clusterRoleBinding.ObjectMeta
	name := formatResourceName(options.Kind, meta.Name, options.WithKind)

	if options.WithNamespace {
		return fmt.Errorf("clusterRoleBinding is not namespaced")
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s",
		name,
		translateTimestamp(meta.CreationTimestamp),
	); err != nil {
		return err
	}

	if options.Wide {
		roleRef := clusterRoleBinding.RoleRef.Name
		users, groups, sas, _ := rbac.SubjectsStrings(clusterRoleBinding.Subjects)
		if _, err := fmt.Fprintf(w, "\t%s\t%v\t%v\t%v",
			roleRef,
			strings.Join(users, ", "),
			strings.Join(groups, ", "),
			strings.Join(sas, ", "),
		); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprint(w, AppendLabels(meta.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, meta.Labels))
	return err
}

// Prints the ClusterRoleBinding in a human-friendly format.
func printClusterRoleBindingList(list *rbac.ClusterRoleBindingList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printClusterRoleBinding(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printCertificateSigningRequest(csr *certificates.CertificateSigningRequest, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, csr.Name, options.WithKind)
	meta := csr.ObjectMeta

	status, err := extractCSRStatus(csr)
	if err != nil {
		return err
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s\t%s\t%s",
		name,
		translateTimestamp(meta.CreationTimestamp),
		csr.Spec.Username,
		status,
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(meta.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err = fmt.Fprint(w, AppendAllLabels(options.ShowLabels, meta.Labels))
	return err
}

func extractCSRStatus(csr *certificates.CertificateSigningRequest) (string, error) {
	var approved, denied bool
	for _, c := range csr.Status.Conditions {
		switch c.Type {
		case certificates.CertificateApproved:
			approved = true
		case certificates.CertificateDenied:
			denied = true
		default:
			return "", fmt.Errorf("unknown csr condition %q", c)
		}
	}
	var status string
	// must be in order of presidence
	if denied {
		status += "Denied"
	} else if approved {
		status += "Approved"
	} else {
		status += "Pending"
	}
	if len(csr.Status.Certificate) > 0 {
		status += ",Issued"
	}
	return status, nil
}

func printCertificateSigningRequestList(list *certificates.CertificateSigningRequestList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printCertificateSigningRequest(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printComponentStatus(item *api.ComponentStatus, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	if options.WithNamespace {
		return fmt.Errorf("componentStatus is not namespaced")
	}
	status := "Unknown"
	message := ""
	error := ""
	for _, condition := range item.Conditions {
		if condition.Type == api.ComponentHealthy {
			if condition.Status == api.ConditionTrue {
				status = "Healthy"
			} else {
				status = "Unhealthy"
			}
			message = condition.Message
			error = condition.Error
			break
		}
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s", name, status, message, error); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(item.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, item.Labels))
	return err
}

func printComponentStatusList(list *api.ComponentStatusList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printComponentStatus(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printThirdPartyResource(rsrc *extensions.ThirdPartyResource, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, rsrc.Name, options.WithKind)

	versions := make([]string, len(rsrc.Versions))
	for ix := range rsrc.Versions {
		version := &rsrc.Versions[ix]
		versions[ix] = fmt.Sprintf("%s", version.Name)
	}
	versionsString := strings.Join(versions, ",")
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n", name, rsrc.Description, versionsString); err != nil {
		return err
	}
	return nil
}

func printThirdPartyResourceList(list *extensions.ThirdPartyResourceList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printThirdPartyResource(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func truncate(str string, maxLen int) string {
	if len(str) > maxLen {
		return str[0:maxLen] + "..."
	}
	return str
}

func printThirdPartyResourceData(rsrc *extensions.ThirdPartyResourceData, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, rsrc.Name, options.WithKind)

	l := labels.FormatLabels(rsrc.Labels)
	truncateCols := 50
	if options.Wide {
		truncateCols = 100
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n", name, l, truncate(string(rsrc.Data), truncateCols)); err != nil {
		return err
	}
	return nil
}

func printThirdPartyResourceDataList(list *extensions.ThirdPartyResourceDataList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printThirdPartyResourceData(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printDeployment(deployment *extensions.Deployment, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, deployment.Name, options.WithKind)

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", deployment.Namespace); err != nil {
			return err
		}
	}

	desiredReplicas := deployment.Spec.Replicas
	currentReplicas := deployment.Status.Replicas
	updatedReplicas := deployment.Status.UpdatedReplicas
	availableReplicas := deployment.Status.AvailableReplicas
	age := translateTimestamp(deployment.CreationTimestamp)
	containers := deployment.Spec.Template.Spec.Containers
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		// this shouldn't happen if LabelSelector passed validation
		return err
	}

	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%d\t%d\t%s", name, desiredReplicas, currentReplicas, updatedReplicas, availableReplicas, age); err != nil {
		return err
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", selector.String()); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprint(w, AppendLabels(deployment.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err = fmt.Fprint(w, AppendAllLabels(options.ShowLabels, deployment.Labels))
	return err
}

func printDeploymentList(list *extensions.DeploymentList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printDeployment(&item, w, options); err != nil {
			return err
		}
	}
	return nil
}

func formatHPAMetrics(specs []autoscaling.MetricSpec, statuses []autoscaling.MetricStatus) string {
	if len(specs) == 0 {
		return "<none>"
	}
	list := []string{}
	max := 2
	more := false
	count := 0
	for i, spec := range specs {
		switch spec.Type {
		case autoscaling.PodsMetricSourceType:
			current := "<unknown>"
			if len(statuses) > i && statuses[i].Pods != nil {
				current = statuses[i].Pods.CurrentAverageValue.String()
			}
			list = append(list, fmt.Sprintf("%s / %s", current, spec.Pods.TargetAverageValue.String()))
		case autoscaling.ObjectMetricSourceType:
			current := "<unknown>"
			if len(statuses) > i && statuses[i].Object != nil {
				current = statuses[i].Object.CurrentValue.String()
			}
			list = append(list, fmt.Sprintf("%s / %s", current, spec.Object.TargetValue.String()))
		case autoscaling.ResourceMetricSourceType:
			if spec.Resource.TargetAverageValue != nil {
				current := "<unknown>"
				if len(statuses) > i && statuses[i].Resource != nil {
					current = statuses[i].Resource.CurrentAverageValue.String()
				}
				list = append(list, fmt.Sprintf("%s / %s", current, spec.Resource.TargetAverageValue.String()))
			} else {
				current := "<unknown>"
				if len(statuses) > i && statuses[i].Resource != nil && statuses[i].Resource.CurrentAverageUtilization != nil {
					current = fmt.Sprintf("%d%%", *statuses[i].Resource.CurrentAverageUtilization)
				}

				target := "<auto>"
				if spec.Resource.TargetAverageUtilization != nil {
					target = fmt.Sprintf("%d%%", *spec.Resource.TargetAverageUtilization)
				}
				list = append(list, fmt.Sprintf("%s / %s", current, target))
			}
		default:
			list = append(list, "<unknown type>")
		}

		count++
	}

	if count > max {
		list = list[:max]
		more = true
	}

	ret := strings.Join(list, ", ")
	if more {
		return fmt.Sprintf("%s + %d more...", ret, count-max)
	}
	return ret
}

func printHorizontalPodAutoscaler(hpa *autoscaling.HorizontalPodAutoscaler, w io.Writer, options printers.PrintOptions) error {
	namespace := hpa.Namespace
	name := formatResourceName(options.Kind, hpa.Name, options.WithKind)

	reference := fmt.Sprintf("%s/%s",
		hpa.Spec.ScaleTargetRef.Kind,
		hpa.Spec.ScaleTargetRef.Name)
	minPods := "<unset>"
	metrics := formatHPAMetrics(hpa.Spec.Metrics, hpa.Status.CurrentMetrics)
	if hpa.Spec.MinReplicas != nil {
		minPods = fmt.Sprintf("%d", *hpa.Spec.MinReplicas)
	}
	maxPods := hpa.Spec.MaxReplicas
	currentReplicas := hpa.Status.CurrentReplicas

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d\t%d\t%s",
		name,
		reference,
		metrics,
		minPods,
		maxPods,
		currentReplicas,
		translateTimestamp(hpa.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(hpa.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, hpa.Labels))
	return err
}

func printHorizontalPodAutoscalerList(list *autoscaling.HorizontalPodAutoscalerList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printHorizontalPodAutoscaler(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printConfigMap(configMap *api.ConfigMap, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, configMap.Name, options.WithKind)

	namespace := configMap.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%v\t%s", name, len(configMap.Data), translateTimestamp(configMap.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(configMap.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, configMap.Labels))
	return err
}

func printConfigMapList(list *api.ConfigMapList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printConfigMap(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printPodSecurityPolicy(item *extensions.PodSecurityPolicy, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	_, err := fmt.Fprintf(w, "%s\t%t\t%v\t%s\t%s\t%s\t%s\t%t\t%v\n", name, item.Spec.Privileged,
		item.Spec.AllowedCapabilities, item.Spec.SELinux.Rule,
		item.Spec.RunAsUser.Rule, item.Spec.FSGroup.Rule, item.Spec.SupplementalGroups.Rule, item.Spec.ReadOnlyRootFilesystem, item.Spec.Volumes)
	return err
}

func printPodSecurityPolicyList(list *extensions.PodSecurityPolicyList, w io.Writer, options printers.PrintOptions) error {
	for _, item := range list.Items {
		if err := printPodSecurityPolicy(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printNetworkPolicy(networkPolicy *extensions.NetworkPolicy, w io.Writer, options printers.PrintOptions) error {
	name := formatResourceName(options.Kind, networkPolicy.Name, options.WithKind)

	namespace := networkPolicy.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%v\t%s", name, metav1.FormatLabelSelector(&networkPolicy.Spec.PodSelector), translateTimestamp(networkPolicy.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(networkPolicy.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, networkPolicy.Labels))
	return err
}

func printNetworkPolicyList(list *extensions.NetworkPolicyList, w io.Writer, options printers.PrintOptions) error {
	for i := range list.Items {
		if err := printNetworkPolicy(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printStorageClass(sc *storage.StorageClass, w io.Writer, options printers.PrintOptions) error {
	name := sc.Name

	if options.WithNamespace {
		return fmt.Errorf("storageclass is not namespaced")
	}

	if storageutil.IsDefaultAnnotation(sc.ObjectMeta) {
		name += " (default)"
	}
	provtype := sc.Provisioner

	if _, err := fmt.Fprintf(w, "%s\t%s\t", name, provtype); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(sc.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, sc.Labels)); err != nil {
		return err
	}

	return nil
}

func printStorageClassList(scList *storage.StorageClassList, w io.Writer, options printers.PrintOptions) error {
	for _, sc := range scList.Items {
		if err := printStorageClass(&sc, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printStatus(status *metav1.Status, w io.Writer, options printers.PrintOptions) error {
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n", status.Status, status.Reason, status.Message); err != nil {
		return err
	}

	return nil
}

func AppendLabels(itemLabels map[string]string, columnLabels []string) string {
	var buffer bytes.Buffer

	for _, cl := range columnLabels {
		buffer.WriteString(fmt.Sprint("\t"))
		if il, ok := itemLabels[cl]; ok {
			buffer.WriteString(fmt.Sprint(il))
		} else {
			buffer.WriteString("<none>")
		}
	}

	return buffer.String()
}

// Append all labels to a single column. We need this even when show-labels flag* is
// false, since this adds newline delimiter to the end of each row.
func AppendAllLabels(showLabels bool, itemLabels map[string]string) string {
	var buffer bytes.Buffer

	if showLabels {
		buffer.WriteString(fmt.Sprint("\t"))
		buffer.WriteString(labels.FormatLabels(itemLabels))
	}
	buffer.WriteString("\n")

	return buffer.String()
}

// Append a set of tabs for each label column.  We need this in the case where
// we have extra lines so that the tabwriter will still line things up.
func AppendLabelTabs(columnLabels []string) string {
	var buffer bytes.Buffer

	for range columnLabels {
		buffer.WriteString("\t")
	}
	buffer.WriteString("\n")

	return buffer.String()
}

// Lay out all the containers on one line if use wide output.
func layoutContainers(containers []api.Container, w io.Writer) error {
	var namesBuffer bytes.Buffer
	var imagesBuffer bytes.Buffer

	for i, container := range containers {
		namesBuffer.WriteString(container.Name)
		imagesBuffer.WriteString(container.Image)
		if i != len(containers)-1 {
			namesBuffer.WriteString(",")
			imagesBuffer.WriteString(",")
		}
	}
	_, err := fmt.Fprintf(w, "\t%s\t%s", namesBuffer.String(), imagesBuffer.String())
	if err != nil {
		return err
	}
	return nil
}

// formatEventSource formats EventSource as a comma separated string excluding Host when empty
func formatEventSource(es api.EventSource) string {
	EventSourceString := []string{es.Component}
	if len(es.Host) > 0 {
		EventSourceString = append(EventSourceString, es.Host)
	}
	return strings.Join(EventSourceString, ", ")
}
