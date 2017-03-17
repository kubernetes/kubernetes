/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/url"
	"reflect"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/federation/apis/federation"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/annotations"
	"k8s.io/kubernetes/pkg/api/events"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	versionedextension "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/storage"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
	versionedclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	coreclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	extensionsclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/printers"
)

// Each level has 2 spaces for PrefixWriter
const (
	LEVEL_0 = iota
	LEVEL_1
	LEVEL_2
	LEVEL_3
)

type PrefixWriter struct {
	out io.Writer
}

func (pw *PrefixWriter) Write(level int, format string, a ...interface{}) {
	levelSpace := "  "
	prefix := ""
	for i := 0; i < level; i++ {
		prefix += levelSpace
	}
	fmt.Fprintf(pw.out, prefix+format, a...)
}

func (pw *PrefixWriter) WriteLine(a ...interface{}) {
	fmt.Fprintln(pw.out, a...)
}

func describerMap(c clientset.Interface) map[schema.GroupKind]printers.Describer {
	m := map[schema.GroupKind]printers.Describer{
		api.Kind("Pod"):                   &PodDescriber{c},
		api.Kind("ReplicationController"): &ReplicationControllerDescriber{c},
		api.Kind("Secret"):                &SecretDescriber{c},
		api.Kind("Service"):               &ServiceDescriber{c},
		api.Kind("ServiceAccount"):        &ServiceAccountDescriber{c},
		api.Kind("Node"):                  &NodeDescriber{c},
		api.Kind("LimitRange"):            &LimitRangeDescriber{c},
		api.Kind("ResourceQuota"):         &ResourceQuotaDescriber{c},
		api.Kind("PersistentVolume"):      &PersistentVolumeDescriber{c},
		api.Kind("PersistentVolumeClaim"): &PersistentVolumeClaimDescriber{c},
		api.Kind("Namespace"):             &NamespaceDescriber{c},
		api.Kind("Endpoints"):             &EndpointsDescriber{c},
		api.Kind("ConfigMap"):             &ConfigMapDescriber{c},

		extensions.Kind("ReplicaSet"):                  &ReplicaSetDescriber{c},
		extensions.Kind("NetworkPolicy"):               &NetworkPolicyDescriber{c},
		autoscaling.Kind("HorizontalPodAutoscaler"):    &HorizontalPodAutoscalerDescriber{c},
		extensions.Kind("DaemonSet"):                   &DaemonSetDescriber{c},
		extensions.Kind("Deployment"):                  &DeploymentDescriber{c, versionedClientsetForDeployment(c)},
		extensions.Kind("Ingress"):                     &IngressDescriber{c},
		batch.Kind("Job"):                              &JobDescriber{c},
		batch.Kind("CronJob"):                          &CronJobDescriber{c},
		apps.Kind("StatefulSet"):                       &StatefulSetDescriber{c},
		apps.Kind("Deployment"):                        &DeploymentDescriber{c, versionedClientsetForDeployment(c)},
		certificates.Kind("CertificateSigningRequest"): &CertificateSigningRequestDescriber{c},
		storage.Kind("StorageClass"):                   &StorageClassDescriber{c},
		policy.Kind("PodDisruptionBudget"):             &PodDisruptionBudgetDescriber{c},
	}

	return m
}

// DescribableResources lists all resource types we can describe.
func DescribableResources() []string {
	keys := make([]string, 0)

	for k := range describerMap(nil) {
		resource := strings.ToLower(k.Kind)
		keys = append(keys, resource)
	}
	return keys
}

// DescriberFor returns the default describe functions for each of the standard
// Kubernetes types.
func DescriberFor(kind schema.GroupKind, c clientset.Interface) (printers.Describer, bool) {
	f, ok := describerMap(c)[kind]
	return f, ok
}

// GenericDescriberFor returns a generic describer for the specified mapping
// that uses only information available from runtime.Unstructured
func GenericDescriberFor(mapping *meta.RESTMapping, dynamic *dynamic.Client, events coreclient.EventsGetter) printers.Describer {
	return &genericDescriber{mapping, dynamic, events}
}

type genericDescriber struct {
	mapping *meta.RESTMapping
	dynamic *dynamic.Client
	events  coreclient.EventsGetter
}

func (g *genericDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (output string, err error) {
	apiResource := &metav1.APIResource{
		Name:       g.mapping.Resource,
		Namespaced: g.mapping.Scope.Name() == meta.RESTScopeNameNamespace,
		Kind:       g.mapping.GroupVersionKind.Kind,
	}
	obj, err := g.dynamic.Resource(apiResource, namespace).Get(name)
	if err != nil {
		return "", err
	}

	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = g.events.Events(namespace).Search(api.Scheme, obj)
	}

	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", obj.GetName())
		w.Write(LEVEL_0, "Namespace:\t%s\n", obj.GetNamespace())
		printLabelsMultiline(w, "Labels", obj.GetLabels())
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

// DefaultObjectDescriber can describe the default Kubernetes objects.
var DefaultObjectDescriber printers.ObjectDescriber

func init() {
	d := &Describers{}
	err := d.Add(
		describeLimitRange,
		describeQuota,
		describePod,
		describeService,
		describeReplicationController,
		describeDaemonSet,
		describeNode,
		describeNamespace,
	)
	if err != nil {
		glog.Fatalf("Cannot register describers: %v", err)
	}
	DefaultObjectDescriber = d
}

// NamespaceDescriber generates information about a namespace
type NamespaceDescriber struct {
	clientset.Interface
}

func (d *NamespaceDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	ns, err := d.Core().Namespaces().Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	resourceQuotaList, err := d.Core().ResourceQuotas(name).List(metav1.ListOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			// Server does not support resource quotas.
			// Not an error, will not show resource quotas information.
			resourceQuotaList = nil
		} else {
			return "", err
		}
	}
	limitRangeList, err := d.Core().LimitRanges(name).List(metav1.ListOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			// Server does not support limit ranges.
			// Not an error, will not show limit ranges information.
			limitRangeList = nil
		} else {
			return "", err
		}
	}
	return describeNamespace(ns, resourceQuotaList, limitRangeList)
}

func describeNamespace(namespace *api.Namespace, resourceQuotaList *api.ResourceQuotaList, limitRangeList *api.LimitRangeList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", namespace.Name)
		printLabelsMultiline(w, "Labels", namespace.Labels)
		printAnnotationsMultiline(w, "Annotations", namespace.Annotations)
		w.Write(LEVEL_0, "Status:\t%s\n", string(namespace.Status.Phase))
		if resourceQuotaList != nil {
			w.Write(LEVEL_0, "\n")
			DescribeResourceQuotas(resourceQuotaList, w)
		}
		if limitRangeList != nil {
			w.Write(LEVEL_0, "\n")
			DescribeLimitRanges(limitRangeList, w)
		}
		return nil
	})
}

func describeLimitRangeSpec(spec api.LimitRangeSpec, prefix string, w *PrefixWriter) {
	for i := range spec.Limits {
		item := spec.Limits[i]
		maxResources := item.Max
		minResources := item.Min
		defaultLimitResources := item.Default
		defaultRequestResources := item.DefaultRequest
		ratio := item.MaxLimitRequestRatio

		set := map[api.ResourceName]bool{}
		for k := range maxResources {
			set[k] = true
		}
		for k := range minResources {
			set[k] = true
		}
		for k := range defaultLimitResources {
			set[k] = true
		}
		for k := range defaultRequestResources {
			set[k] = true
		}
		for k := range ratio {
			set[k] = true
		}

		for k := range set {
			// if no value is set, we output -
			maxValue := "-"
			minValue := "-"
			defaultLimitValue := "-"
			defaultRequestValue := "-"
			ratioValue := "-"

			maxQuantity, maxQuantityFound := maxResources[k]
			if maxQuantityFound {
				maxValue = maxQuantity.String()
			}

			minQuantity, minQuantityFound := minResources[k]
			if minQuantityFound {
				minValue = minQuantity.String()
			}

			defaultLimitQuantity, defaultLimitQuantityFound := defaultLimitResources[k]
			if defaultLimitQuantityFound {
				defaultLimitValue = defaultLimitQuantity.String()
			}

			defaultRequestQuantity, defaultRequestQuantityFound := defaultRequestResources[k]
			if defaultRequestQuantityFound {
				defaultRequestValue = defaultRequestQuantity.String()
			}

			ratioQuantity, ratioQuantityFound := ratio[k]
			if ratioQuantityFound {
				ratioValue = ratioQuantity.String()
			}

			msg := "%s%s\t%v\t%v\t%v\t%v\t%v\t%v\n"
			w.Write(LEVEL_0, msg, prefix, item.Type, k, minValue, maxValue, defaultRequestValue, defaultLimitValue, ratioValue)
		}
	}
}

// DescribeLimitRanges merges a set of limit range items into a single tabular description
func DescribeLimitRanges(limitRanges *api.LimitRangeList, w *PrefixWriter) {
	if len(limitRanges.Items) == 0 {
		w.Write(LEVEL_0, "No resource limits.\n")
		return
	}
	w.Write(LEVEL_0, "Resource Limits\n Type\tResource\tMin\tMax\tDefault Request\tDefault Limit\tMax Limit/Request Ratio\n")
	w.Write(LEVEL_0, " ----\t--------\t---\t---\t---------------\t-------------\t-----------------------\n")
	for _, limitRange := range limitRanges.Items {
		describeLimitRangeSpec(limitRange.Spec, " ", w)
	}
}

// DescribeResourceQuotas merges a set of quota items into a single tabular description of all quotas
func DescribeResourceQuotas(quotas *api.ResourceQuotaList, w *PrefixWriter) {
	if len(quotas.Items) == 0 {
		w.Write(LEVEL_0, "No resource quota.\n")
		return
	}
	sort.Sort(SortableResourceQuotas(quotas.Items))

	w.Write(LEVEL_0, "Resource Quotas")
	for _, q := range quotas.Items {
		w.Write(LEVEL_0, "\n Name:\t%s\n", q.Name)
		if len(q.Spec.Scopes) > 0 {
			scopes := make([]string, 0, len(q.Spec.Scopes))
			for _, scope := range q.Spec.Scopes {
				scopes = append(scopes, string(scope))
			}
			sort.Strings(scopes)
			w.Write(LEVEL_0, " Scopes:\t%s\n", strings.Join(scopes, ", "))
			for _, scope := range scopes {
				helpText := helpTextForResourceQuotaScope(api.ResourceQuotaScope(scope))
				if len(helpText) > 0 {
					w.Write(LEVEL_0, "  * %s\n", helpText)
				}
			}
		}

		w.Write(LEVEL_0, " Resource\tUsed\tHard\n")
		w.Write(LEVEL_0, " --------\t---\t---\n")

		resources := make([]api.ResourceName, 0, len(q.Status.Hard))
		for resource := range q.Status.Hard {
			resources = append(resources, resource)
		}
		sort.Sort(SortableResourceNames(resources))

		for _, resource := range resources {
			hardQuantity := q.Status.Hard[resource]
			usedQuantity := q.Status.Used[resource]
			w.Write(LEVEL_0, " %s\t%s\t%s\n", string(resource), usedQuantity.String(), hardQuantity.String())
		}
	}
}

// LimitRangeDescriber generates information about a limit range
type LimitRangeDescriber struct {
	clientset.Interface
}

func (d *LimitRangeDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	lr := d.Core().LimitRanges(namespace)

	limitRange, err := lr.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return describeLimitRange(limitRange)
}

func describeLimitRange(limitRange *api.LimitRange) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", limitRange.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", limitRange.Namespace)
		w.Write(LEVEL_0, "Type\tResource\tMin\tMax\tDefault Request\tDefault Limit\tMax Limit/Request Ratio\n")
		w.Write(LEVEL_0, "----\t--------\t---\t---\t---------------\t-------------\t-----------------------\n")
		describeLimitRangeSpec(limitRange.Spec, "", w)
		return nil
	})
}

// ResourceQuotaDescriber generates information about a resource quota
type ResourceQuotaDescriber struct {
	clientset.Interface
}

func (d *ResourceQuotaDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	rq := d.Core().ResourceQuotas(namespace)

	resourceQuota, err := rq.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	return describeQuota(resourceQuota)
}

func helpTextForResourceQuotaScope(scope api.ResourceQuotaScope) string {
	switch scope {
	case api.ResourceQuotaScopeTerminating:
		return "Matches all pods that have an active deadline. These pods have a limited lifespan on a node before being actively terminated by the system."
	case api.ResourceQuotaScopeNotTerminating:
		return "Matches all pods that do not have an active deadline. These pods usually include long running pods whose container command is not expected to terminate."
	case api.ResourceQuotaScopeBestEffort:
		return "Matches all pods that do not have resource requirements set. These pods have a best effort quality of service."
	case api.ResourceQuotaScopeNotBestEffort:
		return "Matches all pods that have at least one resource requirement set. These pods have a burstable or guaranteed quality of service."
	default:
		return ""
	}
}
func describeQuota(resourceQuota *api.ResourceQuota) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", resourceQuota.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", resourceQuota.Namespace)
		if len(resourceQuota.Spec.Scopes) > 0 {
			scopes := make([]string, 0, len(resourceQuota.Spec.Scopes))
			for _, scope := range resourceQuota.Spec.Scopes {
				scopes = append(scopes, string(scope))
			}
			sort.Strings(scopes)
			w.Write(LEVEL_0, "Scopes:\t%s\n", strings.Join(scopes, ", "))
			for _, scope := range scopes {
				helpText := helpTextForResourceQuotaScope(api.ResourceQuotaScope(scope))
				if len(helpText) > 0 {
					w.Write(LEVEL_0, " * %s\n", helpText)
				}
			}
		}
		w.Write(LEVEL_0, "Resource\tUsed\tHard\n")
		w.Write(LEVEL_0, "--------\t----\t----\n")

		resources := make([]api.ResourceName, 0, len(resourceQuota.Status.Hard))
		for resource := range resourceQuota.Status.Hard {
			resources = append(resources, resource)
		}
		sort.Sort(SortableResourceNames(resources))

		msg := "%v\t%v\t%v\n"
		for i := range resources {
			resource := resources[i]
			hardQuantity := resourceQuota.Status.Hard[resource]
			usedQuantity := resourceQuota.Status.Used[resource]
			w.Write(LEVEL_0, msg, resource, usedQuantity.String(), hardQuantity.String())
		}
		return nil
	})
}

// PodDescriber generates information about a pod and the replication controllers that
// create it.
type PodDescriber struct {
	clientset.Interface
}

func (d *PodDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	pod, err := d.Core().Pods(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		if describerSettings.ShowEvents {
			eventsInterface := d.Core().Events(namespace)
			selector := eventsInterface.GetFieldSelector(&name, &namespace, nil, nil)
			options := metav1.ListOptions{FieldSelector: selector.String()}
			events, err2 := eventsInterface.List(options)
			if describerSettings.ShowEvents && err2 == nil && len(events.Items) > 0 {
				return tabbedString(func(out io.Writer) error {
					w := &PrefixWriter{out}
					w.Write(LEVEL_0, "Pod '%v': error '%v', but found events.\n", name, err)
					DescribeEvents(events, w)
					return nil
				})
			}
		}
		return "", err
	}

	var events *api.EventList
	if describerSettings.ShowEvents {
		if ref, err := api.GetReference(api.Scheme, pod); err != nil {
			glog.Errorf("Unable to construct reference to '%#v': %v", pod, err)
		} else {
			ref.Kind = ""
			events, _ = d.Core().Events(namespace).Search(api.Scheme, ref)
		}
	}

	return describePod(pod, events)
}

func describePod(pod *api.Pod, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", pod.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", pod.Namespace)
		w.Write(LEVEL_0, "Node:\t%s\n", pod.Spec.NodeName+"/"+pod.Status.HostIP)
		if pod.Status.StartTime != nil {
			w.Write(LEVEL_0, "Start Time:\t%s\n", pod.Status.StartTime.Time.Format(time.RFC1123Z))
		}
		printLabelsMultiline(w, "Labels", pod.Labels)
		printAnnotationsMultiline(w, "Annotations", pod.Annotations)
		if pod.DeletionTimestamp != nil {
			w.Write(LEVEL_0, "Status:\tTerminating (expires %s)\n", pod.DeletionTimestamp.Time.Format(time.RFC1123Z))
			w.Write(LEVEL_0, "Termination Grace Period:\t%ds\n", *pod.DeletionGracePeriodSeconds)
		} else {
			w.Write(LEVEL_0, "Status:\t%s\n", string(pod.Status.Phase))
		}
		if len(pod.Status.Reason) > 0 {
			w.Write(LEVEL_0, "Reason:\t%s\n", pod.Status.Reason)
		}
		if len(pod.Status.Message) > 0 {
			w.Write(LEVEL_0, "Message:\t%s\n", pod.Status.Message)
		}
		w.Write(LEVEL_0, "IP:\t%s\n", pod.Status.PodIP)
		w.Write(LEVEL_0, "Controllers:\t%s\n", printControllers(pod.Annotations))

		if len(pod.Spec.InitContainers) > 0 {
			describeContainers("Init Containers", pod.Spec.InitContainers, pod.Status.InitContainerStatuses, EnvValueRetriever(pod), w, "")
		}
		describeContainers("Containers", pod.Spec.Containers, pod.Status.ContainerStatuses, EnvValueRetriever(pod), w, "")
		if len(pod.Status.Conditions) > 0 {
			w.Write(LEVEL_0, "Conditions:\n  Type\tStatus\n")
			for _, c := range pod.Status.Conditions {
				w.Write(LEVEL_1, "%v \t%v \n",
					c.Type,
					c.Status)
			}
		}
		describeVolumes(pod.Spec.Volumes, w, "")
		if pod.Status.QOSClass != "" {
			w.Write(LEVEL_0, "QoS Class:\t%s\n", pod.Status.QOSClass)
		} else {
			w.Write(LEVEL_0, "QoS Class:\t%s\n", qos.InternalGetPodQOS(pod))
		}
		printLabelsMultiline(w, "Node-Selectors", pod.Spec.NodeSelector)
		printPodTolerationsMultiline(w, "Tolerations", pod.Spec.Tolerations)
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

func printControllers(annotation map[string]string) string {
	value, ok := annotation[api.CreatedByAnnotation]
	if ok {
		var r api.SerializedReference
		err := json.Unmarshal([]byte(value), &r)
		if err == nil {
			return fmt.Sprintf("%s/%s", r.Reference.Kind, r.Reference.Name)
		}
	}
	return "<none>"
}

func describeVolumes(volumes []api.Volume, w *PrefixWriter, space string) {
	if volumes == nil || len(volumes) == 0 {
		w.Write(LEVEL_0, "%sVolumes:\t<none>\n", space)
		return
	}
	w.Write(LEVEL_0, "%sVolumes:\n", space)
	for _, volume := range volumes {
		nameIndent := ""
		if len(space) > 0 {
			nameIndent = " "
		}
		w.Write(LEVEL_1, "%s%v:\n", nameIndent, volume.Name)
		switch {
		case volume.VolumeSource.HostPath != nil:
			printHostPathVolumeSource(volume.VolumeSource.HostPath, w)
		case volume.VolumeSource.EmptyDir != nil:
			printEmptyDirVolumeSource(volume.VolumeSource.EmptyDir, w)
		case volume.VolumeSource.GCEPersistentDisk != nil:
			printGCEPersistentDiskVolumeSource(volume.VolumeSource.GCEPersistentDisk, w)
		case volume.VolumeSource.AWSElasticBlockStore != nil:
			printAWSElasticBlockStoreVolumeSource(volume.VolumeSource.AWSElasticBlockStore, w)
		case volume.VolumeSource.GitRepo != nil:
			printGitRepoVolumeSource(volume.VolumeSource.GitRepo, w)
		case volume.VolumeSource.Secret != nil:
			printSecretVolumeSource(volume.VolumeSource.Secret, w)
		case volume.VolumeSource.ConfigMap != nil:
			printConfigMapVolumeSource(volume.VolumeSource.ConfigMap, w)
		case volume.VolumeSource.NFS != nil:
			printNFSVolumeSource(volume.VolumeSource.NFS, w)
		case volume.VolumeSource.ISCSI != nil:
			printISCSIVolumeSource(volume.VolumeSource.ISCSI, w)
		case volume.VolumeSource.Glusterfs != nil:
			printGlusterfsVolumeSource(volume.VolumeSource.Glusterfs, w)
		case volume.VolumeSource.PersistentVolumeClaim != nil:
			printPersistentVolumeClaimVolumeSource(volume.VolumeSource.PersistentVolumeClaim, w)
		case volume.VolumeSource.RBD != nil:
			printRBDVolumeSource(volume.VolumeSource.RBD, w)
		case volume.VolumeSource.Quobyte != nil:
			printQuobyteVolumeSource(volume.VolumeSource.Quobyte, w)
		case volume.VolumeSource.DownwardAPI != nil:
			printDownwardAPIVolumeSource(volume.VolumeSource.DownwardAPI, w)
		case volume.VolumeSource.AzureDisk != nil:
			printAzureDiskVolumeSource(volume.VolumeSource.AzureDisk, w)
		case volume.VolumeSource.VsphereVolume != nil:
			printVsphereVolumeSource(volume.VolumeSource.VsphereVolume, w)
		case volume.VolumeSource.Cinder != nil:
			printCinderVolumeSource(volume.VolumeSource.Cinder, w)
		case volume.VolumeSource.PhotonPersistentDisk != nil:
			printPhotonPersistentDiskVolumeSource(volume.VolumeSource.PhotonPersistentDisk, w)
		case volume.VolumeSource.PortworxVolume != nil:
			printPortworxVolumeSource(volume.VolumeSource.PortworxVolume, w)
		case volume.VolumeSource.ScaleIO != nil:
			printScaleIOVolumeSource(volume.VolumeSource.ScaleIO, w)
		default:
			w.Write(LEVEL_1, "<unknown>\n")
		}
	}
}

func printHostPathVolumeSource(hostPath *api.HostPathVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tHostPath (bare host directory volume)\n"+
		"    Path:\t%v\n", hostPath.Path)
}

func printEmptyDirVolumeSource(emptyDir *api.EmptyDirVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tEmptyDir (a temporary directory that shares a pod's lifetime)\n"+
		"    Medium:\t%v\n", emptyDir.Medium)
}

func printGCEPersistentDiskVolumeSource(gce *api.GCEPersistentDiskVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tGCEPersistentDisk (a Persistent Disk resource in Google Compute Engine)\n"+
		"    PDName:\t%v\n"+
		"    FSType:\t%v\n"+
		"    Partition:\t%v\n"+
		"    ReadOnly:\t%v\n",
		gce.PDName, gce.FSType, gce.Partition, gce.ReadOnly)
}

func printAWSElasticBlockStoreVolumeSource(aws *api.AWSElasticBlockStoreVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tAWSElasticBlockStore (a Persistent Disk resource in AWS)\n"+
		"    VolumeID:\t%v\n"+
		"    FSType:\t%v\n"+
		"    Partition:\t%v\n"+
		"    ReadOnly:\t%v\n",
		aws.VolumeID, aws.FSType, aws.Partition, aws.ReadOnly)
}

func printGitRepoVolumeSource(git *api.GitRepoVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tGitRepo (a volume that is pulled from git when the pod is created)\n"+
		"    Repository:\t%v\n"+
		"    Revision:\t%v\n",
		git.Repository, git.Revision)
}

func printSecretVolumeSource(secret *api.SecretVolumeSource, w *PrefixWriter) {
	optional := secret.Optional != nil && *secret.Optional
	w.Write(LEVEL_2, "Type:\tSecret (a volume populated by a Secret)\n"+
		"    SecretName:\t%v\n"+
		"    Optional:\t%v\n",
		secret.SecretName, optional)
}

func printConfigMapVolumeSource(configMap *api.ConfigMapVolumeSource, w *PrefixWriter) {
	optional := configMap.Optional != nil && *configMap.Optional
	w.Write(LEVEL_2, "Type:\tConfigMap (a volume populated by a ConfigMap)\n"+
		"    Name:\t%v\n"+
		"    Optional:\t%v\n",
		configMap.Name, optional)
}

func printNFSVolumeSource(nfs *api.NFSVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tNFS (an NFS mount that lasts the lifetime of a pod)\n"+
		"    Server:\t%v\n"+
		"    Path:\t%v\n"+
		"    ReadOnly:\t%v\n",
		nfs.Server, nfs.Path, nfs.ReadOnly)
}

func printQuobyteVolumeSource(quobyte *api.QuobyteVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tQuobyte (a Quobyte mount on the host that shares a pod's lifetime)\n"+
		"    Registry:\t%v\n"+
		"    Volume:\t%v\n"+
		"    ReadOnly:\t%v\n",
		quobyte.Registry, quobyte.Volume, quobyte.ReadOnly)
}

func printPortworxVolumeSource(pwxVolume *api.PortworxVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tPortworxVolume (a Portworx Volume resource)\n"+
		"    VolumeID:\t%v\n",
		pwxVolume.VolumeID)
}

func printISCSIVolumeSource(iscsi *api.ISCSIVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tISCSI (an ISCSI Disk resource that is attached to a kubelet's host machine and then exposed to the pod)\n"+
		"    TargetPortal:\t%v\n"+
		"    IQN:\t%v\n"+
		"    Lun:\t%v\n"+
		"    ISCSIInterface\t%v\n"+
		"    FSType:\t%v\n"+
		"    ReadOnly:\t%v\n",
		iscsi.TargetPortal, iscsi.IQN, iscsi.Lun, iscsi.ISCSIInterface, iscsi.FSType, iscsi.ReadOnly)
}

func printGlusterfsVolumeSource(glusterfs *api.GlusterfsVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tGlusterfs (a Glusterfs mount on the host that shares a pod's lifetime)\n"+
		"    EndpointsName:\t%v\n"+
		"    Path:\t%v\n"+
		"    ReadOnly:\t%v\n",
		glusterfs.EndpointsName, glusterfs.Path, glusterfs.ReadOnly)
}

func printPersistentVolumeClaimVolumeSource(claim *api.PersistentVolumeClaimVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tPersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)\n"+
		"    ClaimName:\t%v\n"+
		"    ReadOnly:\t%v\n",
		claim.ClaimName, claim.ReadOnly)
}

func printRBDVolumeSource(rbd *api.RBDVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tRBD (a Rados Block Device mount on the host that shares a pod's lifetime)\n"+
		"    CephMonitors:\t%v\n"+
		"    RBDImage:\t%v\n"+
		"    FSType:\t%v\n"+
		"    RBDPool:\t%v\n"+
		"    RadosUser:\t%v\n"+
		"    Keyring:\t%v\n"+
		"    SecretRef:\t%v\n"+
		"    ReadOnly:\t%v\n",
		rbd.CephMonitors, rbd.RBDImage, rbd.FSType, rbd.RBDPool, rbd.RadosUser, rbd.Keyring, rbd.SecretRef, rbd.ReadOnly)
}

func printDownwardAPIVolumeSource(d *api.DownwardAPIVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tDownwardAPI (a volume populated by information about the pod)\n    Items:\n")
	for _, mapping := range d.Items {
		if mapping.FieldRef != nil {
			w.Write(LEVEL_3, "%v -> %v\n", mapping.FieldRef.FieldPath, mapping.Path)
		}
		if mapping.ResourceFieldRef != nil {
			w.Write(LEVEL_3, "%v -> %v\n", mapping.ResourceFieldRef.Resource, mapping.Path)
		}
	}
}

func printAzureDiskVolumeSource(d *api.AzureDiskVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tAzureDisk (an Azure Data Disk mount on the host and bind mount to the pod)\n"+
		"    DiskName:\t%v\n"+
		"    DiskURI:\t%v\n"+
		"    FSType:\t%v\n"+
		"    CachingMode:\t%v\n"+
		"    ReadOnly:\t%v\n",
		d.DiskName, d.DataDiskURI, *d.FSType, *d.CachingMode, *d.ReadOnly)
}

func printVsphereVolumeSource(vsphere *api.VsphereVirtualDiskVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tvSphereVolume (a Persistent Disk resource in vSphere)\n"+
		"    VolumePath:\t%v\n"+
		"    FSType:\t%v\n",
		vsphere.VolumePath, vsphere.FSType)
}

func printPhotonPersistentDiskVolumeSource(photon *api.PhotonPersistentDiskVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tPhotonPersistentDisk (a Persistent Disk resource in photon platform)\n"+
		"    PdID:\t%v\n"+
		"    FSType:\t%v\n",
		photon.PdID, photon.FSType)
}

func printCinderVolumeSource(cinder *api.CinderVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tCinder (a Persistent Disk resource in OpenStack)\n"+
		"    VolumeID:\t%v\n"+
		"    FSType:\t%v\n"+
		"    ReadOnly:\t%v\n",
		cinder.VolumeID, cinder.FSType, cinder.ReadOnly)
}

func printScaleIOVolumeSource(sio *api.ScaleIOVolumeSource, w *PrefixWriter) {
	w.Write(LEVEL_2, "Type:\tScaleIO (a persistent volume backed by a block device in ScaleIO)\n"+
		"    Gateway:\t%v\n"+
		"    System:\t%v\n"+
		"    Protection Domain:\t%v\n"+
		"    Storage Pool:\t%v\n"+
		"    Storage Mode:\t%v\n"+
		"    VolumeName:\t%v\n"+
		"    FSType:\t%v\n"+
		"    ReadOnly:\t%v\n",
		sio.Gateway, sio.System, sio.ProtectionDomain, sio.StoragePool, sio.StorageMode, sio.VolumeName, sio.FSType, sio.ReadOnly)
}

type PersistentVolumeDescriber struct {
	clientset.Interface
}

func (d *PersistentVolumeDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := d.Core().PersistentVolumes()

	pv, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	storage := pv.Spec.Capacity[api.ResourceStorage]

	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = d.Core().Events(namespace).Search(api.Scheme, pv)
	}

	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", pv.Name)
		printLabelsMultiline(w, "Labels", pv.Labels)
		printAnnotationsMultiline(w, "Annotations", pv.Annotations)
		w.Write(LEVEL_0, "StorageClass:\t%s\n", api.GetPersistentVolumeClass(pv))
		w.Write(LEVEL_0, "Status:\t%s\n", pv.Status.Phase)
		if pv.Spec.ClaimRef != nil {
			w.Write(LEVEL_0, "Claim:\t%s\n", pv.Spec.ClaimRef.Namespace+"/"+pv.Spec.ClaimRef.Name)
		} else {
			w.Write(LEVEL_0, "Claim:\t%s\n", "")
		}
		w.Write(LEVEL_0, "Reclaim Policy:\t%v\n", pv.Spec.PersistentVolumeReclaimPolicy)
		w.Write(LEVEL_0, "Access Modes:\t%s\n", api.GetAccessModesAsString(pv.Spec.AccessModes))
		w.Write(LEVEL_0, "Capacity:\t%s\n", storage.String())
		w.Write(LEVEL_0, "Message:\t%s\n", pv.Status.Message)
		w.Write(LEVEL_0, "Source:\n")

		switch {
		case pv.Spec.HostPath != nil:
			printHostPathVolumeSource(pv.Spec.HostPath, w)
		case pv.Spec.GCEPersistentDisk != nil:
			printGCEPersistentDiskVolumeSource(pv.Spec.GCEPersistentDisk, w)
		case pv.Spec.AWSElasticBlockStore != nil:
			printAWSElasticBlockStoreVolumeSource(pv.Spec.AWSElasticBlockStore, w)
		case pv.Spec.NFS != nil:
			printNFSVolumeSource(pv.Spec.NFS, w)
		case pv.Spec.ISCSI != nil:
			printISCSIVolumeSource(pv.Spec.ISCSI, w)
		case pv.Spec.Glusterfs != nil:
			printGlusterfsVolumeSource(pv.Spec.Glusterfs, w)
		case pv.Spec.RBD != nil:
			printRBDVolumeSource(pv.Spec.RBD, w)
		case pv.Spec.Quobyte != nil:
			printQuobyteVolumeSource(pv.Spec.Quobyte, w)
		case pv.Spec.VsphereVolume != nil:
			printVsphereVolumeSource(pv.Spec.VsphereVolume, w)
		case pv.Spec.Cinder != nil:
			printCinderVolumeSource(pv.Spec.Cinder, w)
		case pv.Spec.AzureDisk != nil:
			printAzureDiskVolumeSource(pv.Spec.AzureDisk, w)
		case pv.Spec.PhotonPersistentDisk != nil:
			printPhotonPersistentDiskVolumeSource(pv.Spec.PhotonPersistentDisk, w)
		case pv.Spec.PortworxVolume != nil:
			printPortworxVolumeSource(pv.Spec.PortworxVolume, w)
		case pv.Spec.ScaleIO != nil:
			printScaleIOVolumeSource(pv.Spec.ScaleIO, w)
		}

		if events != nil {
			DescribeEvents(events, w)
		}

		return nil
	})
}

type PersistentVolumeClaimDescriber struct {
	clientset.Interface
}

func (d *PersistentVolumeClaimDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := d.Core().PersistentVolumeClaims(namespace)

	pvc, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	storage := pvc.Spec.Resources.Requests[api.ResourceStorage]
	capacity := ""
	accessModes := ""
	if pvc.Spec.VolumeName != "" {
		accessModes = api.GetAccessModesAsString(pvc.Status.AccessModes)
		storage = pvc.Status.Capacity[api.ResourceStorage]
		capacity = storage.String()
	}

	events, _ := d.Core().Events(namespace).Search(api.Scheme, pvc)

	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", pvc.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", pvc.Namespace)
		w.Write(LEVEL_0, "StorageClass:\t%s\n", api.GetPersistentVolumeClaimClass(pvc))
		w.Write(LEVEL_0, "Status:\t%v\n", pvc.Status.Phase)
		w.Write(LEVEL_0, "Volume:\t%s\n", pvc.Spec.VolumeName)
		printLabelsMultiline(w, "Labels", pvc.Labels)
		printAnnotationsMultiline(w, "Annotations", pvc.Annotations)
		w.Write(LEVEL_0, "Capacity:\t%s\n", capacity)
		w.Write(LEVEL_0, "Access Modes:\t%s\n", accessModes)
		if events != nil {
			DescribeEvents(events, w)
		}

		return nil
	})
}

func describeContainers(label string, containers []api.Container, containerStatuses []api.ContainerStatus,
	resolverFn EnvVarResolverFunc, w *PrefixWriter, space string) {
	statuses := map[string]api.ContainerStatus{}
	for _, status := range containerStatuses {
		statuses[status.Name] = status
	}

	describeContainersLabel(containers, label, space, w)

	for _, container := range containers {
		status, ok := statuses[container.Name]
		describeContainerBasicInfo(container, status, ok, space, w)
		describeContainerCommand(container, w)
		if ok {
			describeContainerState(status, w)
		}
		describeContainerResource(container, w)
		describeContainerProbe(container, w)
		if len(container.EnvFrom) > 0 {
			describeContainerEnvFrom(container, resolverFn, w)
		}
		describeContainerEnvVars(container, resolverFn, w)
		describeContainerVolumes(container, w)
	}
}

func describeContainersLabel(containers []api.Container, label, space string, w *PrefixWriter) {
	none := ""
	if len(containers) == 0 {
		none = " <none>"
	}
	w.Write(LEVEL_0, "%s%s:%s\n", space, label, none)
}

func describeContainerBasicInfo(container api.Container, status api.ContainerStatus, ok bool, space string, w *PrefixWriter) {
	nameIndent := ""
	if len(space) > 0 {
		nameIndent = " "
	}
	w.Write(LEVEL_1, "%s%v:\n", nameIndent, container.Name)
	if ok {
		w.Write(LEVEL_2, "Container ID:\t%s\n", status.ContainerID)
	}
	w.Write(LEVEL_2, "Image:\t%s\n", container.Image)
	if ok {
		w.Write(LEVEL_2, "Image ID:\t%s\n", status.ImageID)
	}
	portString := describeContainerPorts(container.Ports)
	if strings.Contains(portString, ",") {
		w.Write(LEVEL_2, "Ports:\t%s\n", portString)
	} else {
		w.Write(LEVEL_2, "Port:\t%s\n", portString)
	}
}

func describeContainerPorts(cPorts []api.ContainerPort) string {
	ports := make([]string, 0, len(cPorts))
	for _, cPort := range cPorts {
		ports = append(ports, fmt.Sprintf("%d/%s", cPort.ContainerPort, cPort.Protocol))
	}
	return strings.Join(ports, ", ")
}

func describeContainerCommand(container api.Container, w *PrefixWriter) {
	if len(container.Command) > 0 {
		w.Write(LEVEL_2, "Command:\n")
		for _, c := range container.Command {
			w.Write(LEVEL_3, "%s\n", c)
		}
	}
	if len(container.Args) > 0 {
		w.Write(LEVEL_2, "Args:\n")
		for _, arg := range container.Args {
			w.Write(LEVEL_3, "%s\n", arg)
		}
	}
}

func describeContainerResource(container api.Container, w *PrefixWriter) {
	resources := container.Resources
	if len(resources.Limits) > 0 {
		w.Write(LEVEL_2, "Limits:\n")
	}
	for _, name := range SortedResourceNames(resources.Limits) {
		quantity := resources.Limits[name]
		w.Write(LEVEL_3, "%s:\t%s\n", name, quantity.String())
	}

	if len(resources.Requests) > 0 {
		w.Write(LEVEL_2, "Requests:\n")
	}
	for _, name := range SortedResourceNames(resources.Requests) {
		quantity := resources.Requests[name]
		w.Write(LEVEL_3, "%s:\t%s\n", name, quantity.String())
	}
}

func describeContainerState(status api.ContainerStatus, w *PrefixWriter) {
	describeStatus("State", status.State, w)
	if status.LastTerminationState.Terminated != nil {
		describeStatus("Last State", status.LastTerminationState, w)
	}
	w.Write(LEVEL_2, "Ready:\t%v\n", printBool(status.Ready))
	w.Write(LEVEL_2, "Restart Count:\t%d\n", status.RestartCount)
}

func describeContainerProbe(container api.Container, w *PrefixWriter) {
	if container.LivenessProbe != nil {
		probe := DescribeProbe(container.LivenessProbe)
		w.Write(LEVEL_2, "Liveness:\t%s\n", probe)
	}
	if container.ReadinessProbe != nil {
		probe := DescribeProbe(container.ReadinessProbe)
		w.Write(LEVEL_2, "Readiness:\t%s\n", probe)
	}
}

func describeContainerVolumes(container api.Container, w *PrefixWriter) {
	none := ""
	if len(container.VolumeMounts) == 0 {
		none = "\t<none>"
	}
	w.Write(LEVEL_2, "Mounts:%s\n", none)
	sort.Sort(SortableVolumeMounts(container.VolumeMounts))
	for _, mount := range container.VolumeMounts {
		flags := []string{}
		switch {
		case mount.ReadOnly:
			flags = append(flags, "ro")
		case !mount.ReadOnly:
			flags = append(flags, "rw")
		case len(mount.SubPath) > 0:
			flags = append(flags, fmt.Sprintf("path=%q", mount.SubPath))
		}
		w.Write(LEVEL_3, "%s from %s (%s)\n", mount.MountPath, mount.Name, strings.Join(flags, ","))
	}
}

func describeContainerEnvVars(container api.Container, resolverFn EnvVarResolverFunc, w *PrefixWriter) {
	none := ""
	if len(container.Env) == 0 {
		none = "\t<none>"
	}
	w.Write(LEVEL_2, "Environment:%s\n", none)

	for _, e := range container.Env {
		if e.ValueFrom == nil {
			w.Write(LEVEL_3, "%s:\t%s\n", e.Name, e.Value)
			continue
		}

		switch {
		case e.ValueFrom.FieldRef != nil:
			var valueFrom string
			if resolverFn != nil {
				valueFrom = resolverFn(e)
			}
			w.Write(LEVEL_3, "%s:\t%s (%s:%s)\n", e.Name, valueFrom, e.ValueFrom.FieldRef.APIVersion, e.ValueFrom.FieldRef.FieldPath)
		case e.ValueFrom.ResourceFieldRef != nil:
			valueFrom, err := fieldpath.InternalExtractContainerResourceValue(e.ValueFrom.ResourceFieldRef, &container)
			if err != nil {
				valueFrom = ""
			}
			resource := e.ValueFrom.ResourceFieldRef.Resource
			if valueFrom == "0" && (resource == "limits.cpu" || resource == "limits.memory") {
				valueFrom = "node allocatable"
			}
			w.Write(LEVEL_3, "%s:\t%s (%s)\n", e.Name, valueFrom, resource)
		case e.ValueFrom.SecretKeyRef != nil:
			optional := e.ValueFrom.SecretKeyRef.Optional != nil && *e.ValueFrom.SecretKeyRef.Optional
			w.Write(LEVEL_3, "%s:\t<set to the key '%s' in secret '%s'>\tOptional: %t\n", e.Name, e.ValueFrom.SecretKeyRef.Key, e.ValueFrom.SecretKeyRef.Name, optional)
		case e.ValueFrom.ConfigMapKeyRef != nil:
			optional := e.ValueFrom.ConfigMapKeyRef.Optional != nil && *e.ValueFrom.ConfigMapKeyRef.Optional
			w.Write(LEVEL_3, "%s:\t<set to the key '%s' of config map '%s'>\tOptional: %t\n", e.Name, e.ValueFrom.ConfigMapKeyRef.Key, e.ValueFrom.ConfigMapKeyRef.Name, optional)
		}
	}
}

func describeContainerEnvFrom(container api.Container, resolverFn EnvVarResolverFunc, w *PrefixWriter) {
	none := ""
	if len(container.EnvFrom) == 0 {
		none = "\t<none>"
	}
	w.Write(LEVEL_2, "Environment Variables from:%s\n", none)

	for _, e := range container.EnvFrom {
		from := ""
		name := ""
		optional := false
		if e.ConfigMapRef != nil {
			from = "ConfigMap"
			name = e.ConfigMapRef.Name
			optional = e.ConfigMapRef.Optional != nil && *e.ConfigMapRef.Optional
		} else if e.SecretRef != nil {
			from = "Secret"
			name = e.SecretRef.Name
			optional = e.SecretRef.Optional != nil && *e.SecretRef.Optional
		}
		if len(e.Prefix) == 0 {
			w.Write(LEVEL_3, "%s\t%s\tOptional: %t\n", name, from, optional)
		} else {
			w.Write(LEVEL_3, "%s\t%s with prefix '%s'\tOptional: %t\n", name, from, e.Prefix, optional)
		}
	}
}

// DescribeProbe is exported for consumers in other API groups that have probes
func DescribeProbe(probe *api.Probe) string {
	attrs := fmt.Sprintf("delay=%ds timeout=%ds period=%ds #success=%d #failure=%d", probe.InitialDelaySeconds, probe.TimeoutSeconds, probe.PeriodSeconds, probe.SuccessThreshold, probe.FailureThreshold)
	switch {
	case probe.Exec != nil:
		return fmt.Sprintf("exec %v %s", probe.Exec.Command, attrs)
	case probe.HTTPGet != nil:
		url := &url.URL{}
		url.Scheme = strings.ToLower(string(probe.HTTPGet.Scheme))
		if len(probe.HTTPGet.Port.String()) > 0 {
			url.Host = net.JoinHostPort(probe.HTTPGet.Host, probe.HTTPGet.Port.String())
		} else {
			url.Host = probe.HTTPGet.Host
		}
		url.Path = probe.HTTPGet.Path
		return fmt.Sprintf("http-get %s %s", url.String(), attrs)
	case probe.TCPSocket != nil:
		return fmt.Sprintf("tcp-socket :%s %s", probe.TCPSocket.Port.String(), attrs)
	}
	return fmt.Sprintf("unknown %s", attrs)
}

type EnvVarResolverFunc func(e api.EnvVar) string

// EnvValueFrom is exported for use by describers in other packages
func EnvValueRetriever(pod *api.Pod) EnvVarResolverFunc {
	return func(e api.EnvVar) string {
		internalFieldPath, _, err := api.Scheme.ConvertFieldLabel(e.ValueFrom.FieldRef.APIVersion, "Pod", e.ValueFrom.FieldRef.FieldPath, "")
		if err != nil {
			return "" // pod validation should catch this on create
		}

		valueFrom, err := fieldpath.ExtractFieldPathAsString(pod, internalFieldPath)
		if err != nil {
			return "" // pod validation should catch this on create
		}

		return valueFrom
	}
}

func describeStatus(stateName string, state api.ContainerState, w *PrefixWriter) {
	switch {
	case state.Running != nil:
		w.Write(LEVEL_2, "%s:\tRunning\n", stateName)
		w.Write(LEVEL_3, "Started:\t%v\n", state.Running.StartedAt.Time.Format(time.RFC1123Z))
	case state.Waiting != nil:
		w.Write(LEVEL_2, "%s:\tWaiting\n", stateName)
		if state.Waiting.Reason != "" {
			w.Write(LEVEL_3, "Reason:\t%s\n", state.Waiting.Reason)
		}
	case state.Terminated != nil:
		w.Write(LEVEL_2, "%s:\tTerminated\n", stateName)
		if state.Terminated.Reason != "" {
			w.Write(LEVEL_3, "Reason:\t%s\n", state.Terminated.Reason)
		}
		if state.Terminated.Message != "" {
			w.Write(LEVEL_3, "Message:\t%s\n", state.Terminated.Message)
		}
		w.Write(LEVEL_3, "Exit Code:\t%d\n", state.Terminated.ExitCode)
		if state.Terminated.Signal > 0 {
			w.Write(LEVEL_3, "Signal:\t%d\n", state.Terminated.Signal)
		}
		w.Write(LEVEL_3, "Started:\t%s\n", state.Terminated.StartedAt.Time.Format(time.RFC1123Z))
		w.Write(LEVEL_3, "Finished:\t%s\n", state.Terminated.FinishedAt.Time.Format(time.RFC1123Z))
	default:
		w.Write(LEVEL_2, "%s:\tWaiting\n", stateName)
	}
}

func describeVolumeClaimTemplates(templates []api.PersistentVolumeClaim, w *PrefixWriter) {
	if len(templates) == 0 {
		w.Write(LEVEL_0, "Volume Claims:\t<none>\n")
		return
	}
	w.Write(LEVEL_0, "Volume Claims:\n")
	for _, pvc := range templates {
		w.Write(LEVEL_1, "Name:\t%s\n", pvc.Name)
		w.Write(LEVEL_1, "StorageClass:\t%s\n", api.GetPersistentVolumeClaimClass(&pvc))
		printLabelsMultilineWithIndent(w, "  ", "Labels", "\t", pvc.Labels, sets.NewString())
		printLabelsMultilineWithIndent(w, "  ", "Annotations", "\t", pvc.Annotations, sets.NewString())
		if capacity, ok := pvc.Spec.Resources.Requests[api.ResourceStorage]; ok {
			w.Write(LEVEL_1, "Capacity:\t%s\n", capacity)
		} else {
			w.Write(LEVEL_1, "Capacity:\t%s\n", "<default>")
		}
		w.Write(LEVEL_1, "Access Modes:\t%s\n", pvc.Spec.AccessModes)
	}
}

func printBoolPtr(value *bool) string {
	if value != nil {
		return printBool(*value)
	}

	return "<unset>"
}

func printBool(value bool) string {
	if value {
		return "True"
	}

	return "False"
}

// ReplicationControllerDescriber generates information about a replication controller
// and the pods it has created.
type ReplicationControllerDescriber struct {
	clientset.Interface
}

func (d *ReplicationControllerDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	rc := d.Core().ReplicationControllers(namespace)
	pc := d.Core().Pods(namespace)

	controller, err := rc.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	running, waiting, succeeded, failed, err := getPodStatusForController(pc, labels.SelectorFromSet(controller.Spec.Selector))
	if err != nil {
		return "", err
	}

	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = d.Core().Events(namespace).Search(api.Scheme, controller)
	}

	return describeReplicationController(controller, events, running, waiting, succeeded, failed)
}

func describeReplicationController(controller *api.ReplicationController, events *api.EventList, running, waiting, succeeded, failed int) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", controller.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", controller.Namespace)
		w.Write(LEVEL_0, "Selector:\t%s\n", labels.FormatLabels(controller.Spec.Selector))
		printLabelsMultiline(w, "Labels", controller.Labels)
		printAnnotationsMultiline(w, "Annotations", controller.Annotations)
		w.Write(LEVEL_0, "Replicas:\t%d current / %d desired\n", controller.Status.Replicas, controller.Spec.Replicas)
		w.Write(LEVEL_0, "Pods Status:\t%d Running / %d Waiting / %d Succeeded / %d Failed\n", running, waiting, succeeded, failed)

		DescribePodTemplate(controller.Spec.Template, out)
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

func DescribePodTemplate(template *api.PodTemplateSpec, out io.Writer) {
	w := &PrefixWriter{out}
	w.Write(LEVEL_0, "Pod Template:\n")
	if template == nil {
		w.Write(LEVEL_1, "<unset>")
		return
	}
	printLabelsMultiline(w, "  Labels", template.Labels)
	if len(template.Annotations) > 0 {
		printAnnotationsMultiline(w, "  Annotations", template.Annotations)
	}
	if len(template.Spec.ServiceAccountName) > 0 {
		w.Write(LEVEL_1, "Service Account:\t%s\n", template.Spec.ServiceAccountName)
	}
	if len(template.Spec.InitContainers) > 0 {
		describeContainers("Init Containers", template.Spec.InitContainers, nil, nil, w, "  ")
	}
	describeContainers("Containers", template.Spec.Containers, nil, nil, w, "  ")
	describeVolumes(template.Spec.Volumes, w, "  ")
}

// ReplicaSetDescriber generates information about a ReplicaSet and the pods it has created.
type ReplicaSetDescriber struct {
	clientset.Interface
}

func (d *ReplicaSetDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	rsc := d.Extensions().ReplicaSets(namespace)
	pc := d.Core().Pods(namespace)

	rs, err := rsc.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
	if err != nil {
		return "", err
	}

	running, waiting, succeeded, failed, getPodErr := getPodStatusForController(pc, selector)

	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = d.Core().Events(namespace).Search(api.Scheme, rs)
	}

	return describeReplicaSet(rs, events, running, waiting, succeeded, failed, getPodErr)
}

func describeReplicaSet(rs *extensions.ReplicaSet, events *api.EventList, running, waiting, succeeded, failed int, getPodErr error) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", rs.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", rs.Namespace)
		w.Write(LEVEL_0, "Selector:\t%s\n", metav1.FormatLabelSelector(rs.Spec.Selector))
		printLabelsMultiline(w, "Labels", rs.Labels)
		printAnnotationsMultiline(w, "Annotations", rs.Annotations)
		w.Write(LEVEL_0, "Replicas:\t%d current / %d desired\n", rs.Status.Replicas, rs.Spec.Replicas)
		w.Write(LEVEL_0, "Pods Status:\t")
		if getPodErr != nil {
			w.Write(LEVEL_0, "error in fetching pods: %s\n", getPodErr)
		} else {
			w.Write(LEVEL_0, "%d Running / %d Waiting / %d Succeeded / %d Failed\n", running, waiting, succeeded, failed)
		}
		DescribePodTemplate(&rs.Spec.Template, out)
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

// JobDescriber generates information about a job and the pods it has created.
type JobDescriber struct {
	clientset.Interface
}

func (d *JobDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	job, err := d.Batch().Jobs(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = d.Core().Events(namespace).Search(api.Scheme, job)
	}

	return describeJob(job, events)
}

func describeJob(job *batch.Job, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", job.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", job.Namespace)
		selector, _ := metav1.LabelSelectorAsSelector(job.Spec.Selector)
		w.Write(LEVEL_0, "Selector:\t%s\n", selector)
		printLabelsMultiline(w, "Labels", job.Labels)
		printAnnotationsMultiline(w, "Annotations", job.Annotations)
		w.Write(LEVEL_0, "Parallelism:\t%d\n", *job.Spec.Parallelism)
		if job.Spec.Completions != nil {
			w.Write(LEVEL_0, "Completions:\t%d\n", *job.Spec.Completions)
		} else {
			w.Write(LEVEL_0, "Completions:\t<unset>\n")
		}
		if job.Status.StartTime != nil {
			w.Write(LEVEL_0, "Start Time:\t%s\n", job.Status.StartTime.Time.Format(time.RFC1123Z))
		}
		if job.Spec.ActiveDeadlineSeconds != nil {
			w.Write(LEVEL_0, "Active Deadline Seconds:\t%ds\n", *job.Spec.ActiveDeadlineSeconds)
		}
		w.Write(LEVEL_0, "Pods Statuses:\t%d Running / %d Succeeded / %d Failed\n", job.Status.Active, job.Status.Succeeded, job.Status.Failed)
		DescribePodTemplate(&job.Spec.Template, out)
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

// CronJobDescriber generates information about a scheduled job and the jobs it has created.
type CronJobDescriber struct {
	clientset.Interface
}

func (d *CronJobDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	scheduledJob, err := d.Batch().CronJobs(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = d.Core().Events(namespace).Search(api.Scheme, scheduledJob)
	}

	return describeCronJob(scheduledJob, events)
}

func describeCronJob(scheduledJob *batch.CronJob, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", scheduledJob.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", scheduledJob.Namespace)
		printLabelsMultiline(w, "Labels", scheduledJob.Labels)
		printAnnotationsMultiline(w, "Annotations", scheduledJob.Annotations)
		w.Write(LEVEL_0, "Schedule:\t%s\n", scheduledJob.Spec.Schedule)
		w.Write(LEVEL_0, "Concurrency Policy:\t%s\n", scheduledJob.Spec.ConcurrencyPolicy)
		w.Write(LEVEL_0, "Suspend:\t%s\n", printBoolPtr(scheduledJob.Spec.Suspend))
		if scheduledJob.Spec.StartingDeadlineSeconds != nil {
			w.Write(LEVEL_0, "Starting Deadline Seconds:\t%ds\n", *scheduledJob.Spec.StartingDeadlineSeconds)
		} else {
			w.Write(LEVEL_0, "Starting Deadline Seconds:\t<unset>\n")
		}
		describeJobTemplate(scheduledJob.Spec.JobTemplate, w)
		if scheduledJob.Status.LastScheduleTime != nil {
			w.Write(LEVEL_0, "Last Schedule Time:\t%s\n", scheduledJob.Status.LastScheduleTime.Time.Format(time.RFC1123Z))
		} else {
			w.Write(LEVEL_0, "Last Schedule Time:\t<unset>\n")
		}
		printActiveJobs(w, "Active Jobs", scheduledJob.Status.Active)
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

func describeJobTemplate(jobTemplate batch.JobTemplateSpec, w *PrefixWriter) {
	if jobTemplate.Spec.Selector != nil {
		selector, _ := metav1.LabelSelectorAsSelector(jobTemplate.Spec.Selector)
		w.Write(LEVEL_0, "Selector:\t%s\n", selector)
	} else {
		w.Write(LEVEL_0, "Selector:\t<unset>\n")
	}
	if jobTemplate.Spec.Parallelism != nil {
		w.Write(LEVEL_0, "Parallelism:\t%d\n", *jobTemplate.Spec.Parallelism)
	} else {
		w.Write(LEVEL_0, "Parallelism:\t<unset>\n")
	}
	if jobTemplate.Spec.Completions != nil {
		w.Write(LEVEL_0, "Completions:\t%d\n", *jobTemplate.Spec.Completions)
	} else {
		w.Write(LEVEL_0, "Completions:\t<unset>\n")
	}
	if jobTemplate.Spec.ActiveDeadlineSeconds != nil {
		w.Write(LEVEL_0, "Active Deadline Seconds:\t%ds\n", *jobTemplate.Spec.ActiveDeadlineSeconds)
	}
	DescribePodTemplate(&jobTemplate.Spec.Template, w.out)
}

func printActiveJobs(w *PrefixWriter, title string, jobs []api.ObjectReference) {
	w.Write(LEVEL_0, "%s:\t", title)
	if len(jobs) == 0 {
		w.WriteLine("<none>")
		return
	}

	for i, job := range jobs {
		if i != 0 {
			w.Write(LEVEL_0, ", ")
		}
		w.Write(LEVEL_0, "%s", job.Name)
	}
	w.WriteLine("")
}

// DaemonSetDescriber generates information about a daemon set and the pods it has created.
type DaemonSetDescriber struct {
	clientset.Interface
}

func (d *DaemonSetDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	dc := d.Extensions().DaemonSets(namespace)
	pc := d.Core().Pods(namespace)

	daemon, err := dc.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	selector, err := metav1.LabelSelectorAsSelector(daemon.Spec.Selector)
	if err != nil {
		return "", err
	}
	running, waiting, succeeded, failed, err := getPodStatusForController(pc, selector)
	if err != nil {
		return "", err
	}

	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = d.Core().Events(namespace).Search(api.Scheme, daemon)
	}

	return describeDaemonSet(daemon, events, running, waiting, succeeded, failed)
}

func describeDaemonSet(daemon *extensions.DaemonSet, events *api.EventList, running, waiting, succeeded, failed int) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", daemon.Name)
		selector, err := metav1.LabelSelectorAsSelector(daemon.Spec.Selector)
		if err != nil {
			// this shouldn't happen if LabelSelector passed validation
			return err
		}
		w.Write(LEVEL_0, "Selector:\t%s\n", selector)
		w.Write(LEVEL_0, "Node-Selector:\t%s\n", labels.FormatLabels(daemon.Spec.Template.Spec.NodeSelector))
		printLabelsMultiline(w, "Labels", daemon.Labels)
		printAnnotationsMultiline(w, "Annotations", daemon.Annotations)
		w.Write(LEVEL_0, "Desired Number of Nodes Scheduled: %d\n", daemon.Status.DesiredNumberScheduled)
		w.Write(LEVEL_0, "Current Number of Nodes Scheduled: %d\n", daemon.Status.CurrentNumberScheduled)
		w.Write(LEVEL_0, "Number of Nodes Scheduled with Up-to-date Pods: %d\n", daemon.Status.UpdatedNumberScheduled)
		w.Write(LEVEL_0, "Number of Nodes Scheduled with Available Pods: %d\n", daemon.Status.NumberAvailable)
		w.Write(LEVEL_0, "Number of Nodes Misscheduled: %d\n", daemon.Status.NumberMisscheduled)
		w.Write(LEVEL_0, "Pods Status:\t%d Running / %d Waiting / %d Succeeded / %d Failed\n", running, waiting, succeeded, failed)
		DescribePodTemplate(&daemon.Spec.Template, out)
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

// SecretDescriber generates information about a secret
type SecretDescriber struct {
	clientset.Interface
}

func (d *SecretDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := d.Core().Secrets(namespace)

	secret, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	return describeSecret(secret)
}

func describeSecret(secret *api.Secret) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", secret.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", secret.Namespace)
		printLabelsMultiline(w, "Labels", secret.Labels)
		skipAnnotations := sets.NewString(annotations.LastAppliedConfigAnnotation)
		printAnnotationsMultilineWithFilter(w, "Annotations", secret.Annotations, skipAnnotations)

		w.Write(LEVEL_0, "\nType:\t%s\n", secret.Type)

		w.Write(LEVEL_0, "\nData\n====\n")
		for k, v := range secret.Data {
			switch {
			case k == api.ServiceAccountTokenKey && secret.Type == api.SecretTypeServiceAccountToken:
				w.Write(LEVEL_0, "%s:\t%s\n", k, string(v))
			default:
				w.Write(LEVEL_0, "%s:\t%d bytes\n", k, len(v))
			}
		}

		return nil
	})
}

type IngressDescriber struct {
	clientset.Interface
}

func (i *IngressDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := i.Extensions().Ingresses(namespace)
	ing, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return i.describeIngress(ing, describerSettings)
}

func (i *IngressDescriber) describeBackend(ns string, backend *extensions.IngressBackend) string {
	endpoints, _ := i.Core().Endpoints(ns).Get(backend.ServiceName, metav1.GetOptions{})
	service, _ := i.Core().Services(ns).Get(backend.ServiceName, metav1.GetOptions{})
	spName := ""
	for i := range service.Spec.Ports {
		sp := &service.Spec.Ports[i]
		switch backend.ServicePort.Type {
		case intstr.String:
			if backend.ServicePort.StrVal == sp.Name {
				spName = sp.Name
			}
		case intstr.Int:
			if int32(backend.ServicePort.IntVal) == sp.Port {
				spName = sp.Name
			}
		}
	}
	return formatEndpoints(endpoints, sets.NewString(spName))
}

func (i *IngressDescriber) describeIngress(ing *extensions.Ingress, describerSettings printers.DescriberSettings) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%v\n", ing.Name)
		w.Write(LEVEL_0, "Namespace:\t%v\n", ing.Namespace)
		w.Write(LEVEL_0, "Address:\t%v\n", loadBalancerStatusStringer(ing.Status.LoadBalancer, true))
		def := ing.Spec.Backend
		ns := ing.Namespace
		if def == nil {
			// Ingresses that don't specify a default backend inherit the
			// default backend in the kube-system namespace.
			def = &extensions.IngressBackend{
				ServiceName: "default-http-backend",
				ServicePort: intstr.IntOrString{Type: intstr.Int, IntVal: 80},
			}
			ns = metav1.NamespaceSystem
		}
		w.Write(LEVEL_0, "Default backend:\t%s (%s)\n", backendStringer(def), i.describeBackend(ns, def))
		if len(ing.Spec.TLS) != 0 {
			describeIngressTLS(w, ing.Spec.TLS)
		}
		w.Write(LEVEL_0, "Rules:\n  Host\tPath\tBackends\n")
		w.Write(LEVEL_1, "----\t----\t--------\n")
		count := 0
		for _, rules := range ing.Spec.Rules {
			if rules.HTTP == nil {
				continue
			}
			count++
			host := rules.Host
			if len(host) == 0 {
				host = "*"
			}
			w.Write(LEVEL_1, "%s\t\n", host)
			for _, path := range rules.HTTP.Paths {
				w.Write(LEVEL_2, "\t%s \t%s (%s)\n", path.Path, backendStringer(&path.Backend), i.describeBackend(ns, &path.Backend))
			}
		}
		if count == 0 {
			w.Write(LEVEL_1, "%s\t%s \t%s (%s)\n", "*", "*", backendStringer(def), i.describeBackend(ns, def))
		}
		describeIngressAnnotations(w, ing.Annotations)

		if describerSettings.ShowEvents {
			events, _ := i.Core().Events(ing.Namespace).Search(api.Scheme, ing)
			if events != nil {
				DescribeEvents(events, w)
			}
		}
		return nil
	})
}

func describeIngressTLS(w *PrefixWriter, ingTLS []extensions.IngressTLS) {
	w.Write(LEVEL_0, "TLS:\n")
	for _, t := range ingTLS {
		if t.SecretName == "" {
			w.Write(LEVEL_1, "SNI routes %v\n", strings.Join(t.Hosts, ","))
		} else {
			w.Write(LEVEL_1, "%v terminates %v\n", t.SecretName, strings.Join(t.Hosts, ","))
		}
	}
	return
}

// TODO: Move from annotations into Ingress status.
func describeIngressAnnotations(w *PrefixWriter, annotations map[string]string) {
	w.Write(LEVEL_0, "Annotations:\n")
	for k, v := range annotations {
		if !strings.HasPrefix(k, "ingress") {
			continue
		}
		parts := strings.Split(k, "/")
		name := parts[len(parts)-1]
		w.Write(LEVEL_1, "%v:\t%s\n", name, v)
	}
	return
}

// ServiceDescriber generates information about a service.
type ServiceDescriber struct {
	clientset.Interface
}

func (d *ServiceDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := d.Core().Services(namespace)

	service, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	endpoints, _ := d.Core().Endpoints(namespace).Get(name, metav1.GetOptions{})
	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = d.Core().Events(namespace).Search(api.Scheme, service)
	}
	return describeService(service, endpoints, events)
}

func buildIngressString(ingress []api.LoadBalancerIngress) string {
	var buffer bytes.Buffer

	for i := range ingress {
		if i != 0 {
			buffer.WriteString(", ")
		}
		if ingress[i].IP != "" {
			buffer.WriteString(ingress[i].IP)
		} else {
			buffer.WriteString(ingress[i].Hostname)
		}
	}
	return buffer.String()
}

func describeService(service *api.Service, endpoints *api.Endpoints, events *api.EventList) (string, error) {
	if endpoints == nil {
		endpoints = &api.Endpoints{}
	}
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", service.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", service.Namespace)
		printLabelsMultiline(w, "Labels", service.Labels)
		printAnnotationsMultiline(w, "Annotations", service.Annotations)
		w.Write(LEVEL_0, "Selector:\t%s\n", labels.FormatLabels(service.Spec.Selector))
		w.Write(LEVEL_0, "Type:\t%s\n", service.Spec.Type)
		w.Write(LEVEL_0, "IP:\t%s\n", service.Spec.ClusterIP)
		if len(service.Spec.ExternalIPs) > 0 {
			w.Write(LEVEL_0, "External IPs:\t%v\n", strings.Join(service.Spec.ExternalIPs, ","))
		}
		if service.Spec.ExternalName != "" {
			w.Write(LEVEL_0, "External Name:\t%s\n", service.Spec.ExternalName)
		}
		if len(service.Status.LoadBalancer.Ingress) > 0 {
			list := buildIngressString(service.Status.LoadBalancer.Ingress)
			w.Write(LEVEL_0, "LoadBalancer Ingress:\t%s\n", list)
		}
		for i := range service.Spec.Ports {
			sp := &service.Spec.Ports[i]

			name := sp.Name
			if name == "" {
				name = "<unset>"
			}
			w.Write(LEVEL_0, "Port:\t%s\t%d/%s\n", name, sp.Port, sp.Protocol)
			if sp.NodePort != 0 {
				w.Write(LEVEL_0, "NodePort:\t%s\t%d/%s\n", name, sp.NodePort, sp.Protocol)
			}
			w.Write(LEVEL_0, "Endpoints:\t%s\n", formatEndpoints(endpoints, sets.NewString(sp.Name)))
		}
		w.Write(LEVEL_0, "Session Affinity:\t%s\n", service.Spec.SessionAffinity)
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

// EndpointsDescriber generates information about an Endpoint.
type EndpointsDescriber struct {
	clientset.Interface
}

func (d *EndpointsDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := d.Core().Endpoints(namespace)

	ep, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	var events *api.EventList
	if describerSettings.ShowEvents {
		events, _ = d.Core().Events(namespace).Search(api.Scheme, ep)
	}

	return describeEndpoints(ep, events)
}

func describeEndpoints(ep *api.Endpoints, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", ep.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", ep.Namespace)
		printLabelsMultiline(w, "Labels", ep.Labels)
		printAnnotationsMultiline(w, "Annotations", ep.Annotations)

		w.Write(LEVEL_0, "Subsets:\n")
		for i := range ep.Subsets {
			subset := &ep.Subsets[i]

			addresses := make([]string, 0, len(subset.Addresses))
			for _, addr := range subset.Addresses {
				addresses = append(addresses, addr.IP)
			}
			addressesString := strings.Join(addresses, ",")
			if len(addressesString) == 0 {
				addressesString = "<none>"
			}
			w.Write(LEVEL_1, "Addresses:\t%s\n", addressesString)

			notReadyAddresses := make([]string, 0, len(subset.NotReadyAddresses))
			for _, addr := range subset.NotReadyAddresses {
				notReadyAddresses = append(notReadyAddresses, addr.IP)
			}
			notReadyAddressesString := strings.Join(notReadyAddresses, ",")
			if len(notReadyAddressesString) == 0 {
				notReadyAddressesString = "<none>"
			}
			w.Write(LEVEL_1, "NotReadyAddresses:\t%s\n", notReadyAddressesString)

			if len(subset.Ports) > 0 {
				w.Write(LEVEL_1, "Ports:\n")
				w.Write(LEVEL_2, "Name\tPort\tProtocol\n")
				w.Write(LEVEL_2, "----\t----\t--------\n")
				for _, port := range subset.Ports {
					name := port.Name
					if len(name) == 0 {
						name = "<unset>"
					}
					w.Write(LEVEL_2, "%s\t%d\t%s\n", name, port.Port, port.Protocol)
				}
			}
			w.Write(LEVEL_0, "\n")
		}

		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

// ServiceAccountDescriber generates information about a service.
type ServiceAccountDescriber struct {
	clientset.Interface
}

func (d *ServiceAccountDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := d.Core().ServiceAccounts(namespace)

	serviceAccount, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	tokens := []api.Secret{}

	// missingSecrets is the set of all secrets present in the
	// serviceAccount but not present in the set of existing secrets.
	missingSecrets := sets.NewString()
	secrets, err := d.Core().Secrets(namespace).List(metav1.ListOptions{})

	// errors are tolerated here in order to describe the serviceAccount with all
	// of the secrets that it references, even if those secrets cannot be fetched.
	if err == nil {
		// existingSecrets is the set of all secrets remaining on a
		// service account that are not present in the "tokens" slice.
		existingSecrets := sets.NewString()

		for _, s := range secrets.Items {
			if s.Type == api.SecretTypeServiceAccountToken {
				name, _ := s.Annotations[api.ServiceAccountNameKey]
				uid, _ := s.Annotations[api.ServiceAccountUIDKey]
				if name == serviceAccount.Name && uid == string(serviceAccount.UID) {
					tokens = append(tokens, s)
				}
			}
			existingSecrets.Insert(s.Name)
		}

		for _, s := range serviceAccount.Secrets {
			if !existingSecrets.Has(s.Name) {
				missingSecrets.Insert(s.Name)
			}
		}
		for _, s := range serviceAccount.ImagePullSecrets {
			if !existingSecrets.Has(s.Name) {
				missingSecrets.Insert(s.Name)
			}
		}
	}

	return describeServiceAccount(serviceAccount, tokens, missingSecrets)
}

func describeServiceAccount(serviceAccount *api.ServiceAccount, tokens []api.Secret, missingSecrets sets.String) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", serviceAccount.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", serviceAccount.Namespace)
		printLabelsMultiline(w, "Labels", serviceAccount.Labels)
		printAnnotationsMultiline(w, "Annotations", serviceAccount.Annotations)
		w.WriteLine()

		var (
			emptyHeader = "                   "
			pullHeader  = "Image pull secrets:"
			mountHeader = "Mountable secrets: "
			tokenHeader = "Tokens:            "

			pullSecretNames  = []string{}
			mountSecretNames = []string{}
			tokenSecretNames = []string{}
		)

		for _, s := range serviceAccount.ImagePullSecrets {
			pullSecretNames = append(pullSecretNames, s.Name)
		}
		for _, s := range serviceAccount.Secrets {
			mountSecretNames = append(mountSecretNames, s.Name)
		}
		for _, s := range tokens {
			tokenSecretNames = append(tokenSecretNames, s.Name)
		}

		types := map[string][]string{
			pullHeader:  pullSecretNames,
			mountHeader: mountSecretNames,
			tokenHeader: tokenSecretNames,
		}
		for header, names := range types {
			if len(names) == 0 {
				w.Write(LEVEL_0, "%s\t<none>\n", header)
			} else {
				prefix := header
				for _, name := range names {
					if missingSecrets.Has(name) {
						w.Write(LEVEL_0, "%s\t%s (not found)\n", prefix, name)
					} else {
						w.Write(LEVEL_0, "%s\t%s\n", prefix, name)
					}
					prefix = emptyHeader
				}
			}
			w.WriteLine()
		}

		return nil
	})
}

// NodeDescriber generates information about a node.
type NodeDescriber struct {
	clientset.Interface
}

func (d *NodeDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	mc := d.Core().Nodes()
	node, err := mc.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	fieldSelector, err := fields.ParseSelector("spec.nodeName=" + name + ",status.phase!=" + string(api.PodSucceeded) + ",status.phase!=" + string(api.PodFailed))
	if err != nil {
		return "", err
	}
	// in a policy aware setting, users may have access to a node, but not all pods
	// in that case, we note that the user does not have access to the pods
	canViewPods := true
	nodeNonTerminatedPodsList, err := d.Core().Pods(namespace).List(metav1.ListOptions{FieldSelector: fieldSelector.String()})
	if err != nil {
		if !errors.IsForbidden(err) {
			return "", err
		}
		canViewPods = false
	}

	var events *api.EventList
	if describerSettings.ShowEvents {
		if ref, err := api.GetReference(api.Scheme, node); err != nil {
			glog.Errorf("Unable to construct reference to '%#v': %v", node, err)
		} else {
			// TODO: We haven't decided the namespace for Node object yet.
			ref.UID = types.UID(ref.Name)
			events, _ = d.Core().Events("").Search(api.Scheme, ref)
		}
	}

	return describeNode(node, nodeNonTerminatedPodsList, events, canViewPods)
}

func describeNode(node *api.Node, nodeNonTerminatedPodsList *api.PodList, events *api.EventList, canViewPods bool) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", node.Name)
		w.Write(LEVEL_0, "Role:\t%s\n", findNodeRole(node))
		printLabelsMultiline(w, "Labels", node.Labels)
		printAnnotationsMultiline(w, "Annotations", node.Annotations)
		printNodeTaintsMultiline(w, "Taints", node.Spec.Taints)
		w.Write(LEVEL_0, "CreationTimestamp:\t%s\n", node.CreationTimestamp.Time.Format(time.RFC1123Z))
		w.Write(LEVEL_0, "Phase:\t%v\n", node.Status.Phase)
		if len(node.Status.Conditions) > 0 {
			w.Write(LEVEL_0, "Conditions:\n  Type\tStatus\tLastHeartbeatTime\tLastTransitionTime\tReason\tMessage\n")
			w.Write(LEVEL_1, "----\t------\t-----------------\t------------------\t------\t-------\n")
			for _, c := range node.Status.Conditions {
				w.Write(LEVEL_1, "%v \t%v \t%s \t%s \t%v \t%v\n",
					c.Type,
					c.Status,
					c.LastHeartbeatTime.Time.Format(time.RFC1123Z),
					c.LastTransitionTime.Time.Format(time.RFC1123Z),
					c.Reason,
					c.Message)
			}
		}
		addresses := make([]string, 0, len(node.Status.Addresses))
		for _, address := range node.Status.Addresses {
			addresses = append(addresses, address.Address)
		}

		printResourceList := func(resourceList api.ResourceList) {
			resources := make([]api.ResourceName, 0, len(resourceList))
			for resource := range resourceList {
				resources = append(resources, resource)
			}
			sort.Sort(SortableResourceNames(resources))
			for _, resource := range resources {
				value := resourceList[resource]
				w.Write(LEVEL_0, " %s:\t%s\n", resource, value.String())
			}
		}

		w.Write(LEVEL_0, "Addresses:\t%s\n", strings.Join(addresses, ","))
		if len(node.Status.Capacity) > 0 {
			w.Write(LEVEL_0, "Capacity:\n")
			printResourceList(node.Status.Capacity)
		}
		if len(node.Status.Allocatable) > 0 {
			w.Write(LEVEL_0, "Allocatable:\n")
			printResourceList(node.Status.Allocatable)
		}

		w.Write(LEVEL_0, "System Info:\n")
		w.Write(LEVEL_0, " Machine ID:\t%s\n", node.Status.NodeInfo.MachineID)
		w.Write(LEVEL_0, " System UUID:\t%s\n", node.Status.NodeInfo.SystemUUID)
		w.Write(LEVEL_0, " Boot ID:\t%s\n", node.Status.NodeInfo.BootID)
		w.Write(LEVEL_0, " Kernel Version:\t%s\n", node.Status.NodeInfo.KernelVersion)
		w.Write(LEVEL_0, " OS Image:\t%s\n", node.Status.NodeInfo.OSImage)
		w.Write(LEVEL_0, " Operating System:\t%s\n", node.Status.NodeInfo.OperatingSystem)
		w.Write(LEVEL_0, " Architecture:\t%s\n", node.Status.NodeInfo.Architecture)
		w.Write(LEVEL_0, " Container Runtime Version:\t%s\n", node.Status.NodeInfo.ContainerRuntimeVersion)
		w.Write(LEVEL_0, " Kubelet Version:\t%s\n", node.Status.NodeInfo.KubeletVersion)
		w.Write(LEVEL_0, " Kube-Proxy Version:\t%s\n", node.Status.NodeInfo.KubeProxyVersion)

		if len(node.Spec.PodCIDR) > 0 {
			w.Write(LEVEL_0, "PodCIDR:\t%s\n", node.Spec.PodCIDR)
		}
		if len(node.Spec.ExternalID) > 0 {
			w.Write(LEVEL_0, "ExternalID:\t%s\n", node.Spec.ExternalID)
		}
		if canViewPods && nodeNonTerminatedPodsList != nil {
			if err := describeNodeResource(nodeNonTerminatedPodsList, node, w); err != nil {
				return err
			}
		} else {
			w.Write(LEVEL_0, "Pods:\tnot authorized\n")
		}
		if events != nil {
			DescribeEvents(events, w)
		}
		return nil
	})
}

type StatefulSetDescriber struct {
	client clientset.Interface
}

func (p *StatefulSetDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	ps, err := p.client.Apps().StatefulSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	pc := p.client.Core().Pods(namespace)

	selector, err := metav1.LabelSelectorAsSelector(ps.Spec.Selector)
	if err != nil {
		return "", err
	}

	running, waiting, succeeded, failed, err := getPodStatusForController(pc, selector)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", ps.ObjectMeta.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", ps.ObjectMeta.Namespace)
		w.Write(LEVEL_0, "CreationTimestamp:\t%s\n", ps.CreationTimestamp.Time.Format(time.RFC1123Z))
		w.Write(LEVEL_0, "Selector:\t%s\n", selector)
		printLabelsMultiline(w, "Labels", ps.Labels)
		printAnnotationsMultiline(w, "Annotations", ps.Annotations)
		w.Write(LEVEL_0, "Replicas:\t%d desired | %d total\n", ps.Spec.Replicas, ps.Status.Replicas)
		w.Write(LEVEL_0, "Pods Status:\t%d Running / %d Waiting / %d Succeeded / %d Failed\n", running, waiting, succeeded, failed)
		DescribePodTemplate(&ps.Spec.Template, out)
		describeVolumeClaimTemplates(ps.Spec.VolumeClaimTemplates, w)
		if describerSettings.ShowEvents {
			events, _ := p.client.Core().Events(namespace).Search(api.Scheme, ps)
			if events != nil {
				DescribeEvents(events, w)
			}
		}
		return nil
	})
}

type CertificateSigningRequestDescriber struct {
	client clientset.Interface
}

func (p *CertificateSigningRequestDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	csr, err := p.client.Certificates().CertificateSigningRequests().Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	cr, err := certificates.ParseCSR(csr)
	if err != nil {
		return "", fmt.Errorf("Error parsing CSR: %v", err)
	}
	status, err := extractCSRStatus(csr)
	if err != nil {
		return "", err
	}

	printListHelper := func(w *PrefixWriter, prefix, name string, values []string) {
		if len(values) == 0 {
			return
		}
		w.Write(LEVEL_0, prefix+name+":\t")
		w.Write(LEVEL_0, strings.Join(values, "\n"+prefix+"\t"))
		w.Write(LEVEL_0, "\n")
	}

	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", csr.Name)
		w.Write(LEVEL_0, "Labels:\t%s\n", labels.FormatLabels(csr.Labels))
		w.Write(LEVEL_0, "Annotations:\t%s\n", labels.FormatLabels(csr.Annotations))
		w.Write(LEVEL_0, "CreationTimestamp:\t%s\n", csr.CreationTimestamp.Time.Format(time.RFC1123Z))
		w.Write(LEVEL_0, "Requesting User:\t%s\n", csr.Spec.Username)
		w.Write(LEVEL_0, "Status:\t%s\n", status)

		w.Write(LEVEL_0, "Subject:\n")
		w.Write(LEVEL_0, "\tCommon Name:\t%s\n", cr.Subject.CommonName)
		w.Write(LEVEL_0, "\tSerial Number:\t%s\n", cr.Subject.SerialNumber)
		printListHelper(w, "\t", "Organization", cr.Subject.Organization)
		printListHelper(w, "\t", "Organizational Unit", cr.Subject.OrganizationalUnit)
		printListHelper(w, "\t", "Country", cr.Subject.Country)
		printListHelper(w, "\t", "Locality", cr.Subject.Locality)
		printListHelper(w, "\t", "Province", cr.Subject.Province)
		printListHelper(w, "\t", "StreetAddress", cr.Subject.StreetAddress)
		printListHelper(w, "\t", "PostalCode", cr.Subject.PostalCode)

		if len(cr.DNSNames)+len(cr.EmailAddresses)+len(cr.IPAddresses) > 0 {
			w.Write(LEVEL_0, "Subject Alternative Names:\n")
			printListHelper(w, "\t", "DNS Names", cr.DNSNames)
			printListHelper(w, "\t", "Email Addresses", cr.EmailAddresses)
			var ipaddrs []string
			for _, ipaddr := range cr.IPAddresses {
				ipaddrs = append(ipaddrs, ipaddr.String())
			}
			printListHelper(w, "\t", "IP Addresses", ipaddrs)
		}

		if describerSettings.ShowEvents {
			events, _ := p.client.Core().Events(namespace).Search(api.Scheme, csr)
			if events != nil {
				DescribeEvents(events, w)
			}
		}
		return nil
	})
}

// HorizontalPodAutoscalerDescriber generates information about a horizontal pod autoscaler.
type HorizontalPodAutoscalerDescriber struct {
	client clientset.Interface
}

func (d *HorizontalPodAutoscalerDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	hpa, err := d.client.Autoscaling().HorizontalPodAutoscalers(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", hpa.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", hpa.Namespace)
		printLabelsMultiline(w, "Labels", hpa.Labels)
		printAnnotationsMultiline(w, "Annotations", hpa.Annotations)
		w.Write(LEVEL_0, "CreationTimestamp:\t%s\n", hpa.CreationTimestamp.Time.Format(time.RFC1123Z))
		w.Write(LEVEL_0, "Reference:\t%s/%s\n",
			hpa.Spec.ScaleTargetRef.Kind,
			hpa.Spec.ScaleTargetRef.Name)
		w.Write(LEVEL_0, "Metrics:\t( current / target )\n")
		for i, metric := range hpa.Spec.Metrics {
			switch metric.Type {
			case autoscaling.PodsMetricSourceType:
				current := "<unknown>"
				if len(hpa.Status.CurrentMetrics) > i && hpa.Status.CurrentMetrics[i].Pods != nil {
					current = hpa.Status.CurrentMetrics[i].Pods.CurrentAverageValue.String()
				}
				w.Write(LEVEL_1, "%q on pods:\t%s / %s\n", metric.Pods.MetricName, current, metric.Pods.TargetAverageValue.String())
			case autoscaling.ObjectMetricSourceType:
				current := "<unknown>"
				if len(hpa.Status.CurrentMetrics) > i && hpa.Status.CurrentMetrics[i].Object != nil {
					current = hpa.Status.CurrentMetrics[i].Object.CurrentValue.String()
				}
				w.Write(LEVEL_1, "%q on %s/%s:\t%s / %s\n", metric.Object.MetricName, metric.Object.Target.Kind, metric.Object.Target.Name, current, metric.Object.TargetValue.String())
			case autoscaling.ResourceMetricSourceType:
				w.Write(LEVEL_1, "resource %s on pods", string(metric.Resource.Name))
				if metric.Resource.TargetAverageValue != nil {
					current := "<unknown>"
					if len(hpa.Status.CurrentMetrics) > i && hpa.Status.CurrentMetrics[i].Resource != nil {
						current = hpa.Status.CurrentMetrics[i].Resource.CurrentAverageValue.String()
					}
					w.Write(LEVEL_0, ":\t%s / %s\n", current, metric.Resource.TargetAverageValue.String())
				} else {
					current := "<unknown>"
					if len(hpa.Status.CurrentMetrics) > i && hpa.Status.CurrentMetrics[i].Resource != nil && hpa.Status.CurrentMetrics[i].Resource.CurrentAverageUtilization != nil {
						current = fmt.Sprintf("%d%% (%s)", *hpa.Status.CurrentMetrics[i].Resource.CurrentAverageUtilization, hpa.Status.CurrentMetrics[i].Resource.CurrentAverageValue.String())
					}

					target := "<auto>"
					if metric.Resource.TargetAverageUtilization != nil {
						target = fmt.Sprintf("%d%%", *metric.Resource.TargetAverageUtilization)
					}
					w.Write(LEVEL_1, "(as a percentage of request):\t%s / %s\n", current, target)
				}
			default:
				w.Write(LEVEL_1, "<unknown metric type %q>", string(metric.Type))
			}
		}
		minReplicas := "<unset>"
		if hpa.Spec.MinReplicas != nil {
			minReplicas = fmt.Sprintf("%d", *hpa.Spec.MinReplicas)
		}
		w.Write(LEVEL_0, "Min replicas:\t%s\n", minReplicas)
		w.Write(LEVEL_0, "Max replicas:\t%d\n", hpa.Spec.MaxReplicas)

		// TODO: switch to scale subresource once the required code is submitted.
		if strings.ToLower(hpa.Spec.ScaleTargetRef.Kind) == "replicationcontroller" {
			w.Write(LEVEL_0, "ReplicationController pods:\t")
			rc, err := d.client.Core().ReplicationControllers(hpa.Namespace).Get(hpa.Spec.ScaleTargetRef.Name, metav1.GetOptions{})
			if err == nil {
				w.Write(LEVEL_0, "%d current / %d desired\n", rc.Status.Replicas, rc.Spec.Replicas)
			} else {
				w.Write(LEVEL_0, "failed to check Replication Controller\n")
			}
		}

		if describerSettings.ShowEvents {
			events, _ := d.client.Core().Events(namespace).Search(api.Scheme, hpa)
			if events != nil {
				DescribeEvents(events, w)
			}
		}
		return nil
	})
}

func describeNodeResource(nodeNonTerminatedPodsList *api.PodList, node *api.Node, w *PrefixWriter) error {
	w.Write(LEVEL_0, "Non-terminated Pods:\t(%d in total)\n", len(nodeNonTerminatedPodsList.Items))
	w.Write(LEVEL_1, "Namespace\tName\t\tCPU Requests\tCPU Limits\tMemory Requests\tMemory Limits\n")
	w.Write(LEVEL_1, "---------\t----\t\t------------\t----------\t---------------\t-------------\n")
	allocatable := node.Status.Capacity
	if len(node.Status.Allocatable) > 0 {
		allocatable = node.Status.Allocatable
	}

	for _, pod := range nodeNonTerminatedPodsList.Items {
		req, limit, err := api.PodRequestsAndLimits(&pod)
		if err != nil {
			return err
		}
		cpuReq, cpuLimit, memoryReq, memoryLimit := req[api.ResourceCPU], limit[api.ResourceCPU], req[api.ResourceMemory], limit[api.ResourceMemory]
		fractionCpuReq := float64(cpuReq.MilliValue()) / float64(allocatable.Cpu().MilliValue()) * 100
		fractionCpuLimit := float64(cpuLimit.MilliValue()) / float64(allocatable.Cpu().MilliValue()) * 100
		fractionMemoryReq := float64(memoryReq.Value()) / float64(allocatable.Memory().Value()) * 100
		fractionMemoryLimit := float64(memoryLimit.Value()) / float64(allocatable.Memory().Value()) * 100
		w.Write(LEVEL_1, "%s\t%s\t\t%s (%d%%)\t%s (%d%%)\t%s (%d%%)\t%s (%d%%)\n", pod.Namespace, pod.Name,
			cpuReq.String(), int64(fractionCpuReq), cpuLimit.String(), int64(fractionCpuLimit),
			memoryReq.String(), int64(fractionMemoryReq), memoryLimit.String(), int64(fractionMemoryLimit))
	}

	w.Write(LEVEL_0, "Allocated resources:\n  (Total limits may be over 100 percent, i.e., overcommitted.)\n  CPU Requests\tCPU Limits\tMemory Requests\tMemory Limits\n")
	w.Write(LEVEL_1, "------------\t----------\t---------------\t-------------\n")
	reqs, limits, err := getPodsTotalRequestsAndLimits(nodeNonTerminatedPodsList)
	if err != nil {
		return err
	}
	cpuReqs, cpuLimits, memoryReqs, memoryLimits := reqs[api.ResourceCPU], limits[api.ResourceCPU], reqs[api.ResourceMemory], limits[api.ResourceMemory]
	fractionCpuReqs := float64(cpuReqs.MilliValue()) / float64(allocatable.Cpu().MilliValue()) * 100
	fractionCpuLimits := float64(cpuLimits.MilliValue()) / float64(allocatable.Cpu().MilliValue()) * 100
	fractionMemoryReqs := float64(memoryReqs.Value()) / float64(allocatable.Memory().Value()) * 100
	fractionMemoryLimits := float64(memoryLimits.Value()) / float64(allocatable.Memory().Value()) * 100
	w.Write(LEVEL_1, "%s (%d%%)\t%s (%d%%)\t%s (%d%%)\t%s (%d%%)\n",
		cpuReqs.String(), int64(fractionCpuReqs), cpuLimits.String(), int64(fractionCpuLimits),
		memoryReqs.String(), int64(fractionMemoryReqs), memoryLimits.String(), int64(fractionMemoryLimits))
	return nil
}

func filterTerminatedPods(pods []*api.Pod) []*api.Pod {
	if len(pods) == 0 {
		return pods
	}
	result := []*api.Pod{}
	for _, pod := range pods {
		if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
			continue
		}
		result = append(result, pod)
	}
	return result
}

func getPodsTotalRequestsAndLimits(podList *api.PodList) (reqs map[api.ResourceName]resource.Quantity, limits map[api.ResourceName]resource.Quantity, err error) {
	reqs, limits = map[api.ResourceName]resource.Quantity{}, map[api.ResourceName]resource.Quantity{}
	for _, pod := range podList.Items {
		podReqs, podLimits, err := api.PodRequestsAndLimits(&pod)
		if err != nil {
			return nil, nil, err
		}
		for podReqName, podReqValue := range podReqs {
			if value, ok := reqs[podReqName]; !ok {
				reqs[podReqName] = *podReqValue.Copy()
			} else {
				value.Add(podReqValue)
				reqs[podReqName] = value
			}
		}
		for podLimitName, podLimitValue := range podLimits {
			if value, ok := limits[podLimitName]; !ok {
				limits[podLimitName] = *podLimitValue.Copy()
			} else {
				value.Add(podLimitValue)
				limits[podLimitName] = value
			}
		}
	}
	return
}

func DescribeEvents(el *api.EventList, w *PrefixWriter) {
	if len(el.Items) == 0 {
		w.Write(LEVEL_0, "Events:\t<none>\n")
		return
	}
	sort.Sort(events.SortableEvents(el.Items))
	w.Write(LEVEL_0, "Events:\n  FirstSeen\tLastSeen\tCount\tFrom\tSubObjectPath\tType\tReason\tMessage\n")
	w.Write(LEVEL_1, "---------\t--------\t-----\t----\t-------------\t--------\t------\t-------\n")
	for _, e := range el.Items {
		w.Write(LEVEL_1, "%s\t%s\t%d\t%v\t%v\t%v\t%v\t%v\n",
			translateTimestamp(e.FirstTimestamp),
			translateTimestamp(e.LastTimestamp),
			e.Count,
			formatEventSource(e.Source),
			e.InvolvedObject.FieldPath,
			e.Type,
			e.Reason,
			e.Message)
	}
}

// DeploymentDescriber generates information about a deployment.
type DeploymentDescriber struct {
	clientset.Interface
	versionedClient versionedclientset.Interface
}

func (dd *DeploymentDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	d, err := dd.versionedClient.Extensions().Deployments(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	selector, err := metav1.LabelSelectorAsSelector(d.Spec.Selector)
	if err != nil {
		return "", err
	}
	internalDeployment := &extensions.Deployment{}
	if err := api.Scheme.Convert(d, internalDeployment, extensions.SchemeGroupVersion); err != nil {
		return "", err
	}
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", d.ObjectMeta.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", d.ObjectMeta.Namespace)
		w.Write(LEVEL_0, "CreationTimestamp:\t%s\n", d.CreationTimestamp.Time.Format(time.RFC1123Z))
		printLabelsMultiline(w, "Labels", d.Labels)
		printAnnotationsMultiline(w, "Annotations", d.Annotations)
		w.Write(LEVEL_0, "Selector:\t%s\n", selector)
		w.Write(LEVEL_0, "Replicas:\t%d desired | %d updated | %d total | %d available | %d unavailable\n", *(d.Spec.Replicas), d.Status.UpdatedReplicas, d.Status.Replicas, d.Status.AvailableReplicas, d.Status.UnavailableReplicas)
		w.Write(LEVEL_0, "StrategyType:\t%s\n", d.Spec.Strategy.Type)
		w.Write(LEVEL_0, "MinReadySeconds:\t%d\n", d.Spec.MinReadySeconds)
		if d.Spec.Strategy.RollingUpdate != nil {
			ru := d.Spec.Strategy.RollingUpdate
			w.Write(LEVEL_0, "RollingUpdateStrategy:\t%s max unavailable, %s max surge\n", ru.MaxUnavailable.String(), ru.MaxSurge.String())
		}
		DescribePodTemplate(&internalDeployment.Spec.Template, out)
		if len(d.Status.Conditions) > 0 {
			w.Write(LEVEL_0, "Conditions:\n  Type\tStatus\tReason\n")
			w.Write(LEVEL_1, "----\t------\t------\n")
			for _, c := range d.Status.Conditions {
				w.Write(LEVEL_1, "%v \t%v\t%v\n", c.Type, c.Status, c.Reason)
			}
		}
		oldRSs, _, newRS, err := deploymentutil.GetAllReplicaSets(d, dd.versionedClient)
		if err == nil {
			w.Write(LEVEL_0, "OldReplicaSets:\t%s\n", printReplicaSetsByLabels(oldRSs))
			var newRSs []*versionedextension.ReplicaSet
			if newRS != nil {
				newRSs = append(newRSs, newRS)
			}
			w.Write(LEVEL_0, "NewReplicaSet:\t%s\n", printReplicaSetsByLabels(newRSs))
		}
		overlapWith := d.Annotations[deploymentutil.OverlapAnnotation]
		if len(overlapWith) > 0 {
			w.Write(LEVEL_0, "!!!WARNING!!! This deployment has overlapping label selector with deployment %q and won't behave as expected. Please fix it before continuing.\n", overlapWith)
		}
		if describerSettings.ShowEvents {
			events, err := dd.Core().Events(namespace).Search(api.Scheme, d)
			if err == nil && events != nil {
				DescribeEvents(events, w)
			}
		}
		return nil
	})
}

// Get all daemon set whose selectors would match a given set of labels.
// TODO: Move this to pkg/client and ideally implement it server-side (instead
// of getting all DS's and searching through them manually).
// TODO: write an interface for controllers and fuse getReplicationControllersForLabels
// and getDaemonSetsForLabels.
func getDaemonSetsForLabels(c extensionsclient.DaemonSetInterface, labelsToMatch labels.Labels) ([]extensions.DaemonSet, error) {
	// Get all daemon sets
	// TODO: this needs a namespace scope as argument
	dss, err := c.List(metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("error getting daemon set: %v", err)
	}

	// Find the ones that match labelsToMatch.
	var matchingDaemonSets []extensions.DaemonSet
	for _, ds := range dss.Items {
		selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
		if err != nil {
			// this should never happen if the DaemonSet passed validation
			return nil, err
		}
		if selector.Matches(labelsToMatch) {
			matchingDaemonSets = append(matchingDaemonSets, ds)
		}
	}
	return matchingDaemonSets, nil
}

func printReplicationControllersByLabels(matchingRCs []*api.ReplicationController) string {
	// Format the matching RC's into strings.
	rcStrings := make([]string, 0, len(matchingRCs))
	for _, controller := range matchingRCs {
		rcStrings = append(rcStrings, fmt.Sprintf("%s (%d/%d replicas created)", controller.Name, controller.Status.Replicas, controller.Spec.Replicas))
	}

	list := strings.Join(rcStrings, ", ")
	if list == "" {
		return "<none>"
	}
	return list
}

func printReplicaSetsByLabels(matchingRSs []*versionedextension.ReplicaSet) string {
	// Format the matching ReplicaSets into strings.
	rsStrings := make([]string, 0, len(matchingRSs))
	for _, rs := range matchingRSs {
		rsStrings = append(rsStrings, fmt.Sprintf("%s (%d/%d replicas created)", rs.Name, rs.Status.Replicas, *rs.Spec.Replicas))
	}

	list := strings.Join(rsStrings, ", ")
	if list == "" {
		return "<none>"
	}
	return list
}

func getPodStatusForController(c coreclient.PodInterface, selector labels.Selector) (running, waiting, succeeded, failed int, err error) {
	options := metav1.ListOptions{LabelSelector: selector.String()}
	rcPods, err := c.List(options)
	if err != nil {
		return
	}
	for _, pod := range rcPods.Items {
		switch pod.Status.Phase {
		case api.PodRunning:
			running++
		case api.PodPending:
			waiting++
		case api.PodSucceeded:
			succeeded++
		case api.PodFailed:
			failed++
		}
	}
	return
}

// ConfigMapDescriber generates information about a ConfigMap
type ConfigMapDescriber struct {
	clientset.Interface
}

func (d *ConfigMapDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := d.Core().ConfigMaps(namespace)

	configMap, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	return describeConfigMap(configMap)
}

func describeConfigMap(configMap *api.ConfigMap) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", configMap.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", configMap.Namespace)
		printLabelsMultiline(w, "Labels", configMap.Labels)
		printAnnotationsMultiline(w, "Annotations", configMap.Annotations)

		w.Write(LEVEL_0, "\nData\n====\n")
		for k, v := range configMap.Data {
			w.Write(LEVEL_0, "%s:\n----\n", k)
			w.Write(LEVEL_0, "%s\n", string(v))
		}

		return nil
	})
}

type ClusterDescriber struct {
	fedclientset.Interface
}

func (d *ClusterDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	cluster, err := d.Federation().Clusters().Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return describeCluster(cluster)
}

func describeCluster(cluster *federation.Cluster) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", cluster.Name)
		w.Write(LEVEL_0, "Labels:\t%s\n", labels.FormatLabels(cluster.Labels))

		w.Write(LEVEL_0, "ServerAddressByClientCIDRs:\n  ClientCIDR\tServerAddress\n")
		w.Write(LEVEL_1, "----\t----\n")
		for _, cidrAddr := range cluster.Spec.ServerAddressByClientCIDRs {
			w.Write(LEVEL_1, "%v \t%v\n\n", cidrAddr.ClientCIDR, cidrAddr.ServerAddress)
		}

		if len(cluster.Status.Conditions) > 0 {
			w.Write(LEVEL_0, "Conditions:\n  Type\tStatus\tLastUpdateTime\tLastTransitionTime\tReason\tMessage\n")
			w.Write(LEVEL_1, "----\t------\t-----------------\t------------------\t------\t-------\n")
			for _, c := range cluster.Status.Conditions {
				w.Write(LEVEL_1, "%v \t%v \t%s \t%s \t%v \t%v\n",
					c.Type,
					c.Status,
					c.LastProbeTime.Time.Format(time.RFC1123Z),
					c.LastTransitionTime.Time.Format(time.RFC1123Z),
					c.Reason,
					c.Message)
			}
		}
		return nil
	})
}

// NetworkPolicyDescriber generates information about a NetworkPolicy
type NetworkPolicyDescriber struct {
	clientset.Interface
}

func (d *NetworkPolicyDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	c := d.Extensions().NetworkPolicies(namespace)

	networkPolicy, err := c.Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	return describeNetworkPolicy(networkPolicy)
}

func describeNetworkPolicy(networkPolicy *extensions.NetworkPolicy) (string, error) {
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", networkPolicy.Name)
		w.Write(LEVEL_0, "Namespace:\t%s\n", networkPolicy.Namespace)
		printLabelsMultiline(w, "Labels", networkPolicy.Labels)
		printAnnotationsMultiline(w, "Annotations", networkPolicy.Annotations)

		return nil
	})
}

type StorageClassDescriber struct {
	clientset.Interface
}

func (s *StorageClassDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	sc, err := s.Storage().StorageClasses().Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", sc.Name)
		w.Write(LEVEL_0, "IsDefaultClass:\t%s\n", storageutil.IsDefaultAnnotationText(sc.ObjectMeta))
		w.Write(LEVEL_0, "Annotations:\t%s\n", labels.FormatLabels(sc.Annotations))
		w.Write(LEVEL_0, "Provisioner:\t%s\n", sc.Provisioner)
		w.Write(LEVEL_0, "Parameters:\t%s\n", labels.FormatLabels(sc.Parameters))
		if describerSettings.ShowEvents {
			events, err := s.Core().Events(namespace).Search(api.Scheme, sc)
			if err != nil {
				return err
			}
			if events != nil {
				DescribeEvents(events, w)
			}
		}
		return nil
	})
}

type PodDisruptionBudgetDescriber struct {
	clientset.Interface
}

func (p *PodDisruptionBudgetDescriber) Describe(namespace, name string, describerSettings printers.DescriberSettings) (string, error) {
	pdb, err := p.Policy().PodDisruptionBudgets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return tabbedString(func(out io.Writer) error {
		w := &PrefixWriter{out}
		w.Write(LEVEL_0, "Name:\t%s\n", pdb.Name)
		w.Write(LEVEL_0, "Min available:\t%s\n", pdb.Spec.MinAvailable.String())
		if pdb.Spec.Selector != nil {
			w.Write(LEVEL_0, "Selector:\t%s\n", metav1.FormatLabelSelector(pdb.Spec.Selector))
		} else {
			w.Write(LEVEL_0, "Selector:\t<unset>\n")
		}
		w.Write(LEVEL_0, "Status:\n")
		w.Write(LEVEL_2, "Allowed disruptions:\t%d\n", pdb.Status.PodDisruptionsAllowed)
		w.Write(LEVEL_2, "Current:\t%d\n", pdb.Status.CurrentHealthy)
		w.Write(LEVEL_2, "Desired:\t%d\n", pdb.Status.DesiredHealthy)
		w.Write(LEVEL_2, "Total:\t%d\n", pdb.Status.ExpectedPods)
		if describerSettings.ShowEvents {
			events, err := p.Core().Events(namespace).Search(api.Scheme, pdb)
			if err != nil {
				return err
			}
			if events != nil {
				DescribeEvents(events, w)
			}
		}
		return nil
	})
}

// newErrNoDescriber creates a new ErrNoDescriber with the names of the provided types.
func newErrNoDescriber(types ...reflect.Type) error {
	names := make([]string, 0, len(types))
	for _, t := range types {
		names = append(names, t.String())
	}
	return printers.ErrNoDescriber{Types: names}
}

// Describers implements ObjectDescriber against functions registered via Add. Those functions can
// be strongly typed. Types are exactly matched (no conversion or assignable checks).
type Describers struct {
	searchFns map[reflect.Type][]typeFunc
}

// DescribeObject implements ObjectDescriber and will attempt to print the provided object to a string,
// if at least one describer function has been registered with the exact types passed, or if any
// describer can print the exact object in its first argument (the remainder will be provided empty
// values). If no function registered with Add can satisfy the passed objects, an ErrNoDescriber will
// be returned
// TODO: reorder and partial match extra.
func (d *Describers) DescribeObject(exact interface{}, extra ...interface{}) (string, error) {
	exactType := reflect.TypeOf(exact)
	fns, ok := d.searchFns[exactType]
	if !ok {
		return "", newErrNoDescriber(exactType)
	}
	if len(extra) == 0 {
		for _, typeFn := range fns {
			if len(typeFn.Extra) == 0 {
				return typeFn.Describe(exact, extra...)
			}
		}
		typeFn := fns[0]
		for _, t := range typeFn.Extra {
			v := reflect.New(t).Elem()
			extra = append(extra, v.Interface())
		}
		return fns[0].Describe(exact, extra...)
	}

	types := make([]reflect.Type, 0, len(extra))
	for _, obj := range extra {
		types = append(types, reflect.TypeOf(obj))
	}
	for _, typeFn := range fns {
		if typeFn.Matches(types) {
			return typeFn.Describe(exact, extra...)
		}
	}
	return "", newErrNoDescriber(append([]reflect.Type{exactType}, types...)...)
}

// Add adds one or more describer functions to the printers.Describer. The passed function must
// match the signature:
//
//     func(...) (string, error)
//
// Any number of arguments may be provided.
func (d *Describers) Add(fns ...interface{}) error {
	for _, fn := range fns {
		fv := reflect.ValueOf(fn)
		ft := fv.Type()
		if ft.Kind() != reflect.Func {
			return fmt.Errorf("expected func, got: %v", ft)
		}
		numIn := ft.NumIn()
		if numIn == 0 {
			return fmt.Errorf("expected at least one 'in' params, got: %v", ft)
		}
		if ft.NumOut() != 2 {
			return fmt.Errorf("expected two 'out' params - (string, error), got: %v", ft)
		}
		types := make([]reflect.Type, 0, numIn)
		for i := 0; i < numIn; i++ {
			types = append(types, ft.In(i))
		}
		if ft.Out(0) != reflect.TypeOf(string("")) {
			return fmt.Errorf("expected string return, got: %v", ft)
		}
		var forErrorType error
		// This convolution is necessary, otherwise TypeOf picks up on the fact
		// that forErrorType is nil.
		errorType := reflect.TypeOf(&forErrorType).Elem()
		if ft.Out(1) != errorType {
			return fmt.Errorf("expected error return, got: %v", ft)
		}

		exact := types[0]
		extra := types[1:]
		if d.searchFns == nil {
			d.searchFns = make(map[reflect.Type][]typeFunc)
		}
		fns := d.searchFns[exact]
		fn := typeFunc{Extra: extra, Fn: fv}
		fns = append(fns, fn)
		d.searchFns[exact] = fns
	}
	return nil
}

// typeFunc holds information about a describer function and the types it accepts
type typeFunc struct {
	Extra []reflect.Type
	Fn    reflect.Value
}

// Matches returns true when the passed types exactly match the Extra list.
func (fn typeFunc) Matches(types []reflect.Type) bool {
	if len(fn.Extra) != len(types) {
		return false
	}
	// reorder the items in array types and fn.Extra
	// convert the type into string and sort them, check if they are matched
	varMap := make(map[reflect.Type]bool)
	for i := range fn.Extra {
		varMap[fn.Extra[i]] = true
	}
	for i := range types {
		if _, found := varMap[types[i]]; !found {
			return false
		}
	}
	return true
}

// Describe invokes the nested function with the exact number of arguments.
func (fn typeFunc) Describe(exact interface{}, extra ...interface{}) (string, error) {
	values := []reflect.Value{reflect.ValueOf(exact)}
	for _, obj := range extra {
		values = append(values, reflect.ValueOf(obj))
	}
	out := fn.Fn.Call(values)
	s := out[0].Interface().(string)
	var err error
	if !out[1].IsNil() {
		err = out[1].Interface().(error)
	}
	return s, err
}

// printLabelsMultilineWithFilter prints filtered multiple labels with a proper alignment.
func printLabelsMultilineWithFilter(w *PrefixWriter, title string, labels map[string]string, skip sets.String) {
	printLabelsMultilineWithIndent(w, "", title, "\t", labels, skip)
}

// printLabelsMultiline prints multiple labels with a proper alignment.
func printLabelsMultiline(w *PrefixWriter, title string, labels map[string]string) {
	printLabelsMultilineWithIndent(w, "", title, "\t", labels, sets.NewString())
}

// printLabelsMultiline prints multiple labels with a user-defined alignment.
func printLabelsMultilineWithIndent(w *PrefixWriter, initialIndent, title, innerIndent string, labels map[string]string, skip sets.String) {

	w.Write(LEVEL_0, "%s%s:%s", initialIndent, title, innerIndent)

	if labels == nil || len(labels) == 0 {
		w.WriteLine("<none>")
		return
	}

	// to print labels in the sorted order
	keys := make([]string, 0, len(labels))
	for key := range labels {
		if skip.Has(key) {
			continue
		}
		keys = append(keys, key)
	}
	if len(keys) == 0 {
		w.WriteLine("<none>")
		return
	}
	sort.Strings(keys)

	for i, key := range keys {
		if i != 0 {
			w.Write(LEVEL_0, "%s", initialIndent)
			w.Write(LEVEL_0, "%s", innerIndent)
		}
		w.Write(LEVEL_0, "%s=%s\n", key, labels[key])
		i++
	}
}

// printTaintsMultiline prints multiple taints with a proper alignment.
func printNodeTaintsMultiline(w *PrefixWriter, title string, taints []api.Taint) {
	printTaintsMultilineWithIndent(w, "", title, "\t", taints)
}

// printTaintsMultilineWithIndent prints multiple taints with a user-defined alignment.
func printTaintsMultilineWithIndent(w *PrefixWriter, initialIndent, title, innerIndent string, taints []api.Taint) {
	w.Write(LEVEL_0, "%s%s:%s", initialIndent, title, innerIndent)

	if taints == nil || len(taints) == 0 {
		w.WriteLine("<none>")
		return
	}

	// to print taints in the sorted order
	keys := make([]string, 0, len(taints))
	for _, taint := range taints {
		keys = append(keys, string(taint.Effect)+","+taint.Key)
	}
	sort.Strings(keys)

	for i, key := range keys {
		for _, taint := range taints {
			if string(taint.Effect)+","+taint.Key == key {
				if i != 0 {
					w.Write(LEVEL_0, "%s", initialIndent)
					w.Write(LEVEL_0, "%s", innerIndent)
				}
				w.Write(LEVEL_0, "%s\n", taint.ToString())
				i++
			}
		}
	}
}

// printPodTolerationsMultiline prints multiple tolerations with a proper alignment.
func printPodTolerationsMultiline(w *PrefixWriter, title string, tolerations []api.Toleration) {
	printTolerationsMultilineWithIndent(w, "", title, "\t", tolerations)
}

// printTolerationsMultilineWithIndent prints multiple tolerations with a user-defined alignment.
func printTolerationsMultilineWithIndent(w *PrefixWriter, initialIndent, title, innerIndent string, tolerations []api.Toleration) {
	w.Write(LEVEL_0, "%s%s:%s", initialIndent, title, innerIndent)

	if tolerations == nil || len(tolerations) == 0 {
		w.WriteLine("<none>")
		return
	}

	// to print tolerations in the sorted order
	keys := make([]string, 0, len(tolerations))
	for _, toleration := range tolerations {
		keys = append(keys, toleration.Key)
	}
	sort.Strings(keys)

	for i, key := range keys {
		for _, toleration := range tolerations {
			if toleration.Key == key {
				if i != 0 {
					w.Write(LEVEL_0, "%s", initialIndent)
					w.Write(LEVEL_0, "%s", innerIndent)
				}
				w.Write(LEVEL_0, "%s=%s", toleration.Key, toleration.Value)
				if len(toleration.Operator) != 0 {
					w.Write(LEVEL_0, ":%s", toleration.Operator)
				}
				if len(toleration.Effect) != 0 {
					w.Write(LEVEL_0, ":%s", toleration.Effect)
				}
				if toleration.TolerationSeconds != nil {
					w.Write(LEVEL_0, " for %ds", *toleration.TolerationSeconds)
				}
				w.Write(LEVEL_0, "\n")
				i++
			}
		}
	}
}

func tabbedString(f func(io.Writer) error) (string, error) {
	out := new(tabwriter.Writer)
	buf := &bytes.Buffer{}
	out.Init(buf, 0, 8, 1, '\t', 0)

	err := f(out)
	if err != nil {
		return "", err
	}

	out.Flush()
	str := string(buf.String())
	return str, nil
}

type SortableResourceNames []api.ResourceName

func (list SortableResourceNames) Len() int {
	return len(list)
}

func (list SortableResourceNames) Swap(i, j int) {
	list[i], list[j] = list[j], list[i]
}

func (list SortableResourceNames) Less(i, j int) bool {
	return list[i] < list[j]
}

// SortedResourceNames returns the sorted resource names of a resource list.
func SortedResourceNames(list api.ResourceList) []api.ResourceName {
	resources := make([]api.ResourceName, 0, len(list))
	for res := range list {
		resources = append(resources, res)
	}
	sort.Sort(SortableResourceNames(resources))
	return resources
}

type SortableResourceQuotas []api.ResourceQuota

func (list SortableResourceQuotas) Len() int {
	return len(list)
}

func (list SortableResourceQuotas) Swap(i, j int) {
	list[i], list[j] = list[j], list[i]
}

func (list SortableResourceQuotas) Less(i, j int) bool {
	return list[i].Name < list[j].Name
}

type SortableVolumeMounts []api.VolumeMount

func (list SortableVolumeMounts) Len() int {
	return len(list)
}

func (list SortableVolumeMounts) Swap(i, j int) {
	list[i], list[j] = list[j], list[i]
}

func (list SortableVolumeMounts) Less(i, j int) bool {
	return list[i].MountPath < list[j].MountPath
}

// SortedQoSResourceNames returns the sorted resource names of a QoS list.
func SortedQoSResourceNames(list qos.QOSList) []api.ResourceName {
	resources := make([]api.ResourceName, 0, len(list))
	for res := range list {
		resources = append(resources, api.ResourceName(res))
	}
	sort.Sort(SortableResourceNames(resources))
	return resources
}

func versionedClientsetForDeployment(internalClient clientset.Interface) versionedclientset.Interface {
	if internalClient == nil {
		return &versionedclientset.Clientset{}
	}
	return &versionedclientset.Clientset{
		CoreV1Client:            coreclientset.New(internalClient.Core().RESTClient()),
		ExtensionsV1beta1Client: extensionsclientset.New(internalClient.Extensions().RESTClient()),
	}
}

var maxAnnotationLen = 200

// printAnnotationsMultilineWithFilter prints filtered multiple annotations with a proper alignment.
func printAnnotationsMultilineWithFilter(w *PrefixWriter, title string, annotations map[string]string, skip sets.String) {
	printAnnotationsMultilineWithIndent(w, "", title, "\t", annotations, skip)
}

// printAnnotationsMultiline prints multiple annotations with a proper alignment.
func printAnnotationsMultiline(w *PrefixWriter, title string, annotations map[string]string) {
	printAnnotationsMultilineWithIndent(w, "", title, "\t", annotations, sets.NewString())
}

// printAnnotationsMultilineWithIndent prints multiple annotations with a user-defined alignment.
// If annotation string is too long, we omit chars more than 200 length.
func printAnnotationsMultilineWithIndent(w *PrefixWriter, initialIndent, title, innerIndent string, annotations map[string]string, skip sets.String) {

	w.Write(LEVEL_0, "%s%s:%s", initialIndent, title, innerIndent)

	if len(annotations) == 0 {
		w.WriteLine("<none>")
		return
	}

	// to print labels in the sorted order
	keys := make([]string, 0, len(annotations))
	for key := range annotations {
		if skip.Has(key) {
			continue
		}
		keys = append(keys, key)
	}
	if len(annotations) == 0 {
		w.WriteLine("<none>")
		return
	}
	sort.Strings(keys)

	for i, key := range keys {
		if i != 0 {
			w.Write(LEVEL_0, initialIndent)
			w.Write(LEVEL_0, innerIndent)
		}
		line := fmt.Sprintf("%s=%s", key, annotations[key])
		if len(line) > maxAnnotationLen {
			w.Write(LEVEL_0, "%s...\n", line[:maxAnnotationLen])
		} else {
			w.Write(LEVEL_0, "%s\n", line)
		}
		i++
	}
}
