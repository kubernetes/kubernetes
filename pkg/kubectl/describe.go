/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"bytes"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/fields"
	qosutil "k8s.io/kubernetes/pkg/kubelet/qos/util"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/types"
	deploymentUtil "k8s.io/kubernetes/pkg/util/deployment"
	"k8s.io/kubernetes/pkg/util/sets"
)

// Describer generates output for the named resource or an error
// if the output could not be generated. Implementors typically
// abstract the retrieval of the named object from a remote server.
type Describer interface {
	Describe(namespace, name string) (output string, err error)
}

// ObjectDescriber is an interface for displaying arbitrary objects with extra
// information. Use when an object is in hand (on disk, or already retrieved).
// Implementors may ignore the additional information passed on extra, or use it
// by default. ObjectDescribers may return ErrNoDescriber if no suitable describer
// is found.
type ObjectDescriber interface {
	DescribeObject(object interface{}, extra ...interface{}) (output string, err error)
}

// ErrNoDescriber is a structured error indicating the provided object or objects
// cannot be described.
type ErrNoDescriber struct {
	Types []string
}

// Error implements the error interface.
func (e ErrNoDescriber) Error() string {
	return fmt.Sprintf("no describer has been defined for %v", e.Types)
}

func describerMap(c *client.Client) map[string]Describer {
	m := map[string]Describer{
		"Pod": &PodDescriber{c},
		"ReplicationController": &ReplicationControllerDescriber{c},
		"Secret":                &SecretDescriber{c},
		"Service":               &ServiceDescriber{c},
		"ServiceAccount":        &ServiceAccountDescriber{c},
		"Minion":                &NodeDescriber{c},
		"Node":                  &NodeDescriber{c},
		"LimitRange":            &LimitRangeDescriber{c},
		"ResourceQuota":         &ResourceQuotaDescriber{c},
		"PersistentVolume":      &PersistentVolumeDescriber{c},
		"PersistentVolumeClaim": &PersistentVolumeClaimDescriber{c},
		"Namespace":             &NamespaceDescriber{c},
	}
	return m
}

func expDescriberMap(c *client.Client) map[string]Describer {
	return map[string]Describer{
		"HorizontalPodAutoscaler": &HorizontalPodAutoscalerDescriber{c},
		"DaemonSet":               &DaemonSetDescriber{c},
		"Job":                     &JobDescriber{c},
		"Deployment":              &DeploymentDescriber{c},
		"Ingress":                 &IngressDescriber{c},
	}
}

// List of all resource types we can describe
func DescribableResources() []string {
	keys := make([]string, 0)

	for k := range describerMap(nil) {
		resource := strings.ToLower(k)
		keys = append(keys, resource)
	}
	return keys
}

// Describer returns the default describe functions for each of the standard
// Kubernetes types.
func DescriberFor(group string, kind string, c *client.Client) (Describer, bool) {
	var f Describer
	var ok bool

	switch group {
	case "":
		f, ok = describerMap(c)[kind]
	case "extensions":
		f, ok = expDescriberMap(c)[kind]
	}

	return f, ok
}

// DefaultObjectDescriber can describe the default Kubernetes objects.
var DefaultObjectDescriber ObjectDescriber

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
	client.Interface
}

func (d *NamespaceDescriber) Describe(namespace, name string) (string, error) {
	ns, err := d.Namespaces().Get(name)
	if err != nil {
		return "", err
	}
	resourceQuotaList, err := d.ResourceQuotas(name).List(labels.Everything())
	if err != nil {
		return "", err
	}
	limitRangeList, err := d.LimitRanges(name).List(labels.Everything())
	if err != nil {
		return "", err
	}

	return describeNamespace(ns, resourceQuotaList, limitRangeList)
}

func describeNamespace(namespace *api.Namespace, resourceQuotaList *api.ResourceQuotaList, limitRangeList *api.LimitRangeList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", namespace.Name)
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(namespace.Labels))
		fmt.Fprintf(out, "Status:\t%s\n", string(namespace.Status.Phase))
		if resourceQuotaList != nil {
			fmt.Fprintf(out, "\n")
			DescribeResourceQuotas(resourceQuotaList, out)
		}
		if limitRangeList != nil {
			fmt.Fprintf(out, "\n")
			DescribeLimitRanges(limitRangeList, out)
		}
		return nil
	})
}

// DescribeLimitRanges merges a set of limit range items into a single tabular description
func DescribeLimitRanges(limitRanges *api.LimitRangeList, w io.Writer) {
	if len(limitRanges.Items) == 0 {
		fmt.Fprint(w, "No resource limits.\n")
		return
	}
	fmt.Fprintf(w, "Resource Limits\n Type\tResource\tMin\tMax\tRequest\tLimit\tLimit/Request\n")
	fmt.Fprintf(w, " ----\t--------\t---\t---\t-------\t-----\t-------------\n")
	for _, limitRange := range limitRanges.Items {
		for i := range limitRange.Spec.Limits {
			item := limitRange.Spec.Limits[i]
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

				msg := " %s\t%v\t%v\t%v\t%v\t%v\t%v\n"
				fmt.Fprintf(w, msg, item.Type, k, minValue, maxValue, defaultRequestValue, defaultLimitValue, ratioValue)
			}
		}
	}
}

// DescribeResourceQuotas merges a set of quota items into a single tabular description of all quotas
func DescribeResourceQuotas(quotas *api.ResourceQuotaList, w io.Writer) {
	if len(quotas.Items) == 0 {
		fmt.Fprint(w, "No resource quota.\n")
		return
	}
	resources := []api.ResourceName{}
	hard := map[api.ResourceName]resource.Quantity{}
	used := map[api.ResourceName]resource.Quantity{}
	for _, q := range quotas.Items {
		for resource := range q.Status.Hard {
			resources = append(resources, resource)
			hardQuantity := q.Status.Hard[resource]
			usedQuantity := q.Status.Used[resource]

			// if for some reason there are multiple quota documents, we take least permissive
			prevQuantity, ok := hard[resource]
			if ok {
				if hardQuantity.Value() < prevQuantity.Value() {
					hard[resource] = hardQuantity
				}
			} else {
				hard[resource] = hardQuantity
			}
			used[resource] = usedQuantity
		}
	}

	sort.Sort(SortableResourceNames(resources))
	fmt.Fprint(w, "Resource Quotas\n Resource\tUsed\tHard\n")
	fmt.Fprint(w, " ---\t---\t---\n")
	for _, resource := range resources {
		hardQuantity := hard[resource]
		usedQuantity := used[resource]
		fmt.Fprintf(w, " %s\t%s\t%s\n", string(resource), usedQuantity.String(), hardQuantity.String())
	}
}

// LimitRangeDescriber generates information about a limit range
type LimitRangeDescriber struct {
	client.Interface
}

func (d *LimitRangeDescriber) Describe(namespace, name string) (string, error) {
	lr := d.LimitRanges(namespace)

	limitRange, err := lr.Get(name)
	if err != nil {
		return "", err
	}
	return describeLimitRange(limitRange)
}

func describeLimitRange(limitRange *api.LimitRange) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", limitRange.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", limitRange.Namespace)
		fmt.Fprintf(out, "Type\tResource\tMin\tMax\tRequest\tLimit\tLimit/Request\n")
		fmt.Fprintf(out, "----\t--------\t---\t---\t-------\t-----\t-------------\n")
		for i := range limitRange.Spec.Limits {
			item := limitRange.Spec.Limits[i]
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

				msg := "%v\t%v\t%v\t%v\t%v\t%v\t%v\n"
				fmt.Fprintf(out, msg, item.Type, k, minValue, maxValue, defaultRequestValue, defaultLimitValue, ratioValue)
			}
		}
		return nil
	})
}

// ResourceQuotaDescriber generates information about a resource quota
type ResourceQuotaDescriber struct {
	client.Interface
}

func (d *ResourceQuotaDescriber) Describe(namespace, name string) (string, error) {
	rq := d.ResourceQuotas(namespace)

	resourceQuota, err := rq.Get(name)
	if err != nil {
		return "", err
	}

	return describeQuota(resourceQuota)
}

func describeQuota(resourceQuota *api.ResourceQuota) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", resourceQuota.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", resourceQuota.Namespace)
		fmt.Fprintf(out, "Resource\tUsed\tHard\n")
		fmt.Fprintf(out, "--------\t----\t----\n")

		resources := []api.ResourceName{}
		for resource := range resourceQuota.Status.Hard {
			resources = append(resources, resource)
		}
		sort.Sort(SortableResourceNames(resources))

		msg := "%v\t%v\t%v\n"
		for i := range resources {
			resource := resources[i]
			hardQuantity := resourceQuota.Status.Hard[resource]
			usedQuantity := resourceQuota.Status.Used[resource]
			fmt.Fprintf(out, msg, resource, usedQuantity.String(), hardQuantity.String())
		}
		return nil
	})
}

// PodDescriber generates information about a pod and the replication controllers that
// create it.
type PodDescriber struct {
	client.Interface
}

func (d *PodDescriber) Describe(namespace, name string) (string, error) {
	rc := d.ReplicationControllers(namespace)
	pc := d.Pods(namespace)

	pod, err := pc.Get(name)
	if err != nil {
		eventsInterface := d.Events(namespace)
		events, err2 := eventsInterface.List(
			labels.Everything(),
			eventsInterface.GetFieldSelector(&name, &namespace, nil, nil))
		if err2 == nil && len(events.Items) > 0 {
			return tabbedString(func(out io.Writer) error {
				fmt.Fprintf(out, "Pod '%v': error '%v', but found events.\n", name, err)
				DescribeEvents(events, out)
				return nil
			})
		}
		return "", err
	}

	var events *api.EventList
	if ref, err := api.GetReference(pod); err != nil {
		glog.Errorf("Unable to construct reference to '%#v': %v", pod, err)
	} else {
		ref.Kind = ""
		events, _ = d.Events(namespace).Search(ref)
	}

	rcs, err := getReplicationControllersForLabels(rc, labels.Set(pod.Labels))
	if err != nil {
		return "", err
	}

	return describePod(pod, rcs, events)
}

func describePod(pod *api.Pod, rcs []api.ReplicationController, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", pod.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", pod.Namespace)
		fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(&pod.Spec))
		fmt.Fprintf(out, "Node:\t%s\n", pod.Spec.NodeName+"/"+pod.Status.HostIP)
		if pod.Status.StartTime != nil {
			fmt.Fprintf(out, "Start Time:\t%s\n", pod.Status.StartTime.Time.Format(time.RFC1123Z))
		}
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(pod.Labels))
		if pod.DeletionTimestamp != nil {
			fmt.Fprintf(out, "Status:\tTerminating (expires %s)\n", pod.DeletionTimestamp.Time.Format(time.RFC1123Z))
			fmt.Fprintf(out, "Termination Grace Period:\t%ds\n", *pod.DeletionGracePeriodSeconds)
		} else {
			fmt.Fprintf(out, "Status:\t%s\n", string(pod.Status.Phase))
		}
		fmt.Fprintf(out, "Reason:\t%s\n", pod.Status.Reason)
		fmt.Fprintf(out, "Message:\t%s\n", pod.Status.Message)
		fmt.Fprintf(out, "IP:\t%s\n", pod.Status.PodIP)
		var matchingRCs []*api.ReplicationController
		for _, rc := range rcs {
			matchingRCs = append(matchingRCs, &rc)
		}
		fmt.Fprintf(out, "Replication Controllers:\t%s\n", printReplicationControllersByLabels(matchingRCs))
		fmt.Fprintf(out, "Containers:\n")
		describeContainers(pod, out)
		if len(pod.Status.Conditions) > 0 {
			fmt.Fprint(out, "Conditions:\n  Type\tStatus\n")
			for _, c := range pod.Status.Conditions {
				fmt.Fprintf(out, "  %v \t%v \n",
					c.Type,
					c.Status)
			}
		}
		describeVolumes(pod.Spec.Volumes, out)
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

func describeVolumes(volumes []api.Volume, out io.Writer) {
	if volumes == nil || len(volumes) == 0 {
		fmt.Fprint(out, "No volumes.\n")
		return
	}
	fmt.Fprint(out, "Volumes:\n")
	for _, volume := range volumes {
		fmt.Fprintf(out, "  %v:\n", volume.Name)
		switch {
		case volume.VolumeSource.HostPath != nil:
			printHostPathVolumeSource(volume.VolumeSource.HostPath, out)
		case volume.VolumeSource.EmptyDir != nil:
			printEmptyDirVolumeSource(volume.VolumeSource.EmptyDir, out)
		case volume.VolumeSource.GCEPersistentDisk != nil:
			printGCEPersistentDiskVolumeSource(volume.VolumeSource.GCEPersistentDisk, out)
		case volume.VolumeSource.AWSElasticBlockStore != nil:
			printAWSElasticBlockStoreVolumeSource(volume.VolumeSource.AWSElasticBlockStore, out)
		case volume.VolumeSource.GitRepo != nil:
			printGitRepoVolumeSource(volume.VolumeSource.GitRepo, out)
		case volume.VolumeSource.Secret != nil:
			printSecretVolumeSource(volume.VolumeSource.Secret, out)
		case volume.VolumeSource.NFS != nil:
			printNFSVolumeSource(volume.VolumeSource.NFS, out)
		case volume.VolumeSource.ISCSI != nil:
			printISCSIVolumeSource(volume.VolumeSource.ISCSI, out)
		case volume.VolumeSource.Glusterfs != nil:
			printGlusterfsVolumeSource(volume.VolumeSource.Glusterfs, out)
		case volume.VolumeSource.PersistentVolumeClaim != nil:
			printPersistentVolumeClaimVolumeSource(volume.VolumeSource.PersistentVolumeClaim, out)
		case volume.VolumeSource.RBD != nil:
			printRBDVolumeSource(volume.VolumeSource.RBD, out)
		default:
			fmt.Fprintf(out, "  <Volume Type Not Found>\n")
		}
	}
}

func printHostPathVolumeSource(hostPath *api.HostPathVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tHostPath (bare host directory volume)\n"+
		"    Path:\t%v\n", hostPath.Path)
}

func printEmptyDirVolumeSource(emptyDir *api.EmptyDirVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tEmptyDir (a temporary directory that shares a pod's lifetime)\n"+
		"    Medium:\t%v\n", emptyDir.Medium)
}

func printGCEPersistentDiskVolumeSource(gce *api.GCEPersistentDiskVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tGCEPersistentDisk (a Persistent Disk resource in Google Compute Engine)\n"+
		"    PDName:\t%v\n"+
		"    FSType:\t%v\n"+
		"    Partition:\t%v\n"+
		"    ReadOnly:\t%v\n",
		gce.PDName, gce.FSType, gce.Partition, gce.ReadOnly)
}

func printAWSElasticBlockStoreVolumeSource(aws *api.AWSElasticBlockStoreVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tAWSElasticBlockStore (a Persistent Disk resource in AWS)\n"+
		"    VolumeID:\t%v\n"+
		"    FSType:\t%v\n"+
		"    Partition:\t%v\n"+
		"    ReadOnly:\t%v\n",
		aws.VolumeID, aws.FSType, aws.Partition, aws.ReadOnly)
}

func printGitRepoVolumeSource(git *api.GitRepoVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tGitRepo (a volume that is pulled from git when the pod is created)\n"+
		"    Repository:\t%v\n"+
		"    Revision:\t%v\n",
		git.Repository, git.Revision)
}

func printSecretVolumeSource(secret *api.SecretVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tSecret (a secret that should populate this volume)\n"+
		"    SecretName:\t%v\n", secret.SecretName)
}

func printNFSVolumeSource(nfs *api.NFSVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tNFS (an NFS mount that lasts the lifetime of a pod)\n"+
		"    Server:\t%v\n"+
		"    Path:\t%v\n"+
		"    ReadOnly:\t%v\n",
		nfs.Server, nfs.Path, nfs.ReadOnly)
}

func printISCSIVolumeSource(iscsi *api.ISCSIVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tISCSI (an ISCSI Disk resource that is attached to a kubelet's host machine and then exposed to the pod)\n"+
		"    TargetPortal:\t%v\n"+
		"    IQN:\t%v\n"+
		"    Lun:\t%v\n"+
		"    FSType:\t%v\n"+
		"    ReadOnly:\t%v\n",
		iscsi.TargetPortal, iscsi.IQN, iscsi.Lun, iscsi.FSType, iscsi.ReadOnly)
}

func printGlusterfsVolumeSource(glusterfs *api.GlusterfsVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tGlusterfs (a Glusterfs mount on the host that shares a pod's lifetime)\n"+
		"    EndpointsName:\t%v\n"+
		"    Path:\t%v\n"+
		"    ReadOnly:\t%v\n",
		glusterfs.EndpointsName, glusterfs.Path, glusterfs.ReadOnly)
}

func printPersistentVolumeClaimVolumeSource(claim *api.PersistentVolumeClaimVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tPersistentVolumeClaim (a reference to a PersistentVolumeClaim in the same namespace)\n"+
		"    ClaimName:\t%v\n"+
		"    ReadOnly:\t%v\n",
		claim.ClaimName, claim.ReadOnly)
}

func printRBDVolumeSource(rbd *api.RBDVolumeSource, out io.Writer) {
	fmt.Fprintf(out, "    Type:\tRBD (a Rados Block Device mount on the host that shares a pod's lifetime)\n"+
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

type PersistentVolumeDescriber struct {
	client.Interface
}

func (d *PersistentVolumeDescriber) Describe(namespace, name string) (string, error) {
	c := d.PersistentVolumes()

	pv, err := c.Get(name)
	if err != nil {
		return "", err
	}

	storage := pv.Spec.Capacity[api.ResourceStorage]

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", pv.Name)
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(pv.Labels))
		fmt.Fprintf(out, "Status:\t%s\n", pv.Status.Phase)
		if pv.Spec.ClaimRef != nil {
			fmt.Fprintf(out, "Claim:\t%s\n", pv.Spec.ClaimRef.Namespace+"/"+pv.Spec.ClaimRef.Name)
		} else {
			fmt.Fprintf(out, "Claim:\t%s\n", "")
		}
		fmt.Fprintf(out, "Reclaim Policy:\t%v\n", pv.Spec.PersistentVolumeReclaimPolicy)
		fmt.Fprintf(out, "Access Modes:\t%s\n", api.GetAccessModesAsString(pv.Spec.AccessModes))
		fmt.Fprintf(out, "Capacity:\t%s\n", storage.String())
		fmt.Fprintf(out, "Message:\t%s\n", pv.Status.Message)
		fmt.Fprintf(out, "Source:\n")

		switch {
		case pv.Spec.HostPath != nil:
			printHostPathVolumeSource(pv.Spec.HostPath, out)
		case pv.Spec.GCEPersistentDisk != nil:
			printGCEPersistentDiskVolumeSource(pv.Spec.GCEPersistentDisk, out)
		case pv.Spec.AWSElasticBlockStore != nil:
			printAWSElasticBlockStoreVolumeSource(pv.Spec.AWSElasticBlockStore, out)
		case pv.Spec.NFS != nil:
			printNFSVolumeSource(pv.Spec.NFS, out)
		case pv.Spec.ISCSI != nil:
			printISCSIVolumeSource(pv.Spec.ISCSI, out)
		case pv.Spec.Glusterfs != nil:
			printGlusterfsVolumeSource(pv.Spec.Glusterfs, out)
		case pv.Spec.RBD != nil:
			printRBDVolumeSource(pv.Spec.RBD, out)
		}

		return nil
	})
}

type PersistentVolumeClaimDescriber struct {
	client.Interface
}

func (d *PersistentVolumeClaimDescriber) Describe(namespace, name string) (string, error) {
	c := d.PersistentVolumeClaims(namespace)

	pvc, err := c.Get(name)
	if err != nil {
		return "", err
	}

	labels := labels.FormatLabels(pvc.Labels)
	storage := pvc.Spec.Resources.Requests[api.ResourceStorage]
	capacity := ""
	accessModes := ""
	if pvc.Spec.VolumeName != "" {
		accessModes = api.GetAccessModesAsString(pvc.Status.AccessModes)
		storage = pvc.Status.Capacity[api.ResourceStorage]
		capacity = storage.String()
	}

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", pvc.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", pvc.Namespace)
		fmt.Fprintf(out, "Status:\t%v\n", pvc.Status.Phase)
		fmt.Fprintf(out, "Volume:\t%s\n", pvc.Spec.VolumeName)
		fmt.Fprintf(out, "Labels:\t%s\n", labels)
		fmt.Fprintf(out, "Capacity:\t%s\n", capacity)
		fmt.Fprintf(out, "Access Modes:\t%s\n", accessModes)
		return nil
	})
}

func describeContainers(pod *api.Pod, out io.Writer) {
	statuses := map[string]api.ContainerStatus{}
	for _, status := range pod.Status.ContainerStatuses {
		statuses[status.Name] = status
	}

	for _, container := range pod.Spec.Containers {
		status := statuses[container.Name]
		state := status.State

		fmt.Fprintf(out, "  %v:\n", container.Name)
		fmt.Fprintf(out, "    Container ID:\t%s\n", status.ContainerID)
		fmt.Fprintf(out, "    Image:\t%s\n", container.Image)
		fmt.Fprintf(out, "    Image ID:\t%s\n", status.ImageID)

		resourceToQoS := qosutil.GetQoS(&container)
		if len(resourceToQoS) > 0 {
			fmt.Fprintf(out, "    QoS Tier:\n")
		}
		for resource, qos := range resourceToQoS {
			fmt.Fprintf(out, "      %s:\t%s\n", resource, qos)
		}

		if len(container.Resources.Limits) > 0 {
			fmt.Fprintf(out, "    Limits:\n")
		}
		for name, quantity := range container.Resources.Limits {
			fmt.Fprintf(out, "      %s:\t%s\n", name, quantity.String())
		}

		if len(container.Resources.Requests) > 0 {
			fmt.Fprintf(out, "    Requests:\n")
		}
		for name, quantity := range container.Resources.Requests {
			fmt.Fprintf(out, "      %s:\t%s\n", name, quantity.String())
		}

		describeStatus("State", state, out)
		if status.LastTerminationState.Terminated != nil {
			describeStatus("Last Termination State", status.LastTerminationState, out)
		}
		fmt.Fprintf(out, "    Ready:\t%v\n", printBool(status.Ready))
		fmt.Fprintf(out, "    Restart Count:\t%d\n", status.RestartCount)
		fmt.Fprintf(out, "    Environment Variables:\n")
		for _, e := range container.Env {
			if e.ValueFrom != nil && e.ValueFrom.FieldRef != nil {
				valueFrom := envValueFrom(pod, e)
				fmt.Fprintf(out, "      %s:\t%s (%s:%s)\n", e.Name, valueFrom, e.ValueFrom.FieldRef.APIVersion, e.ValueFrom.FieldRef.FieldPath)
			} else {
				fmt.Fprintf(out, "      %s:\t%s\n", e.Name, e.Value)
			}
		}
	}
}

func envValueFrom(pod *api.Pod, e api.EnvVar) string {
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

func describeStatus(stateName string, state api.ContainerState, out io.Writer) {
	switch {
	case state.Running != nil:
		fmt.Fprintf(out, "    %s:\tRunning\n", stateName)
		fmt.Fprintf(out, "      Started:\t%v\n", state.Running.StartedAt.Time.Format(time.RFC1123Z))
	case state.Waiting != nil:
		fmt.Fprintf(out, "    %s:\tWaiting\n", stateName)
		if state.Waiting.Reason != "" {
			fmt.Fprintf(out, "      Reason:\t%s\n", state.Waiting.Reason)
		}
	case state.Terminated != nil:
		fmt.Fprintf(out, "    %s:\tTerminated\n", stateName)
		if state.Terminated.Reason != "" {
			fmt.Fprintf(out, "      Reason:\t%s\n", state.Terminated.Reason)
		}
		if state.Terminated.Message != "" {
			fmt.Fprintf(out, "      Message:\t%s\n", state.Terminated.Message)
		}
		fmt.Fprintf(out, "      Exit Code:\t%d\n", state.Terminated.ExitCode)
		if state.Terminated.Signal > 0 {
			fmt.Fprintf(out, "      Signal:\t%d\n", state.Terminated.Signal)
		}
		fmt.Fprintf(out, "      Started:\t%s\n", state.Terminated.StartedAt.Time.Format(time.RFC1123Z))
		fmt.Fprintf(out, "      Finished:\t%s\n", state.Terminated.FinishedAt.Time.Format(time.RFC1123Z))
	default:
		fmt.Fprintf(out, "    %s:\tWaiting\n", stateName)
	}
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
	client.Interface
}

func (d *ReplicationControllerDescriber) Describe(namespace, name string) (string, error) {
	rc := d.ReplicationControllers(namespace)
	pc := d.Pods(namespace)

	controller, err := rc.Get(name)
	if err != nil {
		return "", err
	}

	running, waiting, succeeded, failed, err := getPodStatusForController(pc, controller.Spec.Selector)
	if err != nil {
		return "", err
	}

	events, _ := d.Events(namespace).Search(controller)

	return describeReplicationController(controller, events, running, waiting, succeeded, failed)
}

func describeReplicationController(controller *api.ReplicationController, events *api.EventList, running, waiting, succeeded, failed int) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", controller.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", controller.Namespace)
		if controller.Spec.Template != nil {
			fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(&controller.Spec.Template.Spec))
		} else {
			fmt.Fprintf(out, "Image(s):\t%s\n", "<no template>")
		}
		fmt.Fprintf(out, "Selector:\t%s\n", labels.FormatLabels(controller.Spec.Selector))
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(controller.Labels))
		fmt.Fprintf(out, "Replicas:\t%d current / %d desired\n", controller.Status.Replicas, controller.Spec.Replicas)
		fmt.Fprintf(out, "Pods Status:\t%d Running / %d Waiting / %d Succeeded / %d Failed\n", running, waiting, succeeded, failed)
		if controller.Spec.Template != nil {
			describeVolumes(controller.Spec.Template.Spec.Volumes, out)
		}
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// JobDescriber generates information about a job and the pods it has created.
type JobDescriber struct {
	client *client.Client
}

func (d *JobDescriber) Describe(namespace, name string) (string, error) {
	job, err := d.client.Extensions().Jobs(namespace).Get(name)
	if err != nil {
		return "", err
	}

	events, _ := d.client.Events(namespace).Search(job)

	return describeJob(job, events)
}

func describeJob(job *extensions.Job, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", job.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", job.Namespace)
		fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(&job.Spec.Template.Spec))
		selector, _ := extensions.PodSelectorAsSelector(job.Spec.Selector)
		fmt.Fprintf(out, "Selector:\t%s\n", selector)
		fmt.Fprintf(out, "Parallelism:\t%d\n", *job.Spec.Parallelism)
		fmt.Fprintf(out, "Completions:\t%d\n", *job.Spec.Completions)
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(job.Labels))
		fmt.Fprintf(out, "Pods Statuses:\t%d Running / %d Succeeded / %d Failed\n", job.Status.Active, job.Status.Succeeded, job.Status.Failed)
		describeVolumes(job.Spec.Template.Spec.Volumes, out)
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// DaemonSetDescriber generates information about a daemon set and the pods it has created.
type DaemonSetDescriber struct {
	client.Interface
}

func (d *DaemonSetDescriber) Describe(namespace, name string) (string, error) {
	dc := d.Extensions().DaemonSets(namespace)
	pc := d.Pods(namespace)

	daemon, err := dc.Get(name)
	if err != nil {
		return "", err
	}

	running, waiting, succeeded, failed, err := getPodStatusForController(pc, daemon.Spec.Selector)
	if err != nil {
		return "", err
	}

	events, _ := d.Events(namespace).Search(daemon)

	return describeDaemonSet(daemon, events, running, waiting, succeeded, failed)
}

func describeDaemonSet(daemon *extensions.DaemonSet, events *api.EventList, running, waiting, succeeded, failed int) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", daemon.Name)
		if daemon.Spec.Template != nil {
			fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(&daemon.Spec.Template.Spec))
		} else {
			fmt.Fprintf(out, "Image(s):\t%s\n", "<no template>")
		}
		fmt.Fprintf(out, "Selector:\t%s\n", labels.FormatLabels(daemon.Spec.Selector))
		fmt.Fprintf(out, "Node-Selector:\t%s\n", labels.FormatLabels(daemon.Spec.Template.Spec.NodeSelector))
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(daemon.Labels))
		fmt.Fprintf(out, "Desired Number of Nodes Scheduled: %d\n", daemon.Status.DesiredNumberScheduled)
		fmt.Fprintf(out, "Current Number of Nodes Scheduled: %d\n", daemon.Status.CurrentNumberScheduled)
		fmt.Fprintf(out, "Number of Nodes Misscheduled: %d\n", daemon.Status.NumberMisscheduled)
		fmt.Fprintf(out, "Pods Status:\t%d Running / %d Waiting / %d Succeeded / %d Failed\n", running, waiting, succeeded, failed)
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// SecretDescriber generates information about a secret
type SecretDescriber struct {
	client.Interface
}

func (d *SecretDescriber) Describe(namespace, name string) (string, error) {
	c := d.Secrets(namespace)

	secret, err := c.Get(name)
	if err != nil {
		return "", err
	}

	return describeSecret(secret)
}

func describeSecret(secret *api.Secret) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", secret.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", secret.Namespace)
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(secret.Labels))
		fmt.Fprintf(out, "Annotations:\t%s\n", labels.FormatLabels(secret.Annotations))

		fmt.Fprintf(out, "\nType:\t%s\n", secret.Type)

		fmt.Fprintf(out, "\nData\n====\n")
		for k, v := range secret.Data {
			switch {
			case k == api.ServiceAccountTokenKey && secret.Type == api.SecretTypeServiceAccountToken:
				fmt.Fprintf(out, "%s:\t%s\n", k, string(v))
			default:
				fmt.Fprintf(out, "%s:\t%d bytes\n", k, len(v))
			}
		}

		return nil
	})
}

type IngressDescriber struct {
	client.Interface
}

func (i *IngressDescriber) Describe(namespace, name string) (string, error) {
	c := i.Extensions().Ingress(namespace)
	ing, err := c.Get(name)
	if err != nil {
		return "", err
	}
	events, _ := i.Events(namespace).Search(ing)
	return describeIngress(ing, events)
}

func describeIngress(ing *extensions.Ingress, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		describeIngressAnnotations(out, ing.Annotations)
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// TODO: Move from annotations into Ingress status.
func describeIngressAnnotations(out io.Writer, annotations map[string]string) {
	for k, v := range annotations {
		if !strings.HasPrefix(k, "ingress") {
			continue
		}
		parts := strings.Split(k, "/")
		name := parts[len(parts)-1]
		fmt.Fprintf(out, "%v:\t%s\n", name, v)
	}
	return
}

// ServiceDescriber generates information about a service.
type ServiceDescriber struct {
	client.Interface
}

func (d *ServiceDescriber) Describe(namespace, name string) (string, error) {
	c := d.Services(namespace)

	service, err := c.Get(name)
	if err != nil {
		return "", err
	}

	endpoints, _ := d.Endpoints(namespace).Get(name)
	events, _ := d.Events(namespace).Search(service)

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
		fmt.Fprintf(out, "Name:\t%s\n", service.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", service.Namespace)
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(service.Labels))
		fmt.Fprintf(out, "Selector:\t%s\n", labels.FormatLabels(service.Spec.Selector))
		fmt.Fprintf(out, "Type:\t%s\n", service.Spec.Type)
		fmt.Fprintf(out, "IP:\t%s\n", service.Spec.ClusterIP)
		if len(service.Status.LoadBalancer.Ingress) > 0 {
			list := buildIngressString(service.Status.LoadBalancer.Ingress)
			fmt.Fprintf(out, "LoadBalancer Ingress:\t%s\n", list)
		}
		for i := range service.Spec.Ports {
			sp := &service.Spec.Ports[i]

			name := sp.Name
			if name == "" {
				name = "<unnamed>"
			}
			fmt.Fprintf(out, "Port:\t%s\t%d/%s\n", name, sp.Port, sp.Protocol)
			if sp.NodePort != 0 {
				fmt.Fprintf(out, "NodePort:\t%s\t%d/%s\n", name, sp.NodePort, sp.Protocol)
			}
			fmt.Fprintf(out, "Endpoints:\t%s\n", formatEndpoints(endpoints, sets.NewString(sp.Name)))
		}
		fmt.Fprintf(out, "Session Affinity:\t%s\n", service.Spec.SessionAffinity)
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// ServiceAccountDescriber generates information about a service.
type ServiceAccountDescriber struct {
	client.Interface
}

func (d *ServiceAccountDescriber) Describe(namespace, name string) (string, error) {
	c := d.ServiceAccounts(namespace)

	serviceAccount, err := c.Get(name)
	if err != nil {
		return "", err
	}

	tokens := []api.Secret{}

	tokenSelector := fields.SelectorFromSet(map[string]string{client.SecretType: string(api.SecretTypeServiceAccountToken)})
	secrets, err := d.Secrets(namespace).List(labels.Everything(), tokenSelector)
	if err == nil {
		for _, s := range secrets.Items {
			name, _ := s.Annotations[api.ServiceAccountNameKey]
			uid, _ := s.Annotations[api.ServiceAccountUIDKey]
			if name == serviceAccount.Name && uid == string(serviceAccount.UID) {
				tokens = append(tokens, s)
			}
		}
	}

	return describeServiceAccount(serviceAccount, tokens)
}

func describeServiceAccount(serviceAccount *api.ServiceAccount, tokens []api.Secret) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", serviceAccount.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", serviceAccount.Namespace)
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(serviceAccount.Labels))
		fmt.Fprintln(out)

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
				fmt.Fprintf(out, "%s\t<none>\n", header)
			} else {
				prefix := header
				for _, name := range names {
					fmt.Fprintf(out, "%s\t%s\n", prefix, name)
					prefix = emptyHeader
				}
			}
			fmt.Fprintln(out)
		}

		return nil
	})
}

// NodeDescriber generates information about a node.
type NodeDescriber struct {
	client.Interface
}

func (d *NodeDescriber) Describe(namespace, name string) (string, error) {
	mc := d.Nodes()
	node, err := mc.Get(name)
	if err != nil {
		return "", err
	}

	var pods []*api.Pod
	allPods, err := d.Pods(namespace).List(labels.Everything(), fields.Everything())
	if err != nil {
		return "", err
	}
	for i := range allPods.Items {
		pod := &allPods.Items[i]
		if pod.Spec.NodeName != name {
			continue
		}
		pods = append(pods, pod)
	}

	var events *api.EventList
	if ref, err := api.GetReference(node); err != nil {
		glog.Errorf("Unable to construct reference to '%#v': %v", node, err)
	} else {
		// TODO: We haven't decided the namespace for Node object yet.
		ref.UID = types.UID(ref.Name)
		events, _ = d.Events("").Search(ref)
	}

	return describeNode(node, pods, events)
}

func describeNode(node *api.Node, pods []*api.Pod, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", node.Name)
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(node.Labels))
		fmt.Fprintf(out, "CreationTimestamp:\t%s\n", node.CreationTimestamp.Time.Format(time.RFC1123Z))
		fmt.Fprintf(out, "Phase:\t%v\n", node.Status.Phase)
		if len(node.Status.Conditions) > 0 {
			fmt.Fprint(out, "Conditions:\n  Type\tStatus\tLastHeartbeatTime\tLastTransitionTime\tReason\tMessage\n")
			fmt.Fprint(out, "  ────\t──────\t─────────────────\t──────────────────\t──────\t───────\n")
			for _, c := range node.Status.Conditions {
				fmt.Fprintf(out, "  %v \t%v \t%s \t%s \t%v \t%v\n",
					c.Type,
					c.Status,
					c.LastHeartbeatTime.Time.Format(time.RFC1123Z),
					c.LastTransitionTime.Time.Format(time.RFC1123Z),
					c.Reason,
					c.Message)
			}
		}
		var addresses []string
		for _, address := range node.Status.Addresses {
			addresses = append(addresses, address.Address)
		}
		fmt.Fprintf(out, "Addresses:\t%s\n", strings.Join(addresses, ","))
		if len(node.Status.Capacity) > 0 {
			fmt.Fprintf(out, "Capacity:\n")
			for resource, value := range node.Status.Capacity {
				fmt.Fprintf(out, " %s:\t%s\n", resource, value.String())
			}
		}

		fmt.Fprintf(out, "System Info:\n")
		fmt.Fprintf(out, " Machine ID:\t%s\n", node.Status.NodeInfo.MachineID)
		fmt.Fprintf(out, " System UUID:\t%s\n", node.Status.NodeInfo.SystemUUID)
		fmt.Fprintf(out, " Boot ID:\t%s\n", node.Status.NodeInfo.BootID)
		fmt.Fprintf(out, " Kernel Version:\t%s\n", node.Status.NodeInfo.KernelVersion)
		fmt.Fprintf(out, " OS Image:\t%s\n", node.Status.NodeInfo.OsImage)
		fmt.Fprintf(out, " Container Runtime Version:\t%s\n", node.Status.NodeInfo.ContainerRuntimeVersion)
		fmt.Fprintf(out, " Kubelet Version:\t%s\n", node.Status.NodeInfo.KubeletVersion)
		fmt.Fprintf(out, " Kube-Proxy Version:\t%s\n", node.Status.NodeInfo.KubeProxyVersion)

		if len(node.Spec.PodCIDR) > 0 {
			fmt.Fprintf(out, "PodCIDR:\t%s\n", node.Spec.PodCIDR)
		}
		if len(node.Spec.ExternalID) > 0 {
			fmt.Fprintf(out, "ExternalID:\t%s\n", node.Spec.ExternalID)
		}
		if err := describeNodeResource(pods, node, out); err != nil {
			return err
		}

		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// HorizontalPodAutoscalerDescriber generates information about a horizontal pod autoscaler.
type HorizontalPodAutoscalerDescriber struct {
	client *client.Client
}

func (d *HorizontalPodAutoscalerDescriber) Describe(namespace, name string) (string, error) {
	hpa, err := d.client.Extensions().HorizontalPodAutoscalers(namespace).Get(name)
	if err != nil {
		return "", err
	}
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", hpa.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", hpa.Namespace)
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(hpa.Labels))
		fmt.Fprintf(out, "CreationTimestamp:\t%s\n", hpa.CreationTimestamp.Time.Format(time.RFC1123Z))
		fmt.Fprintf(out, "Reference:\t%s/%s/%s/%s\n",
			hpa.Spec.ScaleRef.Kind,
			hpa.Spec.ScaleRef.Namespace,
			hpa.Spec.ScaleRef.Name,
			hpa.Spec.ScaleRef.Subresource)
		if hpa.Spec.CPUUtilization != nil {
			fmt.Fprintf(out, "Target CPU utilization:\t%d%%\n", hpa.Spec.CPUUtilization.TargetPercentage)
			fmt.Fprintf(out, "Current CPU utilization:\t")
			if hpa.Status.CurrentCPUUtilizationPercentage != nil {
				fmt.Fprintf(out, "%d%%\n", *hpa.Status.CurrentCPUUtilizationPercentage)
			} else {
				fmt.Fprintf(out, "<not available>\n")
			}
		}
		fmt.Fprintf(out, "Min pods:\t%d\n", hpa.Spec.MinReplicas)
		fmt.Fprintf(out, "Max pods:\t%d\n", hpa.Spec.MaxReplicas)

		// TODO: switch to scale subresource once the required code is submitted.
		if strings.ToLower(hpa.Spec.ScaleRef.Kind) == "replicationcontroller" {
			fmt.Fprintf(out, "ReplicationController pods:\t")
			rc, err := d.client.ReplicationControllers(hpa.Spec.ScaleRef.Namespace).Get(hpa.Spec.ScaleRef.Name)
			if err == nil {
				fmt.Fprintf(out, "%d current / %d desired\n", rc.Status.Replicas, rc.Spec.Replicas)
			} else {
				fmt.Fprintf(out, "failed to check Replication Controller\n")
			}
		}
		return nil
	})
}

func describeNodeResource(pods []*api.Pod, node *api.Node, out io.Writer) error {
	nonTerminatedPods := filterTerminatedPods(pods)
	fmt.Fprintf(out, "Non-terminated Pods:\t(%d in total)\n", len(nonTerminatedPods))
	fmt.Fprint(out, "  Namespace\tName\t\tCPU Requests\tCPU Limits\tMemory Requests\tMemory Limits\n")
	fmt.Fprint(out, "  ─────────\t────\t\t────────────\t──────────\t───────────────\t─────────────\n")
	for _, pod := range nonTerminatedPods {
		req, limit, err := getSinglePodTotalRequestsAndLimits(pod)
		if err != nil {
			return err
		}
		cpuReq, cpuLimit, memoryReq, memoryLimit := req[api.ResourceCPU], limit[api.ResourceCPU], req[api.ResourceMemory], limit[api.ResourceMemory]
		fractionCpuReq := float64(cpuReq.MilliValue()) / float64(node.Status.Capacity.Cpu().MilliValue()) * 100
		fractionCpuLimit := float64(cpuLimit.MilliValue()) / float64(node.Status.Capacity.Cpu().MilliValue()) * 100
		fractionMemoryReq := float64(memoryReq.MilliValue()) / float64(node.Status.Capacity.Memory().MilliValue()) * 100
		fractionMemoryLimit := float64(memoryLimit.MilliValue()) / float64(node.Status.Capacity.Memory().MilliValue()) * 100
		fmt.Fprintf(out, "  %s\t%s\t\t%s (%d%%)\t%s (%d%%)\t%s (%d%%)\t%s (%d%%)\n", pod.Namespace, pod.Name,
			cpuReq.String(), int64(fractionCpuReq), cpuLimit.String(), int64(fractionCpuLimit),
			memoryReq.String(), int64(fractionMemoryReq), memoryLimit.String(), int64(fractionMemoryLimit))
	}

	fmt.Fprint(out, "Allocated resources:\n  (Total limits may be over 100%, i.e., overcommitted. More info: http://releases.k8s.io/HEAD/docs/user-guide/compute-resources.md)\n  CPU Requests\tCPU Limits\tMemory Requests\tMemory Limits\n")
	fmt.Fprint(out, "  ────────────\t──────────\t───────────────\t─────────────\n")
	reqs, limits, err := getPodsTotalRequestsAndLimits(nonTerminatedPods)
	if err != nil {
		return err
	}
	cpuReqs, cpuLimits, memoryReqs, memoryLimits := reqs[api.ResourceCPU], limits[api.ResourceCPU], reqs[api.ResourceMemory], limits[api.ResourceMemory]
	fractionCpuReqs := float64(cpuReqs.MilliValue()) / float64(node.Status.Capacity.Cpu().MilliValue()) * 100
	fractionCpuLimits := float64(cpuLimits.MilliValue()) / float64(node.Status.Capacity.Cpu().MilliValue()) * 100
	fractionMemoryReqs := float64(memoryReqs.MilliValue()) / float64(node.Status.Capacity.Memory().MilliValue()) * 100
	fractionMemoryLimits := float64(memoryLimits.MilliValue()) / float64(node.Status.Capacity.Memory().MilliValue()) * 100
	fmt.Fprintf(out, "  %s (%d%%)\t%s (%d%%)\t%s (%d%%)\t%s (%d%%)\n",
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

func getPodsTotalRequestsAndLimits(pods []*api.Pod) (reqs map[api.ResourceName]resource.Quantity, limits map[api.ResourceName]resource.Quantity, err error) {
	reqs, limits = map[api.ResourceName]resource.Quantity{}, map[api.ResourceName]resource.Quantity{}
	for _, pod := range pods {
		podReqs, podLimits, err := getSinglePodTotalRequestsAndLimits(pod)
		if err != nil {
			return nil, nil, err
		}
		for podReqName, podReqValue := range podReqs {
			if value, ok := reqs[podReqName]; !ok {
				reqs[podReqName] = *podReqValue.Copy()
			} else if err = value.Add(podReqValue); err != nil {
				return nil, nil, err
			}
		}
		for podLimitName, podLimitValue := range podLimits {
			if value, ok := limits[podLimitName]; !ok {
				limits[podLimitName] = *podLimitValue.Copy()
			} else if err = value.Add(podLimitValue); err != nil {
				return nil, nil, err
			}
		}
	}
	return
}

func getSinglePodTotalRequestsAndLimits(pod *api.Pod) (reqs map[api.ResourceName]resource.Quantity, limits map[api.ResourceName]resource.Quantity, err error) {
	reqs, limits = map[api.ResourceName]resource.Quantity{}, map[api.ResourceName]resource.Quantity{}
	for _, container := range pod.Spec.Containers {
		for name, quantity := range container.Resources.Requests {
			if value, ok := reqs[name]; !ok {
				reqs[name] = *quantity.Copy()
			} else if err = value.Add(quantity); err != nil {
				return nil, nil, err
			}
		}
		for name, quantity := range container.Resources.Limits {
			if value, ok := limits[name]; !ok {
				limits[name] = *quantity.Copy()
			} else if err = value.Add(quantity); err != nil {
				return nil, nil, err
			}
		}
	}
	return
}

func DescribeEvents(el *api.EventList, w io.Writer) {
	if len(el.Items) == 0 {
		fmt.Fprint(w, "No events.")
		return
	}
	sort.Sort(SortableEvents(el.Items))
	fmt.Fprint(w, "Events:\n  FirstSeen\tLastSeen\tCount\tFrom\tSubobjectPath\tReason\tMessage\n")
	fmt.Fprint(w, "  ─────────\t────────\t─────\t────\t─────────────\t──────\t───────\n")
	for _, e := range el.Items {
		fmt.Fprintf(w, "  %s\t%s\t%d\t%v\t%v\t%v\t%v\n",
			translateTimestamp(e.FirstTimestamp),
			translateTimestamp(e.LastTimestamp),
			e.Count,
			e.Source,
			e.InvolvedObject.FieldPath,
			e.Reason,
			e.Message)
	}
}

// DeploymentDescriber generates information about a deployment.
type DeploymentDescriber struct {
	client.Interface
}

func (dd *DeploymentDescriber) Describe(namespace, name string) (string, error) {
	d, err := dd.Extensions().Deployments(namespace).Get(name)
	if err != nil {
		return "", err
	}
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", d.ObjectMeta.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", d.ObjectMeta.Namespace)
		fmt.Fprintf(out, "CreationTimestamp:\t%s\n", d.CreationTimestamp.Time.Format(time.RFC1123Z))
		fmt.Fprintf(out, "Labels:\t%s\n", labels.FormatLabels(d.Labels))
		fmt.Fprintf(out, "Selector:\t%s\n", labels.FormatLabels(d.Spec.Selector))
		fmt.Fprintf(out, "Replicas:\t%d updated / %d total\n", d.Status.UpdatedReplicas, d.Spec.Replicas)
		fmt.Fprintf(out, "StrategyType:\t%s\n", d.Spec.Strategy.Type)
		if d.Spec.Strategy.RollingUpdate != nil {
			ru := d.Spec.Strategy.RollingUpdate
			fmt.Fprintf(out, "RollingUpdateStrategy:\t%s max unavailable, %s max surge, %d min ready seconds\n", ru.MaxUnavailable.String(), ru.MaxSurge.String(), ru.MinReadySeconds)
		}
		oldRCs, err := deploymentUtil.GetOldRCs(*d, dd)
		if err == nil {
			fmt.Fprintf(out, "OldReplicationControllers:\t%s\n", printReplicationControllersByLabels(oldRCs))
		}
		newRC, err := deploymentUtil.GetNewRC(*d, dd)
		if err == nil {
			var newRCs []*api.ReplicationController
			if newRC != nil {
				newRCs = append(newRCs, newRC)
			}
			fmt.Fprintf(out, "NewReplicationController:\t%s\n", printReplicationControllersByLabels(newRCs))
		}
		events, err := dd.Events(namespace).Search(d)
		if err == nil && events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// Get all daemon set whose selectors would match a given set of labels.
// TODO: Move this to pkg/client and ideally implement it server-side (instead
// of getting all DS's and searching through them manually).
// TODO: write an interface for controllers and fuse getReplicationControllersForLabels
// and getDaemonSetsForLabels.
func getDaemonSetsForLabels(c client.DaemonSetInterface, labelsToMatch labels.Labels) ([]extensions.DaemonSet, error) {
	// Get all daemon sets
	// TODO: this needs a namespace scope as argument
	dss, err := c.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("error getting daemon set: %v", err)
	}

	// Find the ones that match labelsToMatch.
	var matchingDaemonSets []extensions.DaemonSet
	for _, ds := range dss.Items {
		selector := labels.SelectorFromSet(ds.Spec.Selector)
		if selector.Matches(labelsToMatch) {
			matchingDaemonSets = append(matchingDaemonSets, ds)
		}
	}
	return matchingDaemonSets, nil
}

// Get all replication controllers whose selectors would match a given set of
// labels.
// TODO Move this to pkg/client and ideally implement it server-side (instead
// of getting all RC's and searching through them manually).
func getReplicationControllersForLabels(c client.ReplicationControllerInterface, labelsToMatch labels.Labels) ([]api.ReplicationController, error) {
	// Get all replication controllers.
	// TODO this needs a namespace scope as argument
	rcs, err := c.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("error getting replication controllers: %v", err)
	}

	// Find the ones that match labelsToMatch.
	var matchingRCs []api.ReplicationController
	for _, controller := range rcs.Items {
		selector := labels.SelectorFromSet(controller.Spec.Selector)
		if selector.Matches(labelsToMatch) {
			matchingRCs = append(matchingRCs, controller)
		}
	}
	return matchingRCs, nil
}

func printReplicationControllersByLabels(matchingRCs []*api.ReplicationController) string {
	// Format the matching RC's into strings.
	var rcStrings []string
	for _, controller := range matchingRCs {
		rcStrings = append(rcStrings, fmt.Sprintf("%s (%d/%d replicas created)", controller.Name, controller.Status.Replicas, controller.Spec.Replicas))
	}

	list := strings.Join(rcStrings, ", ")
	if list == "" {
		return "<none>"
	}
	return list
}

func getPodStatusForController(c client.PodInterface, selector map[string]string) (running, waiting, succeeded, failed int, err error) {
	rcPods, err := c.List(labels.SelectorFromSet(selector), fields.Everything())
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

// newErrNoDescriber creates a new ErrNoDescriber with the names of the provided types.
func newErrNoDescriber(types ...reflect.Type) error {
	names := []string{}
	for _, t := range types {
		names = append(names, t.String())
	}
	return ErrNoDescriber{Types: names}
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

	types := []reflect.Type{}
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

// Add adds one or more describer functions to the Describer. The passed function must
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
		if ft.NumIn() == 0 {
			return fmt.Errorf("expected at least one 'in' params, got: %v", ft)
		}
		if ft.NumOut() != 2 {
			return fmt.Errorf("expected two 'out' params - (string, error), got: %v", ft)
		}
		types := []reflect.Type{}
		for i := 0; i < ft.NumIn(); i++ {
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
// TODO: allow unordered types to be matched and reorderd.
func (fn typeFunc) Matches(types []reflect.Type) bool {
	if len(fn.Extra) != len(types) {
		return false
	}
	for i := range types {
		if fn.Extra[i] != types[i] {
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
