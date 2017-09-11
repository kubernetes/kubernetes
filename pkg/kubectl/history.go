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

package kubectl

import (
	"bytes"
	"fmt"
	"io"
	"text/tabwriter"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	clientappsv1beta1 "k8s.io/client-go/kubernetes/typed/apps/v1beta1"
	clientextensionsv1beta1 "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	sliceutil "k8s.io/kubernetes/pkg/kubectl/util/slice"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
)

const (
	ChangeCauseAnnotation = "kubernetes.io/change-cause"
)

// HistoryViewer provides an interface for resources have historical information.
type HistoryViewer interface {
	ViewHistory(namespace, name string, revision int64) (string, error)
}

func HistoryViewerFor(kind schema.GroupKind, c clientset.Interface) (HistoryViewer, error) {
	switch kind {
	case extensions.Kind("Deployment"), apps.Kind("Deployment"):
		return &DeploymentHistoryViewer{c}, nil
	case apps.Kind("StatefulSet"):
		return &StatefulSetHistoryViewer{c}, nil
	case extensions.Kind("DaemonSet"), apps.Kind("DaemonSet"):
		return &DaemonSetHistoryViewer{c}, nil
	}
	return nil, fmt.Errorf("no history viewer has been implemented for %q", kind)
}

type DeploymentHistoryViewer struct {
	c clientset.Interface
}

// ViewHistory returns a revision-to-replicaset map as the revision history of a deployment
// TODO: this should be a describer
func (h *DeploymentHistoryViewer) ViewHistory(namespace, name string, revision int64) (string, error) {
	versionedExtensionsClient := versionedExtensionsClientV1beta1(h.c)
	deployment, err := versionedExtensionsClient.Deployments(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to retrieve deployment %s: %v", name, err)
	}
	_, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(deployment, versionedExtensionsClient)
	if err != nil {
		return "", fmt.Errorf("failed to retrieve replica sets from deployment %s: %v", name, err)
	}
	allRSs := allOldRSs
	if newRS != nil {
		allRSs = append(allRSs, newRS)
	}

	historyInfo := make(map[int64]*v1.PodTemplateSpec)
	for _, rs := range allRSs {
		v, err := deploymentutil.Revision(rs)
		if err != nil {
			continue
		}
		historyInfo[v] = &rs.Spec.Template
		changeCause := getChangeCause(rs)
		if historyInfo[v].Annotations == nil {
			historyInfo[v].Annotations = make(map[string]string)
		}
		if len(changeCause) > 0 {
			historyInfo[v].Annotations[ChangeCauseAnnotation] = changeCause
		}
	}

	if len(historyInfo) == 0 {
		return "No rollout history found.", nil
	}

	if revision > 0 {
		// Print details of a specific revision
		template, ok := historyInfo[revision]
		if !ok {
			return "", fmt.Errorf("unable to find the specified revision")
		}
		return printTemplate(template)
	}

	// Sort the revisionToChangeCause map by revision
	revisions := make([]int64, 0, len(historyInfo))
	for r := range historyInfo {
		revisions = append(revisions, r)
	}
	sliceutil.SortInts64(revisions)

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "REVISION\tCHANGE-CAUSE\n")
		for _, r := range revisions {
			// Find the change-cause of revision r
			changeCause := historyInfo[r].Annotations[ChangeCauseAnnotation]
			if len(changeCause) == 0 {
				changeCause = "<none>"
			}
			fmt.Fprintf(out, "%d\t%s\n", r, changeCause)
		}
		return nil
	})
}

func printTemplate(template *v1.PodTemplateSpec) (string, error) {
	buf := bytes.NewBuffer([]byte{})
	internalTemplate := &api.PodTemplateSpec{}
	if err := apiv1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(template, internalTemplate, nil); err != nil {
		return "", fmt.Errorf("failed to convert podtemplate, %v", err)
	}
	w := printersinternal.NewPrefixWriter(buf)
	printersinternal.DescribePodTemplate(internalTemplate, w)
	return buf.String(), nil
}

type DaemonSetHistoryViewer struct {
	c clientset.Interface
}

// ViewHistory returns a revision-to-history map as the revision history of a deployment
// TODO: this should be a describer
func (h *DaemonSetHistoryViewer) ViewHistory(namespace, name string, revision int64) (string, error) {
	versionedAppsClient := versionedAppsClientV1beta1(h.c)
	versionedExtensionsClient := versionedExtensionsClientV1beta1(h.c)
	versionedObj, allHistory, err := controlledHistories(versionedAppsClient, versionedExtensionsClient, namespace, name, "DaemonSet")
	if err != nil {
		return "", fmt.Errorf("unable to find history controlled by DaemonSet %s: %v", name, err)
	}
	historyInfo := make(map[int64]*appsv1beta1.ControllerRevision)
	for _, history := range allHistory {
		// TODO: for now we assume revisions don't overlap, we may need to handle it
		historyInfo[history.Revision] = history
	}

	if len(historyInfo) == 0 {
		return "No rollout history found.", nil
	}

	// Print details of a specific revision
	if revision > 0 {
		history, ok := historyInfo[revision]
		if !ok {
			return "", fmt.Errorf("unable to find the specified revision")
		}

		versionedDS, ok := versionedObj.(*extensionsv1beta1.DaemonSet)
		if !ok {
			return "", fmt.Errorf("unexpected non-DaemonSet object returned: %v", versionedDS)
		}

		dsOfHistory, err := applyHistory(versionedDS, history)
		if err != nil {
			return "", fmt.Errorf("unable to parse history %s", history.Name)
		}
		return printTemplate(&dsOfHistory.Spec.Template)
	}

	// Print an overview of all Revisions
	// Sort the revisionToChangeCause map by revision
	revisions := make([]int64, 0, len(historyInfo))
	for r := range historyInfo {
		revisions = append(revisions, r)
	}
	sliceutil.SortInts64(revisions)

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "REVISION\tCHANGE-CAUSE\n")
		for _, r := range revisions {
			// Find the change-cause of revision r
			changeCause := historyInfo[r].Annotations[ChangeCauseAnnotation]
			if len(changeCause) == 0 {
				changeCause = "<none>"
			}
			fmt.Fprintf(out, "%d\t%s\n", r, changeCause)
		}
		return nil
	})
}

type StatefulSetHistoryViewer struct {
	c clientset.Interface
}

func getOwner(revision apps.ControllerRevision) *metav1.OwnerReference {
	ownerRefs := revision.GetOwnerReferences()
	for i := range ownerRefs {
		owner := &ownerRefs[i]
		if owner.Controller != nil && *owner.Controller == true {
			return owner
		}
	}
	return nil
}

// ViewHistory returns a list of the revision history of a statefulset
// TODO: this should be a describer
// TODO: needs to implement detailed revision view
func (h *StatefulSetHistoryViewer) ViewHistory(namespace, name string, revision int64) (string, error) {

	sts, err := h.c.Apps().StatefulSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to retrieve statefulset %s", err)
	}
	selector, err := metav1.LabelSelectorAsSelector(sts.Spec.Selector)
	if err != nil {
		return "", fmt.Errorf("failed to retrieve statefulset history %s", err)
	}
	revisions, err := h.c.Apps().ControllerRevisions(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return "", fmt.Errorf("failed to retrieve statefulset history %s", err)
	}
	if len(revisions.Items) <= 0 {
		return "No rollout history found.", nil
	}
	revisionNumbers := make([]int64, len(revisions.Items))
	for i := range revisions.Items {
		if owner := getOwner(revisions.Items[i]); owner != nil && owner.UID == sts.UID {
			revisionNumbers[i] = revisions.Items[i].Revision
		}
	}
	sliceutil.SortInts64(revisionNumbers)

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "REVISION\n")
		for _, r := range revisionNumbers {
			fmt.Fprintf(out, "%d\n", r)
		}
		return nil
	})
}

// controlledHistories returns all ControllerRevisions controlled by the given API object
func controlledHistories(apps clientappsv1beta1.AppsV1beta1Interface, extensions clientextensionsv1beta1.ExtensionsV1beta1Interface, namespace, name, kind string) (runtime.Object, []*appsv1beta1.ControllerRevision, error) {
	var obj runtime.Object
	var labelSelector *metav1.LabelSelector

	switch kind {
	case "DaemonSet":
		ds, err := extensions.DaemonSets(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return nil, nil, fmt.Errorf("failed to retrieve DaemonSet %s: %v", name, err)
		}
		labelSelector = ds.Spec.Selector
		obj = ds
	case "StatefulSet":
		ss, err := apps.StatefulSets(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return nil, nil, fmt.Errorf("failed to retrieve StatefulSet %s: %v", name, err)
		}
		labelSelector = ss.Spec.Selector
		obj = ss
	default:
		return nil, nil, fmt.Errorf("unsupported API object kind: %s", kind)
	}

	var result []*appsv1beta1.ControllerRevision
	selector, err := metav1.LabelSelectorAsSelector(labelSelector)
	if err != nil {
		return nil, nil, err
	}
	historyList, err := apps.ControllerRevisions(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return nil, nil, err
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to obtain accessor for %s named %s: %v", kind, name, err)
	}
	for i := range historyList.Items {
		history := historyList.Items[i]
		// Only add history that belongs to the API object
		if metav1.IsControlledBy(&history, accessor) {
			result = append(result, &history)
		}
	}
	return obj, result, nil
}

// applyHistory returns a specific revision of DaemonSet by applying the given history to a copy of the given DaemonSet
func applyHistory(ds *extensionsv1beta1.DaemonSet, history *appsv1beta1.ControllerRevision) (*extensionsv1beta1.DaemonSet, error) {
	obj, err := api.Scheme.New(ds.GroupVersionKind())
	if err != nil {
		return nil, err
	}
	clone := obj.(*extensionsv1beta1.DaemonSet)
	cloneBytes, err := json.Marshal(clone)
	if err != nil {
		return nil, err
	}
	patched, err := strategicpatch.StrategicMergePatch(cloneBytes, history.Data.Raw, clone)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(patched, clone)
	if err != nil {
		return nil, err
	}
	return clone, nil
}

// TODO: copied here until this becomes a describer
func tabbedString(f func(io.Writer) error) (string, error) {
	out := new(tabwriter.Writer)
	buf := &bytes.Buffer{}
	out.Init(buf, 0, 8, 2, ' ', 0)

	err := f(out)
	if err != nil {
		return "", err
	}

	out.Flush()
	str := string(buf.String())
	return str, nil
}

// getChangeCause returns the change-cause annotation of the input object
func getChangeCause(obj runtime.Object) string {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return ""
	}
	return accessor.GetAnnotations()[ChangeCauseAnnotation]
}
