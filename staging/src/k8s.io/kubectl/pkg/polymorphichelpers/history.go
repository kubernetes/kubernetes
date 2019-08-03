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

package polymorphichelpers

import (
	"bytes"
	"fmt"
	"io"
	"text/tabwriter"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/client-go/kubernetes"
	clientappsv1 "k8s.io/client-go/kubernetes/typed/apps/v1"
	"k8s.io/kubectl/pkg/apps"
	describe "k8s.io/kubectl/pkg/describe/versioned"
	deploymentutil "k8s.io/kubectl/pkg/util/deployment"
	sliceutil "k8s.io/kubectl/pkg/util/slice"
)

const (
	ChangeCauseAnnotation = "kubernetes.io/change-cause"
)

// HistoryViewer provides an interface for resources have historical information.
type HistoryViewer interface {
	ViewHistory(namespace, name string, revision int64) (string, error)
}

type HistoryVisitor struct {
	clientset kubernetes.Interface
	result    HistoryViewer
}

func (v *HistoryVisitor) VisitDeployment(elem apps.GroupKindElement) {
	v.result = &DeploymentHistoryViewer{v.clientset}
}

func (v *HistoryVisitor) VisitStatefulSet(kind apps.GroupKindElement) {
	v.result = &StatefulSetHistoryViewer{v.clientset}
}

func (v *HistoryVisitor) VisitDaemonSet(kind apps.GroupKindElement) {
	v.result = &DaemonSetHistoryViewer{v.clientset}
}

func (v *HistoryVisitor) VisitJob(kind apps.GroupKindElement)                   {}
func (v *HistoryVisitor) VisitPod(kind apps.GroupKindElement)                   {}
func (v *HistoryVisitor) VisitReplicaSet(kind apps.GroupKindElement)            {}
func (v *HistoryVisitor) VisitReplicationController(kind apps.GroupKindElement) {}
func (v *HistoryVisitor) VisitCronJob(kind apps.GroupKindElement)               {}

// HistoryViewerFor returns an implementation of HistoryViewer interface for the given schema kind
func HistoryViewerFor(kind schema.GroupKind, c kubernetes.Interface) (HistoryViewer, error) {
	elem := apps.GroupKindElement(kind)
	visitor := &HistoryVisitor{
		clientset: c,
	}

	// Determine which HistoryViewer we need here
	err := elem.Accept(visitor)

	if err != nil {
		return nil, fmt.Errorf("error retrieving history for %q, %v", kind.String(), err)
	}

	if visitor.result == nil {
		return nil, fmt.Errorf("no history viewer has been implemented for %q", kind.String())
	}

	return visitor.result, nil
}

type DeploymentHistoryViewer struct {
	c kubernetes.Interface
}

// ViewHistory returns a revision-to-replicaset map as the revision history of a deployment
// TODO: this should be a describer
func (h *DeploymentHistoryViewer) ViewHistory(namespace, name string, revision int64) (string, error) {
	versionedAppsClient := h.c.AppsV1()
	deployment, err := versionedAppsClient.Deployments(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to retrieve deployment %s: %v", name, err)
	}
	_, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(deployment, versionedAppsClient)
	if err != nil {
		return "", fmt.Errorf("failed to retrieve replica sets from deployment %s: %v", name, err)
	}
	allRSs := allOldRSs
	if newRS != nil {
		allRSs = append(allRSs, newRS)
	}

	historyInfo := make(map[int64]*corev1.PodTemplateSpec)
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

func printTemplate(template *corev1.PodTemplateSpec) (string, error) {
	buf := bytes.NewBuffer([]byte{})
	w := describe.NewPrefixWriter(buf)
	describe.DescribePodTemplate(template, w)
	return buf.String(), nil
}

type DaemonSetHistoryViewer struct {
	c kubernetes.Interface
}

// ViewHistory returns a revision-to-history map as the revision history of a deployment
// TODO: this should be a describer
func (h *DaemonSetHistoryViewer) ViewHistory(namespace, name string, revision int64) (string, error) {
	ds, history, err := daemonSetHistory(h.c.AppsV1(), namespace, name)
	if err != nil {
		return "", err
	}
	historyInfo := make(map[int64]*appsv1.ControllerRevision)
	for _, history := range history {
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
		dsOfHistory, err := applyDaemonSetHistory(ds, history)
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
	c kubernetes.Interface
}

// ViewHistory returns a list of the revision history of a statefulset
// TODO: this should be a describer
// TODO: needs to implement detailed revision view
func (h *StatefulSetHistoryViewer) ViewHistory(namespace, name string, revision int64) (string, error) {
	_, history, err := statefulSetHistory(h.c.AppsV1(), namespace, name)
	if err != nil {
		return "", err
	}

	if len(history) <= 0 {
		return "No rollout history found.", nil
	}
	revisions := make([]int64, len(history))
	for _, revision := range history {
		revisions = append(revisions, revision.Revision)
	}
	sliceutil.SortInts64(revisions)

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "REVISION\n")
		for _, r := range revisions {
			fmt.Fprintf(out, "%d\n", r)
		}
		return nil
	})
}

// controlledHistories returns all ControllerRevisions in namespace that selected by selector and owned by accessor
// TODO: Rename this to controllerHistory when other controllers have been upgraded
func controlledHistoryV1(
	apps clientappsv1.AppsV1Interface,
	namespace string,
	selector labels.Selector,
	accessor metav1.Object) ([]*appsv1.ControllerRevision, error) {
	var result []*appsv1.ControllerRevision
	historyList, err := apps.ControllerRevisions(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return nil, err
	}
	for i := range historyList.Items {
		history := historyList.Items[i]
		// Only add history that belongs to the API object
		if metav1.IsControlledBy(&history, accessor) {
			result = append(result, &history)
		}
	}
	return result, nil
}

// controlledHistories returns all ControllerRevisions in namespace that selected by selector and owned by accessor
func controlledHistory(
	apps clientappsv1.AppsV1Interface,
	namespace string,
	selector labels.Selector,
	accessor metav1.Object) ([]*appsv1.ControllerRevision, error) {
	var result []*appsv1.ControllerRevision
	historyList, err := apps.ControllerRevisions(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return nil, err
	}
	for i := range historyList.Items {
		history := historyList.Items[i]
		// Only add history that belongs to the API object
		if metav1.IsControlledBy(&history, accessor) {
			result = append(result, &history)
		}
	}
	return result, nil
}

// daemonSetHistory returns the DaemonSet named name in namespace and all ControllerRevisions in its history.
func daemonSetHistory(
	apps clientappsv1.AppsV1Interface,
	namespace, name string) (*appsv1.DaemonSet, []*appsv1.ControllerRevision, error) {
	ds, err := apps.DaemonSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to retrieve DaemonSet %s: %v", name, err)
	}
	selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create selector for DaemonSet %s: %v", ds.Name, err)
	}
	accessor, err := meta.Accessor(ds)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create accessor for DaemonSet %s: %v", ds.Name, err)
	}
	history, err := controlledHistory(apps, ds.Namespace, selector, accessor)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to find history controlled by DaemonSet %s: %v", ds.Name, err)
	}
	return ds, history, nil
}

// statefulSetHistory returns the StatefulSet named name in namespace and all ControllerRevisions in its history.
func statefulSetHistory(
	apps clientappsv1.AppsV1Interface,
	namespace, name string) (*appsv1.StatefulSet, []*appsv1.ControllerRevision, error) {
	sts, err := apps.StatefulSets(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to retrieve Statefulset %s: %s", name, err.Error())
	}
	selector, err := metav1.LabelSelectorAsSelector(sts.Spec.Selector)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create selector for StatefulSet %s: %s", name, err.Error())
	}
	accessor, err := meta.Accessor(sts)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to obtain accessor for StatefulSet %s: %s", name, err.Error())
	}
	history, err := controlledHistoryV1(apps, namespace, selector, accessor)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to find history controlled by StatefulSet %s: %v", name, err)
	}
	return sts, history, nil
}

// applyDaemonSetHistory returns a specific revision of DaemonSet by applying the given history to a copy of the given DaemonSet
func applyDaemonSetHistory(ds *appsv1.DaemonSet, history *appsv1.ControllerRevision) (*appsv1.DaemonSet, error) {
	clone := ds.DeepCopy()
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
