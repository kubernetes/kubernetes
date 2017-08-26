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
	"os"
	"os/signal"
	"sort"
	"syscall"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	externalextensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/daemon"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/controller/statefulset"
	sliceutil "k8s.io/kubernetes/pkg/kubectl/util/slice"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
)

const (
	rollbackSuccess = "rolled back"
	rollbackSkipped = "skipped rollback"
)

// Rollbacker provides an interface for resources that can be rolled back.
type Rollbacker interface {
	Rollback(obj runtime.Object, updatedAnnotations map[string]string, toRevision int64, dryRun bool) (string, error)
}

func RollbackerFor(kind schema.GroupKind, c clientset.Interface) (Rollbacker, error) {
	switch kind {
	case extensions.Kind("Deployment"), apps.Kind("Deployment"):
		return &DeploymentRollbacker{c}, nil
	case extensions.Kind("DaemonSet"), apps.Kind("DaemonSet"):
		return &DaemonSetRollbacker{c}, nil
	case apps.Kind("StatefulSet"):
		return &StatefulSetRollbacker{c}, nil
	}
	return nil, fmt.Errorf("no rollbacker has been implemented for %q", kind)
}

type DeploymentRollbacker struct {
	c clientset.Interface
}

func (r *DeploymentRollbacker) Rollback(obj runtime.Object, updatedAnnotations map[string]string, toRevision int64, dryRun bool) (string, error) {
	d, ok := obj.(*extensions.Deployment)
	if !ok {
		return "", fmt.Errorf("passed object is not a Deployment: %#v", obj)
	}
	if dryRun {
		return simpleDryRun(d, r.c, toRevision)
	}
	if d.Spec.Paused {
		return "", fmt.Errorf("you cannot rollback a paused deployment; resume it first with 'kubectl rollout resume deployment/%s' and try again", d.Name)
	}
	deploymentRollback := &extensions.DeploymentRollback{
		Name:               d.Name,
		UpdatedAnnotations: updatedAnnotations,
		RollbackTo: extensions.RollbackConfig{
			Revision: toRevision,
		},
	}
	result := ""

	// Get current events
	events, err := r.c.Core().Events(d.Namespace).List(metav1.ListOptions{})
	if err != nil {
		return result, err
	}
	// Do the rollback
	if err := r.c.Extensions().Deployments(d.Namespace).Rollback(deploymentRollback); err != nil {
		return result, err
	}
	// Watch for the changes of events
	watch, err := r.c.Core().Events(d.Namespace).Watch(metav1.ListOptions{Watch: true, ResourceVersion: events.ResourceVersion})
	if err != nil {
		return result, err
	}
	result = watchRollbackEvent(watch)
	return result, err
}

// watchRollbackEvent watches for rollback events and returns rollback result
func watchRollbackEvent(w watch.Interface) string {
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, os.Kill, syscall.SIGTERM)
	for {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				return ""
			}
			obj, ok := event.Object.(*api.Event)
			if !ok {
				w.Stop()
				return ""
			}
			isRollback, result := isRollbackEvent(obj)
			if isRollback {
				w.Stop()
				return result
			}
		case <-signals:
			w.Stop()
		}
	}
}

// isRollbackEvent checks if the input event is about rollback, and returns true and
// related result string back if it is.
func isRollbackEvent(e *api.Event) (bool, string) {
	rollbackEventReasons := []string{deploymentutil.RollbackRevisionNotFound, deploymentutil.RollbackTemplateUnchanged, deploymentutil.RollbackDone}
	for _, reason := range rollbackEventReasons {
		if e.Reason == reason {
			if reason == deploymentutil.RollbackDone {
				return true, rollbackSuccess
			}
			return true, fmt.Sprintf("%s (%s: %s)", rollbackSkipped, e.Reason, e.Message)
		}
	}
	return false, ""
}

func simpleDryRun(deployment *extensions.Deployment, c clientset.Interface, toRevision int64) (string, error) {
	externalDeployment := &externalextensions.Deployment{}
	if err := api.Scheme.Convert(deployment, externalDeployment, nil); err != nil {
		return "", fmt.Errorf("failed to convert deployment, %v", err)
	}
	versionedExtensionsClient := versionedExtensionsClientV1beta1(c)
	_, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(externalDeployment, versionedExtensionsClient)
	if err != nil {
		return "", fmt.Errorf("failed to retrieve replica sets from deployment %s: %v", deployment.Name, err)
	}
	allRSs := allOldRSs
	if newRS != nil {
		allRSs = append(allRSs, newRS)
	}

	revisionToSpec := make(map[int64]*v1.PodTemplateSpec)
	for _, rs := range allRSs {
		v, err := deploymentutil.Revision(rs)
		if err != nil {
			continue
		}
		revisionToSpec[v] = &rs.Spec.Template
	}

	if len(revisionToSpec) < 2 {
		return "", fmt.Errorf("no rollout history found for deployment %q", deployment.Name)
	}

	if toRevision > 0 {
		template, ok := revisionToSpec[toRevision]
		if !ok {
			return "", revisionNotFoundErr(toRevision)
		}
		buf := bytes.NewBuffer([]byte{})
		internalTemplate := &api.PodTemplateSpec{}
		if err := apiv1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(template, internalTemplate, nil); err != nil {
			return "", fmt.Errorf("failed to convert podtemplate, %v", err)
		}
		w := printersinternal.NewPrefixWriter(buf)
		printersinternal.DescribePodTemplate(internalTemplate, w)
		return buf.String(), nil
	}

	// Sort the revisionToSpec map by revision
	revisions := make([]int64, 0, len(revisionToSpec))
	for r := range revisionToSpec {
		revisions = append(revisions, r)
	}
	sliceutil.SortInts64(revisions)

	template, _ := revisionToSpec[revisions[len(revisions)-2]]
	buf := bytes.NewBuffer([]byte{})
	buf.WriteString("\n")
	internalTemplate := &api.PodTemplateSpec{}
	if err := apiv1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(template, internalTemplate, nil); err != nil {
		return "", fmt.Errorf("failed to convert podtemplate, %v", err)
	}
	w := printersinternal.NewPrefixWriter(buf)
	printersinternal.DescribePodTemplate(internalTemplate, w)
	return buf.String(), nil
}

type DaemonSetRollbacker struct {
	c clientset.Interface
}

func (r *DaemonSetRollbacker) Rollback(obj runtime.Object, updatedAnnotations map[string]string, toRevision int64, dryRun bool) (string, error) {
	if toRevision < 0 {
		return "", revisionNotFoundErr(toRevision)
	}

	ds, ok := obj.(*extensions.DaemonSet)
	if !ok {
		return "", fmt.Errorf("passed object is not a DaemonSet: %#v", obj)
	}
	versionedAppsClient := versionedAppsClientV1beta1(r.c)
	versionedExtensionsClient := versionedExtensionsClientV1beta1(r.c)
	versionedObj, allHistory, err := controlledHistories(versionedAppsClient, versionedExtensionsClient, ds.Namespace, ds.Name, "DaemonSet")
	if err != nil {
		return "", fmt.Errorf("unable to find history controlled by DaemonSet %s: %v", ds.Name, err)
	}
	versionedDS, ok := versionedObj.(*externalextensions.DaemonSet)
	if !ok {
		return "", fmt.Errorf("unexpected non-DaemonSet object returned: %v", versionedDS)
	}

	if toRevision == 0 && len(allHistory) <= 1 {
		return "", fmt.Errorf("no last revision to roll back to")
	}

	toHistory := findHistory(toRevision, allHistory)
	if toHistory == nil {
		return "", revisionNotFoundErr(toRevision)
	}

	if dryRun {
		appliedDS, err := applyHistory(versionedDS, toHistory)
		if err != nil {
			return "", err
		}
		return printPodTemplate(&appliedDS.Spec.Template)
	}

	// Skip if the revision already matches current DaemonSet
	done, err := daemon.Match(versionedDS, toHistory)
	if err != nil {
		return "", err
	}
	if done {
		return fmt.Sprintf("%s (current template already matches revision %d)", rollbackSkipped, toRevision), nil
	}

	// Restore revision
	if _, err = versionedExtensionsClient.DaemonSets(ds.Namespace).Patch(ds.Name, types.StrategicMergePatchType, toHistory.Data.Raw); err != nil {
		return "", fmt.Errorf("failed restoring revision %d: %v", toRevision, err)
	}

	return rollbackSuccess, nil
}

type StatefulSetRollbacker struct {
	c clientset.Interface
}

// toRevision is a non-negative integer, with 0 being reserved to indicate rolling back to previous configuration
func (r *StatefulSetRollbacker) Rollback(obj runtime.Object, updatedAnnotations map[string]string, toRevision int64, dryRun bool) (string, error) {
	if toRevision < 0 {
		return "", revisionNotFoundErr(toRevision)
	}

	ss, ok := obj.(*apps.StatefulSet)
	if !ok {
		return "", fmt.Errorf("passed object is not a StatefulSet: %#v", obj)
	}
	versionedAppsClient := versionedAppsClientV1beta1(r.c)
	versionedExtensionsClient := versionedExtensionsClientV1beta1(r.c)
	versionedObj, allHistory, err := controlledHistories(versionedAppsClient, versionedExtensionsClient, ss.Namespace, ss.Name, "StatefulSet")
	if err != nil {
		return "", fmt.Errorf("unable to find history controlled by StatefulSet %s: %v", ss.Name, err)
	}

	versionedSS, ok := versionedObj.(*appsv1beta1.StatefulSet)
	if !ok {
		return "", fmt.Errorf("unexpected non-StatefulSet object returned: %v", versionedSS)
	}

	if toRevision == 0 && len(allHistory) <= 1 {
		return "", fmt.Errorf("no last revision to roll back to")
	}

	toHistory := findHistory(toRevision, allHistory)
	if toHistory == nil {
		return "", revisionNotFoundErr(toRevision)
	}

	if dryRun {
		appliedSS, err := statefulset.ApplyRevision(versionedSS, toHistory)
		if err != nil {
			return "", err
		}
		return printPodTemplate(&appliedSS.Spec.Template)
	}

	// Skip if the revision already matches current StatefulSet
	done, err := statefulset.Match(versionedSS, toHistory)
	if err != nil {
		return "", err
	}
	if done {
		return fmt.Sprintf("%s (current template already matches revision %d)", rollbackSkipped, toRevision), nil
	}

	// Restore revision
	if _, err = versionedAppsClient.StatefulSets(ss.Namespace).Patch(ss.Name, types.StrategicMergePatchType, toHistory.Data.Raw); err != nil {
		return "", fmt.Errorf("failed restoring revision %d: %v", toRevision, err)
	}

	return rollbackSuccess, nil
}

// findHistory returns a controllerrevision of a specific revision from the given controllerrevisions.
// It returns nil if no such controllerrevision exists.
// If toRevision is 0, the last previously used history is returned.
func findHistory(toRevision int64, allHistory []*appsv1beta1.ControllerRevision) *appsv1beta1.ControllerRevision {
	if toRevision == 0 && len(allHistory) <= 1 {
		return nil
	}

	// Find the history to rollback to
	var toHistory *appsv1beta1.ControllerRevision
	if toRevision == 0 {
		// If toRevision == 0, find the latest revision (2nd max)
		sort.Sort(historiesByRevision(allHistory))
		toHistory = allHistory[len(allHistory)-2]
	} else {
		for _, h := range allHistory {
			if h.Revision == toRevision {
				// If toRevision != 0, find the history with matching revision
				return h
			}
		}
	}

	return toHistory
}

// printPodTemplate converts a given pod template into a human-readable string.
func printPodTemplate(specTemplate *v1.PodTemplateSpec) (string, error) {
	content := bytes.NewBuffer([]byte{})
	w := printersinternal.NewPrefixWriter(content)
	internalTemplate := &api.PodTemplateSpec{}
	if err := apiv1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(specTemplate, internalTemplate, nil); err != nil {
		return "", fmt.Errorf("failed to convert podtemplate while printing: %v", err)
	}
	printersinternal.DescribePodTemplate(internalTemplate, w)
	return fmt.Sprintf("will roll back to %s", content.String()), nil
}

func revisionNotFoundErr(r int64) error {
	return fmt.Errorf("unable to find specified revision %v in history", r)
}

// TODO: copied from daemon controller, should extract to a library
type historiesByRevision []*appsv1beta1.ControllerRevision

func (h historiesByRevision) Len() int      { return len(h) }
func (h historiesByRevision) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h historiesByRevision) Less(i, j int) bool {
	return h[i].Revision < h[j].Revision
}
