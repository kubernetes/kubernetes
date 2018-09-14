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

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	apiv1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	kapps "k8s.io/kubernetes/pkg/kubectl/apps"
	sliceutil "k8s.io/kubernetes/pkg/kubectl/util/slice"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	// kubectl should not be taking dependencies on logic in the controllers
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
)

const (
	rollbackSuccess = "rolled back"
	rollbackSkipped = "skipped rollback"
)

// Rollbacker provides an interface for resources that can be rolled back.
type Rollbacker interface {
	Rollback(obj runtime.Object, updatedAnnotations map[string]string, toRevision int64, dryRun bool) (string, error)
}

type RollbackVisitor struct {
	clientset kubernetes.Interface
	result    Rollbacker
}

func (v *RollbackVisitor) VisitDeployment(elem kapps.GroupKindElement) {
	v.result = &DeploymentRollbacker{v.clientset}
}

func (v *RollbackVisitor) VisitStatefulSet(kind kapps.GroupKindElement) {
	v.result = &StatefulSetRollbacker{v.clientset}
}

func (v *RollbackVisitor) VisitDaemonSet(kind kapps.GroupKindElement) {
	v.result = &DaemonSetRollbacker{v.clientset}
}

func (v *RollbackVisitor) VisitJob(kind kapps.GroupKindElement)                   {}
func (v *RollbackVisitor) VisitPod(kind kapps.GroupKindElement)                   {}
func (v *RollbackVisitor) VisitReplicaSet(kind kapps.GroupKindElement)            {}
func (v *RollbackVisitor) VisitReplicationController(kind kapps.GroupKindElement) {}
func (v *RollbackVisitor) VisitCronJob(kind kapps.GroupKindElement)               {}

// RollbackerFor returns an implementation of Rollbacker interface for the given schema kind
func RollbackerFor(kind schema.GroupKind, c kubernetes.Interface) (Rollbacker, error) {
	elem := kapps.GroupKindElement(kind)
	visitor := &RollbackVisitor{
		clientset: c,
	}

	err := elem.Accept(visitor)

	if err != nil {
		return nil, fmt.Errorf("error retrieving rollbacker for %q, %v", kind.String(), err)
	}

	if visitor.result == nil {
		return nil, fmt.Errorf("no rollbacker has been implemented for %q", kind)
	}

	return visitor.result, nil
}

type DeploymentRollbacker struct {
	c kubernetes.Interface
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
	deploymentRollback := &extensionsv1beta1.DeploymentRollback{
		Name:               d.Name,
		UpdatedAnnotations: updatedAnnotations,
		RollbackTo: extensionsv1beta1.RollbackConfig{
			Revision: toRevision,
		},
	}
	result := ""

	// Get current events
	events, err := r.c.CoreV1().Events(d.Namespace).List(metav1.ListOptions{})
	if err != nil {
		return result, err
	}
	// Do the rollback
	if err := r.c.ExtensionsV1beta1().Deployments(d.Namespace).Rollback(deploymentRollback); err != nil {
		return result, err
	}
	// Watch for the changes of events
	watch, err := r.c.CoreV1().Events(d.Namespace).Watch(metav1.ListOptions{Watch: true, ResourceVersion: events.ResourceVersion})
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

func simpleDryRun(deployment *extensions.Deployment, c kubernetes.Interface, toRevision int64) (string, error) {
	externalDeployment := &appsv1.Deployment{}
	if err := legacyscheme.Scheme.Convert(deployment, externalDeployment, nil); err != nil {
		return "", fmt.Errorf("failed to convert deployment, %v", err)
	}

	_, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSets(externalDeployment, c.AppsV1())
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
		if err := apiv1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(template, internalTemplate, nil); err != nil {
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
	if err := apiv1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(template, internalTemplate, nil); err != nil {
		return "", fmt.Errorf("failed to convert podtemplate, %v", err)
	}
	w := printersinternal.NewPrefixWriter(buf)
	printersinternal.DescribePodTemplate(internalTemplate, w)
	return buf.String(), nil
}

type DaemonSetRollbacker struct {
	c kubernetes.Interface
}

func (r *DaemonSetRollbacker) Rollback(obj runtime.Object, updatedAnnotations map[string]string, toRevision int64, dryRun bool) (string, error) {
	if toRevision < 0 {
		return "", revisionNotFoundErr(toRevision)
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return "", fmt.Errorf("failed to create accessor for kind %v: %s", obj.GetObjectKind(), err.Error())
	}
	ds, history, err := daemonSetHistory(r.c.AppsV1(), accessor.GetNamespace(), accessor.GetName())
	if err != nil {
		return "", err
	}
	if toRevision == 0 && len(history) <= 1 {
		return "", fmt.Errorf("no last revision to roll back to")
	}

	toHistory := findHistory(toRevision, history)
	if toHistory == nil {
		return "", revisionNotFoundErr(toRevision)
	}

	if dryRun {
		appliedDS, err := applyDaemonSetHistory(ds, toHistory)
		if err != nil {
			return "", err
		}
		return printPodTemplate(&appliedDS.Spec.Template)
	}

	// Skip if the revision already matches current DaemonSet
	done, err := daemonSetMatch(ds, toHistory)
	if err != nil {
		return "", err
	}
	if done {
		return fmt.Sprintf("%s (current template already matches revision %d)", rollbackSkipped, toRevision), nil
	}

	// Restore revision
	if _, err = r.c.ExtensionsV1beta1().DaemonSets(accessor.GetNamespace()).Patch(accessor.GetName(), types.StrategicMergePatchType, toHistory.Data.Raw); err != nil {
		return "", fmt.Errorf("failed restoring revision %d: %v", toRevision, err)
	}

	return rollbackSuccess, nil
}

// daemonMatch check if the given DaemonSet's template matches the template stored in the given history.
func daemonSetMatch(ds *appsv1.DaemonSet, history *appsv1.ControllerRevision) (bool, error) {
	patch, err := getDaemonSetPatch(ds)
	if err != nil {
		return false, err
	}
	return bytes.Equal(patch, history.Data.Raw), nil
}

// getPatch returns a strategic merge patch that can be applied to restore a Daemonset to a
// previous version. If the returned error is nil the patch is valid. The current state that we save is just the
// PodSpecTemplate. We can modify this later to encompass more state (or less) and remain compatible with previously
// recorded patches.
func getDaemonSetPatch(ds *appsv1.DaemonSet) ([]byte, error) {
	dsBytes, err := json.Marshal(ds)
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	err = json.Unmarshal(dsBytes, &raw)
	if err != nil {
		return nil, err
	}
	objCopy := make(map[string]interface{})
	specCopy := make(map[string]interface{})

	// Create a patch of the DaemonSet that replaces spec.template
	spec := raw["spec"].(map[string]interface{})
	template := spec["template"].(map[string]interface{})
	specCopy["template"] = template
	template["$patch"] = "replace"
	objCopy["spec"] = specCopy
	patch, err := json.Marshal(objCopy)
	return patch, err
}

type StatefulSetRollbacker struct {
	c kubernetes.Interface
}

// toRevision is a non-negative integer, with 0 being reserved to indicate rolling back to previous configuration
func (r *StatefulSetRollbacker) Rollback(obj runtime.Object, updatedAnnotations map[string]string, toRevision int64, dryRun bool) (string, error) {
	if toRevision < 0 {
		return "", revisionNotFoundErr(toRevision)
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return "", fmt.Errorf("failed to create accessor for kind %v: %s", obj.GetObjectKind(), err.Error())
	}
	sts, history, err := statefulSetHistory(r.c.AppsV1(), accessor.GetNamespace(), accessor.GetName())
	if err != nil {
		return "", err
	}
	if toRevision == 0 && len(history) <= 1 {
		return "", fmt.Errorf("no last revision to roll back to")
	}

	toHistory := findHistory(toRevision, history)
	if toHistory == nil {
		return "", revisionNotFoundErr(toRevision)
	}

	if dryRun {
		appliedSS, err := applyRevision(sts, toHistory)
		if err != nil {
			return "", err
		}
		return printPodTemplate(&appliedSS.Spec.Template)
	}

	// Skip if the revision already matches current StatefulSet
	done, err := statefulsetMatch(sts, toHistory)
	if err != nil {
		return "", err
	}
	if done {
		return fmt.Sprintf("%s (current template already matches revision %d)", rollbackSkipped, toRevision), nil
	}

	// Restore revision
	if _, err = r.c.AppsV1().StatefulSets(sts.Namespace).Patch(sts.Name, types.StrategicMergePatchType, toHistory.Data.Raw); err != nil {
		return "", fmt.Errorf("failed restoring revision %d: %v", toRevision, err)
	}

	return rollbackSuccess, nil
}

var appsCodec = legacyscheme.Codecs.LegacyCodec(appsv1.SchemeGroupVersion)

// applyRevision returns a new StatefulSet constructed by restoring the state in revision to set. If the returned error
// is nil, the returned StatefulSet is valid.
func applyRevision(set *appsv1.StatefulSet, revision *appsv1.ControllerRevision) (*appsv1.StatefulSet, error) {
	clone := set.DeepCopy()
	patched, err := strategicpatch.StrategicMergePatch([]byte(runtime.EncodeOrDie(appsCodec, clone)), revision.Data.Raw, clone)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(patched, clone)
	if err != nil {
		return nil, err
	}
	return clone, nil
}

// statefulsetMatch check if the given StatefulSet's template matches the template stored in the given history.
func statefulsetMatch(ss *appsv1.StatefulSet, history *appsv1.ControllerRevision) (bool, error) {
	patch, err := getStatefulSetPatch(ss)
	if err != nil {
		return false, err
	}
	return bytes.Equal(patch, history.Data.Raw), nil
}

// getStatefulSetPatch returns a strategic merge patch that can be applied to restore a StatefulSet to a
// previous version. If the returned error is nil the patch is valid. The current state that we save is just the
// PodSpecTemplate. We can modify this later to encompass more state (or less) and remain compatible with previously
// recorded patches.
func getStatefulSetPatch(set *appsv1.StatefulSet) ([]byte, error) {
	str, err := runtime.Encode(appsCodec, set)
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(str), &raw); err != nil {
		return nil, err
	}
	objCopy := make(map[string]interface{})
	specCopy := make(map[string]interface{})
	spec := raw["spec"].(map[string]interface{})
	template := spec["template"].(map[string]interface{})
	specCopy["template"] = template
	template["$patch"] = "replace"
	objCopy["spec"] = specCopy
	patch, err := json.Marshal(objCopy)
	return patch, err
}

// findHistory returns a controllerrevision of a specific revision from the given controllerrevisions.
// It returns nil if no such controllerrevision exists.
// If toRevision is 0, the last previously used history is returned.
func findHistory(toRevision int64, allHistory []*appsv1.ControllerRevision) *appsv1.ControllerRevision {
	if toRevision == 0 && len(allHistory) <= 1 {
		return nil
	}

	// Find the history to rollback to
	var toHistory *appsv1.ControllerRevision
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
	if err := apiv1.Convert_v1_PodTemplateSpec_To_core_PodTemplateSpec(specTemplate, internalTemplate, nil); err != nil {
		return "", fmt.Errorf("failed to convert podtemplate while printing: %v", err)
	}
	printersinternal.DescribePodTemplate(internalTemplate, w)
	return fmt.Sprintf("will roll back to %s", content.String()), nil
}

func revisionNotFoundErr(r int64) error {
	return fmt.Errorf("unable to find specified revision %v in history", r)
}

// TODO: copied from daemon controller, should extract to a library
type historiesByRevision []*appsv1.ControllerRevision

func (h historiesByRevision) Len() int      { return len(h) }
func (h historiesByRevision) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h historiesByRevision) Less(i, j int) bool {
	return h[i].Revision < h[j].Revision
}
