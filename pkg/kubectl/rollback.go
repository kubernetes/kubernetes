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
	"syscall"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/extensions"
	externalextensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	sliceutil "k8s.io/kubernetes/pkg/util/slice"
)

// Rollbacker provides an interface for resources that can be rolled back.
type Rollbacker interface {
	Rollback(obj runtime.Object, updatedAnnotations map[string]string, toRevision int64, dryRun bool) (string, error)
}

func RollbackerFor(kind schema.GroupKind, c clientset.Interface) (Rollbacker, error) {
	switch kind {
	case extensions.Kind("Deployment"), apps.Kind("Deployment"):
		return &DeploymentRollbacker{c}, nil
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
				return true, "rolled back"
			}
			return true, fmt.Sprintf("skipped rollback (%s: %s)", e.Reason, e.Message)
		}
	}
	return false, ""
}

func simpleDryRun(deployment *extensions.Deployment, c clientset.Interface, toRevision int64) (string, error) {
	externalDeployment := &externalextensions.Deployment{}
	if err := api.Scheme.Convert(deployment, externalDeployment, nil); err != nil {
		return "", fmt.Errorf("failed to convert deployment, %v", err)
	}
	versionedClient := versionedClientsetForDeployment(c)
	_, allOldRSs, newRS, err := deploymentutil.GetAllReplicaSetsV15(externalDeployment, versionedClient)
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
			return "", fmt.Errorf("unable to find specified revision")
		}
		buf := bytes.NewBuffer([]byte{})
		internalTemplate := &api.PodTemplateSpec{}
		if err := v1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(template, internalTemplate, nil); err != nil {
			return "", fmt.Errorf("failed to convert podtemplate, %v", err)
		}
		printersinternal.DescribePodTemplate(internalTemplate, buf)
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
	if err := v1.Convert_v1_PodTemplateSpec_To_api_PodTemplateSpec(template, internalTemplate, nil); err != nil {
		return "", fmt.Errorf("failed to convert podtemplate, %v", err)
	}
	printersinternal.DescribePodTemplate(internalTemplate, buf)
	return buf.String(), nil
}
