/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io"
	"sort"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/runtime"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	"k8s.io/kubernetes/pkg/util/errors"
)

const (
	ChangeCauseAnnotation = "kubernetes.io/change-cause"
)

// HistoryViewer provides an interface for resources that can be rolled back.
type HistoryViewer interface {
	History(namespace, name string) (HistoryInfo, error)
}

func HistoryViewerFor(kind unversioned.GroupKind, c clientset.Interface) (HistoryViewer, error) {
	switch kind {
	case extensions.Kind("Deployment"):
		return &DeploymentHistoryViewer{c}, nil
	}
	return nil, fmt.Errorf("no history viewer has been implemented for %q", kind)
}

// HistoryInfo stores the mapping from revision to podTemplate;
// note that change-cause annotation should be copied to podTemplate
type HistoryInfo struct {
	RevisionToTemplate map[int64]*api.PodTemplateSpec
}

type DeploymentHistoryViewer struct {
	c clientset.Interface
}

// History returns a revision-to-replicaset map as the revision history of a deployment
func (h *DeploymentHistoryViewer) History(namespace, name string) (HistoryInfo, error) {
	historyInfo := HistoryInfo{
		RevisionToTemplate: make(map[int64]*api.PodTemplateSpec),
	}
	deployment, err := h.c.Extensions().Deployments(namespace).Get(name)
	if err != nil {
		return historyInfo, fmt.Errorf("failed to retrieve deployment %s: %v", name, err)
	}
	_, allOldRSs, err := deploymentutil.GetOldReplicaSets(deployment, h.c)
	if err != nil {
		return historyInfo, fmt.Errorf("failed to retrieve old replica sets from deployment %s: %v", name, err)
	}
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, h.c)
	if err != nil {
		return historyInfo, fmt.Errorf("failed to retrieve new replica set from deployment %s: %v", name, err)
	}
	allRSs := append(allOldRSs, newRS)
	for _, rs := range allRSs {
		v, err := deploymentutil.Revision(rs)
		if err != nil {
			continue
		}
		historyInfo.RevisionToTemplate[v] = &rs.Spec.Template
		changeCause := getChangeCause(rs)
		if historyInfo.RevisionToTemplate[v].Annotations == nil {
			historyInfo.RevisionToTemplate[v].Annotations = make(map[string]string)
		}
		if len(changeCause) > 0 {
			historyInfo.RevisionToTemplate[v].Annotations[ChangeCauseAnnotation] = changeCause
		}
	}
	return historyInfo, nil
}

// PrintRolloutHistory prints a formatted table of the input revision history of the deployment
func PrintRolloutHistory(historyInfo HistoryInfo, resource, name string) (string, error) {
	if len(historyInfo.RevisionToTemplate) == 0 {
		return fmt.Sprintf("No rollout history found in %s %q", resource, name), nil
	}
	// Sort the revisionToChangeCause map by revision
	var revisions []string
	for k := range historyInfo.RevisionToTemplate {
		revisions = append(revisions, strconv.FormatInt(k, 10))
	}
	sort.Strings(revisions)

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "%s %q:\n", resource, name)
		fmt.Fprintf(out, "REVISION\tCHANGE-CAUSE\n")
		errs := []error{}
		for _, r := range revisions {
			// Find the change-cause of revision r
			r64, err := strconv.ParseInt(r, 10, 64)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			changeCause := historyInfo.RevisionToTemplate[r64].Annotations[ChangeCauseAnnotation]
			if len(changeCause) == 0 {
				changeCause = "<none>"
			}
			fmt.Fprintf(out, "%s\t%s\n", r, changeCause)
		}
		return errors.NewAggregate(errs)
	})
}

// getChangeCause returns the change-cause annotation of the input object
func getChangeCause(obj runtime.Object) string {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return ""
	}
	return accessor.GetAnnotations()[ChangeCauseAnnotation]
}
