/*
Copyright 2015 The Kubernetes Authors.

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
package daemon

import (
	"fmt"
	"strconv"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	podutil "k8s.io/kubernetes/pkg/util/pod"
)

const (
	// RevisionAnnotation is the revision annotation of a daemon set pod template which records its rollout sequence
	RevisionAnnotation = "daemonset.kubernetes.io/revision"
	PodTemplateLabel   = "damonset.kubernetes.io/daemon-name"

	// RollbackRevisionNotFound is not found rollback event reason
	RollbackRevisionNotFound = "DaemonRollbackRevisionNotFound"
	// RollbackTemplateUnchanged is the template unchanged rollback event reason
	RollbackTemplateUnchanged = "DaemonRollbackTemplateUnchanged"
	// RollbackDone is the done rollback event reason
	RollbackDone = "DaemonSetRollback"
)

type podTemplateListFunc func(string, api.ListOptions) (*api.PodTemplateList, error)

func GetOrCreatePodTemplate(ptc PodTemplateControllerInterface, ds *extensions.DaemonSet, c clientset.Interface) (*api.PodTemplate, error) {
	var podTemplate *api.PodTemplate

	podsTemplates, err := listPodTemplates(ds, c)
	if err != nil {
		return nil, err
	}
	if len(podsTemplates.Items) == 0 {
		podTemplate, err = CreatePodTemplateFromDS(ptc, ds, "1")
		if err != nil {
			return nil, err
		}
	} else {
		daemonTemplateHash := podutil.GetPodTemplateSpecHash(ds.Spec.Template)
		daemonTemplateHashStr := strconv.FormatUint(uint64(daemonTemplateHash), 10)
		for _, podTmpl := range podsTemplates.Items {
			podTemplateSpecHash, hashExists := podTmpl.ObjectMeta.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
			if hashExists && daemonTemplateHashStr == podTemplateSpecHash {
				return &podTmpl, nil
			}
		}
		revision := MaxRevision(podsTemplates)
		newRevision := strconv.FormatInt(revision+1, 10)
		podTemplate, err = CreatePodTemplateFromDS(ptc, ds, newRevision)
		if err != nil {
			return nil, err
		}
	}
	return podTemplate, nil
}

func CreatePodTemplateFromDS(ptc PodTemplateControllerInterface, ds *extensions.DaemonSet, revision string) (*api.PodTemplate, error) {
	template := ds.Spec.Template
	namespace := ds.ObjectMeta.Namespace
	podTemplateSpecHash := podutil.GetPodTemplateSpecHash(template)
	// TODO: copy annotations from DaemonSet to podtemplate

	template.ObjectMeta.Labels = labelsutil.CloneAndAddLabel(
		ds.Spec.Template.ObjectMeta.Labels,
		extensions.DefaultDaemonSetUniqueLabelKey,
		podTemplateSpecHash,
	)

	newPodTemplate := api.PodTemplate{
		ObjectMeta: api.ObjectMeta{
			Name:        ds.Name + "-" + fmt.Sprintf("%d", podTemplateSpecHash),
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
		Template: template,
	}
	newPodTemplate.ObjectMeta.Labels = labelsutil.CloneAndAddLabel(
		ds.Spec.Template.ObjectMeta.Labels,
		extensions.DefaultDaemonSetUniqueLabelKey,
		podTemplateSpecHash,
	)
	newPodTemplate.ObjectMeta.Annotations[RevisionAnnotation] = revision
	createdPodTemplate, err := ptc.CreatePodTemplate(&newPodTemplate, namespace)
	if err != nil {
		return nil, fmt.Errorf("error creating pod template for DaemonSet %v: %v", ds.Name, err)
	}
	return createdPodTemplate, nil
}

// MaxRevision finds the highest revision in the pod templates
func MaxRevision(templates *api.PodTemplateList) int64 {
	max := int64(0)
	for _, template := range templates.Items {
		if v, err := Revision(&template); err != nil {
			// Skip the PodTemplate  when it failed to parse their revision information
			glog.V(4).Infof("Error: %v. Couldn't parse revision for PodTemplate %#v, daemonset controller will skip it when reconciling revisions.", err, template)
		} else if v > max {
			max = v
		}
	}
	return max
}

// Revision returns the revision number of the input replica set
func Revision(template *api.PodTemplate) (int64, error) {
	v, ok := template.ObjectMeta.Annotations[RevisionAnnotation]
	if !ok {
		return 0, fmt.Errorf("Missing revision annotation in PodTemplate: ", template.Name)
	}
	return strconv.ParseInt(v, 10, 64)
}

// LastRevision finds the second max revision number in all PodTemplates (the last revision)
func LastRevision(podTemplates *api.PodTemplateList) int64 {
	max, secMax := int64(0), int64(0)
	for _, template := range podTemplates.Items {
		if v, err := Revision(&template); err != nil {
			// Skip the pod templates when it failed to parse their revision information
			glog.V(4).Infof("Error: %v. Couldn't parse revision for pod template %#v, daemon controller will skip it when reconciling revisions.", err, template)
		} else if v >= max {
			secMax = max
			max = v
		} else if v > secMax {
			secMax = v
		}
	}
	return secMax
}

// ListPodTemplates returns a list of PodTemplates the given daemon targets.
func ListPodTemplates(ds *extensions.DaemonSet, getPodTemplateList podTemplateListFunc) (*api.PodTemplateList, error) {
	namespace := ds.Namespace
	selector, err := unversioned.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		return nil, err
	}
	options := api.ListOptions{LabelSelector: selector}
	return getPodTemplateList(namespace, options)
}

// listPodTemplates lists all PodTemplates the given daemon targets with the given client interface.
func listPodTemplates(ds *extensions.DaemonSet, c clientset.Interface) (*api.PodTemplateList, error) {
	return ListPodTemplates(ds,
		func(namespace string, options api.ListOptions) (*api.PodTemplateList, error) {
			return c.Core().PodTemplates(namespace).List(options)
		})
}

func GetAllPodTemplates(daemon *extensions.DaemonSet, c clientset.Interface) (*api.PodTemplateList, error) {
	return listPodTemplates(daemon, c)
}

// SetFromPodTemplate sets the desired PodTemplateSpec from a PodTemplate template to the given deployment.
func SetFromPodTemplate(daemon *extensions.DaemonSet, podTemplate *api.PodTemplate) *extensions.DaemonSet {
	daemon.Spec.Template.ObjectMeta = podTemplate.Template.ObjectMeta
	daemon.Spec.Template.Spec = podTemplate.Template.Spec
	daemon.Spec.Template.ObjectMeta.Labels = labelsutil.CloneAndRemoveLabel(
		daemon.Spec.Template.ObjectMeta.Labels,
		extensions.DefaultDaemonSetUniqueLabelKey)
	return daemon
}

// SetiDaemonSetAnnotationsTo sets daemon set's annotations as given PodTemplate's annotations.
// This action should be done if and only if the daemon set is rolling back to this PodTemplate.
// Note that apply and revision annotations are not changed.
func SetDaemonSetAnnotationsTo(daemon *extensions.DaemonSet, rollbackToPT *api.PodTemplate) {
	// XXX:	deployment.Annotations = getSkippedAnnotations(deployment.Annotations)
	daemon.Annotations = make(map[string]string)
	for k, v := range rollbackToPT.Annotations {
		// XXX:if !skipCopyAnnotation(k) {
		daemon.Annotations[k] = v
		// XXX:	}
	}
}

// TODO: Should I make it a real controller?
type PodTemplateControllerInterface interface {
	CreatePodTemplate(podTemplate *api.PodTemplate, namespace string) (*api.PodTemplate, error)
}

type PodTemplateController struct {
	KubeClient clientset.Interface
}

func (ptc *PodTemplateController) CreatePodTemplate(podTemplate *api.PodTemplate, namespace string) (*api.PodTemplate, error) {
	return ptc.KubeClient.Core().PodTemplates(namespace).Create(podTemplate)
}
