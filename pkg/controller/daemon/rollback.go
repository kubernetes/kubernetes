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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	daemonutil "k8s.io/kubernetes/pkg/controller/daemon/util"
)

// Rolling back to a revision; no-op if the toRevision is daemon's current revision
func (dsc *DaemonSetsController) rollback(daemon *extensions.DaemonSet, toRevision *int64) (*extensions.DaemonSet, error) {
	podTemplates, err := daemonutil.GetAllPodTemplates(daemon, dsc.kubeClient)
	if err != nil {
		return nil, err
	}
	// If rollback revision is 0, rollback to the last revision
	if *toRevision == 0 {
		if *toRevision = daemonutil.LastRevision(podTemplates); *toRevision == 0 {
			dsc.emitRollbackWarningEvent(daemon, daemonutil.RollbackRevisionNotFound, "Unable to find last revision.")
			// Gives up rollback
			return dsc.updateDaemonSetAndClearRollbackTo(daemon)
		}
	}
	for _, template := range podTemplates.Items {
		v, err := daemonutil.Revision(&template)
		if err != nil {
			glog.V(4).Infof("Unable to extract revision from daemon's podTemplate %q: %v", template.Name, err)
			continue
		}
		if v == *toRevision {
			glog.V(4).Infof("Found podTemplate %q with desired revision %d", template.Name, v)
			// rollback by copying Template from the podTemplate, and increment revision number by 1
			// no-op if the the spec matches current daemon's podTemplate.Spec
			daemon, performedRollback, err := dsc.rollbackToTemplate(daemon, &template)
			if performedRollback && err == nil {
				dsc.emitRollbackNormalEvent(daemon, fmt.Sprintf("Rolled back daemon %q to revision %d", daemon.Name, *toRevision))
			}
			return daemon, err
		}
	}
	dsc.emitRollbackWarningEvent(daemon, daemonutil.RollbackRevisionNotFound, "Unable to find the revision to rollback to.")
	// Gives up rollback
	return dsc.updateDaemonSetAndClearRollbackTo(daemon)
}

func (dsc *DaemonSetsController) rollbackToTemplate(daemon *extensions.DaemonSet, podTemplate *api.PodTemplate) (d *extensions.DaemonSet, performedRollback bool, err error) {
	glog.Infof("Rolling back daemon set %s to template spec %+v", daemon.Name, podTemplate.Template.Spec)
	daemonutil.SetFromPodTemplate(daemon, podTemplate)
	daemonutil.SetDaemonSetAnnotationsTo(daemon, podTemplate)
	performedRollback = true
	d, err = dsc.updateDaemonSetAndClearRollbackTo(daemon)
	return
}

// updateDaemonSetAndClearRollbackTo sets .spec.rollbackTo to nil and update the input daemon
func (dsc *DaemonSetsController) updateDaemonSetAndClearRollbackTo(daemon *extensions.DaemonSet) (*extensions.DaemonSet, error) {
	glog.V(4).Infof("Cleans up rollbackTo of daemon %s", daemon.Name)
	daemon.Spec.RollbackTo = nil
	return dsc.kubeClient.Extensions().DaemonSets(daemon.ObjectMeta.Namespace).Update(daemon)
}

func (dsc *DaemonSetsController) emitRollbackNormalEvent(daemon *extensions.DaemonSet, message string) {
	dsc.eventRecorder.Eventf(daemon, api.EventTypeNormal, daemonutil.RollbackDone, message)
}

func (dsc *DaemonSetsController) emitRollbackWarningEvent(daemon *extensions.DaemonSet, reason, message string) {
	dsc.eventRecorder.Eventf(daemon, api.EventTypeWarning, reason, message)
}
