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

package deployment

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
)

// Rolling back to a revision; no-op if the toRevision is deployment's current revision
func (dc *DeploymentController) rollback(deployment *extensions.Deployment, toRevision *int64) (*extensions.Deployment, error) {
	newRS, allOldRSs, err := dc.getAllReplicaSetsAndSyncRevision(deployment, true)
	if err != nil {
		return nil, err
	}
	allRSs := append(allOldRSs, newRS)
	// If rollback revision is 0, rollback to the last revision
	if *toRevision == 0 {
		if *toRevision = deploymentutil.LastRevision(allRSs); *toRevision == 0 {
			// If we still can't find the last revision, gives up rollback
			dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackRevisionNotFound, "Unable to find last revision.")
			// Gives up rollback
			return dc.updateDeploymentAndClearRollbackTo(deployment)
		}
	}
	for _, rs := range allRSs {
		v, err := deploymentutil.Revision(rs)
		if err != nil {
			glog.V(4).Infof("Unable to extract revision from deployment's replica set %q: %v", rs.Name, err)
			continue
		}
		if v == *toRevision {
			glog.V(4).Infof("Found replica set %q with desired revision %d", rs.Name, v)
			// rollback by copying podTemplate.Spec from the replica set
			// revision number will be incremented during the next getAllReplicaSetsAndSyncRevision call
			// no-op if the the spec matches current deployment's podTemplate.Spec
			deployment, performedRollback, err := dc.rollbackToTemplate(deployment, rs)
			if performedRollback && err == nil {
				dc.emitRollbackNormalEvent(deployment, fmt.Sprintf("Rolled back deployment %q to revision %d", deployment.Name, *toRevision))
			}
			return deployment, err
		}
	}
	dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackRevisionNotFound, "Unable to find the revision to rollback to.")
	// Gives up rollback
	return dc.updateDeploymentAndClearRollbackTo(deployment)
}

func (dc *DeploymentController) rollbackToTemplate(deployment *extensions.Deployment, rs *extensions.ReplicaSet) (d *extensions.Deployment, performedRollback bool, err error) {
	if !deploymentutil.EqualIgnoreHash(deployment.Spec.Template, rs.Spec.Template) {
		glog.Infof("Rolling back deployment %s to template spec %+v", deployment.Name, rs.Spec.Template.Spec)
		deploymentutil.SetFromReplicaSetTemplate(deployment, rs.Spec.Template)
		// set RS (the old RS we'll rolling back to) annotations back to the deployment;
		// otherwise, the deployment's current annotations (should be the same as current new RS) will be copied to the RS after the rollback.
		//
		// For example,
		// A Deployment has old RS1 with annotation {change-cause:create}, and new RS2 {change-cause:edit}.
		// Note that both annotations are copied from Deployment, and the Deployment should be annotated {change-cause:edit} as well.
		// Now, rollback Deployment to RS1, we should update Deployment's pod-template and also copy annotation from RS1.
		// Deployment is now annotated {change-cause:create}, and we have new RS1 {change-cause:create}, old RS2 {change-cause:edit}.
		//
		// If we don't copy the annotations back from RS to deployment on rollback, the Deployment will stay as {change-cause:edit},
		// and new RS1 becomes {change-cause:edit} (copied from deployment after rollback), old RS2 {change-cause:edit}, which is not correct.
		deploymentutil.SetDeploymentAnnotationsTo(deployment, rs)
		performedRollback = true
	} else {
		glog.V(4).Infof("Rolling back to a revision that contains the same template as current deployment %s, skipping rollback...", deployment.Name)
		dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackTemplateUnchanged, fmt.Sprintf("The rollback revision contains the same template as current deployment %q", deployment.Name))
	}
	d, err = dc.updateDeploymentAndClearRollbackTo(deployment)
	return
}

func (dc *DeploymentController) emitRollbackWarningEvent(deployment *extensions.Deployment, reason, message string) {
	dc.eventRecorder.Eventf(deployment, v1.EventTypeWarning, reason, message)
}

func (dc *DeploymentController) emitRollbackNormalEvent(deployment *extensions.Deployment, message string) {
	dc.eventRecorder.Eventf(deployment, v1.EventTypeNormal, deploymentutil.RollbackDone, message)
}

// updateDeploymentAndClearRollbackTo sets .spec.rollbackTo to nil and update the input deployment
func (dc *DeploymentController) updateDeploymentAndClearRollbackTo(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	glog.V(4).Infof("Cleans up rollbackTo of deployment %s", deployment.Name)
	deployment.Spec.RollbackTo = nil
	return dc.client.Extensions().Deployments(deployment.ObjectMeta.Namespace).Update(deployment)
}
