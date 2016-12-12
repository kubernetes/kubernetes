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

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

func (dc *DeploymentController) migrateDeployment(deployment *extensions.Deployment) ([]*extensions.ReplicaSet, error) {
	_, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(deployment, false)
	if err != nil {
		return nil, err
	}

	return dc.migrateOldReplicaSets(deployment, oldRSs)
}

// migrateOldReplicaSets will recreate all old replica sets for a deployment to use the new hashing algorithm
// and return every newly migrated replica set.
func (dc *DeploymentController) migrateOldReplicaSets(deployment *extensions.Deployment, oldRSs []*extensions.ReplicaSet) ([]*extensions.ReplicaSet, error) {
	var (
		errs          []error
		colliding     []*extensions.ReplicaSet
		newlyMigrated []*extensions.ReplicaSet
	)

	// Loop over all old replica sets for this deployment and slowly migrate all of them to use the
	// new hashing algorithm. Colliding replica sets can be retried after the initial iteration.
	for i := range oldRSs {
		rs := oldRSs[i]

		// Avoid old replica sets that already use the new algorithm.
		if len(rs.Annotations[deploymentutil.MigratedFromAnnotation]) > 0 {
			// TODO: Maybe try to see if the replica set we migrated from exists and delete it.
			continue
		}

		// We shouldn't migrate running pods, doesn't feel good.
		if *(rs.Spec.Replicas) > 0 {
			continue
		}

		// Use fnv to create a new hash for the present replica set. Then do a cache lookup to determine
		// if we can directly migrate this replica set (CREATE new + DELETE old), or defer to try again
		// later due to a possible collision.
		podTemplateSpecHash := fmt.Sprintf("%d", deploymentutil.GetPodTemplateSpecHashFnv(rs.Spec.Template))
		collision, err := dc.rsLister.ReplicaSets(deployment.Namespace).Get(deployment.Name + "-" + podTemplateSpecHash)
		switch {
		case errors.IsNotFound(err):
			// Try to migrate this replica set to a new hash.
			migratedRs, err := dc.migrateReplicaSet(deployment, rs, podTemplateSpecHash)
			if err != nil {
				errs = append(errs, err)
			}
			if migratedRs != nil {
				newlyMigrated = append(newlyMigrated, migratedRs)
			}

		case err == nil:
			// Already migrated. Either rs was failed to be deleted in a previous migration or our caches
			// are lagging. Either way retry the deletion.
			if len(collision.Annotations[deploymentutil.MigratedFromAnnotation]) > 0 {
				if err := dc.client.Extensions().ReplicaSets(deployment.Namespace).Delete(rs.Name, &v1.DeleteOptions{}); err != nil && !errors.IsNotFound(err) {
					errs = append(errs, err)
				}
				continue
			}
			if *(collision.Spec.Replicas) > 0 {
				// Old replica set has running pods. Nothing we can do here.
				errs = append(errs, fmt.Errorf("cannot migrate ReplicaSet %q because it collides with ReplicaSet %q that has %d pods",
					rs.Name, collision.Name, *(collision.Spec.Replicas)))
			} else {
				// The new hash collides with an old hash. We can try to migrate this replica set after we have migrated
				// all non-colliding replica sets.
				colliding = append(colliding, rs)
			}

		default: /*err != nil && !errors.IsNotFound(err)*/
			// Not sure how we can end up here but it's a case.
			errs = append(errs, err)
		}
	}

	// Try to migrate colliding replica sets. I don't think we will need to recurse deeper than this.
	for i := range colliding {
		rs := colliding[i]

		podTemplateSpecHash := fmt.Sprintf("%d", deploymentutil.GetPodTemplateSpecHashFnv(rs.Spec.Template))
		collidingMigrated, err := dc.migrateReplicaSet(deployment, rs, podTemplateSpecHash)
		if err != nil {
			errs = append(errs, err)
		}
		if collidingMigrated != nil {
			newlyMigrated = append(newlyMigrated, collidingMigrated)
		}
	}

	return newlyMigrated, utilerrors.NewAggregate(errs)
}

func (dc *DeploymentController) migrateReplicaSet(d *extensions.Deployment, rs *extensions.ReplicaSet, podTemplateSpecHash string) (*extensions.ReplicaSet, error) {
	zero := int32(0)
	// Go ahead and create replica set with the new hashing algorithm. What about quota here?
	// So far there are no replica set quotas implemented so we can get away with that?
	oldRSSelector := labelsutil.CloneSelectorAndAddLabel(rs.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey, podTemplateSpecHash)
	oldRSTemplate := rs.Spec.Template
	oldRSTemplate.Labels = labelsutil.CloneAndAddLabel(rs.Spec.Template.Labels, extensions.DefaultDeploymentUniqueLabelKey, podTemplateSpecHash)
	oldRSLabels := labelsutil.CloneAndAddLabel(rs.Labels, extensions.DefaultDeploymentUniqueLabelKey, podTemplateSpecHash)
	oldRSAnnotations := make(map[string]string)
	for k, v := range rs.Annotations {
		oldRSAnnotations[k] = v
	}
	// Note which replica set resulted in creating this new replica set. The migration annotation is useful
	// for two reasons:
	//
	// 1) we can keep track of all the migrated replica sets. Once all replica sets owned by all deployments
	//    in the cluster have migrated to the new algorithm we should somehow inform the admin that they can
	//    restart the controller manager by using just the new hashing algorithm.
	// 2) until we use transactions we need to make this process re-entrant. In case a new replica set is
	//    successfully created but the old replica set is failed to be deleted, we can simply try to get
	//    replica sets from this annotation and retry the deletion.
	oldRSAnnotations[deploymentutil.MigratedFromAnnotation] = rs.Name

	oldReplicaSetWithNewHash := extensions.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Name:      d.Name + "-" + podTemplateSpecHash,
			Namespace: d.Namespace,
			// TODO: Add owner reference.
			Annotations: oldRSAnnotations,
			Labels:      oldRSLabels,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas:        &zero,
			MinReadySeconds: rs.Spec.MinReadySeconds,
			Selector:        oldRSSelector,
			Template:        oldRSTemplate,
		},
	}
	migratedRs, err := dc.client.Extensions().ReplicaSets(d.Namespace).Create(&oldReplicaSetWithNewHash)
	if err != nil {
		// Let's not ignore AlreadyExists errors. The cache may be lagging behind but the replica set should
		// eventually show up.
		return nil, err
	}
	// Delete the old replica set.
	if err := dc.client.Extensions().ReplicaSets(d.Namespace).Delete(rs.Name, &v1.DeleteOptions{}); err != nil && !errors.IsNotFound(err) {
		return migratedRs, err
	}
	return migratedRs, nil
}
