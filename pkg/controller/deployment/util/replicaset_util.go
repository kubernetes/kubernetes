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

package util

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/retry"
	errorsutil "k8s.io/kubernetes/pkg/util/errors"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	"k8s.io/kubernetes/pkg/util/wait"
)

// TODO: use client library instead when it starts to support update retries
//       see https://github.com/kubernetes/kubernetes/issues/21479
type updateRSFunc func(rs *extensions.ReplicaSet) error

// UpdateRSWithRetries updates a RS with given applyUpdate function. Note that RS not found error is ignored.
// The returned bool value can be used to tell if the RS is actually updated.
func UpdateRSWithRetries(rsClient unversionedextensions.ReplicaSetInterface, rsLister *cache.StoreToReplicaSetLister, namespace, name string, applyUpdate updateRSFunc) (*extensions.ReplicaSet, error) {
	var err error
	var rs *extensions.ReplicaSet

	pollErr := wait.ExponentialBackoff(retry.DefaultBackoff, func() (bool, error) {
		var getErr error
		rs, getErr = rsLister.ReplicaSets(namespace).Get(name)
		if getErr != nil {
			return false, getErr
		}
		obj, deepCopyErr := api.Scheme.DeepCopy(rs)
		if deepCopyErr != nil {
			return false, deepCopyErr
		}
		rs = obj.(*extensions.ReplicaSet)
		// Apply the update, then attempt to push it to the apiserver.
		if applyErr := applyUpdate(rs); applyErr != nil {
			return false, applyErr
		}
		rs, err = rsClient.Update(rs)
		if err == nil {
			// Update successful.
			return true, nil
		}
		if !errors.IsConflict(err) {
			// Do not retry if the error is not an update conflict.
			return false, err
		}
		// Update could have failed due to conflict error. Try again.
		return false, nil
	})

	// Handle returned error from wait poll
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("timed out trying to update replica set %q: %v", name, err)
	}
	// Ignore the precondition violated error, but the RS isn't updated.
	if pollErr == errorsutil.ErrPreconditionViolated {
		glog.V(4).Infof("Replica set %s/%s precondition doesn't hold, skip updating it.", namespace, name)
		pollErr = nil
	}

	// If pollErr is non-nil the returned replica set cannot be trusted.
	// Otherwise the returned pod contains the applied update.
	return rs, pollErr
}

// GetReplicaSetHash returns the pod template hash of a ReplicaSet's pod template space
func GetReplicaSetHash(rs *extensions.ReplicaSet) string {
	meta := rs.Spec.Template.ObjectMeta
	meta.Labels = labelsutil.CloneAndRemoveLabel(meta.Labels, extensions.DefaultDeploymentUniqueLabelKey)
	return fmt.Sprintf("%d", GetPodTemplateSpecHash(v1.PodTemplateSpec{
		ObjectMeta: meta,
		Spec:       rs.Spec.Template.Spec,
	}))
}
