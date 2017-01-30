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

	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/retry"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

// TODO: use client library instead when it starts to support update retries
//       see https://github.com/kubernetes/kubernetes/issues/21479
type updateRSFunc func(rs *extensions.ReplicaSet) error

// UpdateRSWithRetries updates a RS with given applyUpdate function. Note that RS not found error is ignored.
// The returned bool value can be used to tell if the RS is actually updated.
func UpdateRSWithRetries(rsClient unversionedextensions.ReplicaSetInterface, rsLister *cache.StoreToReplicaSetLister, namespace, name string, applyUpdate updateRSFunc) (*extensions.ReplicaSet, error) {
	var rs *extensions.ReplicaSet

	retryErr := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		var err error
		rs, err = rsLister.ReplicaSets(namespace).Get(name)
		if err != nil {
			return err
		}
		obj, deepCopyErr := api.Scheme.DeepCopy(rs)
		if deepCopyErr != nil {
			return deepCopyErr
		}
		rs = obj.(*extensions.ReplicaSet)
		// Apply the update, then attempt to push it to the apiserver.
		if applyErr := applyUpdate(rs); applyErr != nil {
			return applyErr
		}
		rs, err = rsClient.Update(rs)
		return err
	})

	// Ignore the precondition violated error, but the RS isn't updated.
	if retryErr == errorsutil.ErrPreconditionViolated {
		glog.V(4).Infof("Replica set %s/%s precondition doesn't hold, skip updating it.", namespace, name)
		retryErr = nil
	}

	return rs, retryErr
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

// GetReplicaSetHashFnv returns the pod template hash of a ReplicaSet's pod template spec.
func GetReplicaSetHashFnv(rs *extensions.ReplicaSet) string {
	meta := rs.Spec.Template.ObjectMeta
	meta.Labels = labelsutil.CloneAndRemoveLabel(meta.Labels, extensions.DefaultDeploymentUniqueLabelKey)
	return fmt.Sprintf("%d", GetPodTemplateSpecHashFnv(v1.PodTemplateSpec{
		ObjectMeta: meta,
		Spec:       rs.Spec.Template.Spec,
	}))
}
