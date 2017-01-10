/*
Copyright 2014 The Kubernetes Authors.

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

package api

import (
	metav1 "k8s.io/client-go/pkg/apis/meta/v1"
	genericapirequest "k8s.io/client-go/pkg/genericapiserver/api/request"
	"k8s.io/client-go/pkg/util/uuid"
)

// FillObjectMetaSystemFields populates fields that are managed by the system on ObjectMeta.
func FillObjectMetaSystemFields(ctx genericapirequest.Context, meta *metav1.ObjectMeta) {
	meta.CreationTimestamp = metav1.Now()
	// allows admission controllers to assign a UID earlier in the request processing
	// to support tracking resources pending creation.
	uid, found := genericapirequest.UIDFrom(ctx)
	if !found {
		uid = uuid.NewUUID()
	}
	meta.UID = uid
	meta.SelfLink = ""
}

// HasObjectMetaSystemFieldValues returns true if fields that are managed by the system on ObjectMeta have values.
func HasObjectMetaSystemFieldValues(meta *metav1.ObjectMeta) bool {
	return !meta.CreationTimestamp.Time.IsZero() ||
		len(meta.UID) != 0
}
