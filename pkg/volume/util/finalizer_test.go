/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	arbitraryTime = metav1.Date(2017, 11, 1, 14, 28, 47, 0, time.FixedZone("CET", 0))
)

func TestIsPVCBeingDeleted(t *testing.T) {
	tests := []struct {
		pvc  *v1.PersistentVolumeClaim
		want bool
	}{
		{
			pvc: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: nil,
				},
			},
			want: false,
		},
		{
			pvc: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: &arbitraryTime,
				},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		if got := IsPVCBeingDeleted(tt.pvc); got != tt.want {
			t.Errorf("IsPVCBeingDeleted(%v) = %v WANT %v", tt.pvc, got, tt.want)
		}
	}
}
