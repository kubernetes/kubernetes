/*
Copyright 2018 The Kubernetes Authors.

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

package table

import (
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/duration"
)

// MetaToTableRow converts a list or object into one or more table rows. The provided rowFn is invoked for
// each accessed item, with name and age being passed to each.
func MetaToTableRow(obj runtime.Object, rowFn func(obj runtime.Object, m metav1.Object, name, age string) ([]interface{}, error)) ([]metav1beta1.TableRow, error) {
	if meta.IsListType(obj) {
		rows := make([]metav1beta1.TableRow, 0, 16)
		err := meta.EachListItem(obj, func(obj runtime.Object) error {
			nestedRows, err := MetaToTableRow(obj, rowFn)
			if err != nil {
				return err
			}
			rows = append(rows, nestedRows...)
			return nil
		})
		if err != nil {
			return nil, err
		}
		return rows, nil
	}

	rows := make([]metav1beta1.TableRow, 0, 1)
	m, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}
	row := metav1beta1.TableRow{
		Object: runtime.RawExtension{Object: obj},
	}
	row.Cells, err = rowFn(obj, m, m.GetName(), ConvertToHumanReadableDateType(m.GetCreationTimestamp()))
	if err != nil {
		return nil, err
	}
	rows = append(rows, row)
	return rows, nil
}

// ConvertToHumanReadableDateType returns the elapsed time since timestamp in
// human-readable approximation.
func ConvertToHumanReadableDateType(timestamp metav1.Time) string {
	if timestamp.IsZero() {
		return "<unknown>"
	}
	return duration.ShortHumanDuration(time.Now().Sub(timestamp.Time))
}
