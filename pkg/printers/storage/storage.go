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

package storage

import (
	"context"
	"fmt"

	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/printers"
)

// TableConvertor struct - converts objects to metav1beta1.Table using printers.TableGenerator
type TableConvertor struct {
	printers.TableGenerator
}

// ConvertToTable method - converts objects to metav1beta1.Table objects using TableGenerator
func (c TableConvertor) ConvertToTable(ctx context.Context, obj runtime.Object, tableOptions runtime.Object) (*metav1beta1.Table, error) {
	noHeaders := false
	if tableOptions != nil {
		switch t := tableOptions.(type) {
		case *metav1beta1.TableOptions:
			if t != nil {
				noHeaders = t.NoHeaders
			}
		default:
			return nil, fmt.Errorf("unrecognized type %T for table options, can't display tabular output", tableOptions)
		}
	}
	return c.TableGenerator.GenerateTable(obj, printers.PrintOptions{Wide: true, NoHeaders: noHeaders})
}
