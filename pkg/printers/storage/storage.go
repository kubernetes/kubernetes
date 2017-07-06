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
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/printers"
)

type TableConvertor struct {
	printers.TablePrinter
}

func (c TableConvertor) ConvertToTable(ctx genericapirequest.Context, obj runtime.Object, tableOptions runtime.Object) (*metav1alpha1.Table, error) {
	return c.TablePrinter.PrintTable(obj, printers.PrintOptions{Wide: true})
}
