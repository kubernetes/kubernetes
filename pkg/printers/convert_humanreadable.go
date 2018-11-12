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

package printers

import (
	"fmt"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"reflect"
)

var _ PrintHandler = &ConvertingHumanReadablePrinter{}
var _ TablePrinter = &ConvertingHumanReadablePrinter{}

type ConvertingHumanReadablePrinter struct {
	delegate                     *HumanReadablePrinter
	convertor                    runtime.ObjectConvertor
	typer                        runtime.ObjectTyper
	receivingVersionForGroupKind map[schema.GroupKind]string
}

func NewConvertingHumanReadablePrinterWithLegacyScheme() *ConvertingHumanReadablePrinter {
	return NewConvertingHumanReadablePrinter(legacyscheme.Scheme)
}

func NewConvertingHumanReadablePrinter(scheme *runtime.Scheme) *ConvertingHumanReadablePrinter {
	return &ConvertingHumanReadablePrinter{
		delegate:                     NewTablePrinter(),
		typer:                        scheme,
		convertor:                    scheme,
		receivingVersionForGroupKind: make(map[schema.GroupKind]string),
	}
}

func (h *ConvertingHumanReadablePrinter) Handler(columns, columnsWithWide []string, printFunc interface{}) error {
	if err := h.registerPrintFunc(printFunc); err != nil {
		return err
	}
	return h.delegate.Handler(columns, columnsWithWide, printFunc)
}

func (h *ConvertingHumanReadablePrinter) TableHandler(columns []metav1beta1.TableColumnDefinition, printFunc interface{}) error {
	if err := h.registerPrintFunc(printFunc); err != nil {
		return err
	}
	return h.delegate.TableHandler(columns, printFunc)
}

func (h *ConvertingHumanReadablePrinter) DefaultTableHandler(columns []metav1beta1.TableColumnDefinition, printFunc interface{}) error {
	if err := h.registerPrintFunc(printFunc); err != nil {
		return err
	}
	return h.DefaultTableHandler(columns, printFunc)
}

func (h *ConvertingHumanReadablePrinter) registerPrintFunc(printFunc interface{}) error {
	gvk, err := h.getPrintFuncReceivingGroupVersionKind(reflect.ValueOf(printFunc))
	if err != nil {
		return err
	}
	h.receivingVersionForGroupKind[gvk.GroupKind()] = gvk.Version
	return nil
}

func (h *ConvertingHumanReadablePrinter) With(fns ...func(PrintHandler)) *ConvertingHumanReadablePrinter {
	for _, fn := range fns {
		fn(h)
	}
	return h
}

func (h *ConvertingHumanReadablePrinter) PrintTable(obj runtime.Object, options PrintOptions) (*metav1beta1.Table, error) {
	gvks, isUnversioned, err := h.typer.ObjectKinds(obj)
	if err != nil {
		return nil, err
	}
	if isUnversioned {
		return nil, fmt.Errorf("unexpect call with unversioned object in human readable printer")
	}
	gk := gvks[0].GroupKind()
	_, exist := h.receivingVersionForGroupKind[gk]
	if !exist {
		return nil, fmt.Errorf("no print handler registered for %v", gk)
	}
	targetObj, err := h.convertor.ConvertToVersion(obj, schema.GroupVersion{
		Group:   gk.Group,
		Version: h.receivingVersionForGroupKind[gk]})
	return h.delegate.PrintTable(targetObj, options)
}

func (h *ConvertingHumanReadablePrinter) getPrintFuncReceivingGroupVersionKind(printFuncValue reflect.Value) (schema.GroupVersionKind, error) {
	objType := printFuncValue.Type().In(0).Elem()
	object := reflect.New(objType).Interface().(runtime.Object)
	gvks, _, err := h.typer.ObjectKinds(object)
	if err != nil {
		return schema.GroupVersionKind{}, err
	}
	return gvks[0], nil
}
