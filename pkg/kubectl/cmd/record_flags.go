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

package cmd

import (
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/kubectl"
)

// RecordFlags contains all flags associated with the "--record" operation
type RecordFlags struct {
	Record *bool

	changeCause string
}

// ToRecorder returns a ChangeCause recorder if --record=false was not
// explicitly given by the user
func (f *RecordFlags) ToRecorder() (Recorder, error) {
	shouldRecord := false
	if f.Record != nil {
		shouldRecord = *f.Record
	}

	// if flag was explicitly set to false by the user,
	// do not record
	if !shouldRecord {
		return &NoopRecorder{}, nil
	}

	return &ChangeCauseRecorder{
		changeCause: f.changeCause,
	}, nil
}

func (f *RecordFlags) Complete(changeCause string) error {
	f.changeCause = changeCause
	return nil
}

func (f *RecordFlags) AddFlags(cmd *cobra.Command) {
	if f.Record != nil {
		cmd.Flags().BoolVar(f.Record, "record", *f.Record, "Record current kubectl command in the resource annotation. If set to false, do not record the command. If set to true, record the command. If not set, default to updating the existing annotation value only if one already exists.")
	}
}

func NewRecordFlags() *RecordFlags {
	record := false

	return &RecordFlags{
		Record: &record,
	}
}

type Recorder interface {
	Record(runtime.Object) error
}

type NoopRecorder struct{}

func (r *NoopRecorder) Record(obj runtime.Object) error {
	return nil
}

// ChangeCauseRecorder annotates a "change-cause" to an input runtime object
type ChangeCauseRecorder struct {
	changeCause string
}

// Record annotates a "change-cause" to a given info if either "shouldRecord" is true,
// or the resource info previously contained a "change-cause" annotation.
func (r *ChangeCauseRecorder) Record(obj runtime.Object) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	annotations := accessor.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[kubectl.ChangeCauseAnnotation] = r.changeCause
	accessor.SetAnnotations(annotations)
	return nil
}
