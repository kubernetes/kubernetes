package genericclioptions

import (
	"testing"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestRecordFlags_ToRecorder(t *testing.T) {
	tests := []struct {
		name      string
		record    bool
		expectErr bool
	}{
		{
			name:      "Record is true",
			record:    true,
			expectErr: false,
		},
		{
			name:      "Record is false",
			record:    false,
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags := &RecordFlags{
				Record: &tt.record,
			}
			recorder, err := flags.ToRecorder()
			if (err != nil) != tt.expectErr {
				t.Errorf("ToRecorder() error = %v, expectErr %v", err, tt.expectErr)
				return
			}

			if tt.record {
				if _, ok := recorder.(*ChangeCauseRecorder); !ok {
					t.Errorf("expected ChangeCauseRecorder, got %T", recorder)
				}
			} else {
				if _, ok := recorder.(NoopRecorder); !ok {
					t.Errorf("expected NoopRecorder, got %T", recorder)
				}
			}
		})
	}
}

func TestRecordFlags_Complete(t *testing.T) {
	cmd := &cobra.Command{
		Use: "test",
		Run: func(cmd *cobra.Command, args []string) {},
	}
	flags := NewRecordFlags()
	flags.AddFlags(cmd)

	cmd.SetArgs([]string{"--record=true"})

	err := cmd.Execute()
	if err != nil {
		t.Fatalf("Execute() failed: %v", err)
	}

	err = flags.Complete(cmd)
	if err != nil {
		t.Fatalf("Complete() failed: %v", err)
	}

	if flags.changeCause == "" {
		t.Errorf("expected changeCause to be set, got empty string")
	}
}

func TestChangeCauseRecorder_Record(t *testing.T) {
	obj := &unstructured.Unstructured{}
	recorder := ChangeCauseRecorder{
		changeCause: "test change",
	}

	err := recorder.Record(obj)
	if err != nil {
		t.Fatalf("Record() failed: %v", err)
	}

	annotations := obj.GetAnnotations()
	if annotations[ChangeCauseAnnotation] != "test change" {
		t.Errorf("expected changeCause annotation to be set, got %v", annotations[ChangeCauseAnnotation])
	}
}

func TestChangeCauseRecorder_MakeRecordMergePatch(t *testing.T) {
	obj := &unstructured.Unstructured{}
	recorder := ChangeCauseRecorder{
		changeCause: "test change",
	}

	patch, err := recorder.MakeRecordMergePatch(obj)
	if err != nil {
		t.Fatalf("MakeRecordMergePatch() failed: %v", err)
	}

	if len(patch) == 0 {
		t.Errorf("expected non-empty patch, got empty patch")
	}
}
