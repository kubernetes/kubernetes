package printers

import (
	"bytes"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestPrintObj(t *testing.T) {
	tests := []struct {
		name          string
		operation     string
		shortOutput   bool
		showKind      bool
		groupKind     schema.GroupKind
		expected      string
		expectedError bool
	}{
		// must be: [kind.group/name operation]
		{
			name:        "testing-pod",
			operation:   "edited",
			showKind:    true,
			groupKind: schema.GroupKind{
				Group: "test",
				Kind:  "pod",
			},
			expected: "pod.test/testing-pod edited\n",
		},
		// when shortOutput is true, operation is omitted
		{
			name:        "testing-pod",
			operation:   "edited",
			shortOutput: true,
			showKind:    true,
			groupKind: schema.GroupKind{
				Group: "test",
				Kind:  "pod",
			},
			expected: "pod.test/testing-pod\n",
		},
		// when Group is empty, it's omitted
		{
			name:        "testing-pod",
			operation:   "edited",
			showKind:    true,
			groupKind: schema.GroupKind{
				Kind:  "pod",
			},
			expected: "pod/testing-pod edited\n",
		},
		// when showKind is false, it's omitted
		{
			name:        "testing-pod",
			operation:   "edited",
			groupKind: schema.GroupKind{
				Group: "test",
				Kind:  "pod",
			},
			expected: "testing-pod edited\n",
		},
		// when Kind is empty, error should be responded
		{
			name:        "testing-pod",
			operation:   "edited",
			groupKind: schema.GroupKind{
				Group: "test",
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		out := bytes.NewBuffer([]byte{})
		err := printObj(out, test.name, test.operation, test.shortOutput, test.showKind, test.groupKind)

		if test.expectedError != (err != nil) {
			t.Errorf("Expected Error fail. Should be (%v); got (%s)", test.expectedError, err)
		}

		if out.String() != test.expected {
			t.Errorf("Error output. Should be (%s); got (%s)", test.expected, out.String())
		}
	}
}
