package cmdproto

import (
	"reflect"
	"testing"

	"github.com/spf13/cobra"

	cmdproto "k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto/k8s_io_kubectl_cmd"
)

func TestFlags2Proto(t *testing.T) {

	//please add new test field also in this struct
	type Test struct {
		BoolFlag   bool
		StringFlag string
		Int32Flag  int32
		ArrayFlag  []string
		Int64Flag  int64
		TimeFlag   int64
	}

	tests := []struct {
		name                string
		flagName, flagValue []string
		expectedErr         []bool
		expectedOut         Test
	}{
		{
			name:        "test with default value",
			flagName:    []string{},
			flagValue:   []string{},
			expectedErr: []bool{},
			expectedOut: Test{
				BoolFlag:   false,
				StringFlag: "",
				Int32Flag:  0,
				ArrayFlag:  []string{"default1", "default2"},
				Int64Flag:  1234,
				TimeFlag:   1234565666,
			},
		},
		{
			name:        "test with normal value",
			flagName:    []string{"bool-flag", "string-flag", "int32-flag", "repeated-string-flag", "repeated-string-flag", "int64-flag", "time-flag"},
			flagValue:   []string{"true", "testProto", "4544", "123", "456", "111111111", "123s"},
			expectedErr: []bool{false, false, false, false, false, false, false},
			expectedOut: Test{
				BoolFlag:   true,
				StringFlag: "testProto",
				Int32Flag:  4544,
				ArrayFlag:  []string{"123", "456"},
				Int64Flag:  111111111,
				TimeFlag:   123000000000,
			},
		},
		{
			name:        "test with abnormal value",
			flagName:    []string{"bool-flag", "int32-flag", "int64-flag", "time-flag"},
			flagValue:   []string{"not-true", "not-int32", "not-int64", "not-time"},
			expectedErr: []bool{true, true, true, true},
			expectedOut: Test{
				BoolFlag:   false,
				StringFlag: "",
				Int32Flag:  0,
				ArrayFlag:  []string{"default1", "default2"},
				Int64Flag:  0,
				TimeFlag:   0,
			},
		},
	}

	for _, test := range tests {
		testProto := new(cmdproto.TestCmd)
		cmd := &cobra.Command{}

		//Setup flags
		FlagsSetup(cmd, testProto)
		//stimulate user to input flags
		for index, v := range test.flagName {
			err := cmd.Flags().Set(v, test.flagValue[index])
			if test.expectedErr[index] && err == nil {
				t.Errorf("%s: unexpected non-error: %s", test.name, err)
			}
		}

		if testProto.GetBoolFlag() != test.expectedOut.BoolFlag {
			t.Errorf("%s: unexpected value, expected: %v, actually got: %v", test.name, test.expectedOut.BoolFlag, testProto.GetBoolFlag())
		}
		if testProto.GetStringFlag() != test.expectedOut.StringFlag {
			t.Errorf("%s: unexpected value, expected: %v, actually got: %v", test.name, test.expectedOut.StringFlag, testProto.GetStringFlag())
		}
		if testProto.GetInt32Flag() != test.expectedOut.Int32Flag {
			t.Errorf("%s: unexpected value, expected: %v, actually got: %v", test.name, test.expectedOut.Int32Flag, testProto.GetInt32Flag())
		}

		if !reflect.DeepEqual(testProto.GetArrayFlag().Array, test.expectedOut.ArrayFlag) {
			t.Errorf("%s: unexpected value, expected: %v, actually got: %v", test.name, test.expectedOut.ArrayFlag, testProto.GetArrayFlag().Array)
		}
		if testProto.GetInt64Flag() != test.expectedOut.Int64Flag {
			t.Errorf("%s: unexpected value, expected: %v, actually got: %v", test.name, test.expectedOut.Int64Flag, testProto.GetInt64Flag())
		}
		if testProto.GetTimeFlag() != test.expectedOut.TimeFlag {
			t.Errorf("%s: unexpected value, expected: %v, actually got: %v", test.name, test.expectedOut.TimeFlag, testProto.GetTimeFlag())
		}
	}

}

func TestFlagsSetup(t *testing.T) {
}

