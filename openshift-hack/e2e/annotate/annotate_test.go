package annotate

import (
	"fmt"
	"os"
	"testing"
)

func Test_checkBalancedBrackets(t *testing.T) {
	tests := []struct {
		testCase string
		testName string
		wantErr  bool
	}{
		{
			testCase: "balanced brackets succeeds",
			testName: "[sig-storage] Test that storage [apigroup:storage.openshift.io] actually works [Driver:azure][Serial][Late]",
			wantErr:  false,
		},
		{
			testCase: "unbalanced brackets errors",
			testName: "[sig-storage] Test that storage [apigroup:storage.openshift.io actually works [Driver:azure][Serial][Late]",
			wantErr:  true,
		},
		{
			testCase: "start with close bracket errors",
			testName: "[sig-storage] test with a random bracket ]",
			wantErr:  true,
		},
		{
			testCase: "multiple unbalanced brackets errors",
			testName: "[sig-storage Test that storage [apigroup:storage.openshift.io actually works [Driver:azure]",
			wantErr:  true,
		},
		{
			testCase: "balanced deeply nested brackets succeeds",
			testName: "[[[[[[some weird test with deeply nested brackets]]]]]]",
			wantErr:  false,
		},
		{
			testCase: "unbalanced deeply nested brackets errors",
			testName: "[[[[[[some weird test with deeply nested brackets]]]]]",
			wantErr:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.testCase, func(t *testing.T) {
			if err := checkBalancedBrackets(tt.testName); (err != nil) != tt.wantErr {
				t.Errorf("checkBalancedBrackets() error = %v, wantErr %v", err, tt.wantErr)
			} else if err != nil {
				fmt.Fprintf(os.Stderr, "checkBalancedBrackets() success, found expected err = \n%s\n", err.Error())
			}
		})
	}
}
