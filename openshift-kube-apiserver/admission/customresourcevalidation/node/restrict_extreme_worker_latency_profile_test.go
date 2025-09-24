package node

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	configv1 "github.com/openshift/api/config/v1"
)

func TestValidateConfigNodeForExtremeLatencyProfile(t *testing.T) {
	testCases := []struct {
		fromProfile  configv1.WorkerLatencyProfileType
		toProfile    configv1.WorkerLatencyProfileType
		shouldReject bool
	}{
		// no rejections
		{fromProfile: "", toProfile: "", shouldReject: false},
		{fromProfile: "", toProfile: configv1.DefaultUpdateDefaultReaction, shouldReject: false},
		{fromProfile: "", toProfile: configv1.MediumUpdateAverageReaction, shouldReject: false},
		{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: "", shouldReject: false},
		{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: configv1.DefaultUpdateDefaultReaction, shouldReject: false},
		{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: configv1.MediumUpdateAverageReaction, shouldReject: false},
		{fromProfile: configv1.MediumUpdateAverageReaction, toProfile: "", shouldReject: false},
		{fromProfile: configv1.MediumUpdateAverageReaction, toProfile: configv1.DefaultUpdateDefaultReaction, shouldReject: false},
		{fromProfile: configv1.MediumUpdateAverageReaction, toProfile: configv1.MediumUpdateAverageReaction, shouldReject: false},
		{fromProfile: configv1.MediumUpdateAverageReaction, toProfile: configv1.LowUpdateSlowReaction, shouldReject: false},
		{fromProfile: configv1.LowUpdateSlowReaction, toProfile: configv1.MediumUpdateAverageReaction, shouldReject: false},
		{fromProfile: configv1.LowUpdateSlowReaction, toProfile: configv1.LowUpdateSlowReaction, shouldReject: false},

		// rejections
		{fromProfile: "", toProfile: configv1.LowUpdateSlowReaction, shouldReject: true},
		{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: configv1.LowUpdateSlowReaction, shouldReject: true},
		{fromProfile: configv1.LowUpdateSlowReaction, toProfile: "", shouldReject: true},
		{fromProfile: configv1.LowUpdateSlowReaction, toProfile: configv1.DefaultUpdateDefaultReaction, shouldReject: true},
	}

	for _, testCase := range testCases {
		shouldStr := "should not be"
		if testCase.shouldReject {
			shouldStr = "should be"
		}
		testCaseName := fmt.Sprintf("update from profile %s to %s %s rejected", testCase.fromProfile, testCase.toProfile, shouldStr)
		t.Run(testCaseName, func(t *testing.T) {
			// config node objects
			oldObject := configv1.Node{
				Spec: configv1.NodeSpec{
					WorkerLatencyProfile: testCase.fromProfile,
				},
			}
			newObject := configv1.Node{
				Spec: configv1.NodeSpec{
					WorkerLatencyProfile: testCase.toProfile,
				},
			}

			fieldErr := validateConfigNodeForExtremeLatencyProfile(&oldObject, &newObject)
			assert.Equal(t, testCase.shouldReject, fieldErr != nil, "latency profile from %q to %q %s rejected", testCase.fromProfile, testCase.toProfile, shouldStr)

			if testCase.shouldReject {
				assert.Equal(t, "spec.workerLatencyProfile", fieldErr.Field, "field name during for latency profile should be spec.workerLatencyProfile")
				assert.Contains(t, fieldErr.Detail, testCase.fromProfile, "error message should contain %q", testCase.fromProfile)
				assert.Contains(t, fieldErr.Detail, testCase.toProfile, "error message should contain %q", testCase.toProfile)
			}
		})
	}
}
