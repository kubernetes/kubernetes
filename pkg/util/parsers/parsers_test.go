/*
Copyright 2016 The Kubernetes Authors.

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

package parsers

import (
	"strings"
	"testing"
	"time"
)

// Based on Docker test case removed in:
// https://github.com/docker/docker/commit/4352da7803d182a6013a5238ce20a7c749db979a
func TestParseImageName(t *testing.T) {
	testCases := []struct {
		Input         string
		Repo          string
		Tag           string
		Digest        string
		expectedError string
	}{
		{Input: "root", Repo: "docker.io/library/root", Tag: "latest"},
		{Input: "root:tag", Repo: "docker.io/library/root", Tag: "tag"},
		{Input: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "docker.io/library/root", Digest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{Input: "user/repo", Repo: "docker.io/user/repo", Tag: "latest"},
		{Input: "user/repo:tag", Repo: "docker.io/user/repo", Tag: "tag"},
		{Input: "user/repo@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "docker.io/user/repo", Digest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{Input: "url:5000/repo", Repo: "url:5000/repo", Tag: "latest"},
		{Input: "url:5000/repo:tag", Repo: "url:5000/repo", Tag: "tag"},
		{Input: "url:5000/repo@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "url:5000/repo", Digest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{Input: "url:5000/repo:latest@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Tag: "latest", Repo: "url:5000/repo", Digest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{Input: "ROOT", expectedError: "must be lowercase"},
		{Input: "http://root", expectedError: "invalid reference format"},
		{Input: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", expectedError: "cannot specify 64-byte hexadecimal strings"},
	}
	for _, testCase := range testCases {
		repo, tag, digest, err := ParseImageName(testCase.Input)
		switch {
		case testCase.expectedError != "" && !strings.Contains(err.Error(), testCase.expectedError):
			t.Errorf("ParseImageName(%s) expects error %v but did not get one", testCase.Input, err)
		case testCase.expectedError == "" && err != nil:
			t.Errorf("ParseImageName(%s) failed: %v", testCase.Input, err)
		case repo != testCase.Repo || tag != testCase.Tag || digest != testCase.Digest:
			t.Errorf("Expected repo: %q, tag: %q and digest: %q, got %q, %q and %q", testCase.Repo, testCase.Tag, testCase.Digest,
				repo, tag, digest)
		}
	}
}

func TestParseCronSchedule(t *testing.T) {
	testCases := []struct {
		name          string
		schedule      string
		expectError   bool
		expectedError string
		expectPanic   bool
	}{
		{
			name:        "valid schedule without timezone",
			schedule:    "0 0 * * *",
			expectError: false,
		},
		{
			name:        "valid schedule with timezone",
			schedule:    "TZ=UTC 0 0 * * *",
			expectError: false,
		},
		{
			name:        "valid schedule with CRON_TZ",
			schedule:    "CRON_TZ=America/New_York 0 0 * * *",
			expectError: false,
		},
		{
			name:          "TZ=0 without space should panic and be recovered",
			schedule:      "TZ=0",
			expectError:   true,
			expectedError: "invalid schedule format",
			expectPanic:   true,
		},
		{
			name:          "TZ= without value should panic and be recovered",
			schedule:      "TZ=",
			expectError:   true,
			expectedError: "invalid schedule format",
			expectPanic:   true,
		},
		{
			name:          "CRON_TZ= without space should panic and be recovered",
			schedule:      "CRON_TZ=UTC",
			expectError:   true,
			expectedError: "invalid schedule format",
			expectPanic:   true,
		},
		{
			name:          "malformed timezone spec should panic and be recovered",
			schedule:      "TZ=Invalid/Timezone",
			expectError:   true,
			expectedError: "invalid schedule format",
			expectPanic:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// This should not panic even with malformed schedules
			sched, err := ParseCronScheduleWithPanicRecovery(tc.schedule)

			if tc.expectError {
				if err == nil {
					t.Errorf("Expected error for schedule %q, but got none", tc.schedule)
					return
				}

				if tc.expectedError != "" {
					errMsg := err.Error()
					if !strings.Contains(errMsg, tc.expectedError) {
						t.Errorf("Expected error message to contain %q, but got: %s", tc.expectedError, errMsg)
					}
				}

				if sched != nil {
					t.Errorf("Expected nil schedule when error occurs, but got: %v", sched)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error for schedule %q, but got: %v", tc.schedule, err)
					return
				}

				if sched == nil {
					t.Errorf("Expected valid schedule for %q, but got nil", tc.schedule)
					return
				}

				// Verify that the schedule is actually valid by calling Next
				// This ensures we got a real cron.Schedule, not just a non-nil value
				next := sched.Next(time.Now())
				if next.IsZero() {
					t.Errorf("Expected valid next execution time for schedule %q, but got zero time", tc.schedule)
				}
			}
		})
	}
}

func TestParseCronSchedulePanicRecovery(t *testing.T) {
	// Test that panics are properly recovered and converted to errors
	panicSchedules := []string{
		"TZ=0",
		"TZ=",
		"CRON_TZ=UTC",
		"TZ=Invalid/Timezone",
	}

	for _, schedule := range panicSchedules {
		t.Run("panic_recovery_"+schedule, func(t *testing.T) {
			// This should not panic
			sched, err := ParseCronScheduleWithPanicRecovery(schedule)

			// Should get an error
			if err == nil {
				t.Errorf("Expected error for panic-causing schedule %q, but got none", schedule)
			}

			// Should get nil schedule
			if sched != nil {
				t.Errorf("Expected nil schedule for panic-causing schedule %q, but got: %v", schedule, sched)
			}

			// Error message should contain "invalid schedule format"
			errMsg := err.Error()
			if !strings.Contains(errMsg, "invalid schedule format") {
				t.Errorf("Expected error message to contain 'invalid schedule format', but got: %s", errMsg)
			}
		})
	}
}
