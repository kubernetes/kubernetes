/*
Copyright 2014 The Kubernetes Authors.

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

package run

import (
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"

	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

func TestGetRestartPolicy(t *testing.T) {
	tests := []struct {
		input       string
		interactive bool
		expected    corev1.RestartPolicy
		expectErr   bool
	}{
		{
			input:    "",
			expected: corev1.RestartPolicyAlways,
		},
		{
			input:       "",
			interactive: true,
			expected:    corev1.RestartPolicyOnFailure,
		},
		{
			input:       string(corev1.RestartPolicyAlways),
			interactive: true,
			expected:    corev1.RestartPolicyAlways,
		},
		{
			input:       string(corev1.RestartPolicyNever),
			interactive: true,
			expected:    corev1.RestartPolicyNever,
		},
		{
			input:    string(corev1.RestartPolicyAlways),
			expected: corev1.RestartPolicyAlways,
		},
		{
			input:    string(corev1.RestartPolicyNever),
			expected: corev1.RestartPolicyNever,
		},
		{
			input:     "foo",
			expectErr: true,
		},
	}
	for _, test := range tests {
		policy, err := getRestartPolicy(test.input, test.interactive)
		if test.expectErr && err == nil {
			t.Error("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !test.expectErr && policy != test.expected {
			t.Errorf("expected: %s, saw: %s (%s:%v)", test.expected, policy, test.input, test.interactive)
		}
	}
}

func TestRunArgsFollowDashRules(t *testing.T) {
	tests := []struct {
		args          []string
		argsLenAtDash int
		expectError   bool
		name          string
	}{
		{
			args:          []string{},
			argsLenAtDash: -1,
			expectError:   true,
			name:          "empty",
		},
		{
			args:          []string{"foo"},
			argsLenAtDash: -1,
			expectError:   false,
			name:          "no cmd",
		},
		{
			args:          []string{"foo", "sleep"},
			argsLenAtDash: -1,
			expectError:   false,
			name:          "cmd no dash",
		},
		{
			args:          []string{"foo", "sleep"},
			argsLenAtDash: 1,
			expectError:   false,
			name:          "cmd has dash",
		},
		{
			args:          []string{"foo", "sleep"},
			argsLenAtDash: 0,
			expectError:   true,
			name:          "no name",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			opts := &RunOptions{
				Image:         "nginx",
				ArgsLenAtDash: test.argsLenAtDash,
			}

			err := opts.Validate(test.args)
			if test.expectError && err == nil {
				t.Errorf("unexpected non-error (%s)", test.name)
			}
			if !test.expectError && err != nil {
				t.Errorf("unexpected error: %v (%s)", err, test.name)
			}
		})
	}
}

func TestExpose(t *testing.T) {
	tests := []struct {
		name           string
		args           []string
		command        bool
		imageName      string
		labels         string
		port           string
		expectedOutput string
	}{
		{
			name:      "basic",
			args:      []string{"test-pod"},
			imageName: "test-image",
			port:      "80",
			expectedOutput: `apiVersion: v1
kind: Pod
metadata:
  labels:
    run: test-pod
  name: test-pod
  namespace: ns
spec:
  containers:
  - image: test-image
    name: test-pod
    ports:
    - containerPort: 80
    resources: {}
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
---
apiVersion: v1
kind: Service
metadata:
  name: test-pod
  namespace: ns
spec:
  ports:
  - port: 80
    protocol: TCP
    targetPort: 80
  selector:
    run: test-pod
status:
  loadBalancer: {}
`,
		},
		{
			name:      "custom labels",
			args:      []string{"test-pod"},
			imageName: "test-image",
			labels:    "color=red,shape=square",
			port:      "80",
			expectedOutput: `apiVersion: v1
kind: Pod
metadata:
  labels:
    color: red
    shape: square
  name: test-pod
  namespace: ns
spec:
  containers:
  - image: test-image
    name: test-pod
    ports:
    - containerPort: 80
    resources: {}
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
---
apiVersion: v1
kind: Service
metadata:
  labels:
    color: red
    shape: square
  name: test-pod
  namespace: ns
spec:
  ports:
  - port: 80
    protocol: TCP
    targetPort: 80
  selector:
    color: red
    shape: square
status:
  loadBalancer: {}
`,
		},
		{
			name:      "with args",
			args:      []string{"test-pod", "run-cmd", "args"},
			imageName: "test-image",
			port:      "80",
			expectedOutput: `apiVersion: v1
kind: Pod
metadata:
  labels:
    run: test-pod
  name: test-pod
  namespace: ns
spec:
  containers:
  - args:
    - run-cmd
    - args
    image: test-image
    name: test-pod
    ports:
    - containerPort: 80
    resources: {}
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
---
apiVersion: v1
kind: Service
metadata:
  name: test-pod
  namespace: ns
spec:
  ports:
  - port: 80
    protocol: TCP
    targetPort: 80
  selector:
    run: test-pod
status:
  loadBalancer: {}
`,
		},
		{
			name:      "with args and command",
			args:      []string{"test-pod", "run-cmd", "args"},
			command:   true,
			imageName: "test-image",
			port:      "80",
			expectedOutput: `apiVersion: v1
kind: Pod
metadata:
  labels:
    run: test-pod
  name: test-pod
  namespace: ns
spec:
  containers:
  - command:
    - run-cmd
    - args
    image: test-image
    name: test-pod
    ports:
    - containerPort: 80
    resources: {}
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
---
apiVersion: v1
kind: Service
metadata:
  name: test-pod
  namespace: ns
spec:
  ports:
  - port: 80
    protocol: TCP
    targetPort: 80
  selector:
    run: test-pod
status:
  loadBalancer: {}
`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("ns")
			defer tf.Cleanup()

			streams, _, bufOut, _ := genericiooptions.NewTestIOStreams()

			cmd := NewCmdRun(tf, streams)
			cmd.Flags().Set("dry-run", "client")     // nolint:errcheck
			cmd.Flags().Set("output", "yaml")        // nolint:errcheck
			cmd.Flags().Set("image", test.imageName) // nolint:errcheck
			cmd.Flags().Set("labels", test.labels)   // nolint:errcheck
			cmd.Flags().Set("expose", "true")        // nolint:errcheck
			cmd.Flags().Set("port", test.port)       // nolint:errcheck
			if test.command {
				cmd.Flags().Set("command", "true") // nolint:errcheck
			}
			cmd.Run(cmd, test.args)
			actualOutput := bufOut.String()
			if actualOutput != test.expectedOutput {
				t.Errorf("unexpected output.\n\nExpected:\n%v\nActual:\n%v", test.expectedOutput, actualOutput)
			}
		})
	}
}

func TestRunValidations(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		image       string
		rm          bool
		attach      bool
		stdin       bool
		tty         bool
		dryRun      cmdutil.DryRunStrategy
		expectedErr string
	}{
		{
			name:        "test missing name error",
			expectedErr: "NAME is required",
		},
		{
			name:        "test missing --image error",
			args:        []string{"test"},
			expectedErr: "--image is required",
		},
		{
			name:        "test invalid image name error",
			args:        []string{"test"},
			image:       "#",
			expectedErr: "invalid image name",
		},
		{
			name:        "test rm errors when used on non-attached containers",
			args:        []string{"test"},
			image:       "busybox",
			rm:          true,
			expectedErr: "rm should only be used for attached containers",
		},
		{
			name:        "test error on attached containers options",
			args:        []string{"test"},
			image:       "busybox",
			attach:      true,
			dryRun:      cmdutil.DryRunClient,
			expectedErr: "can't be used with attached containers options",
		},
		{
			name:        "test error on attached containers options, with value from stdin",
			args:        []string{"test"},
			image:       "busybox",
			stdin:       true,
			dryRun:      cmdutil.DryRunClient,
			expectedErr: "can't be used with attached containers options",
		},
		{
			name:        "test error on attached containers options, with value from stdin and tty",
			args:        []string{"test"},
			image:       "busybox",
			tty:         true,
			stdin:       true,
			dryRun:      cmdutil.DryRunClient,
			expectedErr: "can't be used with attached containers options",
		},
		{
			name:        "test error when tty=true and no stdin provided",
			args:        []string{"test"},
			image:       "busybox",
			tty:         true,
			expectedErr: "stdin is required for containers with -t/--tty",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			opts := &RunOptions{
				Image:          test.image,
				Remove:         test.rm,
				Attach:         test.attach,
				Interactive:    test.stdin,
				TTY:            test.tty,
				DryRunStrategy: test.dryRun,
				ArgsLenAtDash:  1,
			}
			err := opts.Validate(test.args)
			if err != nil && len(test.expectedErr) > 0 {
				if !strings.Contains(err.Error(), test.expectedErr) {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestOverride(t *testing.T) {
	tests := []struct {
		name           string
		podName        string
		imageName      string
		overrides      string
		overrideType   string
		expectedOutput string
	}{
		{
			name:         "run with merge override type should replace spec",
			podName:      "test",
			imageName:    "busybox",
			overrides:    `{"spec":{"containers":[{"name":"test","resources":{"limits":{"cpu":"200m"}}}]}}`,
			overrideType: "merge",
			expectedOutput: `apiVersion: v1
kind: Pod
metadata:
  labels:
    run: test
  name: test
  namespace: ns
spec:
  containers:
  - name: test
    resources:
      limits:
        cpu: 200m
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
`,
		},
		{
			name:         "run with no override type specified, should perform an RFC7396 JSON Merge Patch",
			podName:      "test",
			imageName:    "busybox",
			overrides:    `{"spec":{"containers":[{"name":"test","resources":{"limits":{"cpu":"200m"}}}]}}`,
			overrideType: "",
			expectedOutput: `apiVersion: v1
kind: Pod
metadata:
  labels:
    run: test
  name: test
  namespace: ns
spec:
  containers:
  - name: test
    resources:
      limits:
        cpu: 200m
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
`,
		},
		{
			name:         "run with strategic override type should merge spec, preserving container image",
			podName:      "test",
			imageName:    "busybox",
			overrides:    `{"spec":{"containers":[{"name":"test","resources":{"limits":{"cpu":"200m"}}}]}}`,
			overrideType: "strategic",
			expectedOutput: `apiVersion: v1
kind: Pod
metadata:
  labels:
    run: test
  name: test
  namespace: ns
spec:
  containers:
  - image: busybox
    name: test
    resources:
      limits:
        cpu: 200m
  dnsPolicy: ClusterFirst
  restartPolicy: Always
status: {}
`,
		},
		{
			name:      "run with json override type should perform add, replace, and remove operations",
			podName:   "test",
			imageName: "busybox",
			overrides: `[
						{"op": "add", "path": "/metadata/labels/foo", "value": "bar"},
						{"op": "replace", "path": "/spec/containers/0/resources", "value": {"limits": {"cpu": "200m"}}},
						{"op": "remove", "path": "/spec/dnsPolicy"}
					]`,
			overrideType: "json",
			expectedOutput: `apiVersion: v1
kind: Pod
metadata:
  labels:
    foo: bar
    run: test
  name: test
  namespace: ns
spec:
  containers:
  - image: busybox
    name: test
    resources:
      limits:
        cpu: 200m
  restartPolicy: Always
status: {}
`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("ns")
			defer tf.Cleanup()

			streams, _, bufOut, _ := genericiooptions.NewTestIOStreams()

			cmd := NewCmdRun(tf, streams)
			cmd.Flags().Set("dry-run", "client")                // nolint:errcheck
			cmd.Flags().Set("output", "yaml")                   // nolint:errcheck
			cmd.Flags().Set("image", test.imageName)            // nolint:errcheck
			cmd.Flags().Set("overrides", test.overrides)        // nolint:errcheck
			cmd.Flags().Set("override-type", test.overrideType) // nolint:errcheck
			cmd.Run(cmd, []string{test.podName})
			actualOutput := bufOut.String()
			if actualOutput != test.expectedOutput {
				t.Errorf("unexpected output.\n\nExpected:\n%v\nActual:\n%v", test.expectedOutput, actualOutput)
			}
		})
	}
}

func TestParseLabels(t *testing.T) {
	successCases := []struct {
		name     string
		labels   string
		expected map[string]string
	}{
		{
			name:   "test1",
			labels: "foo=false",
			expected: map[string]string{
				"foo": "false",
			},
		},
		{
			name:   "test2",
			labels: "foo=true,bar=123",
			expected: map[string]string{
				"foo": "true",
				"bar": "123",
			},
		},
	}
	for _, tt := range successCases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseLabels(tt.labels)
			if err != nil {
				t.Errorf("unexpected error :%v", err)
			}
			if !reflect.DeepEqual(tt.expected, got) {
				t.Errorf("\nexpected:\n%v\ngot:\n%v", tt.expected, got)
			}
		})
	}

	errorCases := []struct {
		name   string
		labels string
	}{
		{
			name:   "error format",
			labels: "abc=456;bcd=789",
		},
		{
			name:   "error format",
			labels: "abc=456.bcd=789",
		},
		{
			name:   "error format",
			labels: "abc,789",
		},
		{
			name:   "error format",
			labels: "abc",
		},
		{
			name:   "error format",
			labels: "=abc",
		},
	}
	for _, test := range errorCases {
		_, err := parseLabels(test.labels)
		if err == nil {
			t.Errorf("labels %s expect error, reason: %s, got nil", test.labels, test.name)
		}
	}
}

func TestParseEnv(t *testing.T) {
	tests := []struct {
		name      string
		envArray  []string
		expected  []corev1.EnvVar
		expectErr bool
		test      string
	}{
		{
			name: "test1",
			envArray: []string{
				"THIS_ENV=isOK",
				"this.dotted.env=isOKToo",
				"HAS_COMMAS=foo,bar",
				"HAS_EQUALS=jJnro54iUu75xNy==",
			},
			expected: []corev1.EnvVar{
				{
					Name:  "THIS_ENV",
					Value: "isOK",
				},
				{
					Name:  "this.dotted.env",
					Value: "isOKToo",
				},
				{
					Name:  "HAS_COMMAS",
					Value: "foo,bar",
				},
				{
					Name:  "HAS_EQUALS",
					Value: "jJnro54iUu75xNy==",
				},
			},
			expectErr: false,
			test:      "test case 1",
		},
		{
			name: "test2",
			envArray: []string{
				"WITH_OUT_EQUALS",
			},
			expected:  []corev1.EnvVar{},
			expectErr: true,
			test:      "test case 2",
		},
		{
			name: "test3",
			envArray: []string{
				"WITH_OUT_VALUES=",
			},
			expected: []corev1.EnvVar{
				{
					Name:  "WITH_OUT_VALUES",
					Value: "",
				},
			},
			expectErr: false,
			test:      "test case 3",
		},
		{
			name: "test4",
			envArray: []string{
				"=WITH_OUT_NAME",
			},
			expected:  []corev1.EnvVar{},
			expectErr: true,
			test:      "test case 4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			envs, err := parseEnvs(tt.envArray)
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v (%s)", err, tt.test)
			}
			if tt.expectErr && err != nil {
				return
			}
			if !reflect.DeepEqual(envs, tt.expected) {
				t.Errorf("\nexpected:\n%#v\nsaw:\n%#v (%s)", tt.expected, envs, tt.test)
			}
		})
	}
}
