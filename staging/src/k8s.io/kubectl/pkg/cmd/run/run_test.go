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
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubectl/pkg/cmd/delete"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	generateversioned "k8s.io/kubectl/pkg/generate/versioned"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
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
		cmd := &cobra.Command{}
		cmd.Flags().String("restart", "", i18n.T("dummy restart flag)"))
		cmd.Flags().Lookup("restart").Value.Set(test.input)
		policy, err := getRestartPolicy(cmd, test.interactive)
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

func TestGetEnv(t *testing.T) {
	test := struct {
		input    []string
		expected []string
	}{
		input:    []string{"a=b", "c=d"},
		expected: []string{"a=b", "c=d"},
	}
	cmd := &cobra.Command{}
	cmd.Flags().StringSlice("env", test.input, "")

	envStrings := cmdutil.GetFlagStringSlice(cmd, "env")
	if len(envStrings) != 2 || !reflect.DeepEqual(envStrings, test.expected) {
		t.Errorf("expected: %s, saw: %s", test.expected, envStrings)
	}
}

func TestRunArgsFollowDashRules(t *testing.T) {
	one := int32(1)
	rc := &corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "rc1", Namespace: "test", ResourceVersion: "18"},
		Spec: corev1.ReplicationControllerSpec{
			Replicas: &one,
		},
	}

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
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				GroupVersion:         corev1.SchemeGroupVersion,
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					if req.URL.Path == "/namespaces/test/pods" {
						return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, rc)}, nil
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       ioutil.NopCloser(bytes.NewBuffer([]byte("{}"))),
					}, nil
				}),
			}

			tf.ClientConfigVal = &restclient.Config{}

			cmd := NewCmdRun(tf, genericclioptions.NewTestIOStreamsDiscard())
			cmd.Flags().Set("image", "nginx")

			printFlags := genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme)
			printer, err := printFlags.ToPrinter()
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			deleteFlags := delete.NewDeleteFlags("to use to replace the resource.")
			opts := &RunOptions{
				PrintFlags:    printFlags,
				DeleteOptions: deleteFlags.ToOptions(nil, genericclioptions.NewTestIOStreamsDiscard()),

				IOStreams: genericclioptions.NewTestIOStreamsDiscard(),

				Image:     "nginx",
				Generator: generateversioned.RunPodV1GeneratorName,

				PrintObj: func(obj runtime.Object) error {
					return printer.PrintObj(obj, os.Stdout)
				},
				Recorder: genericclioptions.NoopRecorder{},

				ArgsLenAtDash: test.argsLenAtDash,
			}

			err = opts.Run(tf, cmd, test.args)
			if test.expectError && err == nil {
				t.Errorf("unexpected non-error (%s)", test.name)
			}
			if !test.expectError && err != nil {
				t.Errorf("unexpected error: %v (%s)", err, test.name)
			}
		})
	}
}

func TestGenerateService(t *testing.T) {
	tests := []struct {
		name             string
		port             string
		args             []string
		serviceGenerator string
		params           map[string]interface{}
		expectErr        bool
		service          corev1.Service
		expectPOST       bool
	}{
		{
			name:             "basic",
			port:             "80",
			args:             []string{"foo"},
			serviceGenerator: "service/v2",
			params: map[string]interface{}{
				"name": "foo",
			},
			expectErr: false,
			service: corev1.Service{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Service",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt(80),
						},
					},
					Selector: map[string]string{
						"run": "foo",
					},
				},
			},
			expectPOST: true,
		},
		{
			name:             "custom labels",
			port:             "80",
			args:             []string{"foo"},
			serviceGenerator: "service/v2",
			params: map[string]interface{}{
				"name":   "foo",
				"labels": "app=bar",
			},
			expectErr: false,
			service: corev1.Service{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Service",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo",
					Labels: map[string]string{"app": "bar"},
				},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt(80),
						},
					},
					Selector: map[string]string{
						"app": "bar",
					},
				},
			},
			expectPOST: true,
		},
		{
			expectErr:  true,
			name:       "missing port",
			expectPOST: false,
		},
		{
			name:             "dry-run",
			port:             "80",
			args:             []string{"foo"},
			serviceGenerator: "service/v2",
			params: map[string]interface{}{
				"name": "foo",
			},
			expectErr:  false,
			expectPOST: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sawPOST := false
			tf := cmdtesting.NewTestFactory()
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs.WithoutConversion()

			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			tf.Client = &fake.RESTClient{
				GroupVersion:         corev1.SchemeGroupVersion,
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case test.expectPOST && m == "POST" && p == "/namespaces/test/services":
						sawPOST = true
						body := cmdtesting.ObjBody(codec, &test.service)
						data, err := ioutil.ReadAll(req.Body)
						if err != nil {
							t.Fatalf("unexpected error: %v", err)
						}
						defer req.Body.Close()
						svc := &corev1.Service{}
						if err := runtime.DecodeInto(codec, data, svc); err != nil {
							t.Fatalf("unexpected error: %v", err)
						}
						// Copy things that are defaulted by the system
						test.service.Annotations = svc.Annotations

						if !apiequality.Semantic.DeepEqual(&test.service, svc) {
							t.Errorf("expected:\n%v\nsaw:\n%v\n", &test.service, svc)
						}
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", test.name, req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
			}

			printFlags := genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme)
			printer, err := printFlags.ToPrinter()
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			ioStreams, _, buff, _ := genericclioptions.NewTestIOStreams()
			deleteFlags := delete.NewDeleteFlags("to use to replace the resource.")
			opts := &RunOptions{
				PrintFlags:    printFlags,
				DeleteOptions: deleteFlags.ToOptions(nil, genericclioptions.NewTestIOStreamsDiscard()),

				IOStreams: ioStreams,

				Port:     test.port,
				Recorder: genericclioptions.NoopRecorder{},

				PrintObj: func(obj runtime.Object) error {
					return printer.PrintObj(obj, buff)
				},

				Namespace: "test",
			}

			cmd := &cobra.Command{}
			cmd.Flags().Bool(cmdutil.ApplyAnnotationsFlag, false, "")
			cmd.Flags().Bool("record", false, "Record current kubectl command in the resource annotation. If set to false, do not record the command. If set to true, record the command. If not set, default to updating the existing annotation value only if one already exists.")
			addRunFlags(cmd, opts)

			if !test.expectPOST {
				opts.DryRunStrategy = cmdutil.DryRunClient
			}

			if len(test.port) > 0 {
				cmd.Flags().Set("port", test.port)
				test.params["port"] = test.port
			}

			_, err = opts.generateService(tf, cmd, test.serviceGenerator, test.params)
			if test.expectErr {
				if err == nil {
					t.Error("unexpected non-error")
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if test.expectPOST != sawPOST {
				t.Errorf("expectPost: %v, sawPost: %v", test.expectPOST, sawPOST)
			}
		})
	}
}

func TestRunValidations(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		flags       map[string]string
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
			name: "test invalid image name error",
			args: []string{"test"},
			flags: map[string]string{
				"image": "#",
			},
			expectedErr: "Invalid image name",
		},
		{
			name: "test stdin replicas value",
			args: []string{"test"},
			flags: map[string]string{
				"image":    "busybox",
				"stdin":    "true",
				"replicas": "2",
			},
			expectedErr: "stdin requires that replicas is 1",
		},
		{
			name: "test rm errors when used on non-attached containers",
			args: []string{"test"},
			flags: map[string]string{
				"image": "busybox",
				"rm":    "true",
			},
			expectedErr: "rm should only be used for attached containers",
		},
		{
			name: "test error on attached containers options",
			args: []string{"test"},
			flags: map[string]string{
				"image":   "busybox",
				"attach":  "true",
				"dry-run": "client",
			},
			expectedErr: "can't be used with attached containers options",
		},
		{
			name: "test error on attached containers options, with value from stdin",
			args: []string{"test"},
			flags: map[string]string{
				"image":   "busybox",
				"stdin":   "true",
				"dry-run": "client",
			},
			expectedErr: "can't be used with attached containers options",
		},
		{
			name: "test error on attached containers options, with value from stdin and tty",
			args: []string{"test"},
			flags: map[string]string{
				"image":   "busybox",
				"tty":     "true",
				"stdin":   "true",
				"dry-run": "client",
			},
			expectedErr: "can't be used with attached containers options",
		},
		{
			name: "test error when tty=true and no stdin provided",
			args: []string{"test"},
			flags: map[string]string{
				"image": "busybox",
				"tty":   "true",
			},
			expectedErr: "stdin is required for containers with -t/--tty",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			_, _, codec := cmdtesting.NewExternalScheme()
			ns := scheme.Codecs.WithoutConversion()
			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, cmdtesting.NewInternalType("", "", ""))},
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			streams, _, _, bufErr := genericclioptions.NewTestIOStreams()
			cmdutil.BehaviorOnFatal(func(str string, code int) {
				bufErr.Write([]byte(str))
			})

			cmd := NewCmdRun(tf, streams)
			for flagName, flagValue := range test.flags {
				cmd.Flags().Set(flagName, flagValue)
			}
			cmd.Run(cmd, test.args)

			var err error
			if bufErr.Len() > 0 {
				err = fmt.Errorf("%v", bufErr.String())
			}
			if err != nil && len(test.expectedErr) > 0 {
				if !strings.Contains(err.Error(), test.expectedErr) {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}

}
