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

package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	stdstrings "strings"
	"testing"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

func initTestErrorHandler(t *testing.T) {
	cmdutil.BehaviorOnFatal(func(str string, code int) {
		t.Errorf("Error running command (exit code %d): %s", code, str)
	})
}

func defaultHeader() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}

func defaultClientConfig() *restclient.Config {
	return &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: scheme.Codecs,
			ContentType:          runtime.ContentTypeJSON,
			GroupVersion:         &schema.GroupVersion{Version: "v1"},
		},
	}
}

func testData() (*corev1.PodList, *corev1.ServiceList, *corev1.ReplicationControllerList) {
	pods := &corev1.PodList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "15",
		},
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
				Spec:       apitesting.V1DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "11"},
				Spec:       apitesting.V1DeepEqualSafePodSpec(),
			},
		},
	}
	svc := &corev1.ServiceList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "16",
		},
		Items: []corev1.Service{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					SessionAffinity: "None",
					Type:            corev1.ServiceTypeClusterIP,
				},
			},
		},
	}
	rc := &corev1.ReplicationControllerList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "17",
		},
		Items: []corev1.ReplicationController{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "rc1", Namespace: "test", ResourceVersion: "18"},
				Spec: corev1.ReplicationControllerSpec{
					Replicas: int32ptr(1),
				},
			},
		},
	}
	return pods, svc, rc
}

func int32ptr(val int) *int32 {
	t := int32(val)
	return &t
}

func objBody(codec runtime.Codec, obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj))))
}

func policyObjBody(obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(testapi.Policy.Codec(), obj))))
}

func bytesBody(bodyBytes []byte) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader(bodyBytes))
}

func stringBody(body string) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(body)))
}

func TestNormalizationFuncGlobalExistence(t *testing.T) {
	// This test can be safely deleted when we will not support multiple flag formats
	root := NewKubectlCommand(os.Stdin, os.Stdout, os.Stderr)

	if root.Parent() != nil {
		t.Fatal("We expect the root command to be returned")
	}
	if root.GlobalNormalizationFunc() == nil {
		t.Fatal("We expect that root command has a global normalization function")
	}

	if reflect.ValueOf(root.GlobalNormalizationFunc()).Pointer() != reflect.ValueOf(root.Flags().GetNormalizeFunc()).Pointer() {
		t.Fatal("root command seems to have a wrong normalization function")
	}

	sub := root
	for sub.HasSubCommands() {
		sub = sub.Commands()[0]
	}

	// In case of failure of this test check this PR: spf13/cobra#110
	if reflect.ValueOf(sub.Flags().GetNormalizeFunc()).Pointer() != reflect.ValueOf(root.Flags().GetNormalizeFunc()).Pointer() {
		t.Fatal("child and root commands should have the same normalization functions")
	}
}

func genResponseWithJsonEncodedBody(bodyStruct interface{}) (*http.Response, error) {
	jsonBytes, err := json.Marshal(bodyStruct)
	if err != nil {
		return nil, err
	}
	return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bytesBody(jsonBytes)}, nil
}

func Test_deprecatedAlias(t *testing.T) {
	var correctCommandCalled bool
	makeCobraCommand := func() *cobra.Command {
		cobraCmd := new(cobra.Command)
		cobraCmd.Use = "print five lines"
		cobraCmd.Run = func(*cobra.Command, []string) {
			correctCommandCalled = true
		}
		return cobraCmd
	}

	original := makeCobraCommand()
	alias := deprecatedAlias("echo", makeCobraCommand())

	if len(alias.Deprecated) == 0 {
		t.Error("deprecatedAlias should always have a non-empty .Deprecated")
	}
	if !stdstrings.Contains(alias.Deprecated, "print") {
		t.Error("deprecatedAlias should give the name of the new function in its .Deprecated field")
	}
	if !alias.Hidden {
		t.Error("deprecatedAlias should never have .Hidden == false (deprecated aliases should be hidden)")
	}

	if alias.Name() != "echo" {
		t.Errorf("deprecatedAlias has name %q, expected %q",
			alias.Name(), "echo")
	}
	if original.Name() != "print" {
		t.Errorf("original command has name %q, expected %q",
			original.Name(), "print")
	}

	buffer := new(bytes.Buffer)
	alias.SetOutput(buffer)
	alias.Execute()
	str := buffer.String()
	if !stdstrings.Contains(str, "deprecated") || !stdstrings.Contains(str, "print") {
		t.Errorf("deprecation warning %q does not include enough information", str)
	}

	// It would be nice to test to see that original.Run == alias.Run
	// Unfortunately Golang does not allow comparing functions. I could do
	// this with reflect, but that's technically invoking undefined
	// behavior. Best we can do is make sure that the function is called.
	if !correctCommandCalled {
		t.Errorf("original function doesn't appear to have been called by alias")
	}
}

func TestKubectlCommandHandlesPlugins(t *testing.T) {
	tests := []struct {
		name             string
		args             []string
		expectPlugin     string
		expectPluginArgs []string
		expectError      string
	}{
		{
			name:             "test that normal commands are able to be executed, when no plugin overshadows them",
			args:             []string{"kubectl", "get", "foo"},
			expectPlugin:     "",
			expectPluginArgs: []string{},
		},
		{
			name:             "test that a plugin executable is found based on command args",
			args:             []string{"kubectl", "foo", "--bar"},
			expectPlugin:     "testdata/plugin/kubectl-foo",
			expectPluginArgs: []string{"foo", "--bar"},
		},
		{
			name: "test that a plugin does not execute over an existing command by the same name",
			args: []string{"kubectl", "version"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pluginsHandler := &testPluginHandler{
				pluginsDirectory: "testdata/plugin",
			}
			_, in, out, errOut := genericclioptions.NewTestIOStreams()

			cmdutil.BehaviorOnFatal(func(str string, code int) {
				errOut.Write([]byte(str))
			})

			root := NewDefaultKubectlCommandWithArgs(pluginsHandler, test.args, in, out, errOut)
			if err := root.Execute(); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if pluginsHandler.err != nil && pluginsHandler.err.Error() != test.expectError {
				t.Fatalf("unexpected error: expected %q to occur, but got %q", test.expectError, pluginsHandler.err)
			}

			if pluginsHandler.executedPlugin != test.expectPlugin {
				t.Fatalf("unexpected plugin execution: expedcted %q, got %q", test.expectPlugin, pluginsHandler.executedPlugin)
			}

			if len(pluginsHandler.withArgs) != len(test.expectPluginArgs) {
				t.Fatalf("unexpected plugin execution args: expedcted %q, got %q", test.expectPluginArgs, pluginsHandler.withArgs)
			}
		})
	}
}

type testPluginHandler struct {
	pluginsDirectory string

	// execution results
	executedPlugin string
	withArgs       []string
	withEnv        []string

	err error
}

func (h *testPluginHandler) Lookup(filename string) (string, error) {
	dir, err := os.Stat(h.pluginsDirectory)
	if err != nil {
		h.err = err
		return "", err
	}

	if !dir.IsDir() {
		h.err = fmt.Errorf("expected %q to be a directory", h.pluginsDirectory)
		return "", h.err
	}

	plugins, err := ioutil.ReadDir(h.pluginsDirectory)
	if err != nil {
		h.err = err
		return "", err
	}

	for _, p := range plugins {
		if p.Name() == filename {
			return fmt.Sprintf("%s/%s", h.pluginsDirectory, p.Name()), nil
		}
	}

	h.err = fmt.Errorf("unable to find a plugin executable %q", filename)
	return "", h.err
}

func (h *testPluginHandler) Execute(executablePath string, cmdArgs, env []string) error {
	h.executedPlugin = executablePath
	h.withArgs = cmdArgs
	h.withEnv = env
	return nil
}
