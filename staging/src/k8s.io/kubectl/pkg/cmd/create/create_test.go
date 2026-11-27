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

package create

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

func TestExtraArgsFail(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	c := NewCmdCreate(f, genericiooptions.NewTestIOStreamsDiscard())
	ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()
	options := NewCreateOptions(ioStreams)
	if options.Complete(f, c, []string{"rc"}) == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestCreateObject(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	_, _, rc := cmdtesting.TestData()
	rc.Items[0].Name = "redis-master-controller"

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreate(tf, ioStreams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	if buf.String() != "replicationcontroller/redis-master-controller\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateMultipleObject(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	_, svc, rc := cmdtesting.TestData()

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/services" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &svc.Items[0])}, nil
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreate(tf, ioStreams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml")
	cmd.Flags().Set("filename", "../../../testdata/frontend-service.yaml")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// Names should come from the REST response, NOT the files
	if buf.String() != "replicationcontroller/rc1\nservice/baz\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestCreateDirectory(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	_, _, rc := cmdtesting.TestData()
	rc.Items[0].Name = "name"

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/namespaces/test/replicationcontrollers" && m == http.MethodPost:
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &rc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreate(tf, ioStreams)
	cmd.Flags().Set("filename", "../../../testdata/replace/legacy")
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != "replicationcontroller/name\nreplicationcontroller/name\nreplicationcontroller/name\n" {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestEditOptionsInitializedByComplete(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreate(tf, ioStreams)
	cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml") // nolint:errcheck

	t.Run("editOptions is nil when EditBeforeCreate is false", func(t *testing.T) {
		o := NewCreateOptions(ioStreams)
		o.FilenameOptions.Filenames = []string{"../../../testdata/redis-master-controller.yaml"}
		if err := o.Complete(tf, cmd, []string{}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if o.editOptions != nil {
			t.Error("expected editOptions to be nil when EditBeforeCreate is false")
		}
	})

	t.Run("editOptions is initialized when EditBeforeCreate is true", func(t *testing.T) {
		o := NewCreateOptions(ioStreams)
		o.EditBeforeCreate = true
		o.FilenameOptions.Filenames = []string{"../../../testdata/redis-master-controller.yaml"}
		if err := o.Complete(tf, cmd, []string{}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if o.editOptions == nil {
			t.Error("expected editOptions to be initialized when EditBeforeCreate is true")
		}
	})

	t.Run("RunCreate returns error when EditBeforeCreate is true but Complete was not called", func(t *testing.T) {
		o := NewCreateOptions(ioStreams)
		o.EditBeforeCreate = true
		err := o.RunCreate(tf)
		if err == nil {
			t.Fatal("expected error when editOptions is nil")
		}
		if err.Error() != "EditBeforeCreate requires edit options to be initialized via Complete()" {
			t.Errorf("unexpected error message: %v", err)
		}
	})

	t.Run("editOptions inherit dry-run strategy and field manager", func(t *testing.T) {
		cmd := NewCmdCreate(tf, ioStreams)
		cmd.Flags().Set("dry-run", "server") // nolint:errcheck

		o := NewCreateOptions(ioStreams)
		o.EditBeforeCreate = true
		o.fieldManager = "custom-manager"
		o.FilenameOptions.Filenames = []string{"../../../testdata/redis-master-controller.yaml"}
		if err := o.Complete(tf, cmd, []string{}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if o.editOptions == nil {
			t.Fatal("expected editOptions to be initialized")
		}
		if o.editOptions.DryRunStrategy != cmdutil.DryRunServer {
			t.Errorf("expected dry-run strategy %v, got %v", cmdutil.DryRunServer, o.editOptions.DryRunStrategy)
		}
		if o.editOptions.FieldManager != "custom-manager" {
			t.Errorf("expected field manager %q, got %q", "custom-manager", o.editOptions.FieldManager)
		}
	})
}

func TestCreateEditWithDryRun(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	tests := []struct {
		name          string
		dryRun        string
		wantOutput    string
		expectRequest bool
	}{
		{
			name:       "client",
			dryRun:     "client",
			wantOutput: "replicationcontroller/edited-master created (dry run)\n",
		},
		{
			name:          "server",
			dryRun:        "server",
			wantOutput:    "replicationcontroller/edited-master created (server dry run)\n",
			expectRequest: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			requested := false
			tf.UnstructuredClient = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Version: "v1"},
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					requested = true
					if got := req.URL.Query().Get("dryRun"); got != "All" {
						t.Errorf("expected dryRun=All, got %q", got)
					}
					rc := &corev1.ReplicationController{
						ObjectMeta: metav1.ObjectMeta{Name: "edited-master", Namespace: "test"},
					}
					return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, rc)}, nil
				}),
			}

			editor := filepath.Join(t.TempDir(), "editor.sh")
			if err := os.WriteFile(editor, []byte("#!/bin/sh\nsed -i.bak 's/redis-master/edited-master/' \"$1\"\n"), 0o755); err != nil {
				t.Fatalf("failed to write editor script: %v", err)
			}
			t.Setenv("KUBE_EDITOR", editor)

			ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
			cmd := NewCmdCreate(tf, ioStreams)
			cmd.Flags().Set("filename", "../../../testdata/redis-master-controller.yaml") // nolint:errcheck
			cmd.Flags().Set("edit", "true")                                               // nolint:errcheck
			cmd.Flags().Set("dry-run", tc.dryRun)                                         // nolint:errcheck
			cmd.Run(cmd, []string{})

			if requested != tc.expectRequest {
				t.Errorf("expected request: %v, got: %v", tc.expectRequest, requested)
			}
			if got := buf.String(); got != tc.wantOutput {
				t.Errorf("unexpected output: got %q, want %q", got, tc.wantOutput)
			}
		})
	}
}

func TestMissingFilenameError(t *testing.T) {
	var errStr string
	var exitCode int
	cmdutil.BehaviorOnFatal(func(str string, code int) {
		if errStr == "" {
			errStr = str
			exitCode = code
		}
	})

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreate(tf, ioStreams)
	cmd.Run(cmd, []string{})

	if buf.Len() > 0 {
		t.Errorf("unexpected output: %s", buf.String())
	}

	if len(errStr) == 0 {
		t.Errorf("unexpected non-error")
	} else if errStr != "error: must specify one of -f and -k" {
		t.Errorf("unexpected error: %s", errStr)
	}

	if exitCode != 1 {
		t.Errorf("unexpected exit code: %d", exitCode)
	}
}
