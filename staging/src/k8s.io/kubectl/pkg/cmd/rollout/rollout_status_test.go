/*
Copyright 2022 The Kubernetes Authors.

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

package rollout

import (
	"bytes"
	"io"
	"net/http"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	cgtesting "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

var rolloutStatusGroupVersionEncoder = schema.GroupVersion{Group: "apps", Version: "v1"}

func TestRolloutStatus(t *testing.T) {
	deploymentName := "deployment/nginx-deployment"
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutStatusGroupVersionEncoder)
	tf.Client = &fake.RESTClient{
		GroupVersion:         rolloutStatusGroupVersionEncoder,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			dep := &appsv1.Deployment{}
			dep.Name = deploymentName
			body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, dep))))
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
		}),
	}

	tf.FakeDynamicClient.WatchReactionChain = nil
	fw := watch.NewFake()
	tf.FakeDynamicClient.AddWatchReactor("*", func(action cgtesting.Action) (handled bool, ret watch.Interface, err error) {
		dep := &appsv1.Deployment{}
		dep.Name = deploymentName
		dep.Status = appsv1.DeploymentStatus{
			Replicas:            1,
			UpdatedReplicas:     1,
			ReadyReplicas:       1,
			AvailableReplicas:   1,
			UnavailableReplicas: 0,
			Conditions: []appsv1.DeploymentCondition{{
				Type: appsv1.DeploymentAvailable,
			}},
		}
		c, err := runtime.DefaultUnstructuredConverter.ToUnstructured(dep.DeepCopyObject())
		if err != nil {
			t.Errorf("unexpected err %s", err)
		}
		u := &unstructured.Unstructured{}
		u.SetUnstructuredContent(c)
		go func() {
			fw.Add(u)
			<-fw.StopChan()
			fw.Close()
		}()
		return true, fw, nil
	})

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdRolloutStatus(tf, streams)
	cmd.Run(cmd, []string{deploymentName})

	expectedMsg := "deployment \"deployment/nginx-deployment\" successfully rolled out\n"
	if buf.String() != expectedMsg {
		t.Errorf("expected output: %s, but got: %s", expectedMsg, buf.String())
	}

	// Validate the run command stopped the watcher when done
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestRolloutStatusWithSelector(t *testing.T) {
	deploymentName := "deployment"
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutStatusGroupVersionEncoder)
	tf.Client = &fake.RESTClient{
		GroupVersion:         rolloutStatusGroupVersionEncoder,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			dep := &appsv1.Deployment{}
			dep.Name = deploymentName
			dep.Labels = make(map[string]string)
			dep.Labels["app"] = "api"
			body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, dep))))
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
		}),
	}

	tf.FakeDynamicClient.WatchReactionChain = nil
	fw := watch.NewFake()
	tf.FakeDynamicClient.AddWatchReactor("*", func(action cgtesting.Action) (handled bool, ret watch.Interface, err error) {
		dep := &appsv1.Deployment{}
		dep.Name = deploymentName
		dep.Status = appsv1.DeploymentStatus{
			Replicas:            1,
			UpdatedReplicas:     1,
			ReadyReplicas:       1,
			AvailableReplicas:   1,
			UnavailableReplicas: 0,
			Conditions: []appsv1.DeploymentCondition{{
				Type: appsv1.DeploymentAvailable,
			}},
		}
		dep.Labels = make(map[string]string)
		dep.Labels["app"] = "api"
		c, err := runtime.DefaultUnstructuredConverter.ToUnstructured(dep.DeepCopyObject())
		if err != nil {
			t.Errorf("unexpected err %s", err)
		}
		u := &unstructured.Unstructured{}
		u.SetUnstructuredContent(c)
		go func() {
			fw.Add(u)
			<-fw.StopChan()
			fw.Close()
		}()
		return true, fw, nil
	})

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdRolloutStatus(tf, streams)
	cmd.Flags().Set("selector", "app=api")
	cmd.Run(cmd, []string{deploymentName})

	expectedMsg := "deployment \"deployment\" successfully rolled out\n"
	if buf.String() != expectedMsg {
		t.Errorf("expected output: %s, but got: %s", expectedMsg, buf.String())
	}

	// Validate the run command stopped the watcher when done
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestRolloutStatusWatchDisabled(t *testing.T) {
	deploymentName := "deployment/nginx-deployment"
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutStatusGroupVersionEncoder)
	tf.Client = &fake.RESTClient{
		GroupVersion:         rolloutStatusGroupVersionEncoder,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			dep := &appsv1.Deployment{}
			dep.Name = deploymentName
			body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, dep))))
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
		}),
	}

	tf.FakeDynamicClient.WatchReactionChain = nil
	fw := watch.NewFake()
	tf.FakeDynamicClient.AddWatchReactor("*", func(action cgtesting.Action) (handled bool, ret watch.Interface, err error) {
		dep := &appsv1.Deployment{}
		dep.Name = deploymentName
		dep.Status = appsv1.DeploymentStatus{
			Replicas:            1,
			UpdatedReplicas:     1,
			ReadyReplicas:       1,
			AvailableReplicas:   1,
			UnavailableReplicas: 0,
			Conditions: []appsv1.DeploymentCondition{{
				Type: appsv1.DeploymentAvailable,
			}},
		}
		c, err := runtime.DefaultUnstructuredConverter.ToUnstructured(dep.DeepCopyObject())
		if err != nil {
			t.Errorf("unexpected err %s", err)
		}
		u := &unstructured.Unstructured{}
		u.SetUnstructuredContent(c)
		go func() {
			fw.Add(u)
			<-fw.StopChan()
			fw.Close()
		}()
		return true, fw, nil
	})

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdRolloutStatus(tf, streams)
	cmd.Flags().Set("watch", "false")
	cmd.Run(cmd, []string{deploymentName})

	expectedMsg := "deployment \"deployment/nginx-deployment\" successfully rolled out\n"
	if buf.String() != expectedMsg {
		t.Errorf("expected output: %s, but got: %s", expectedMsg, buf.String())
	}

	// Validate the run command stopped the watcher when done
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestRolloutStatusWatchDisabledUnavailable(t *testing.T) {
	deploymentName := "deployment/nginx-deployment"
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutStatusGroupVersionEncoder)
	tf.Client = &fake.RESTClient{
		GroupVersion:         rolloutStatusGroupVersionEncoder,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			dep := &appsv1.Deployment{}
			dep.Name = deploymentName
			body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, dep))))
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
		}),
	}

	tf.FakeDynamicClient.WatchReactionChain = nil
	fw := watch.NewFake()
	tf.FakeDynamicClient.AddWatchReactor("*", func(action cgtesting.Action) (handled bool, ret watch.Interface, err error) {
		dep := &appsv1.Deployment{}
		dep.Name = deploymentName
		dep.Status = appsv1.DeploymentStatus{
			Replicas:            1,
			UpdatedReplicas:     1,
			ReadyReplicas:       1,
			AvailableReplicas:   0,
			UnavailableReplicas: 0,
			Conditions: []appsv1.DeploymentCondition{{
				Type: appsv1.DeploymentAvailable,
			}},
		}
		c, err := runtime.DefaultUnstructuredConverter.ToUnstructured(dep.DeepCopyObject())
		if err != nil {
			t.Errorf("unexpected err %s", err)
		}
		u := &unstructured.Unstructured{}
		u.SetUnstructuredContent(c)
		go func() {
			fw.Add(u)
			<-fw.StopChan()
			fw.Close()
		}()
		return true, fw, nil
	})

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdRolloutStatus(tf, streams)
	cmd.Flags().Set("watch", "false")
	cmd.Run(cmd, []string{deploymentName})

	expectedMsg := "Waiting for deployment \"deployment/nginx-deployment\" rollout to finish: 0 of 1 updated replicas are available...\n"
	if buf.String() != expectedMsg {
		t.Errorf("expected output: %s, but got: %s", expectedMsg, buf.String())
	}

	// Validate the run command stopped the watcher when done
	select {
	case _, ok := <-fw.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Fatalf("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Expected watcher to be stopped")
	}
}

func TestRolloutStatusEmptyList(t *testing.T) {
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutStatusGroupVersionEncoder)
	tf.Client = &fake.RESTClient{
		GroupVersion:         rolloutStatusGroupVersionEncoder,
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			dep := &appsv1.DeploymentList{}
			body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, dep))))
			return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
		}),
	}
	streams, _, _, err := genericiooptions.NewTestIOStreams()
	cmd := NewCmdRolloutStatus(tf, streams)
	cmd.Run(cmd, []string{"deployment"})

	expectedMsg := "No resources found in test namespace.\n"
	if err.String() != expectedMsg {
		t.Errorf("expected output: %s, but got: %s", expectedMsg, err.String())
	}
}
