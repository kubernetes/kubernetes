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

package events

import (
	"io"
	"net/http"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func getFakeEvents() *corev1.EventList {
	return &corev1.EventList{
		Items: []corev1.Event{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-000",
					Namespace: "foo",
				},
				InvolvedObject: corev1.ObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "bar",
					Namespace:  "foo",
					UID:        "00000000-0000-0000-0000-000000000001",
				},
				Type:                corev1.EventTypeNormal,
				Reason:              "ScalingReplicaSet",
				Message:             "Scaled up replica set bar-002 to 1",
				ReportingController: "deployment-controller",
				EventTime:           metav1.NewMicroTime(time.Now().Add(-30 * time.Minute)),
				Series: &corev1.EventSeries{
					Count:            3,
					LastObservedTime: metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-001",
					Namespace: "foo",
				},
				InvolvedObject: corev1.ObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "bar",
					Namespace:  "foo",
					UID:        "00000000-0000-0000-0000-000000000001",
				},
				Type:                corev1.EventTypeWarning,
				Reason:              "ScalingReplicaSet",
				Message:             "Scaled up replica set bar-002 to 1",
				ReportingController: "deployment-controller",
				EventTime:           metav1.NewMicroTime(time.Now().Add(-28 * time.Minute)),
				Series: &corev1.EventSeries{
					Count:            3,
					LastObservedTime: metav1.NewMicroTime(time.Now().Add(-18 * time.Minute)),
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-002",
					Namespace: "otherfoo",
				},
				InvolvedObject: corev1.ObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "bar",
					Namespace:  "otherfoo",
					UID:        "00000000-0000-0000-0000-000000000001",
				},
				Type:                corev1.EventTypeNormal,
				Reason:              "ScalingReplicaSet",
				Message:             "Scaled up replica set bar-002 to 1",
				ReportingController: "deployment-controller",
				EventTime:           metav1.NewMicroTime(time.Now().Add(-25 * time.Minute)),
				Series: &corev1.EventSeries{
					Count:            3,
					LastObservedTime: metav1.NewMicroTime(time.Now().Add(-15 * time.Minute)),
				},
			},
		},
	}
}

func TestEventIsSorted(t *testing.T) {
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	clientset, err := kubernetes.NewForConfig(cmdtesting.DefaultClientConfig())
	if err != nil {
		t.Fatal(err)
	}

	clientset.CoreV1().RESTClient().(*restclient.RESTClient).Client = fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, getFakeEvents())}, nil
	})

	printer := NewEventPrinter(false, true)

	options := &EventsOptions{
		AllNamespaces: true,
		client:        clientset,
		PrintObj: func(object runtime.Object, writer io.Writer) error {
			return printer.PrintObj(object, writer)
		},
		IOStreams: streams,
	}

	err = options.Run()
	if err != nil {
		t.Fatal(err)
	}

	expected := `NAMESPACE   LAST SEEN           TYPE      REASON              OBJECT           MESSAGE
foo         20m (x3 over 30m)   Normal    ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 to 1
foo         18m (x3 over 28m)   Warning   ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 to 1
otherfoo    15m (x3 over 25m)   Normal    ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 to 1
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestEventNoHeaders(t *testing.T) {
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	clientset, err := kubernetes.NewForConfig(cmdtesting.DefaultClientConfig())
	if err != nil {
		t.Fatal(err)
	}

	clientset.CoreV1().RESTClient().(*restclient.RESTClient).Client = fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, getFakeEvents())}, nil
	})

	printer := NewEventPrinter(true, true)

	options := &EventsOptions{
		AllNamespaces: true,
		client:        clientset,
		PrintObj: func(object runtime.Object, writer io.Writer) error {
			return printer.PrintObj(object, writer)
		},
		IOStreams: streams,
	}

	err = options.Run()
	if err != nil {
		t.Fatal(err)
	}

	expected := `foo        20m (x3 over 30m)   Normal    ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 to 1
foo        18m (x3 over 28m)   Warning   ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 to 1
otherfoo   15m (x3 over 25m)   Normal    ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 to 1
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestEventFiltered(t *testing.T) {
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	clientset, err := kubernetes.NewForConfig(cmdtesting.DefaultClientConfig())
	if err != nil {
		t.Fatal(err)
	}

	clientset.CoreV1().RESTClient().(*restclient.RESTClient).Client = fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, getFakeEvents())}, nil
	})

	printer := NewEventPrinter(false, true)

	options := &EventsOptions{
		AllNamespaces: true,
		client:        clientset,
		FilterTypes:   []string{"WARNING"},
		PrintObj: func(object runtime.Object, writer io.Writer) error {
			return printer.PrintObj(object, writer)
		},
		IOStreams: streams,
	}

	err = options.Run()
	if err != nil {
		t.Fatal(err)
	}

	expected := `NAMESPACE   LAST SEEN           TYPE      REASON              OBJECT           MESSAGE
foo         18m (x3 over 28m)   Warning   ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 to 1
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}
