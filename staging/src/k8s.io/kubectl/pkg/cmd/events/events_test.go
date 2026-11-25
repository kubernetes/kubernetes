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
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	restwatch "k8s.io/client-go/rest/watch"
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
				Message:             "Scaled up replica set bar-002 from 0 to 1",
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
				Message:             "Scaled up replica set bar-002 from 0 to 1",
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
				Message:             "Scaled up replica set bar-002 from 0 to 1",
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
foo         20m (x3 over 30m)   Normal    ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 from 0 to 1
foo         18m (x3 over 28m)   Warning   ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 from 0 to 1
otherfoo    15m (x3 over 25m)   Normal    ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 from 0 to 1
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

	expected := `foo        20m (x3 over 30m)   Normal    ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 from 0 to 1
foo        18m (x3 over 28m)   Warning   ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 from 0 to 1
otherfoo   15m (x3 over 25m)   Normal    ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 from 0 to 1
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
foo         18m (x3 over 28m)   Warning   ScalingReplicaSet   Deployment/bar   Scaled up replica set bar-002 from 0 to 1
`
	if e, a := expected, buf.String(); e != a {
		t.Errorf("expected\n%v\ngot\n%v", e, a)
	}
}

func TestEventWatchResumesAfterStreamCloses(t *testing.T) {
	codec := scheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion)
	streams, _, out, _ := genericiooptions.NewTestIOStreams()
	clientset, err := kubernetes.NewForConfig(&restclient.Config{Host: "http://localhost"})
	if err != nil {
		t.Fatal(err)
	}

	initial := getFakeEvents()
	initial.Items = initial.Items[:2]
	initial.ResourceVersion = "10"
	initial.Continue = "second-page"
	initial.Items[0].Message = "initial event"
	initial.Items[1].Message = "filtered initial event"
	secondPage := &corev1.EventList{
		ListMeta: metav1.ListMeta{ResourceVersion: initial.ResourceVersion},
		Items:    []corev1.Event{*initial.Items[0].DeepCopy()},
	}
	secondPage.Items[0].Name = "second-page-event"
	secondPage.Items[0].Message = "second page event"

	added := initial.Items[0].DeepCopy()
	added.ResourceVersion = "11"
	added.Message = "event before reconnect"
	deleted := added.DeepCopy()
	deleted.ResourceVersion = "12"
	deleted.Message = "deleted event"
	filtered := added.DeepCopy()
	filtered.ResourceVersion = "13"
	filtered.Type = corev1.EventTypeWarning
	filtered.Message = "filtered watch event"
	updated := added.DeepCopy()
	updated.ResourceVersion = "14"
	updated.Message = "event after reconnect"
	firstStream := eventWatchBody(t,
		watch.Event{Type: watch.Added, Object: added},
		watch.Event{Type: watch.Deleted, Object: deleted},
		watch.Event{Type: watch.Added, Object: filtered},
	)
	secondStream := eventWatchBody(t,
		watch.Event{Type: watch.Modified, Object: updated},
		watch.Event{Type: watch.Error, Object: &apierrors.NewResourceExpired("resource version expired").ErrStatus},
	)

	requests := []struct {
		watch, resourceVersion, continuation, limit string
		body                                        io.ReadCloser
	}{
		{limit: "2", body: cmdtesting.ObjBody(codec, initial)},
		{limit: "2", continuation: "second-page", body: cmdtesting.ObjBody(codec, secondPage)},
		{watch: "true", resourceVersion: "10", body: firstStream},
		{watch: "true", resourceVersion: "13", body: secondStream},
	}
	const selector = "involvedObject.kind=Deployment,involvedObject.apiVersion=apps/v1,involvedObject.name=bar"
	var nextRequest int
	clientset.CoreV1().RESTClient().(*restclient.RESTClient).Client = fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		query := req.URL.Query()
		if nextRequest >= len(requests) {
			t.Errorf("unexpected request: %s", req.URL)
			return nil, apierrors.NewForbidden(corev1.Resource("events"), "", fmt.Errorf("unexpected request"))
		}
		want := requests[nextRequest]
		nextRequest++
		if req.URL.Path != "/api/v1/namespaces/foo/events" || query.Get("fieldSelector") != selector ||
			query.Get("watch") != want.watch || query.Get("resourceVersion") != want.resourceVersion ||
			query.Get("continue") != want.continuation || query.Get("limit") != want.limit {
			t.Errorf("unexpected request %d: %s", nextRequest, req.URL)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: want.body}, nil
	})
	options := &EventsOptions{
		client:      clientset,
		FilterTypes: []string{"Normal"},
		PrintObj: func(object runtime.Object, writer io.Writer) error {
			event := object.(*corev1.Event)
			_, err := fmt.Fprintln(writer, event.Message)
			return err
		},
		IOStreams: streams,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err = options.runWatch(ctx, "foo", metav1.ListOptions{Limit: 2, FieldSelector: selector})
	if !apierrors.IsResourceExpired(err) {
		t.Fatalf("expected resource version expiration, got %v", err)
	}
	want := "initial event\nsecond page event\nevent before reconnect\nevent after reconnect\n"
	if got := out.String(); got != want {
		t.Errorf("expected output %q, got %q", want, got)
	}
}

func TestEventWatchResumesFromBookmarkWithWatchOnlyAccess(t *testing.T) {
	codec := scheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion)
	client, err := kubernetes.NewForConfig(&restclient.Config{Host: "http://localhost"})
	if err != nil {
		t.Fatal(err)
	}

	initial := &corev1.Event{
		ObjectMeta: metav1.ObjectMeta{Name: "existing", Namespace: "foo", ResourceVersion: "17"},
		Message:    "existing event",
	}
	older := &corev1.Event{
		ObjectMeta: metav1.ObjectMeta{Name: "older", Namespace: "foo", ResourceVersion: "3"},
		Message:    "older existing event",
	}
	bookmark := &corev1.Event{ObjectMeta: metav1.ObjectMeta{
		ResourceVersion: "20",
		Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
	}}
	updated := initial.DeepCopy()
	updated.ResourceVersion = "21"
	updated.Message = "event after bookmark"

	requests := []struct {
		resourceVersion string
		body            io.ReadCloser
	}{
		{body: eventWatchBody(t,
			watch.Event{Type: watch.Added, Object: initial},
			watch.Event{Type: watch.Added, Object: older},
			watch.Event{Type: watch.Bookmark, Object: bookmark},
		)},
		{resourceVersion: "20", body: eventWatchBody(t,
			watch.Event{Type: watch.Modified, Object: updated},
			watch.Event{Type: watch.Error, Object: &apierrors.NewResourceExpired("resource version expired").ErrStatus},
		)},
	}
	const selector = "involvedObject.name=bar"
	var nextRequest int
	client.CoreV1().RESTClient().(*restclient.RESTClient).Client = fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
		query := req.URL.Query()
		if req.URL.Path != "/api/v1/namespaces/foo/events" || query.Get("fieldSelector") != selector {
			t.Errorf("unexpected request: %s", req.URL)
		}
		if query.Get("watch") != "true" {
			forbidden := apierrors.NewForbidden(corev1.Resource("events"), "", fmt.Errorf("list is forbidden"))
			return &http.Response{
				StatusCode: http.StatusForbidden,
				Header:     cmdtesting.DefaultHeader(),
				Body:       cmdtesting.ObjBody(codec, &forbidden.ErrStatus),
			}, nil
		}
		if nextRequest >= len(requests) {
			t.Errorf("unexpected watch request: %s", req.URL)
			return nil, apierrors.NewBadRequest("unexpected watch request")
		}
		want := requests[nextRequest]
		nextRequest++
		if query.Get("resourceVersion") != want.resourceVersion || query.Get("allowWatchBookmarks") != "true" {
			t.Errorf("unexpected watch request %d: %s", nextRequest, req.URL)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: want.body}, nil
	})

	streams, _, out, _ := genericiooptions.NewTestIOStreams()
	options := &EventsOptions{
		client: client,
		PrintObj: func(object runtime.Object, writer io.Writer) error {
			_, err := fmt.Fprintln(writer, object.(*corev1.Event).Message)
			return err
		},
		IOStreams: streams,
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	if err := options.runWatch(ctx, "foo", metav1.ListOptions{FieldSelector: selector}); !apierrors.IsResourceExpired(err) {
		t.Fatalf("expected resource version expiration, got %v", err)
	}
	want := "existing event\nolder existing event\nevent after bookmark\n"
	if got := out.String(); got != want {
		t.Errorf("expected output %q, got %q", want, got)
	}
}

func TestEventWatchReturnsPermanentRequestErrors(t *testing.T) {
	for _, want := range []error{
		apierrors.NewResourceExpired("resource version expired"),
		apierrors.NewBadRequest("invalid watch request"),
		runtime.NegotiateError{ContentType: "text/plain", Stream: true},
	} {
		t.Run(want.Error(), func(t *testing.T) {
			codec := scheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion)
			clientset, err := kubernetes.NewForConfig(&restclient.Config{Host: "http://localhost"})
			if err != nil {
				t.Fatal(err)
			}
			var watched bool
			clientset.CoreV1().RESTClient().(*restclient.RESTClient).Client = fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				if req.URL.Query().Get("watch") == "true" {
					if watched {
						t.Errorf("unexpected retry of permanent error: %s", req.URL)
					}
					watched = true
					if negotiation, ok := want.(runtime.NegotiateError); ok {
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     http.Header{"Content-Type": {negotiation.ContentType}},
							Body:       io.NopCloser(strings.NewReader("not a watch stream")),
						}, nil
					}
					statusErr := want.(*apierrors.StatusError)
					return &http.Response{
						StatusCode: int(statusErr.ErrStatus.Code),
						Header:     cmdtesting.DefaultHeader(),
						Body:       cmdtesting.ObjBody(codec, &statusErr.ErrStatus),
					}, nil
				}
				list := &corev1.EventList{ListMeta: metav1.ListMeta{ResourceVersion: "10"}}
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, list)}, nil
			})
			streams, _, _, _ := genericiooptions.NewTestIOStreams()
			options := &EventsOptions{client: clientset, IOStreams: streams}
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			if err := options.runWatch(ctx, "foo", metav1.ListOptions{}); err == nil ||
				err.Error() != want.Error() || apierrors.ReasonForError(err) != apierrors.ReasonForError(want) {
				t.Errorf("expected %v, got %v", want, err)
			}
		})
	}
}

func eventWatchBody(t *testing.T, events ...watch.Event) io.ReadCloser {
	t.Helper()
	codec := scheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion)
	var data strings.Builder
	encoder := restwatch.NewEncoder(streaming.NewEncoder(&data, codec), codec)
	for _, event := range events {
		if err := encoder.Encode(&event); err != nil {
			t.Fatal(err)
		}
	}
	return io.NopCloser(strings.NewReader(data.String()))
}
