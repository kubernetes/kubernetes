/*
Copyright 2017 The Kubernetes Authors.

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

package garbagecollector

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

type roundTripFunc func(req *http.Request) (*http.Response, error)

func (w roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return w(req)
}

type readDelayer struct {
	delay time.Duration
	io.ReadCloser
}

func (b *readDelayer) Read(p []byte) (n int, err error) {
	defer time.Sleep(b.delay)
	return b.ReadCloser.Read(p)
}

func TestClusterScopedOwners(t *testing.T) {
	// Start the test server and wrap the client to delay PV watch responses
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	server.ClientConfig.WrapTransport = func(rt http.RoundTripper) http.RoundTripper {
		return roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get("watch") != "true" || !strings.Contains(req.URL.String(), "persistentvolumes") {
				return rt.RoundTrip(req)
			}
			resp, err := rt.RoundTrip(req)
			if err != nil {
				return resp, err
			}
			resp.Body = &readDelayer{30 * time.Second, resp.Body}
			return resp, err
		})
	}
	ctx := setupWithServer(t, server, 5)
	defer ctx.tearDown()

	_, clientSet := ctx.gc, ctx.clientSet

	ns := createNamespaceOrDie("gc-cluster-scope-deletion", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	t.Log("Create a pair of objects")
	pv, err := clientSet.CoreV1().PersistentVolumes().Create(context.TODO(), &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "pv-valid"},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/foo"}},
			Capacity:               v1.ResourceList{v1.ResourceStorage: resource.MustParse("1Gi")},
			AccessModes:            []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Create(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "cm-valid",
			OwnerReferences: []metav1.OwnerReference{{Kind: "PersistentVolume", APIVersion: "v1", Name: pv.Name, UID: pv.UID}},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	t.Log("Create a namespaced object with a missing parent")
	if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Create(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "cm-missing",
			Labels:          map[string]string{"missing": "true"},
			OwnerReferences: []metav1.OwnerReference{{Kind: "PersistentVolume", APIVersion: "v1", Name: "missing-name", UID: types.UID("missing-uid")}},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	t.Log("Create a namespaced object with a missing type parent")
	if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Create(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "cm-invalid",
			OwnerReferences: []metav1.OwnerReference{{Kind: "UnknownType", APIVersion: "unknown.group/v1", Name: "invalid-name", UID: types.UID("invalid-uid")}},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	// wait for deletable children to go away
	if err := wait.Poll(5*time.Second, 300*time.Second, func() (bool, error) {
		_, err := clientSet.CoreV1().ConfigMaps(ns.Name).Get(context.TODO(), "cm-missing", metav1.GetOptions{})
		switch {
		case apierrors.IsNotFound(err):
			return true, nil
		case err != nil:
			return false, err
		default:
			t.Logf("cm with missing parent still exists, retrying")
			return false, nil
		}
	}); err != nil {
		t.Fatal(err)
	}
	t.Logf("deletable children removed")

	// Give time for blocked children to be incorrectly cleaned up
	time.Sleep(5 * time.Second)

	// ensure children with unverifiable parents don't get reaped
	if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Get(context.TODO(), "cm-invalid", metav1.GetOptions{}); err != nil {
		t.Fatalf("child with invalid ownerRef is unexpectedly missing: %v", err)
	}

	// ensure children with present parents don't get reaped
	if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Get(context.TODO(), "cm-valid", metav1.GetOptions{}); err != nil {
		t.Fatalf("child with valid ownerRef is unexpectedly missing: %v", err)
	}
}
