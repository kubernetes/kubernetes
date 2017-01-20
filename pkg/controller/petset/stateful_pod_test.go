package petset

import (
	apiErrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/runtime"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"net/http/httptest"
	"testing"
)

func newTestingClient(url string) *clientset.Clientset {
	return clientset.NewForConfigOrDie(
		&restclient.Config{Host: url,
			ContentConfig: restclient.ContentConfig{
				GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})
}


func TestRealStorageController_CreatePersistentVolumeClaim(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	testServer := httptest.NewServer(&utiltesting.FakeHandler{StatusCode: 500})
	defer testServer.Close()
	controller := NewRealStorageController(newTestingClient(testServer.URL))
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		if err := controller.CreatePersistentVolumeClaim(&claim); !apiErrors.IsInternalError(err) {
			t.Errorf("Expected InternalError found %s", err)
		}
	}
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 200,
			ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), &claim)}
		if err := controller.CreatePersistentVolumeClaim(&claim); err != nil {
			t.Errorf("Expected nil found %s", err)
		}
	}
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 409,
			ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), &claim)}
		if err := controller.CreatePersistentVolumeClaim(&claim); !apiErrors.IsAlreadyExists(err) {
			t.Errorf("Expected AlreadyExists"+
				" found %s", err)
		}
	}
	var claim *v1.PersistentVolumeClaim = nil
	if err := controller.CreatePersistentVolumeClaim(claim); err == nil {
		t.Error("Expected error found nil")
	}
}

func TestRealStorageController_GetPersistentVolumeClaim(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	testServer := httptest.NewServer(&utiltesting.FakeHandler{StatusCode: 500})
	defer testServer.Close()
	controller := NewRealStorageController(newTestingClient(testServer.URL))
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		if _, err := controller.GetPersistentVolumeClaim(claim.Namespace, claim.Name); !apiErrors.IsInternalError(err) {
			t.Errorf("Expected InternalError found %s", err)
		}
	}
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		testServer.Config.Handler = &utiltesting.FakeHandler{
			StatusCode:   200,
			ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), &claim)}
		if retrieved, err := controller.GetPersistentVolumeClaim(claim.Namespace, claim.Name); err != nil {
			t.Errorf("Error retrieving claim %s", err)
		} else if claim.UID != retrieved.UID {
			t.Errorf("Expected %v found %v", claim, retrieved)
		}
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 404}
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		if _, err := controller.GetPersistentVolumeClaim(claim.Namespace, claim.Name); !apiErrors.IsNotFound(err) {
			t.Errorf("Expected NotFound found %s", err)

		}
	}
}

func TestRealStorageController_UpdatePersistentVolumeClaim(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	testServer := httptest.NewServer(&utiltesting.FakeHandler{StatusCode: 500})
	defer testServer.Close()
	controller := NewRealStorageController(newTestingClient(testServer.URL))
	for _, claim := range getPersistentVolumeClaims(set,pod){
		if _, err := controller.UpdatePersistentVolumeClaim(&claim); !apiErrors.IsInternalError(err) {
			t.Errorf("Expected InternalError found %s", err)
		}
	}
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		testServer.Config.Handler = &utiltesting.FakeHandler{
			StatusCode:   200,
			ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), &claim)}
		if _, err := controller.UpdatePersistentVolumeClaim(&claim); err != nil {
			t.Errorf("Error creating claim %s", err)
		}
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 409}
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		if _, err := controller.UpdatePersistentVolumeClaim(&claim); !apiErrors.IsConflict(err) {
			t.Errorf("Expected AlreadyExists found %s", err)

		}
	}
	var claim *v1.PersistentVolumeClaim = nil
	if _, err := controller.UpdatePersistentVolumeClaim(claim); err == nil {
		t.Error("Expected error found nil")
	}
}

func TestRealStorageController_DeletePersistentVolumeClaim(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	testServer := httptest.NewServer(&utiltesting.FakeHandler{StatusCode: 500})
	defer testServer.Close()
	controller := NewRealStorageController(newTestingClient(testServer.URL))
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		if err := controller.DeletePersistentVolumeClaim(claim.Namespace, claim.Name); !apiErrors.IsInternalError(err) {
			t.Errorf("Expected InternalError found %s", err)
		}
	}
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		testServer.Config.Handler = &utiltesting.FakeHandler{
			StatusCode:   200,
			ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), &claim)}
		if err := controller.DeletePersistentVolumeClaim(claim.Namespace, claim.Name); err != nil {
			t.Errorf("Error deleting claim %s", err)
		}
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 404}
	for _, claim := range getPersistentVolumeClaims(set,pod) {
		if err := controller.DeletePersistentVolumeClaim(claim.Namespace, claim.Name); err != nil {
			t.Errorf("Expected nil found %s", err)

		}
	}
}

func TestRealPodController_CreatePod(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	testServer := httptest.NewServer(&utiltesting.FakeHandler{StatusCode: 500})
	defer testServer.Close()
	controller := NewRealPodController(newTestingClient(testServer.URL))
	if err := controller.CreatePod(pod); !apiErrors.IsInternalError(err) {
		t.Errorf("Expected InternalError found %s", err)
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 409}
	if err := controller.CreatePod(pod); !apiErrors.IsAlreadyExists(err) {
		t.Errorf("Expected AlreadyExists found %s", err)
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 200,
		ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), pod)}
	if err := controller.CreatePod(pod); err != nil {
		t.Errorf("Error creating Pod %s", err)
	}
}

func TestRealPodController_GetPod(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	testServer := httptest.NewServer(&utiltesting.FakeHandler{StatusCode: 500})
	defer testServer.Close()
	controller := NewRealPodController(newTestingClient(testServer.URL))
	if _, err := controller.GetPod(pod.Namespace, pod.Name); !apiErrors.IsInternalError(err) {
		t.Errorf("Expected InternalError found %s", err)
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 404}
	if _, err := controller.GetPod(pod.Namespace, pod.Name); err != nil {
		t.Errorf("Retrieve Pod failed found %s", err)
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 200,
		ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), pod)}
	if retrieved, err := controller.GetPod(pod.Namespace, pod.Name); err != nil {
		t.Errorf("Error creating Pod %s", err)
	} else if retrieved.UID != pod.UID {
		t.Errorf("Expected %s found %s", pod.String(), retrieved.String())
	}
}

func TestRealPodController_UpdatePod(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	testServer := httptest.NewServer(&utiltesting.FakeHandler{StatusCode: 500})
	defer testServer.Close()
	controller := NewRealPodController(newTestingClient(testServer.URL))
	if _,err := controller.UpdatePod(pod); !apiErrors.IsInternalError(err) {
		t.Errorf("Expected InternalError found %s", err)
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 409}
	if _,err := controller.UpdatePod(pod); !apiErrors.IsConflict(err) {
		t.Errorf("Expected AlreadyExists found %s", err)
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 200,
		ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), pod)}
	if _,err := controller.UpdatePod(pod); err != nil {
		t.Errorf("Error creating Pod %s", err)
	}
}

func TestRealPodController_DeletePod(t *testing.T) {
	set := newStatefulSet(3)
	pod := newStatefulSetPod(set, 0)
	testServer := httptest.NewServer(&utiltesting.FakeHandler{StatusCode: 500})
	defer testServer.Close()
	controller := NewRealPodController(newTestingClient(testServer.URL))
	if err := controller.DeletePod(pod.Namespace, pod.Name); !apiErrors.IsInternalError(err) {
		t.Errorf("Expected InternalError found %s", err)
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 404}
	if err := controller.DeletePod(pod.Namespace, pod.Name); err != nil {
		t.Errorf("Retrieve Pod failed found %s", err)
	}
	testServer.Config.Handler = &utiltesting.FakeHandler{StatusCode: 200,
		ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), pod)}
	if err := controller.DeletePod(pod.Namespace, pod.Name); err != nil {
		t.Errorf("Error creating Pod %s", err)
	}

}
