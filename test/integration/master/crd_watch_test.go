package master

import (
	"os"
	"testing"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/csi/v1alpha1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	extapi "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestCRDWatch(t *testing.T) {
	glog.V(2).Infof("TestCRDWatch started")

	result := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer result.TearDownFn()

	cfg := result.ClientConfig
	if v, found := os.LookupEnv("KUBE_CONTENT_TYPE"); found {
		cfg.ContentType = v
	}
	glog.Infof("Using content-type: %q", cfg.ContentType)
	kubeclient, err := clientset.NewForConfig(cfg)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	apiextensionsclient, err := extapi.NewForConfig(cfg)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Ensure CRD exists
	crd := v1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "csidrivers.csi.storage.k8s.io",
		},
		Spec: v1beta1.CustomResourceDefinitionSpec{
			Group:   "csi.storage.k8s.io",
			Version: "v1alpha1",
			Names: v1beta1.CustomResourceDefinitionNames{
				Kind:   "CSIDriver",
				Plural: "csidrivers",
			},
			Scope: v1beta1.ClusterScoped,
		},
	}
	_, err = apiextensionsclient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(&crd)
	if err != nil {
		t.Fatalf("Failed to create CRD: %s", err)
	}

	if err := waitForEstablishedCRD(apiextensionsclient, crd.Name); err != nil {
		t.Fatalf("Failed to establish csidrivers.csi.storage.k8s.io CRD: %v", err)
	}

	glog.Infof("CRD create OK")

	// Create CR
	driver := &v1alpha1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "driver1",
		},
		Spec: v1alpha1.CSIDriverSpec{
			Driver: "driver1",
		},
	}
	_, err = kubeclient.CsiV1alpha1().CSIDrivers().Create(driver)
	if err != nil {
		t.Errorf("Failed to create CSIDriver: %s", err)
	}

	// Test List()
	l, err := kubeclient.CsiV1alpha1().CSIDrivers().List(metav1.ListOptions{})
	if err != nil {
		t.Errorf("Failed to create CSIDriver: %s", err)
	}
	glog.Infof("Found %d items", len(l.Items))
	if len(l.Items) != 1 {
		t.Errorf("Expected 1 item, got %d", len(l.Items))
	}

	// Create Watch(). With ResourceVersion:0, we should get all CSIDrivers as events
	w, err := kubeclient.CsiV1alpha1().CSIDrivers().Watch(metav1.ListOptions{
		ResourceVersion: "0",
	})
	if err != nil {
		t.Fatalf("Failed to watch: %s", err)
	}

	eventCh := w.ResultChan()
	timer := time.After(time.Minute)
	gotEvent := false
	select {
	case event, ok := <-eventCh:
		if !ok {
			t.Errorf("Watch channel closed!")
			break
		}
		glog.Infof("Got event %+v", event)
		gotEvent = true
	case <-timer:
		t.Error("Timed out waiting for event")
	}
	if !gotEvent {
		t.Error("Expected event, got nothing")
	}
}
