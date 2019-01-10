package manager

import (
	"fmt"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/kubelet/configmap"
)

var updateCallbackFuncErr = fmt.Errorf("updateCallbackFunc not called")

func TestSync(t *testing.T) {
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "test-cm",
		},
		Data: map[string]string{
			"a": "b",
		},
	}
	fakeClient := fake.NewSimpleClientset(cm)
	cmm := configmap.NewSimpleConfigMapManager(fakeClient)
	cmw := NewConfigMapWatcher(cmm, onTestConfigMapUpdate)

	cm.Data["a"] = "c"
	fakeClient.CoreV1().ConfigMaps("test").Update(cm)
	keyStrs := sets.NewString("test/test-cm")
	cmw.Sync(keyStrs)
	if updateCallbackFuncErr != nil {
		t.Fatalf("test sync failed, %v", updateCallbackFuncErr)
	}
}

func onTestConfigMapUpdate(configMap *v1.ConfigMap) {
	if configMap.Namespace != "test" || configMap.Name != "test-cm" {
		updateCallbackFuncErr = fmt.Errorf("invalid namespace or name, namespace: %s, name: %s", configMap.Namespace, configMap.Name)
		return
	}
	if configMap.Data["a"] != "c" {
		updateCallbackFuncErr = fmt.Errorf("configMap.Data[\"a\"] expect %s, got: %s", "c", configMap.Data["a"])
		return
	}
	updateCallbackFuncErr = nil
}
