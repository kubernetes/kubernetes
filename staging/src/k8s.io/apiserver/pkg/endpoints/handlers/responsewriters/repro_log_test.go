package responsewriters

import (
	"bytes"
	"flag"
	"fmt"
	"math/rand"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/klog/v2"
)

func TestSerializeObjectTraceLog(t *testing.T) {
	// Enable klog V(2) to ensure traces are logged
	klog.InitFlags(nil)
	flag.Set("v", "2")
	flag.Parse()

	// Capture klog output
	var buf bytes.Buffer
	klog.SetOutput(&buf)
	defer klog.SetOutput(nil)

	// Prepare large object
	// 100,000 pods should be enough to generate significant JSON
	count := 100000
	pods := make([]v1.Pod, count)
	for i := 0; i < count; i++ {
		pods[i] = v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("pod-%d", i),
				Labels: map[string]string{
					"app": "nginx",
					"tier": "frontend",
					"env": "prod",
				},
				Annotations: map[string]string{
					"description": fmt.Sprintf("random-data-%d-%d-%s", i, rand.Int(), strings.Repeat("x", 100)),
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "nginx",
						Image: "nginx:latest",
						Env: []v1.EnvVar{
							{Name: "VAR1", Value: "VALUE1"},
							{Name: "VAR2", Value: "VALUE2"},
						},
					},
				},
			},
		}
	}
	podList := &v1.PodList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "PodList",
			APIVersion: "v1",
		},
		Items: pods,
	}
	
	scheme := runtime.NewScheme()
	v1.AddToScheme(scheme)
	serializer := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, json.SerializerOptions{})

	req := httptest.NewRequest("GET", "/api/v1/pods", nil)
	// Enable gzip
	req.Header.Set("Accept-Encoding", "gzip")
	
	w := httptest.NewRecorder()

	start := time.Now()
	// Call SerializeObject
	SerializeObject("application/json", serializer, w, req, 200, podList)
	duration := time.Since(start)

	fmt.Printf("Serialization took: %v\n", duration)
	fmt.Printf("Response size: %d bytes\n", w.Body.Len())

	// Print captured log
	fmt.Println("Captured Klog Output:")
	fmt.Println(buf.String())
}
