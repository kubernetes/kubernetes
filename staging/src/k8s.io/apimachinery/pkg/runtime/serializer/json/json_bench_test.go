package json

import (
	"bytes"
	"encoding/json"
	"testing"

	jsoniter "github.com/json-iterator/go"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
)

func testLargePod() api.Pod {
	obj := api.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "ubuntu",
			Namespace: "default",
			Labels: map[string]string{
				"name": "ubuntu",
			},
			SelfLink: "/api/v1/namespaces/default/pods/ubuntu",
			Annotations: map[string]string{
				"foo.bar.baz": "foo",
				"baz.bar.foo": "bar",
				"bar.foo.baz": "baz",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:       "ubuntu",
					Image:      "foo.io/ubuntu",
					Command:    []string{"sleep"},
					Args:       []string{"1d"},
					WorkingDir: "/data/tmp",
					Ports: []api.ContainerPort{
						{
							Name:          "port",
							HostPort:      1,
							ContainerPort: 2,
							Protocol:      api.ProtocolTCP,
							HostIP:        "127.0.0.1",
						},
					},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "default-token-p0000",
							ReadOnly:  true,
							MountPath: "/var/run/secrets/kubernetes.io/serviceaccount",
						},
					},
				},
			},
			RestartPolicy: api.RestartPolicyAlways,
		},
	}
	return obj
}

func BenchmarkJsonEncoding(b *testing.B) {
	obj := testLargePod()
	encoder := json.NewEncoder(bytes.NewBuffer(nil))
	for i := 0; i < b.N; i++ {
		encoder.Encode(obj)
	}
}

func BenchmarkJsoniterEncoding(b *testing.B) {
	obj := testLargePod()
	encoder := jsoniter.ConfigCompatibleWithStandardLibrary.NewEncoder(bytes.NewBuffer(nil))
	for i := 0; i < b.N; i++ {
		encoder.Encode(obj)
	}
}
