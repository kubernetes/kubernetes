/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
)

var podSpec string = `
{
    "metadata": {
        "creationTimestamp": null,
        "labels": {
            "app": "cats"
        }
    },
    "spec": {
        "containers": [
            {
                "name": "cats",
                "image": "registry/test/cats:v0.@@VERSION@@.0",
                "ports": [
                    {
                        "name": "http",
                        "containerPort": 9077,
                        "protocol": "TCP"
                    }
                ],
                "env": [
                    {
                        "name": "DEPLOYMENT_ENVIRONMENT",
                        "value": "cats-stubbed-functional"
                    },
                    {
                        "name": "APP_NAME",
                        "value": "cats"
                    }
                ],
                "resources": {
                    "limits": {
                        "cpu": "1",
                        "memory": "1Gi"
                    },
                    "requests": {
                        "cpu": "1",
                        "memory": "1Gi"
                    }
                },
                "livenessProbe": {
                    "httpGet": {
                        "path": "/private/status",
                        "port": 9077,
                        "scheme": "HTTP"
                    },
                    "initialDelaySeconds": 30,
                    "timeoutSeconds": 1,
                    "periodSeconds": 10,
                    "successThreshold": 1,
                    "failureThreshold": 3
                },
                "readinessProbe": {
                    "httpGet": {
                        "path": "/private/status",
                        "port": 9077,
                        "scheme": "HTTP"
                    },
                    "initialDelaySeconds": 1,
                    "timeoutSeconds": 1,
                    "periodSeconds": 10,
                    "successThreshold": 1,
                    "failureThreshold": 3
                },
                "terminationMessagePath": "/dev/termination-log",
                "imagePullPolicy": "IfNotPresent"
            }
        ],
        "restartPolicy": "Always",
        "terminationGracePeriodSeconds": 30,
        "dnsPolicy": "ClusterFirst",
        "securityContext": {}
    }
}
`

func TestPodTemplateSpecHash(t *testing.T) {
	seenHashes := make(map[uint32]int)
	broken := false

	for i := 0; i < 1000; i++ {
		specJson := strings.Replace(podSpec, "@@VERSION@@", strconv.Itoa(i), 1)
		spec := v1.PodTemplateSpec{}
		json.Unmarshal([]byte(specJson), &spec)
		hash := GetPodTemplateSpecHash(spec)
		if v, ok := seenHashes[hash]; ok {
			broken = true
			t.Logf("Hash collision, old: %d new: %d", v, i)
			break
		}
		seenHashes[hash] = i
	}

	if !broken {
		t.Errorf("expected adler to break but it didn't")
	}
}

func TestPodTemplateSpecHashFnv(t *testing.T) {
	seenHashes := make(map[uint32]int)

	for i := 0; i < 1000; i++ {
		specJson := strings.Replace(podSpec, "@@VERSION@@", strconv.Itoa(i), 1)
		spec := v1.PodTemplateSpec{}
		json.Unmarshal([]byte(specJson), &spec)
		hash := GetPodTemplateSpecHashFnv(spec)
		if v, ok := seenHashes[hash]; ok {
			t.Errorf("Hash collision, old: %d new: %d", v, i)
			break
		}
		seenHashes[hash] = i
	}
}

func BenchmarkAdler(b *testing.B) {
	spec := v1.PodTemplateSpec{}
	json.Unmarshal([]byte(podSpec), &spec)

	for i := 0; i < b.N; i++ {
		GetPodTemplateSpecHash(spec)
	}
}

func BenchmarkFnv(b *testing.B) {
	spec := v1.PodTemplateSpec{}
	json.Unmarshal([]byte(podSpec), &spec)

	for i := 0; i < b.N; i++ {
		GetPodTemplateSpecHashFnv(spec)
	}
}
