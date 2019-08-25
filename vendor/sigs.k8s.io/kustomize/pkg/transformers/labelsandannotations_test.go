// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transformers

import (
	"testing"

	"sigs.k8s.io/kustomize/k8sdeps/kunstruct"
	"sigs.k8s.io/kustomize/pkg/resmaptest"
	"sigs.k8s.io/kustomize/pkg/resource"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

var rf = resource.NewFactory(kunstruct.NewKunstructuredFactoryImpl())
var defaultTransformerConfig = config.MakeDefaultConfig()

func TestLabelsRun(t *testing.T) {
	m := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "cm1",
			},
		}).
		Add(map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"old-label": "old-value",
						},
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name": "svc1",
			},
			"spec": map[string]interface{}{
				"ports": []interface{}{
					map[string]interface{}{
						"name": "port1",
						"port": "12345",
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "batch/v1",
			"kind":       "Job",
			"metadata": map[string]interface{}{
				"name": "job1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "batch/v1",
			"kind":       "Job",
			"metadata": map[string]interface{}{
				"name": "job2",
			},
			"spec": map[string]interface{}{
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"old-label": "old-value",
					},
				},
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "batch/v1beta1",
			"kind":       "CronJob",
			"metadata": map[string]interface{}{
				"name": "cronjob1",
			},
			"spec": map[string]interface{}{
				"schedule": "* 23 * * *",
				"jobTemplate": map[string]interface{}{
					"spec": map[string]interface{}{
						"template": map[string]interface{}{
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nginx",
										"image": "nginx:1.7.9",
									},
								},
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "batch/v1beta1",
			"kind":       "CronJob",
			"metadata": map[string]interface{}{
				"name": "cronjob2",
			},
			"spec": map[string]interface{}{
				"schedule": "* 23 * * *",
				"jobTemplate": map[string]interface{}{
					"spec": map[string]interface{}{
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"old-label": "old-value",
							},
						},
						"template": map[string]interface{}{
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nginx",
										"image": "nginx:1.7.9",
									},
								},
							},
						},
					},
				},
			},
		}).ResMap()

	expected := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "cm1",
				"labels": map[string]interface{}{
					"label-key1": "label-value1",
					"label-key2": "label-value2",
				},
			},
		}).
		Add(map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
				"labels": map[string]interface{}{
					"label-key1": "label-value1",
					"label-key2": "label-value2",
				},
			},
			"spec": map[string]interface{}{
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"label-key1": "label-value1",
						"label-key2": "label-value2",
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"old-label":  "old-value",
							"label-key1": "label-value1",
							"label-key2": "label-value2",
						},
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name": "svc1",
				"labels": map[string]interface{}{
					"label-key1": "label-value1",
					"label-key2": "label-value2",
				},
			},
			"spec": map[string]interface{}{
				"ports": []interface{}{
					map[string]interface{}{
						"name": "port1",
						"port": "12345",
					},
				},
				"selector": map[string]interface{}{
					"label-key1": "label-value1",
					"label-key2": "label-value2",
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "batch/v1",
			"kind":       "Job",
			"metadata": map[string]interface{}{
				"name": "job1",
				"labels": map[string]interface{}{
					"label-key1": "label-value1",
					"label-key2": "label-value2",
				},
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"label-key1": "label-value1",
							"label-key2": "label-value2",
						},
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "batch/v1",
			"kind":       "Job",
			"metadata": map[string]interface{}{
				"name": "job2",
				"labels": map[string]interface{}{
					"label-key1": "label-value1",
					"label-key2": "label-value2",
				},
			},
			"spec": map[string]interface{}{
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"label-key1": "label-value1",
						"label-key2": "label-value2",
						"old-label":  "old-value",
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"label-key1": "label-value1",
							"label-key2": "label-value2",
						},
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "batch/v1beta1",
			"kind":       "CronJob",
			"metadata": map[string]interface{}{
				"name": "cronjob1",
				"labels": map[string]interface{}{
					"label-key1": "label-value1",
					"label-key2": "label-value2",
				},
			},
			"spec": map[string]interface{}{
				"schedule": "* 23 * * *",
				"jobTemplate": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"label-key1": "label-value1",
							"label-key2": "label-value2",
						},
					},
					"spec": map[string]interface{}{
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"label-key1": "label-value1",
									"label-key2": "label-value2",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nginx",
										"image": "nginx:1.7.9",
									},
								},
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "batch/v1beta1",
			"kind":       "CronJob",
			"metadata": map[string]interface{}{
				"name": "cronjob2",
				"labels": map[string]interface{}{
					"label-key1": "label-value1",
					"label-key2": "label-value2",
				},
			},
			"spec": map[string]interface{}{
				"schedule": "* 23 * * *",
				"jobTemplate": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"label-key1": "label-value1",
							"label-key2": "label-value2",
						},
					},
					"spec": map[string]interface{}{
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"old-label":  "old-value",
								"label-key1": "label-value1",
								"label-key2": "label-value2",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"label-key1": "label-value1",
									"label-key2": "label-value2",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nginx",
										"image": "nginx:1.7.9",
									},
								},
							},
						},
					},
				},
			},
		}).ResMap()

	lt, err := NewLabelsMapTransformer(
		map[string]string{"label-key1": "label-value1", "label-key2": "label-value2"},
		defaultTransformerConfig.CommonLabels)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = lt.Transform(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = expected.ErrorIfNotEqualLists(m); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestAnnotationsRun(t *testing.T) {
	m := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "cm1",
			},
		}).
		Add(map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"old-label": "old-value",
						},
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name": "svc1",
			},
			"spec": map[string]interface{}{
				"ports": []interface{}{
					map[string]interface{}{
						"name": "port1",
						"port": "12345",
					},
				},
			},
		}).ResMap()

	expected := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "cm1",
				"annotations": map[string]interface{}{
					"anno-key1": "anno-value1",
					"anno-key2": "anno-value2",
				},
			},
		}).
		Add(map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
				"annotations": map[string]interface{}{
					"anno-key1": "anno-value1",
					"anno-key2": "anno-value2",
				},
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"annotations": map[string]interface{}{
							"anno-key1": "anno-value1",
							"anno-key2": "anno-value2",
						},
						"labels": map[string]interface{}{
							"old-label": "old-value",
						},
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		}).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name": "svc1",
				"annotations": map[string]interface{}{
					"anno-key1": "anno-value1",
					"anno-key2": "anno-value2",
				},
			},
			"spec": map[string]interface{}{
				"ports": []interface{}{
					map[string]interface{}{
						"name": "port1",
						"port": "12345",
					},
				},
			},
		}).ResMap()
	at, err := NewAnnotationsMapTransformer(
		map[string]string{"anno-key1": "anno-value1", "anno-key2": "anno-value2"},
		defaultTransformerConfig.CommonAnnotations)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = at.Transform(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = expected.ErrorIfNotEqualLists(m); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestAnnotationsRunWithNullValue(t *testing.T) {
	m := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name":        "cm1",
				"annotations": nil,
			},
		}).ResMap()

	expected := resmaptest_test.NewRmBuilder(t, rf).
		Add(map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name": "cm1",
				"annotations": map[string]interface{}{
					"anno-key1": "anno-value1",
					"anno-key2": "anno-value2",
				},
			},
		}).ResMap()

	at, err := NewAnnotationsMapTransformer(
		map[string]string{"anno-key1": "anno-value1", "anno-key2": "anno-value2"},
		defaultTransformerConfig.CommonAnnotations)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = at.Transform(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = expected.ErrorIfNotEqualLists(m); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}
