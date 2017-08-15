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

package parse_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"

	"github.com/ghodss/yaml"

	//"github.com/davecgh/go-spew/spew"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/kubectl/apply"
	"k8s.io/kubernetes/pkg/kubectl/apply/parse"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	tst "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
)

var _ = Describe("Creating an Element", func() {
	var instance parse.Factory

	BeforeEach(func() {
		fakeSchema := tst.Fake{Path: filepath.Join("..", "..", "..", "..", "api", "openapi-spec", "swagger.json")}
		s, err := fakeSchema.OpenAPISchema()
		Expect(err).To(BeNil())
		resources, err := openapi.NewOpenAPIData(s)
		Expect(err).To(BeNil())
		instance = parse.Factory{resources}
	})

	Context("with one of the configs missing", func() {
		It("should fail to create an Element", func() {
			config := create(`
			apiVersion: apps/v1beta1
			kind: Deployment
			metadata:
			  name: deployment
			  labels:
			    recorded: a
			    recorded-local: ab
			    recorded-remote: ac
			    recorded-local-remote: abc`)

			By("only passing in the recorded config")
			_, err := instance.CreateElement(config, map[string]interface{}{}, map[string]interface{}{})
			Expect(err).Should(HaveOccurred())

			By("only passing in the local config")
			_, err = instance.CreateElement(map[string]interface{}{}, config, map[string]interface{}{})
			Expect(err).Should(HaveOccurred())

			By("only passing in the remote config")
			_, err = instance.CreateElement(map[string]interface{}{}, map[string]interface{}{}, config)
			Expect(err).Should(HaveOccurred())

			By("passing in the recorded and local config")
			_, err = instance.CreateElement(config, config, map[string]interface{}{})
			Expect(err).Should(HaveOccurred())

			By("passing in the local and remote config")
			_, err = instance.CreateElement(map[string]interface{}{}, config, config)
			Expect(err).Should(HaveOccurred())
		})
	})

	Context("with openapi and recorded, local and remote config", func() {
		It("should combine empty configs", func() {
			recorded := create(`
			apiVersion: apps/v1beta1
			kind: Deployment
			metadata:
			  name: deployment`)

			local := create(`
			apiVersion: apps/v1beta1
			kind: Deployment
			metadata:
			  name: deployment`)

			remote := create(`
			apiVersion: apps/v1beta1
			kind: Deployment
			metadata:
			  name: deployment`)

			result, err := instance.CreateElement(recorded, local, remote)
			Expect(err).Should(Not(HaveOccurred()))

			expected := &apply.TypeElement{
				Recorded:    recorded,
				RecordedSet: true,
				Local:       local,
				LocalSet:    true,
				Remote:      remote,
				RemoteSet:   true,
				Values: map[string]apply.Element{
					"kind": &apply.PrimitiveElement{
						Name:        "kind",
						Recorded:    "Deployment",
						RecordedSet: true,
						Local:       "Deployment",
						LocalSet:    true,
						Remote:      "Deployment",
						RemoteSet:   true,
					},
					"apiVersion": &apply.PrimitiveElement{
						Name:        "apiVersion",
						Recorded:    "apps/v1beta1",
						RecordedSet: true,
						Local:       "apps/v1beta1",
						LocalSet:    true,
						Remote:      "apps/v1beta1",
						RemoteSet:   true,
					},
					"metadata": &apply.TypeElement{
						Name:        "metadata",
						Recorded:    lookupMap("metadata", recorded),
						RecordedSet: true,
						Local:       lookupMap("metadata", local),
						LocalSet:    true,
						Remote:      lookupMap("metadata", remote),
						RemoteSet:   true,
						Values: map[string]apply.Element{
							"name": &apply.PrimitiveElement{
								Name:        "name",
								Recorded:    "deployment",
								RecordedSet: true,
								Local:       "deployment",
								LocalSet:    true,
								Remote:      "deployment",
								RemoteSet:   true,
							},
						},
					},
				},
			}
			ExpectEqual(result, expected)
		})
	})

	XContext("openapi and recorded, local and remote config", func() {
		It("should combine maps with identical keys and different values", func() {
			recorded := create(`
			apiVersion: apps/v1beta1
			kind: Deployment
			metadata:
			  name: deployment
			  labels:
			    recorded: a
			    recorded-local: ab
			    recorded-remote: ac
			    recorded-local-remote: abc`)

			local := create(`
			apiVersion: apps/v1beta1
			kind: Deployment
			metadata:
			  name: deployment
			  labels:
			    local: b
			    recorded-local: ba
			    local-remote: bc
			    recorded-local-remote: bac`)

			remote := create(`
			apiVersion: apps/v1beta1
			kind: Deployment
			metadata:
			  name: deployment
			  labels:
			    remote: c
			    local-remote: cb
			    recorded-remote: ca
			    recorded-local-remote: cab`)

			result, err := instance.CreateElement(recorded, local, remote)
			Expect(err).Should(Not(HaveOccurred()))

			expected := getExpected(recorded, local, remote)
			expected.Values["metadata"] = &apply.TypeElement{
				Name:     "metadata",
				Recorded: getMap(recorded["metadata"]),
				Local:    getMap(local["metadata"]),
				Remote:   getMap(remote["metadata"]),
				Values: map[string]apply.Element{
					"name": &apply.PrimitiveElement{
						Name:     "name",
						Recorded: "deployment",
						Local:    "deployment",
						Remote:   "deployment",
					},
					"labels": &apply.MapElement{
						Name:     "labels",
						Recorded: lookupMap("metadata.labels", recorded),
						Local:    lookupMap("metadata.labels", local),
						Remote:   lookupMap("metadata.labels", remote),
						Values: map[string]apply.Element{
							"recorded": &apply.PrimitiveElement{
								Name:     "recorded",
								Recorded: "a",
							},
							"local": &apply.PrimitiveElement{
								Name:  "local",
								Local: "b",
							},
							"remote": &apply.PrimitiveElement{
								Name:   "remote",
								Remote: "c",
							},
							"recorded-remote": &apply.PrimitiveElement{
								Name:     "recorded-remote",
								Recorded: "ac",
								Remote:   "ca",
							},
							"local-remote": &apply.PrimitiveElement{
								Name:   "local-remote",
								Local:  "bc",
								Remote: "cb",
							},
							"recorded-local": &apply.PrimitiveElement{
								Name:     "recorded-local",
								Recorded: "ab",
								Local:    "ba",
							},
							"recorded-local-remote": &apply.PrimitiveElement{
								Name:     "recorded-local-remote",
								Recorded: "abc",
								Local:    "bac",
								Remote:   "cab",
							},
						},
					},
				},
			}
			ExpectEqual(result, expected)
		})
	})

	XContext("openapi and recorded, local and remote config", func() {
		It("should combine empty fields", func() {
			recorded := create(`
	           apiVersion: apps/v1beta1
	           kind: Deployment
	           spec:`)
			local := create(`
	           apiVersion: apps/v1beta1
	           kind: Deployment
	           spec:`)
			remote := create(`
	           apiVersion: apps/v1beta1
	           kind: Deployment
	           spec:`)

			result, err := instance.CreateElement(recorded, local, remote)
			Expect(err).Should(Not(HaveOccurred()))

			expected := &apply.TypeElement{
				Recorded: recorded,
				Local:    local,
				Remote:   remote,
				Values: map[string]apply.Element{
					"kind": &apply.PrimitiveElement{
						Name:     "kind",
						Recorded: "Deployment",
						Local:    "Deployment",
						Remote:   "Deployment",
					},
					"apiVersion": &apply.PrimitiveElement{
						Name:     "apiVersion",
						Recorded: "apps/v1beta1",
						Local:    "apps/v1beta1",
						Remote:   "apps/v1beta1",
					},
					"spec": &apply.EmptyElement{
						Name: "spec",
					},
				},
			}
			ExpectEqual(result, expected)
		})
	})

	XContext("openapi and recorded, local and remote config", func() {
		It("should combine lists with identical merge keys and different values", func() {
			recorded := create(`
		apiVersion: apps/v1beta1
		kind: Deployment
		spec:
		  template:
		    spec:
		      containers:
		      - name: recorded
		        image: recorded:a
		        timeoutSeconds: 1
		      - name: recorded-local
		        image: recorded:b
		        timeoutSeconds: 2
		      - name: recorded-remote
		        image: recorded:c
		        timeoutSeconds: 3
		      - name: recorded-local-remote
		        image: recorded:d
		        timeoutSeconds: 4
		`)
			local := create(`
		apiVersion: apps/v1beta1
		kind: Deployment
		spec:
		  template:
		    spec:
		      containers:
		        - name: local
		          image: local:a
		          initialDelaySeconds: 15
		        - name: recorded-local-remote
		          image: local:b
		          initialDelaySeconds: 16
		        - name: local-remote
		          image: local:c
		          initialDelaySeconds: 17
		        - name: recorded-local
		          image: local:d
		          initialDelaySeconds: 18
		`)
			remote := create(`
		apiVersion: apps/v1beta1
		kind: Deployment
		spec:
		  template:
		    spec:
		      containers:
		        - name: remote
		          image: remote:a
		          imagePullPolicy: Always
		        - name: recorded-remote
		          image: remote:b
		          imagePullPolicy: Always
		        - name: local-remote
		          image: remote:c
		          imagePullPolicy: Always
		        - name: recorded-local-remote
		          image: remote:d
		          imagePullPolicy: Always
		`)

			result, err := instance.CreateElement(recorded, local, remote)
			Expect(err).Should(Not(HaveOccurred()))

			expected := &apply.TypeElement{
				Recorded: recorded,
				Local:    local,
				Remote:   remote,
				Values: map[string]apply.Element{
					"kind": &apply.PrimitiveElement{
						Name:     "kind",
						Recorded: "Deployment",
						Local:    "Deployment",
						Remote:   "Deployment",
					},
					"apiVersion": &apply.PrimitiveElement{
						Name:     "apiVersion",
						Recorded: "apps/v1beta1",
						Local:    "apps/v1beta1",
						Remote:   "apps/v1beta1",
					},
					"spec": &apply.TypeElement{
						Name:     "spec",
						Recorded: lookupMap("spec", recorded),
						Local:    lookupMap("spec", local),
						Remote:   lookupMap("spec", remote),
						Values: map[string]apply.Element{
							"template": &apply.TypeElement{
								Name:     "template",
								Recorded: lookupMap("spec.template", recorded),
								Local:    lookupMap("spec.template", local),
								Remote:   lookupMap("spec.template", remote),
								Values: map[string]apply.Element{
									"spec": &apply.TypeElement{
										Name:     "spec",
										Recorded: lookupMap("spec.template.spec", recorded),
										Local:    lookupMap("spec.template.spec", local),
										Remote:   lookupMap("spec.template.spec", remote),
										Values: map[string]apply.Element{
											"containers": &apply.ListElement{
												Name: "containers",
												FieldMetaImpl: apply.FieldMetaImpl{
													MergeKey:  []string{"name"},
													MergeType: "merge",
												},
												Recorded: lookupList("spec.template.spec.containers", recorded),
												Local:    lookupList("spec.template.spec.containers", local),
												Remote:   lookupList("spec.template.spec.containers", remote),
												Values: []apply.Element{
													&apply.TypeElement{
														Name:     "0",
														Recorded: nil,
														Local:    lookupMap("spec.template.spec.containers[0]", local),
														Remote:   nil,
														Values: map[string]apply.Element{
															"name": &apply.PrimitiveElement{
																Name:     "name",
																Recorded: nil,
																Local:    "local",
																Remote:   nil,
															},
															"image": &apply.PrimitiveElement{
																Name:     "image",
																Recorded: nil,
																Local:    "local:a",
																Remote:   nil,
															},
															"initialDelaySeconds": &apply.PrimitiveElement{
																Name:     "initialDelaySeconds",
																Recorded: nil,
																Local:    float64(15),
																Remote:   nil,
															},
														},
													},
													&apply.TypeElement{
														Name:     "1",
														Recorded: lookupMap("spec.template.spec.containers[3]", recorded),
														Local:    lookupMap("spec.template.spec.containers[1]", local),
														Remote:   lookupMap("spec.template.spec.containers[3]", remote),
														Values: map[string]apply.Element{
															"name": &apply.PrimitiveElement{
																Name:     "name",
																Recorded: "recorded-local-remote",
																Local:    "recorded-local-remote",
																Remote:   "recorded-local-remote",
															},
															"image": &apply.PrimitiveElement{
																Name:     "image",
																Recorded: "recorded:d",
																Local:    "local:b",
																Remote:   "remote:d",
															},
															"timeoutSeconds": &apply.PrimitiveElement{
																Name:     "timeoutSeconds",
																Recorded: float64(4),
																Local:    nil,
																Remote:   nil,
															},
															"initialDelaySeconds": &apply.PrimitiveElement{
																Name:     "initialDelaySeconds",
																Recorded: nil,
																Local:    float64(16),
																Remote:   nil,
															},
															"imagePullPolicy": &apply.PrimitiveElement{
																Name:     "imagePullPolicy",
																Recorded: nil,
																Local:    nil,
																Remote:   "Always",
															},
														},
													},
													&apply.TypeElement{
														Name:     "2",
														Recorded: nil,
														Local:    lookupMap("spec.template.spec.containers[2]", local),
														Remote:   lookupMap("spec.template.spec.containers[2]", remote),
														Values: map[string]apply.Element{
															"name": &apply.PrimitiveElement{
																Name:     "name",
																Recorded: nil,
																Local:    "local-remote",
																Remote:   "local-remote",
															},
															"image": &apply.PrimitiveElement{
																Name:     "image",
																Recorded: nil,
																Local:    "local:c",
																Remote:   "remote:c",
															},
															"initialDelaySeconds": &apply.PrimitiveElement{
																Name:     "initialDelaySeconds",
																Recorded: nil,
																Local:    float64(17),
																Remote:   nil,
															},
															"imagePullPolicy": &apply.PrimitiveElement{
																Name:     "imagePullPolicy",
																Recorded: nil,
																Local:    nil,
																Remote:   "Always",
															},
														},
													},
													&apply.TypeElement{
														Name:     "3",
														Recorded: lookupMap("spec.template.spec.containers[1]", recorded),
														Local:    lookupMap("spec.template.spec.containers[3]", local),
														Remote:   nil,
														Values: map[string]apply.Element{
															"name": &apply.PrimitiveElement{
																Name:     "name",
																Recorded: "recorded-local",
																Local:    "recorded-local",
																Remote:   nil,
															},
															"image": &apply.PrimitiveElement{
																Name:     "image",
																Recorded: "recorded:b",
																Local:    "local:d",
																Remote:   nil,
															},
															"timeoutSeconds": &apply.PrimitiveElement{
																Name:     "timeoutSeconds",
																Recorded: float64(2),
																Local:    nil,
																Remote:   nil,
															},
															"initialDelaySeconds": &apply.PrimitiveElement{
																Name:     "initialDelaySeconds",
																Recorded: nil,
																Local:    float64(18),
																Remote:   nil,
															},
														},
													},
													&apply.TypeElement{
														Name:     "4",
														Recorded: nil,
														Local:    nil,
														Remote:   lookupMap("spec.template.spec.containers[0]", remote),
														Values: map[string]apply.Element{
															"name": &apply.PrimitiveElement{
																Name:     "name",
																Recorded: nil,
																Local:    nil,
																Remote:   "remote",
															},
															"image": &apply.PrimitiveElement{
																Name:     "image",
																Recorded: nil,
																Local:    nil,
																Remote:   "remote:a",
															},
															"imagePullPolicy": &apply.PrimitiveElement{
																Name:     "imagePullPolicy",
																Recorded: nil,
																Local:    nil,
																Remote:   "Always",
															},
														},
													},
													&apply.TypeElement{
														Name:     "5",
														Recorded: lookupMap("spec.template.spec.containers[2]", recorded),
														Local:    nil,
														Remote:   lookupMap("spec.template.spec.containers[1]", remote),
														Values: map[string]apply.Element{
															"name": &apply.PrimitiveElement{
																Name:     "name",
																Recorded: "recorded-remote",
																Local:    nil,
																Remote:   "recorded-remote",
															},
															"image": &apply.PrimitiveElement{
																Name:     "image",
																Recorded: "recorded:c",
																Local:    nil,
																Remote:   "remote:b",
															},
															"timeoutSeconds": &apply.PrimitiveElement{
																Name:     "timeoutSeconds",
																Recorded: float64(3),
																Local:    nil,
																Remote:   nil,
															},
															"imagePullPolicy": &apply.PrimitiveElement{
																Name:     "imagePullPolicy",
																Recorded: nil,
																Local:    nil,
																Remote:   "Always",
															},
														},
													},
													&apply.TypeElement{
														Name:     "6",
														Recorded: lookupMap("spec.template.spec.containers[0]", recorded),
														Local:    nil,
														Remote:   nil,
														Values: map[string]apply.Element{
															"name": &apply.PrimitiveElement{
																Name:     "name",
																Recorded: "recorded",
																Local:    nil,
																Remote:   nil,
															},
															"image": &apply.PrimitiveElement{
																Name:     "image",
																Recorded: "recorded:a",
																Local:    nil,
																Remote:   nil,
															},
															"timeoutSeconds": &apply.PrimitiveElement{
																Name:     "timeoutSeconds",
																Recorded: float64(1),
																Local:    nil,
																Remote:   nil,
															},
														},
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			}
			ExpectEqual(result, expected)
		})
	})

	// TODO: Write test to replace type array

	// TODO: Write test to merge primitive array
	// TODO: Write test to replace primitive array

	// TODO: Write test to merge map array
	// TODO: Write test to replace map array

	// TODO: Write test to merge map of types
	// TODO: Write test to replace map of types
	// TODO: Write test to replace keys map of types

	// TODO: Write test to merge map of maps
	// TODO: Write test to replace map of maps
	// TODO: Write test to replace keys map of maps

	// TODO: Write test where some fields are not in the openapi
})

func create(config string) map[string]interface{} {
	result := map[string]interface{}{}
	c := []byte(normalize(config))
	Expect(yaml.Unmarshal(c, &result)).Should(Not(HaveOccurred()), fmt.Sprintf("Could not parse config:\n\n%s\n", c))
	return result
}

func normalize(s string) string {
	lines := []string{}
	found := false
	spaces := 0
	for _, line := range strings.Split(s, "\n") {
		// Count the indentation
		if len(line) > 0 && !found {
			for _, v := range line {
				if unicode.IsSpace(v) {
					spaces++
				} else {
					break
				}
			}
			found = true
		}
		// Trim the indentation from the config
		line = line[spaces:]
		Expect(strings.Contains(line, "\t")).To(BeFalse(),
			fmt.Sprintf("yaml config cannot contain tabs: %v", line))
		lines = append(lines, line)
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func getMap(elem interface{}) map[string]interface{} {
	Expect(elem).ShouldNot(BeNil())
	result, ok := elem.(map[string]interface{})
	Expect(ok).To(BeTrue())
	return result
}

func getList(elem interface{}) []interface{} {
	Expect(elem).ShouldNot(BeNil())
	result, ok := elem.([]interface{})
	Expect(ok).To(BeTrue())
	return result
}

func getExpected(recorded, local, remote map[string]interface{}) *apply.TypeElement {
	return &apply.TypeElement{
		Recorded: recorded,
		Local:    local,
		Remote:   remote,
		Values: map[string]apply.Element{
			"kind": &apply.PrimitiveElement{
				Name:     "kind",
				Recorded: "Deployment",
				Local:    "Deployment",
				Remote:   "Deployment",
			},
			"apiVersion": &apply.PrimitiveElement{
				Name:     "apiVersion",
				Recorded: "apps/v1beta1",
				Local:    "apps/v1beta1",
				Remote:   "apps/v1beta1",
			},
		},
	}
}

func lookup(path string, elem interface{}) interface{} {
	items := strings.Split(path, ".")
	for _, i := range items {
		m := getMap(elem)
		// Array
		if strings.Contains(i, "[") {
			// Parse the field and index
			fieldandindex := strings.Split(i, "[")
			field := fieldandindex[0]

			// Lookup the field
			elem = m[field]

			for cnt, index := range fieldandindex {
				if cnt == 0 {
					continue
				}

				// Strip the closing bracket and parse the index into an int
				index, err := strconv.Atoi(strings.Replace(index, "]", "", -1))
				Expect(err).ShouldNot(HaveOccurred())

				// Lookup the element at the index
				curr := getList(elem)
				elem = curr[index]
			}
		} else {
			elem = m[i]
		}
	}
	return elem
}

func lookupMap(path string, elem interface{}) map[string]interface{} {
	return getMap(lookup(path, elem))
}

func lookupList(path string, elem interface{}) []interface{} {
	return getList(lookup(path, elem))
}

func ExpectEqual(result, expected interface{}) {
	Expect(result).Should(Equal(expected),
		fmt.Sprintf("Diff: %s", diff.ObjectReflectDiff(result, expected)))
}
