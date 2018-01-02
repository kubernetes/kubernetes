// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	v2 "github.com/googleapis/gnostic/OpenAPIv2"
)

func buildDocumentV2() *v2.Document {
	d := &v2.Document{}
	d.Swagger = "2.0"
	d.Info = &v2.Info{
		Title:   "Swagger Petstore",
		Version: "1.0.0",
		License: &v2.License{Name: "MIT"},
	}
	d.Host = "petstore.swagger.io"
	d.BasePath = "/v1"
	d.Schemes = []string{"http"}
	d.Consumes = []string{"application/json"}
	d.Produces = []string{"application/json"}
	d.Paths = &v2.Paths{}
	d.Paths.Path = append(d.Paths.Path,
		&v2.NamedPathItem{
			Name: "/pets",
			Value: &v2.PathItem{
				Get: &v2.Operation{
					Summary:     "List all pets",
					OperationId: "listPets",
					Tags:        []string{"pets"},
					Parameters: []*v2.ParametersItem{
						&v2.ParametersItem{
							Oneof: &v2.ParametersItem_Parameter{
								Parameter: &v2.Parameter{
									Oneof: &v2.Parameter_NonBodyParameter{
										NonBodyParameter: &v2.NonBodyParameter{
											Oneof: &v2.NonBodyParameter_QueryParameterSubSchema{
												QueryParameterSubSchema: &v2.QueryParameterSubSchema{
													Name:        "limit",
													In:          "query",
													Description: "How many items to return at one time (max 100)",
													Required:    false,
													Type:        "integer",
													Format:      "int32",
												},
											},
										},
									},
								},
							},
						},
					},
					Responses: &v2.Responses{
						ResponseCode: []*v2.NamedResponseValue{
							&v2.NamedResponseValue{
								Name: "200",
								Value: &v2.ResponseValue{
									Oneof: &v2.ResponseValue_Response{
										Response: &v2.Response{
											Description: "An paged array of pets", // [sic] match other examples
											Schema: &v2.SchemaItem{
												Oneof: &v2.SchemaItem_Schema{
													Schema: &v2.Schema{
														XRef: "#/definitions/Pets",
													},
												},
											},
											Headers: &v2.Headers{
												AdditionalProperties: []*v2.NamedHeader{
													&v2.NamedHeader{
														Name: "x-next",
														Value: &v2.Header{
															Type:        "string",
															Description: "A link to the next page of responses",
														},
													},
												},
											},
										},
									},
								},
							},
							&v2.NamedResponseValue{
								Name: "default",
								Value: &v2.ResponseValue{
									Oneof: &v2.ResponseValue_Response{
										Response: &v2.Response{
											Description: "unexpected error",
											Schema: &v2.SchemaItem{
												Oneof: &v2.SchemaItem_Schema{
													Schema: &v2.Schema{
														XRef: "#/definitions/Error",
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
				Post: &v2.Operation{
					Summary:     "Create a pet",
					OperationId: "createPets",
					Tags:        []string{"pets"},
					Parameters:  []*v2.ParametersItem{},
					Responses: &v2.Responses{
						ResponseCode: []*v2.NamedResponseValue{
							&v2.NamedResponseValue{
								Name: "201",
								Value: &v2.ResponseValue{
									Oneof: &v2.ResponseValue_Response{
										Response: &v2.Response{
											Description: "Null response",
										},
									},
								},
							},
							&v2.NamedResponseValue{
								Name: "default",
								Value: &v2.ResponseValue{
									Oneof: &v2.ResponseValue_Response{
										Response: &v2.Response{
											Description: "unexpected error",
											Schema: &v2.SchemaItem{
												Oneof: &v2.SchemaItem_Schema{
													Schema: &v2.Schema{
														XRef: "#/definitions/Error",
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
			}})
	d.Paths.Path = append(d.Paths.Path,
		&v2.NamedPathItem{
			Name: "/pets/{petId}",
			Value: &v2.PathItem{
				Get: &v2.Operation{
					Summary:     "Info for a specific pet",
					OperationId: "showPetById",
					Tags:        []string{"pets"},
					Parameters: []*v2.ParametersItem{
						&v2.ParametersItem{
							Oneof: &v2.ParametersItem_Parameter{
								Parameter: &v2.Parameter{
									Oneof: &v2.Parameter_NonBodyParameter{
										NonBodyParameter: &v2.NonBodyParameter{
											Oneof: &v2.NonBodyParameter_PathParameterSubSchema{
												PathParameterSubSchema: &v2.PathParameterSubSchema{
													Name:        "petId",
													In:          "path",
													Description: "The id of the pet to retrieve",
													Required:    true,
													Type:        "string",
												},
											},
										},
									},
								},
							},
						},
					},
					Responses: &v2.Responses{
						ResponseCode: []*v2.NamedResponseValue{
							&v2.NamedResponseValue{
								Name: "200",
								Value: &v2.ResponseValue{
									Oneof: &v2.ResponseValue_Response{
										Response: &v2.Response{
											Description: "Expected response to a valid request",
											Schema: &v2.SchemaItem{
												Oneof: &v2.SchemaItem_Schema{
													Schema: &v2.Schema{
														XRef: "#/definitions/Pets",
													},
												},
											},
										},
									},
								},
							},
							&v2.NamedResponseValue{
								Name: "default",
								Value: &v2.ResponseValue{
									Oneof: &v2.ResponseValue_Response{
										Response: &v2.Response{
											Description: "unexpected error",
											Schema: &v2.SchemaItem{
												Oneof: &v2.SchemaItem_Schema{
													Schema: &v2.Schema{
														XRef: "#/definitions/Error",
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
			}})
	d.Definitions = &v2.Definitions{}
	d.Definitions.AdditionalProperties = append(d.Definitions.AdditionalProperties,
		&v2.NamedSchema{
			Name: "Pet",
			Value: &v2.Schema{
				Required: []string{"id", "name"},
				Properties: &v2.Properties{
					AdditionalProperties: []*v2.NamedSchema{
						&v2.NamedSchema{Name: "id", Value: &v2.Schema{
							Type:   &v2.TypeItem{[]string{"integer"}},
							Format: "int64"}},
						&v2.NamedSchema{Name: "name", Value: &v2.Schema{Type: &v2.TypeItem{[]string{"string"}}}},
						&v2.NamedSchema{Name: "tag", Value: &v2.Schema{Type: &v2.TypeItem{[]string{"string"}}}},
					},
				},
			}})
	d.Definitions.AdditionalProperties = append(d.Definitions.AdditionalProperties,
		&v2.NamedSchema{
			Name: "Pets",
			Value: &v2.Schema{
				Type:  &v2.TypeItem{[]string{"array"}},
				Items: &v2.ItemsItem{[]*v2.Schema{&v2.Schema{XRef: "#/definitions/Pet"}}},
			}})
	d.Definitions.AdditionalProperties = append(d.Definitions.AdditionalProperties,
		&v2.NamedSchema{
			Name: "Error",
			Value: &v2.Schema{
				Required: []string{"code", "message"},
				Properties: &v2.Properties{
					AdditionalProperties: []*v2.NamedSchema{
						&v2.NamedSchema{Name: "code", Value: &v2.Schema{
							Type:   &v2.TypeItem{[]string{"integer"}},
							Format: "int32"}},
						&v2.NamedSchema{Name: "message", Value: &v2.Schema{Type: &v2.TypeItem{[]string{"string"}}}},
					},
				},
			}})
	return d
}
