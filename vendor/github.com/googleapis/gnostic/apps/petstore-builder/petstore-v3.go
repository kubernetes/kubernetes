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
	v3 "github.com/googleapis/gnostic/OpenAPIv3"
)

func buildDocumentV3() *v3.Document {
	d := &v3.Document{}
	d.Openapi = "3.0"
	d.Info = &v3.Info{
		Title:   "OpenAPI Petstore",
		Version: "1.0.0",
		License: &v3.License{Name: "MIT"},
	}
	d.Servers = append(d.Servers, &v3.Server{
		Url:         "https://petstore.openapis.org/v1",
		Description: "Development server",
	})
	d.Paths = &v3.Paths{}
	d.Paths.Path = append(d.Paths.Path,
		&v3.NamedPathItem{
			Name: "/pets",
			Value: &v3.PathItem{
				Get: &v3.Operation{
					Summary:     "List all pets",
					OperationId: "listPets",
					Tags:        []string{"pets"},
					Parameters: []*v3.ParameterOrReference{
						&v3.ParameterOrReference{
							Oneof: &v3.ParameterOrReference_Parameter{
								Parameter: &v3.Parameter{
									Name:        "limit",
									In:          "query",
									Description: "How many items to return at one time (max 100)",
									Required:    false,
									Schema: &v3.SchemaOrReference{
										Oneof: &v3.SchemaOrReference_Schema{
											Schema: &v3.Schema{
												Type:   "integer",
												Format: "int32",
											},
										},
									},
								},
							},
						},
					},
					Responses: &v3.Responses{
						Default: &v3.ResponseOrReference{
							Oneof: &v3.ResponseOrReference_Response{
								Response: &v3.Response{
									Description: "unexpected error",
									Content: &v3.MediaTypes{
										AdditionalProperties: []*v3.NamedMediaType{
											&v3.NamedMediaType{
												Name: "application/json",
												Value: &v3.MediaType{
													Schema: &v3.SchemaOrReference{
														Oneof: &v3.SchemaOrReference_Reference{
															Reference: &v3.Reference{
																XRef: "#/components/schemas/Error",
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
						ResponseOrReference: []*v3.NamedResponseOrReference{
							&v3.NamedResponseOrReference{
								Name: "200",
								Value: &v3.ResponseOrReference{
									Oneof: &v3.ResponseOrReference_Response{
										Response: &v3.Response{
											Description: "An paged array of pets", // [sic] match other examples
											Content: &v3.MediaTypes{
												AdditionalProperties: []*v3.NamedMediaType{
													&v3.NamedMediaType{
														Name: "application/json",
														Value: &v3.MediaType{
															Schema: &v3.SchemaOrReference{
																Oneof: &v3.SchemaOrReference_Reference{
																	Reference: &v3.Reference{
																		XRef: "#/components/schemas/Pets",
																	},
																},
															},
														},
													},
												},
											},
											Headers: &v3.HeadersOrReferences{
												AdditionalProperties: []*v3.NamedHeaderOrReference{
													&v3.NamedHeaderOrReference{
														Name: "x-next",
														Value: &v3.HeaderOrReference{
															Oneof: &v3.HeaderOrReference_Header{
																Header: &v3.Header{
																	Description: "A link to the next page of responses",
																	Schema: &v3.SchemaOrReference{
																		Oneof: &v3.SchemaOrReference_Schema{
																			Schema: &v3.Schema{
																				Type: "string",
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
							},
						},
					},
				},
				Post: &v3.Operation{
					Summary:     "Create a pet",
					OperationId: "createPets",
					Tags:        []string{"pets"},
					Responses: &v3.Responses{
						Default: &v3.ResponseOrReference{
							Oneof: &v3.ResponseOrReference_Response{
								Response: &v3.Response{
									Description: "unexpected error",
									Content: &v3.MediaTypes{
										AdditionalProperties: []*v3.NamedMediaType{
											&v3.NamedMediaType{
												Name: "application/json",
												Value: &v3.MediaType{
													Schema: &v3.SchemaOrReference{
														Oneof: &v3.SchemaOrReference_Reference{
															Reference: &v3.Reference{
																XRef: "#/components/schemas/Error",
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
						ResponseOrReference: []*v3.NamedResponseOrReference{
							&v3.NamedResponseOrReference{
								Name: "201",
								Value: &v3.ResponseOrReference{
									Oneof: &v3.ResponseOrReference_Response{
										Response: &v3.Response{
											Description: "Null response",
										},
									},
								},
							},
						},
					},
				},
			}},
		&v3.NamedPathItem{
			Name: "/pets/{petId}",
			Value: &v3.PathItem{
				Get: &v3.Operation{
					Summary:     "Info for a specific pet",
					OperationId: "showPetById",
					Tags:        []string{"pets"},
					Parameters: []*v3.ParameterOrReference{
						&v3.ParameterOrReference{
							Oneof: &v3.ParameterOrReference_Parameter{
								Parameter: &v3.Parameter{
									Name:        "petId",
									In:          "path",
									Description: "The id of the pet to retrieve",
									Required:    true,
									Schema: &v3.SchemaOrReference{
										Oneof: &v3.SchemaOrReference_Schema{
											Schema: &v3.Schema{
												Type: "string",
											},
										},
									},
								},
							},
						},
					},
					Responses: &v3.Responses{
						Default: &v3.ResponseOrReference{
							Oneof: &v3.ResponseOrReference_Response{
								Response: &v3.Response{
									Description: "unexpected error",
									Content: &v3.MediaTypes{
										AdditionalProperties: []*v3.NamedMediaType{
											&v3.NamedMediaType{
												Name: "application/json",
												Value: &v3.MediaType{
													Schema: &v3.SchemaOrReference{
														Oneof: &v3.SchemaOrReference_Reference{
															Reference: &v3.Reference{
																XRef: "#/components/schemas/Error",
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
						ResponseOrReference: []*v3.NamedResponseOrReference{
							&v3.NamedResponseOrReference{
								Name: "200",
								Value: &v3.ResponseOrReference{
									Oneof: &v3.ResponseOrReference_Response{
										Response: &v3.Response{
											Description: "Expected response to a valid request",
											Content: &v3.MediaTypes{
												AdditionalProperties: []*v3.NamedMediaType{
													&v3.NamedMediaType{
														Name: "application/json",
														Value: &v3.MediaType{
															Schema: &v3.SchemaOrReference{
																Oneof: &v3.SchemaOrReference_Reference{
																	Reference: &v3.Reference{
																		XRef: "#/components/schemas/Pets",
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
					},
				},
			}})
	d.Components = &v3.Components{
		Schemas: &v3.SchemasOrReferences{
			AdditionalProperties: []*v3.NamedSchemaOrReference{
				&v3.NamedSchemaOrReference{
					Name: "Pet",
					Value: &v3.SchemaOrReference{
						Oneof: &v3.SchemaOrReference_Schema{
							Schema: &v3.Schema{
								Required: []string{"id", "name"},
								Properties: &v3.Properties{
									AdditionalProperties: []*v3.NamedSchemaOrReference{
										&v3.NamedSchemaOrReference{
											Name: "id",
											Value: &v3.SchemaOrReference{
												Oneof: &v3.SchemaOrReference_Schema{
													Schema: &v3.Schema{
														Type:   "integer",
														Format: "int64",
													},
												},
											},
										},
										&v3.NamedSchemaOrReference{
											Name: "name",
											Value: &v3.SchemaOrReference{
												Oneof: &v3.SchemaOrReference_Schema{
													Schema: &v3.Schema{
														Type: "string",
													},
												},
											},
										},
										&v3.NamedSchemaOrReference{
											Name: "tag",
											Value: &v3.SchemaOrReference{
												Oneof: &v3.SchemaOrReference_Schema{
													Schema: &v3.Schema{
														Type: "string",
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
				&v3.NamedSchemaOrReference{
					Name: "Pets",
					Value: &v3.SchemaOrReference{
						Oneof: &v3.SchemaOrReference_Schema{
							Schema: &v3.Schema{
								Type: "array",
								Items: &v3.ItemsItem{
									SchemaOrReference: []*v3.SchemaOrReference{
										&v3.SchemaOrReference{
											Oneof: &v3.SchemaOrReference_Reference{
												Reference: &v3.Reference{
													XRef: "#/components/schemas/Pet",
												},
											},
										},
									},
								},
							},
						},
					},
				},
				&v3.NamedSchemaOrReference{
					Name: "Error",
					Value: &v3.SchemaOrReference{
						Oneof: &v3.SchemaOrReference_Schema{
							Schema: &v3.Schema{
								Required: []string{"code", "message"},
								Properties: &v3.Properties{
									AdditionalProperties: []*v3.NamedSchemaOrReference{
										&v3.NamedSchemaOrReference{
											Name: "code",
											Value: &v3.SchemaOrReference{
												Oneof: &v3.SchemaOrReference_Schema{
													Schema: &v3.Schema{
														Type:   "integer",
														Format: "int32",
													},
												},
											},
										},
										&v3.NamedSchemaOrReference{
											Name: "message",
											Value: &v3.SchemaOrReference{
												Oneof: &v3.SchemaOrReference_Schema{
													Schema: &v3.Schema{
														Type: "string",
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
	return d
}
