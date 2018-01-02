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

// report is a demo application that displays information about an
// OpenAPI description.
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/printer"

	pb "github.com/googleapis/gnostic/OpenAPIv2"
)

func readDocumentFromFileWithName(filename string) *pb.Document {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Printf("File error: %v\n", err)
		os.Exit(1)
	}
	document := &pb.Document{}
	err = proto.Unmarshal(data, document)
	if err != nil {
		panic(err)
	}
	return document
}

func printDocument(code *printer.Code, document *pb.Document) {
	code.Print("BasePath: %+v", document.BasePath)
	code.Print("Consumes: %+v", document.Consumes)
	code.Print("Definitions:")
	code.Indent()
	if document.Definitions != nil && document.Definitions.AdditionalProperties != nil {
		for _, pair := range document.Definitions.AdditionalProperties {
			code.Print("%s", pair.Name)
			code.Indent()
			printSchema(code, pair.Value)
			code.Outdent()
		}
	}
	code.Outdent()
	code.Print("ExternalDocs: %+v", document.ExternalDocs)
	code.Print("Host: %+v", document.Host)
	if document.Info != nil {
		code.Print("Info:")
		code.Indent()
		code.Print("Title: %s", document.Info.Title)
		code.Print("Description: %s", document.Info.Description)
		code.Print("Version: %s", document.Info.Version)
		code.Print("TermsOfService: %s", document.Info.TermsOfService)
		if document.Info.Contact != nil {
			code.Print("Contact Email: %s", document.Info.Contact.Email)
		}
		if document.Info.License != nil {
			code.Print("License Name: %s", document.Info.License.Name)
			code.Print("License URL: %s", document.Info.License.Url)
		}
		code.Outdent()
	}
	code.Print("Parameters: %+v", document.Parameters)
	code.Print("Paths:")
	code.Indent()
	for _, pair := range document.Paths.Path {
		code.Print("%+v", pair.Name)
		code.Indent()
		v := pair.Value
		if v.Get != nil {
			code.Print("GET")
			code.Indent()
			printOperation(code, v.Get)
			code.Outdent()
		}
		if v.Post != nil {
			code.Print("POST")
			code.Indent()
			printOperation(code, v.Post)
			code.Outdent()
		}
		code.Outdent()
	}
	code.Outdent()
	code.Print("Produces: %+v", document.Produces)
	code.Print("Responses: %+v", document.Responses)
	code.Print("Schemes: %+v", document.Schemes)
	code.Print("Security: %+v", document.Security)
	if document.SecurityDefinitions != nil {
		code.Print("SecurityDefinitions:")
		code.Indent()
		for _, pair := range document.SecurityDefinitions.AdditionalProperties {
			code.Print("%s", pair.Name)
			code.Indent()
			v := pair.Value
			switch t := v.Oneof.(type) {
			default:
				code.Print("unexpected type %T", t) // %T prints whatever type t has
			case *pb.SecurityDefinitionsItem_ApiKeySecurity:
				code.Print("ApiKeySecurity: %+v", t)
			case *pb.SecurityDefinitionsItem_BasicAuthenticationSecurity:
				code.Print("BasicAuthenticationSecurity: %+v", t)
			case *pb.SecurityDefinitionsItem_Oauth2AccessCodeSecurity:
				code.Print("Oauth2AccessCodeSecurity: %+v", t)
			case *pb.SecurityDefinitionsItem_Oauth2ApplicationSecurity:
				code.Print("Oauth2ApplicationSecurity: %+v", t)
			case *pb.SecurityDefinitionsItem_Oauth2ImplicitSecurity:
				code.Print("Oauth2ImplicitSecurity")
				code.Indent()
				code.Print("AuthorizationUrl: %+v", t.Oauth2ImplicitSecurity.AuthorizationUrl)
				code.Print("Flow: %+v", t.Oauth2ImplicitSecurity.Flow)
				code.Print("Scopes:")
				code.Indent()
				for _, pair := range t.Oauth2ImplicitSecurity.Scopes.AdditionalProperties {
					code.Print("%s -> %s", pair.Name, pair.Value)
				}
				code.Outdent()
				code.Outdent()
			case *pb.SecurityDefinitionsItem_Oauth2PasswordSecurity:
				code.Print("Oauth2PasswordSecurity: %+v", t)
			}
			code.Outdent()
		}
		code.Outdent()
	}
	code.Print("Swagger: %+v", document.Swagger)
	code.Print("Tags:")
	code.Indent()
	for _, tag := range document.Tags {
		code.Print("Tag:")
		code.Indent()
		code.Print("Name: %s", tag.Name)
		code.Print("Description: %s", tag.Description)
		code.Print("ExternalDocs: %s", tag.ExternalDocs)
		printVendorExtension(code, tag.VendorExtension)
		code.Outdent()
	}
	code.Outdent()
}

func printOperation(code *printer.Code, operation *pb.Operation) {
	code.Print("Consumes: %+v", operation.Consumes)
	code.Print("Deprecated: %+v", operation.Deprecated)
	code.Print("Description: %+v", operation.Description)
	code.Print("ExternalDocs: %+v", operation.ExternalDocs)
	code.Print("OperationId: %+v", operation.OperationId)
	code.Print("Parameters:")
	code.Indent()
	for _, item := range operation.Parameters {
		switch t := item.Oneof.(type) {
		default:
			code.Print("unexpected type %T", t) // %T prints whatever type t has
		case *pb.ParametersItem_JsonReference:
			code.Print("JsonReference: %+v", t)
		case *pb.ParametersItem_Parameter:
			code.Print("Parameter: %+v", t)
		}
	}
	code.Outdent()
	code.Print("Produces: %+v", operation.Produces)
	code.Print("Responses:")
	code.Indent()
	code.Print("ResponseCode:")
	code.Indent()
	for _, pair := range operation.Responses.ResponseCode {
		code.Print("%s %s", pair.Name, pair.Value)
	}
	code.Outdent()
	printVendorExtension(code, operation.Responses.VendorExtension)
	code.Outdent()
	code.Print("Schemes: %+v", operation.Schemes)
	code.Print("Security: %+v", operation.Security)
	code.Print("Summary: %+v", operation.Summary)
	code.Print("Tags: %+v", operation.Tags)
	printVendorExtension(code, operation.VendorExtension)
}

func printSchema(code *printer.Code, schema *pb.Schema) {
	//code.Print("%+v", schema)
	if schema.Format != "" {
		code.Print("Format: %+v", schema.Format)
	}
	if schema.Properties != nil {
		code.Print("Properties")
		code.Indent()
		for _, pair := range schema.Properties.AdditionalProperties {
			code.Print("%s", pair.Name)
			code.Indent()
			printSchema(code, pair.Value)
			code.Outdent()
		}
		code.Outdent()
	}
	if schema.Type != nil {
		code.Print("Type: %+v", schema.Type)
	}
	if schema.Xml != nil {
		code.Print("Xml: %+v", schema.Xml)
	}
	printVendorExtension(code, schema.VendorExtension)
}

func printVendorExtension(code *printer.Code, vendorExtension []*pb.NamedAny) {
	if len(vendorExtension) > 0 {
		code.Print("VendorExtension: %+v", vendorExtension)
	}
}

func main() {
	flag.Parse()
	args := flag.Args()

	if len(args) != 1 {
		fmt.Printf("Usage: report <file.pb>\n")
		return
	}

	document := readDocumentFromFileWithName(args[0])

	code := &printer.Code{}
	code.Print("API REPORT")
	code.Print("----------")
	printDocument(code, document)
	fmt.Printf("%s", code)
}
