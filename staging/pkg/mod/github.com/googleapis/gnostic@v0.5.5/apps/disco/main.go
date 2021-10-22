// Copyright 2019 Google LLC. All Rights Reserved.
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
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"github.com/docopt/docopt-go"
	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/conversions"
	discovery "github.com/googleapis/gnostic/discovery"
)

func main() {
	usage := `
Usage:
	disco help
	disco list [--raw]
	disco get [<api>] [<version>] [--raw] [--openapi2] [--openapi3] [--features] [--schemas] [--all]
	disco <file> [--openapi2] [--openapi3] [--features] [--schemas]
	`
	arguments, err := docopt.Parse(usage, nil, false, "Disco 1.0", false)
	if err != nil {
		log.Fatalf("%+v", err)
	}

	// Help.
	if arguments["help"].(bool) {
		fmt.Println("\nRead and process Google's Discovery Format for APIs.")
		fmt.Println(usage)
		fmt.Println("To learn more about Discovery Format, visit https://developers.google.com/discovery/.")
		fmt.Println()
	}

	// List APIs.
	if arguments["list"].(bool) {
		// Read the list of APIs from the apis/list service.
		bytes, err := discovery.FetchListBytes()
		if err != nil {
			log.Fatalf("%+v", err)
		}
		if arguments["--raw"].(bool) {
			ioutil.WriteFile("disco-list.json", bytes, 0644)
		} else {
			// Unpack the apis/list response.
			listResponse, err := discovery.ParseList(bytes)
			if err != nil {
				log.Fatalf("%+v", err)
			}
			// List the APIs.
			for _, api := range listResponse.APIs {
				fmt.Printf("%s %s\n", api.Name, api.Version)
			}
		}
	}

	// Get an API description.
	if arguments["get"].(bool) {
		// Read the list of APIs from the apis/list service.
		listResponse, err := discovery.FetchList()
		if err != nil {
			log.Fatalf("%+v", err)
		}
		if arguments["--all"].(bool) {
			if !arguments["--raw"].(bool) &&
				!arguments["--openapi2"].(bool) &&
				!arguments["--openapi3"].(bool) &&
				!arguments["--features"].(bool) &&
				!arguments["--schemas"].(bool) {
				log.Fatalf("Please specify an output option.")
			}
			for _, api := range listResponse.APIs {
				log.Printf("%s/%s", api.Name, api.Version)
				// Fetch the discovery description of the API.
				bytes, err := discovery.FetchDocumentBytes(api.DiscoveryRestURL)
				if err != nil {
					log.Printf("%+v", err)
					continue
				}
				// Export any requested formats.
				_, err = handleExportArgumentsForBytes(arguments, bytes)
				if err != nil {
					log.Printf("%+v", err)
					continue
				}
			}
		} else {
			// Find the matching API
			var apiName string
			if arguments["<api>"] != nil {
				apiName = arguments["<api>"].(string)
			}
			var apiVersion string
			if arguments["<version>"] != nil {
				apiVersion = arguments["<version>"].(string)
			}
			// Get the description of an API.
			api, err := listResponse.APIWithNameAndVersion(apiName, apiVersion)
			if err != nil {
				log.Fatalf("%+v", err)
			}
			// Fetch the discovery description of the API.
			bytes, err := discovery.FetchDocumentBytes(api.DiscoveryRestURL)
			if err != nil {
				log.Fatalf("%+v", err)
			}
			// Export any requested formats.
			handled, err := handleExportArgumentsForBytes(arguments, bytes)
			if err != nil {
				log.Fatalf("%+v", err)
			} else if !handled {
				// If no action was requested, write the document to stdout.
				os.Stdout.Write(bytes)
			}
		}
	}

	// Do something with a local API description.
	if arguments["<file>"] != nil {
		// Read the local file.
		filename := arguments["<file>"].(string)
		bytes, err := ioutil.ReadFile(filename)
		if err != nil {
			log.Fatalf("%+v", err)
		}
		// Export any requested formats.
		_, err = handleExportArgumentsForBytes(arguments, bytes)
		if err != nil {
			log.Fatalf("%+v", err)
		}
	}
}

func handleExportArgumentsForBytes(arguments map[string]interface{}, bytes []byte) (handled bool, err error) {
	// Unpack the discovery document.
	document, err := discovery.ParseDocument(bytes)
	if err != nil {
		return true, err
	}
	if arguments["--raw"].(bool) {
		// Write the Discovery document as a JSON file.
		filename := "disco-" + document.Name + "-" + document.Version + ".json"
		ioutil.WriteFile(filename, bytes, 0644)
		handled = true
	}
	if arguments["--features"].(bool) {
		if len(document.Features) > 0 {
			log.Printf("%s/%s features: %s\n",
				document.Name,
				document.Version,
				strings.Join(document.Features, ","))
		}
	}
	if arguments["--schemas"].(bool) {
		for _, schema := range document.Schemas.AdditionalProperties {
			checkSchema(schema.Name, schema.Value, 0)
		}
	}
	if arguments["--openapi3"].(bool) {
		// Generate the OpenAPI 3 equivalent.
		openAPIDocument, err := conversions.OpenAPIv3(document)
		if err != nil {
			return handled, err
		}
		bytes, err = proto.Marshal(openAPIDocument)
		if err != nil {
			return handled, err
		}
		filename := "openapi3-" + document.Name + "-" + document.Version + ".pb"
		err = ioutil.WriteFile(filename, bytes, 0644)
		if err != nil {
			return handled, err
		}
		handled = true
	}
	if arguments["--openapi2"].(bool) {
		// Generate the OpenAPI 2 equivalent.
		openAPIDocument, err := conversions.OpenAPIv2(document)
		if err != nil {
			return handled, err
		}
		bytes, err = proto.Marshal(openAPIDocument)
		if err != nil {
			return handled, err
		}
		filename := "openapi2-" + document.Name + "-" + document.Version + ".pb"
		err = ioutil.WriteFile(filename, bytes, 0644)
		if err != nil {
			return handled, err
		}
		handled = true
	}

	return handled, err
}

func checkSchema(schemaName string, schema *discovery.Schema, depth int) {
	switch schema.Type {
	case "string":
	case "number":
	case "integer":
	case "boolean":
	case "object": // only objects should have properties...
	case "array":
	case "null":
		log.Printf("NULL TYPE %s %s", schemaName, schema.Type)
	case "any":
		//log.Printf("ANY TYPE %s/%s %s", schemaName, property.Name, propertySchema.Type)
	default:
		//log.Printf("UNKNOWN TYPE %s/%s %s", schemaName, property.Name, propertySchema.Type)
	}
	if (schema.Properties != nil) && (len(schema.Properties.AdditionalProperties) > 0) {
		if depth > 0 {
			log.Printf("ANONYMOUS SCHEMA %s", schemaName)
		}
		for _, property := range schema.Properties.AdditionalProperties {
			propertySchema := property.Value
			ref := propertySchema.XRef
			if ref != "" {
				//log.Printf("REF: %s", ref)
				// assert (propertySchema.Type == "")
			} else {
				checkSchema(schemaName+"/"+property.Name, propertySchema, depth+1)
			}
		}
	}
	if schema.AdditionalProperties != nil {
		log.Printf("ADDITIONAL PROPERTIES %s", schemaName)
		checkSchema(schemaName+"/*", schema.AdditionalProperties, depth+1)
	}
}
