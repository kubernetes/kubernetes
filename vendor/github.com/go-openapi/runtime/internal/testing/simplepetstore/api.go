// Copyright 2015 go-swagger maintainers
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

package simplepetstore

import (
	"encoding/json"
	"net/http"
	"sync"
	"sync/atomic"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/loads"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/runtime/middleware"
	"github.com/go-openapi/runtime/middleware/untyped"
)

// NewPetstore creates a new petstore api handler
func NewPetstore() (http.Handler, error) {
	spec, err := loads.Analyzed(json.RawMessage([]byte(swaggerJSON)), "")
	if err != nil {
		return nil, err
	}
	api := untyped.NewAPI(spec)

	api.RegisterOperation("get", "/pets", getAllPets)
	api.RegisterOperation("post", "/pets", createPet)
	api.RegisterOperation("delete", "/pets/{id}", deletePet)
	api.RegisterOperation("get", "/pets/{id}", getPetByID)

	return middleware.Serve(spec, api), nil
}

var getAllPets = runtime.OperationHandlerFunc(func(data interface{}) (interface{}, error) {
	return pets, nil
})

var createPet = runtime.OperationHandlerFunc(func(data interface{}) (interface{}, error) {
	body := data.(map[string]interface{})["pet"].(map[string]interface{})
	return addPet(Pet{
		Name:   body["name"].(string),
		Status: body["status"].(string),
	}), nil
})

var deletePet = runtime.OperationHandlerFunc(func(data interface{}) (interface{}, error) {
	id := data.(map[string]interface{})["id"].(int64)
	removePet(id)
	return nil, nil
})

var getPetByID = runtime.OperationHandlerFunc(func(data interface{}) (interface{}, error) {
	id := data.(map[string]interface{})["id"].(int64)
	return petByID(id)
})

// Tag the tag model
type Tag struct {
	ID   int64
	Name string
}

// Pet the pet model
type Pet struct {
	ID        int64    `json:"id"`
	Name      string   `json:"name"`
	PhotoURLs []string `json:"photoUrls,omitempty"`
	Status    string   `json:"status,omitempty"`
	Tags      []Tag    `json:"tags,omitempty"`
}

var pets = []Pet{
	{1, "Dog", []string{}, "available", nil},
	{2, "Cat", []string{}, "pending", nil},
}

var petsLock = &sync.Mutex{}
var lastPetID int64 = 2

func newPetID() int64 {
	return atomic.AddInt64(&lastPetID, 1)
}

func addPet(pet Pet) Pet {
	petsLock.Lock()
	defer petsLock.Unlock()
	pet.ID = newPetID()
	pets = append(pets, pet)
	return pet
}

func removePet(id int64) {
	petsLock.Lock()
	defer petsLock.Unlock()
	var newPets []Pet
	for _, pet := range pets {
		if pet.ID != id {
			newPets = append(newPets, pet)
		}
	}
	pets = newPets
}

func petByID(id int64) (*Pet, error) {
	for _, pet := range pets {
		if pet.ID == id {
			return &pet, nil
		}
	}
	return nil, errors.NotFound("not found: pet %d", id)
}

var swaggerJSON = `{
  "swagger": "2.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "description": "A sample API that uses a petstore as an example to demonstrate features in the swagger-2.0 specification",
    "termsOfService": "http://helloreverb.com/terms/",
    "contact": {
      "name": "Wordnik API Team"
    },
    "license": {
      "name": "MIT"
    }
  },
  "host": "localhost:8344",
  "basePath": "/api",
  "schemes": [
    "http"
  ],
  "consumes": [
    "application/json"
  ],
  "produces": [
    "application/json"
  ],
  "paths": {
    "/pets": {
      "get": {
        "description": "Returns all pets from the system that the user has access to",
        "operationId": "findPets",
        "produces": [
          "application/json",
          "application/xml",
          "text/xml",
          "text/html"
        ],
        "parameters": [
          {
            "name": "tags",
            "in": "query",
            "description": "tags to filter by",
            "required": false,
            "type": "array",
            "items": {
              "type": "string"
            },
            "collectionFormat": "csv"
          },
          {
            "name": "limit",
            "in": "query",
            "description": "maximum number of results to return",
            "required": false,
            "type": "integer",
            "format": "int32"
          }
        ],
        "responses": {
          "200": {
            "description": "pet response",
            "schema": {
              "type": "array",
              "items": {
                "$ref": "#/definitions/pet"
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "schema": {
              "$ref": "#/definitions/errorModel"
            }
          }
        }
      },
      "post": {
        "description": "Creates a new pet in the store.  Duplicates are allowed",
        "operationId": "addPet",
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "name": "pet",
            "in": "body",
            "description": "Pet to add to the store",
            "required": true,
            "schema": {
              "$ref": "#/definitions/petInput"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "pet response",
            "schema": {
              "$ref": "#/definitions/pet"
            }
          },
          "default": {
            "description": "unexpected error",
            "schema": {
              "$ref": "#/definitions/errorModel"
            }
          }
        }
      }
    },
    "/pets/{id}": {
      "get": {
        "description": "Returns a user based on a single ID, if the user does not have access to the pet",
        "operationId": "findPetById",
        "produces": [
          "application/json",
          "application/xml",
          "text/xml",
          "text/html"
        ],
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "description": "ID of pet to fetch",
            "required": true,
            "type": "integer",
            "format": "int64"
          }
        ],
        "responses": {
          "200": {
            "description": "pet response",
            "schema": {
              "$ref": "#/definitions/pet"
            }
          },
          "default": {
            "description": "unexpected error",
            "schema": {
              "$ref": "#/definitions/errorModel"
            }
          }
        }
      },
      "delete": {
        "description": "deletes a single pet based on the ID supplied",
        "operationId": "deletePet",
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "description": "ID of pet to delete",
            "required": true,
            "type": "integer",
            "format": "int64"
          }
        ],
        "responses": {
          "204": {
            "description": "pet deleted"
          },
          "default": {
            "description": "unexpected error",
            "schema": {
              "$ref": "#/definitions/errorModel"
            }
          }
        }
      }
    }
  },
  "definitions": {
    "pet": {
      "required": [
        "id",
        "name"
      ],
      "properties": {
        "id": {
          "type": "integer",
          "format": "int64"
        },
        "name": {
          "type": "string"
        },
        "tag": {
          "type": "string"
        }
      }
    },
    "petInput": {
      "allOf": [
        {
          "$ref": "pet"
        },
        {
          "required": [
            "name"
          ],
          "properties": {
            "id": {
              "type": "integer",
              "format": "int64"
            }
          }
        }
      ]
    },
    "errorModel": {
      "required": [
        "code",
        "message"
      ],
      "properties": {
        "code": {
          "type": "integer",
          "format": "int32"
        },
        "message": {
          "type": "string"
        }
      }
    }
  }
}`
