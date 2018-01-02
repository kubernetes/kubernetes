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

package testing

import (
	"encoding/json"
)

// PetStore20YAML yaml doc for swagger 2.0 pet store
const PetStore20YAML = `swagger: '2.0'
info:
  version: '1.0.0'
  title: Swagger Petstore
  description: A sample API that uses a petstore as an example to demonstrate features in the swagger-2.0 specification
  termsOfService: http://helloreverb.com/terms/
  contact:
    name: Swagger API team
    email: foo@example.com
    url: http://swagger.io
  license:
    name: MIT
    url: http://opensource.org/licenses/MIT
host: petstore.swagger.wordnik.com
basePath: /api
schemes:
  - http
consumes:
  - application/json
produces:
  - application/json
paths:
  /pets:
    get:
      description: Returns all pets from the system that the user has access to
      operationId: findPets
      produces:
        - application/json
        - application/xml
        - text/xml
        - text/html
      parameters:
        - name: tags
          in: query
          description: tags to filter by
          required: false
          type: array
          items:
            type: string
          collectionFormat: csv
        - name: limit
          in: query
          description: maximum number of results to return
          required: false
          type: integer
          format: int32
      responses:
        '200':
          description: pet response
          schema:
            type: array
            items:
              $ref: '#/definitions/pet'
        default:
          description: unexpected error
          schema:
            $ref: '#/definitions/errorModel'
    post:
      description: Creates a new pet in the store.  Duplicates are allowed
      operationId: addPet
      produces:
        - application/json
      parameters:
        - name: pet
          in: body
          description: Pet to add to the store
          required: true
          schema:
            $ref: '#/definitions/newPet'
      responses:
        '200':
          description: pet response
          schema:
            $ref: '#/definitions/pet'
        default:
          description: unexpected error
          schema:
            $ref: '#/definitions/errorModel'
  /pets/{id}:
    get:
      description: Returns a user based on a single ID, if the user does not have access to the pet
      operationId: findPetById
      produces:
        - application/json
        - application/xml
        - text/xml
        - text/html
      parameters:
        - name: id
          in: path
          description: ID of pet to fetch
          required: true
          type: integer
          format: int64
      responses:
        '200':
          description: pet response
          schema:
            $ref: '#/definitions/pet'
        default:
          description: unexpected error
          schema:
            $ref: '#/definitions/errorModel'
    delete:
      description: deletes a single pet based on the ID supplied
      operationId: deletePet
      parameters:
        - name: id
          in: path
          description: ID of pet to delete
          required: true
          type: integer
          format: int64
      responses:
        '204':
          description: pet deleted
        default:
          description: unexpected error
          schema:
            $ref: '#/definitions/errorModel'
definitions:
  pet:
    required:
      - id
      - name
    properties:
      id:
        type: integer
        format: int64
      name:
        type: string
      tag:
        type: string
  newPet:
    allOf:
      - $ref: '#/definitions/pet'
      - required:
          - name
        properties:
          id:
            type: integer
            format: int64
          name:
            type: string
  errorModel:
    required:
      - code
      - message
    properties:
      code:
        type: integer
        format: int32
      message:
        type: string
`

// PetStore20 json doc for swagger 2.0 pet store
const PetStore20 = `{
  "swagger": "2.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "contact": {
      "name": "Wordnik API Team",
      "url": "http://developer.wordnik.com"
    },
    "license": {
      "name": "Creative Commons 4.0 International",
      "url": "http://creativecommons.org/licenses/by/4.0/"
    }
  },
  "host": "petstore.swagger.wordnik.com",
  "basePath": "/api",
  "schemes": [
    "http"
  ],
  "paths": {
    "/pets": {
      "get": {
        "security": [
          {
            "basic": []
          }
        ],
        "tags": [ "Pet Operations" ],
        "operationId": "getAllPets",
        "parameters": [
          {
            "name": "status",
            "in": "query",
            "description": "The status to filter by",
            "type": "string"
          },
          {
            "name": "limit",
            "in": "query",
            "description": "The maximum number of results to return",
            "type": "integer",
						"format": "int64"
          }
        ],
        "summary": "Finds all pets in the system",
        "responses": {
          "200": {
            "description": "Pet response",
            "schema": {
              "type": "array",
              "items": {
                "$ref": "#/definitions/Pet"
              }
            }
          },
          "default": {
            "description": "Unexpected error",
            "schema": {
              "$ref": "#/definitions/Error"
            }
          }
        }
      },
      "post": {
        "security": [
          {
            "basic": []
          }
        ],
        "tags": [ "Pet Operations" ],
        "operationId": "createPet",
        "summary": "Creates a new pet",
        "consumes": ["application/x-yaml"],
        "produces": ["application/x-yaml"],
        "parameters": [
          {
            "name": "pet",
            "in": "body",
            "description": "The Pet to create",
            "required": true,
            "schema": {
              "$ref": "#/definitions/newPet"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Created Pet response",
            "schema": {
              "$ref": "#/definitions/Pet"
            }
          },
          "default": {
            "description": "Unexpected error",
            "schema": {
              "$ref": "#/definitions/Error"
            }
          }
        }
      }
    },
    "/pets/{id}": {
      "delete": {
        "security": [
          {
            "apiKey": []
          }
        ],
        "description": "Deletes the Pet by id",
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
              "$ref": "#/definitions/Error"
            }
          }
        }
      },
      "get": {
        "tags": [ "Pet Operations" ],
        "operationId": "getPetById",
        "summary": "Finds the pet by id",
        "responses": {
          "200": {
            "description": "Pet response",
            "schema": {
              "$ref": "#/definitions/Pet"
            }
          },
          "default": {
            "description": "Unexpected error",
            "schema": {
              "$ref": "#/definitions/Error"
            }
          }
        }
      },
      "parameters": [
        {
          "name": "id",
          "in": "path",
          "description": "ID of pet",
          "required": true,
          "type": "integer",
          "format": "int64"
        }
      ]
    }
  },
  "definitions": {
    "Category": {
      "id": "Category",
      "properties": {
        "id": {
          "format": "int64",
          "type": "integer"
        },
        "name": {
          "type": "string"
        }
      }
    },
    "Pet": {
      "id": "Pet",
      "properties": {
        "category": {
          "$ref": "#/definitions/Category"
        },
        "id": {
          "description": "unique identifier for the pet",
          "format": "int64",
          "maximum": 100.0,
          "minimum": 0.0,
          "type": "integer"
        },
        "name": {
          "type": "string"
        },
        "photoUrls": {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        "status": {
          "description": "pet status in the store",
          "enum": [
            "available",
            "pending",
            "sold"
          ],
          "type": "string"
        },
        "tags": {
          "items": {
            "$ref": "#/definitions/Tag"
          },
          "type": "array"
        }
      },
      "required": [
        "id",
        "name"
      ]
    },
    "newPet": {
      "anyOf": [
        {
          "$ref": "#/definitions/Pet"
        },
        {
          "required": [
            "name"
          ]
        }
      ]
    },
    "Tag": {
      "id": "Tag",
      "properties": {
        "id": {
          "format": "int64",
          "type": "integer"
        },
        "name": {
          "type": "string"
        }
      }
    },
    "Error": {
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
  },
  "consumes": [
    "application/json",
    "application/xml"
  ],
  "produces": [
    "application/json",
    "application/xml",
    "text/plain",
    "text/html"
  ],
  "securityDefinitions": {
    "basic": {
      "type": "basic"
    },
    "apiKey": {
      "type": "apiKey",
      "in": "header",
      "name": "X-API-KEY"
    }
  }
}
`

// RootPetStore20 json doc for swagger 2.0 pet store at /
const RootPetStore20 = `{
  "swagger": "2.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "contact": {
      "name": "Wordnik API Team",
      "url": "http://developer.wordnik.com"
    },
    "license": {
      "name": "Creative Commons 4.0 International",
      "url": "http://creativecommons.org/licenses/by/4.0/"
    }
  },
  "host": "petstore.swagger.wordnik.com",
  "basePath": "/",
  "schemes": [
    "http"
  ],
  "paths": {
    "/pets": {
      "get": {
        "security": [
          {
            "basic": []
          }
        ],
        "tags": [ "Pet Operations" ],
        "operationId": "getAllPets",
        "parameters": [
          {
            "name": "status",
            "in": "query",
            "description": "The status to filter by",
            "type": "string"
          }
        ],
        "summary": "Finds all pets in the system",
        "responses": {
          "200": {
            "description": "Pet response",
            "schema": {
              "type": "array",
              "items": {
                "$ref": "#/definitions/Pet"
              }
            }
          },
          "default": {
            "description": "Unexpected error",
            "schema": {
              "$ref": "#/definitions/Error"
            }
          }
        }
      },
      "post": {
        "security": [
          {
            "basic": []
          }
        ],
        "tags": [ "Pet Operations" ],
        "operationId": "createPet",
        "summary": "Creates a new pet",
        "consumes": ["application/x-yaml"],
        "produces": ["application/x-yaml"],
        "parameters": [
          {
            "name": "pet",
            "in": "body",
            "description": "The Pet to create",
            "required": true,
            "schema": {
              "$ref": "#/definitions/newPet"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Created Pet response",
            "schema": {
              "$ref": "#/definitions/Pet"
            }
          },
          "default": {
            "description": "Unexpected error",
            "schema": {
              "$ref": "#/definitions/Error"
            }
          }
        }
      }
    },
    "/pets/{id}": {
      "delete": {
        "security": [
          {
            "apiKey": []
          }
        ],
        "description": "Deletes the Pet by id",
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
              "$ref": "#/definitions/Error"
            }
          }
        }
      },
      "get": {
        "tags": [ "Pet Operations" ],
        "operationId": "getPetById",
        "summary": "Finds the pet by id",
        "responses": {
          "200": {
            "description": "Pet response",
            "schema": {
              "$ref": "#/definitions/Pet"
            }
          },
          "default": {
            "description": "Unexpected error",
            "schema": {
              "$ref": "#/definitions/Error"
            }
          }
        }
      },
      "parameters": [
        {
          "name": "id",
          "in": "path",
          "description": "ID of pet",
          "required": true,
          "type": "integer",
          "format": "int64"
        }
      ]
    }
  },
  "definitions": {
    "Category": {
      "id": "Category",
      "properties": {
        "id": {
          "format": "int64",
          "type": "integer"
        },
        "name": {
          "type": "string"
        }
      }
    },
    "Pet": {
      "id": "Pet",
      "properties": {
        "category": {
          "$ref": "#/definitions/Category"
        },
        "id": {
          "description": "unique identifier for the pet",
          "format": "int64",
          "maximum": 100.0,
          "minimum": 0.0,
          "type": "integer"
        },
        "name": {
          "type": "string"
        },
        "photoUrls": {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        "status": {
          "description": "pet status in the store",
          "enum": [
            "available",
            "pending",
            "sold"
          ],
          "type": "string"
        },
        "tags": {
          "items": {
            "$ref": "#/definitions/Tag"
          },
          "type": "array"
        }
      },
      "required": [
        "id",
        "name"
      ]
    },
    "newPet": {
      "anyOf": [
        {
          "$ref": "#/definitions/Pet"
        },
        {
          "required": [
            "name"
          ]
        }
      ]
    },
    "Tag": {
      "id": "Tag",
      "properties": {
        "id": {
          "format": "int64",
          "type": "integer"
        },
        "name": {
          "type": "string"
        }
      }
    },
    "Error": {
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
  },
  "consumes": [
    "application/json",
    "application/xml"
  ],
  "produces": [
    "application/json",
    "application/xml",
    "text/plain",
    "text/html"
  ],
  "securityDefinitions": {
    "basic": {
      "type": "basic"
    },
    "apiKey": {
      "type": "apiKey",
      "in": "header",
      "name": "X-API-KEY"
    }
  }
}
`

// PetStoreJSONMessage json raw message for Petstore20
var PetStoreJSONMessage = json.RawMessage([]byte(PetStore20))

// RootPetStoreJSONMessage json raw message for Petstore20
var RootPetStoreJSONMessage = json.RawMessage([]byte(RootPetStore20))

// InvalidJSON invalid swagger 2.0 spec in JSON
const InvalidJSON = `{
  "apiVersion": "1.0.0",
  "apis": [
    {
      "description": "Operations about pets",
      "path": "/pet"
    },
    {
      "description": "Operations about user",
      "path": "/user"
    },
    {
      "description": "Operations about store",
      "path": "/store"
    }
  ],
  "authorizations": {
    "oauth2": {
      "grantTypes": {
        "authorization_code": {
          "tokenEndpoint": {
            "tokenName": "auth_code",
            "url": "http://petstore.swagger.wordnik.com/oauth/token"
          },
          "tokenRequestEndpoint": {
            "clientIdName": "client_id",
            "clientSecretName": "client_secret",
            "url": "http://petstore.swagger.wordnik.com/oauth/requestToken"
          }
        },
        "implicit": {
          "loginEndpoint": {
            "url": "http://petstore.swagger.wordnik.com/oauth/dialog"
          },
          "tokenName": "access_token"
        }
      },
      "scopes": [
        {
          "description": "Modify pets in your account",
          "scope": "write:pets"
        },
        {
          "description": "Read your pets",
          "scope": "read:pets"
        },
        {
          "description": "Anything (testing)",
          "scope": "test:anything"
        }
      ],
      "type": "oauth2"
    }
  },
  "info": {
    "contact": "apiteam@wordnik.com",
    "description": "This is a sample server Petstore server.  You can find out more about Swagger \n  at <a href=\"http://swagger.wordnik.com\">http://swagger.wordnik.com</a> or on irc.freenode.net, #swagger.  For this sample,\n  you can use the api key \"special-key\" to test the authorization filters",
    "license": "Apache 2.0",
    "licenseUrl": "http://www.apache.org/licenses/LICENSE-2.0.html",
    "termsOfServiceUrl": "http://helloreverb.com/terms/",
    "title": "Swagger Sample App"
  },
  "swaggerVersion": "1.2"
}
`

// InvalidJSONMessage json raw message for invalid json
var InvalidJSONMessage = json.RawMessage([]byte(InvalidJSON))
