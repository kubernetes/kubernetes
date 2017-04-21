package swagger

import (
	"encoding/json"
	"testing"

	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful-swagger12/test_package"
)

func TestInfoStruct_Issue231(t *testing.T) {
	config := Config{
		Info: Info{
			Title:             "Title",
			Description:       "Description",
			TermsOfServiceUrl: "http://example.com",
			Contact:           "example@example.com",
			License:           "License",
			LicenseUrl:        "http://example.com/license.txt",
		},
	}
	sws := newSwaggerService(config)
	str, err := json.MarshalIndent(sws.produceListing(), "", "    ")
	if err != nil {
		t.Fatal(err)
	}
	compareJson(t, string(str), `
	{
		"apiVersion": "",
		"swaggerVersion": "1.2",
		"apis": null,
		"info": {
			"title": "Title",
			"description": "Description",
			"termsOfServiceUrl": "http://example.com",
			"contact": "example@example.com",
			"license": "License",
			"licenseUrl": "http://example.com/license.txt"
		}
	}
	`)
}

// go test -v -test.run TestThatMultiplePathsOnRootAreHandled ...swagger
func TestThatMultiplePathsOnRootAreHandled(t *testing.T) {
	ws1 := new(restful.WebService)
	ws1.Route(ws1.GET("/_ping").To(dummy))
	ws1.Route(ws1.GET("/version").To(dummy))

	cfg := Config{
		WebServicesUrl: "http://here.com",
		ApiPath:        "/apipath",
		WebServices:    []*restful.WebService{ws1},
	}
	sws := newSwaggerService(cfg)
	decl := sws.composeDeclaration(ws1, "/")
	if got, want := len(decl.Apis), 2; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestWriteSamples(t *testing.T) {
	ws1 := new(restful.WebService)
	ws1.Route(ws1.GET("/object").To(dummy).Writes(test_package.TestStruct{}))
	ws1.Route(ws1.GET("/array").To(dummy).Writes([]test_package.TestStruct{}))
	ws1.Route(ws1.GET("/object_and_array").To(dummy).Writes(struct{ Abc test_package.TestStruct }{}))

	cfg := Config{
		WebServicesUrl: "http://here.com",
		ApiPath:        "/apipath",
		WebServices:    []*restful.WebService{ws1},
	}
	sws := newSwaggerService(cfg)

	decl := sws.composeDeclaration(ws1, "/")

	str, err := json.MarshalIndent(decl.Apis, "", "    ")
	if err != nil {
		t.Fatal(err)
	}

	compareJson(t, string(str), `
	[
		{
			"path": "/object",
			"description": "",
			"operations": [
				{
					"type": "test_package.TestStruct",
					"method": "GET",
					"nickname": "dummy",
					"parameters": []
				}
			]
		},
		{
			"path": "/array",
			"description": "",
			"operations": [
				{
					"type": "array",
					"items": {
						"$ref": "test_package.TestStruct"
					},
					"method": "GET",
					"nickname": "dummy",
					"parameters": []
				}
			]
		},
		{
			"path": "/object_and_array",
			"description": "",
			"operations": [
				{
					"type": "struct { Abc test_package.TestStruct }",
					"method": "GET",
					"nickname": "dummy",
					"parameters": []
				}
			]
		}
    ]`)

	str, err = json.MarshalIndent(decl.Models, "", "    ")
	if err != nil {
		t.Fatal(err)
	}
	compareJson(t, string(str), `
	{
		"test_package.TestStruct": {
			"id": "test_package.TestStruct",
			"required": [
				"TestField"
			],
			"properties": {
				"TestField": {
					"type": "string"
				}
			}
		},
		"||test_package.TestStruct": {
			"id": "||test_package.TestStruct",
			"properties": {}
		},
		"struct { Abc test_package.TestStruct }": {
			"id": "struct { Abc test_package.TestStruct }",
			"required": [
				"Abc"
			],
			"properties": {
				"Abc": {
					"$ref": "test_package.TestStruct"
				}
			}
		}
    }`)
}

func TestRoutesWithCommonPart(t *testing.T) {
	ws1 := new(restful.WebService)
	ws1.Path("/")
	ws1.Route(ws1.GET("/foobar").To(dummy).Writes(test_package.TestStruct{}))
	ws1.Route(ws1.HEAD("/foobar").To(dummy).Writes(test_package.TestStruct{}))
	ws1.Route(ws1.GET("/foo").To(dummy).Writes([]test_package.TestStruct{}))
	ws1.Route(ws1.HEAD("/foo").To(dummy).Writes(test_package.TestStruct{}))

	cfg := Config{
		WebServicesUrl: "http://here.com",
		ApiPath:        "/apipath",
		WebServices:    []*restful.WebService{ws1},
	}
	sws := newSwaggerService(cfg)

	decl := sws.composeDeclaration(ws1, "/foo")

	str, err := json.MarshalIndent(decl.Apis, "", "    ")
	if err != nil {
		t.Fatal(err)
	}

	compareJson(t, string(str), `[
		{
			"path": "/foo",
			"description": "",
			"operations": [
				{
					"type": "array",
					"items": {
						"$ref": "test_package.TestStruct"
					},
					"method": "GET",
					"nickname": "dummy",
					"parameters": []
				},
				{
					"type": "test_package.TestStruct",
					"method": "HEAD",
					"nickname": "dummy",
					"parameters": []
				}
			]
		}
    ]`)
}

// go test -v -test.run TestServiceToApi ...swagger
func TestServiceToApi(t *testing.T) {
	ws := new(restful.WebService)
	ws.Path("/tests")
	ws.Consumes(restful.MIME_JSON)
	ws.Produces(restful.MIME_XML)
	ws.Route(ws.GET("/a").To(dummy).Writes(sample{}))
	ws.Route(ws.PUT("/b").To(dummy).Writes(sample{}))
	ws.Route(ws.POST("/c").To(dummy).Writes(sample{}))
	ws.Route(ws.DELETE("/d").To(dummy).Writes(sample{}))

	ws.Route(ws.GET("/d").To(dummy).Writes(sample{}))
	ws.Route(ws.PUT("/c").To(dummy).Writes(sample{}))
	ws.Route(ws.POST("/b").To(dummy).Writes(sample{}))
	ws.Route(ws.DELETE("/a").To(dummy).Writes(sample{}))
	ws.ApiVersion("1.2.3")
	cfg := Config{
		WebServicesUrl:   "http://here.com",
		ApiPath:          "/apipath",
		WebServices:      []*restful.WebService{ws},
		PostBuildHandler: func(in *ApiDeclarationList) {},
	}
	sws := newSwaggerService(cfg)
	decl := sws.composeDeclaration(ws, "/tests")
	// checks
	if decl.ApiVersion != "1.2.3" {
		t.Errorf("got %v want %v", decl.ApiVersion, "1.2.3")
	}
	if decl.BasePath != "http://here.com" {
		t.Errorf("got %v want %v", decl.BasePath, "http://here.com")
	}
	if len(decl.Apis) != 4 {
		t.Errorf("got %v want %v", len(decl.Apis), 4)
	}
	pathOrder := ""
	for _, each := range decl.Apis {
		pathOrder += each.Path
		for _, other := range each.Operations {
			pathOrder += other.Method
		}
	}

	if pathOrder != "/tests/aGETDELETE/tests/bPUTPOST/tests/cPOSTPUT/tests/dDELETEGET" {
		t.Errorf("got %v want %v", pathOrder, "see test source")
	}
}

func dummy(i *restful.Request, o *restful.Response) {}

// go test -v -test.run TestIssue78 ...swagger
type Response struct {
	Code  int
	Users *[]User
	Items *[]TestItem
}
type User struct {
	Id, Name string
}
type TestItem struct {
	Id, Name string
}

// clear && go test -v -test.run TestComposeResponseMessages ...swagger
func TestComposeResponseMessages(t *testing.T) {
	responseErrors := map[int]restful.ResponseError{}
	responseErrors[400] = restful.ResponseError{Code: 400, Message: "Bad Request", Model: TestItem{}}
	route := restful.Route{ResponseErrors: responseErrors}
	decl := new(ApiDeclaration)
	decl.Models = ModelList{}
	msgs := composeResponseMessages(route, decl, &Config{})
	if msgs[0].ResponseModel != "swagger.TestItem" {
		t.Errorf("got %s want swagger.TestItem", msgs[0].ResponseModel)
	}
}

func TestIssue78(t *testing.T) {
	sws := newSwaggerService(Config{})
	models := new(ModelList)
	sws.addModelFromSampleTo(&Operation{}, true, Response{Items: &[]TestItem{}}, models)
	model, ok := models.At("swagger.Response")
	if !ok {
		t.Fatal("missing response model")
	}
	if "swagger.Response" != model.Id {
		t.Fatal("wrong model id:" + model.Id)
	}
	code, ok := model.Properties.At("Code")
	if !ok {
		t.Fatal("missing code")
	}
	if "integer" != *code.Type {
		t.Fatal("wrong code type:" + *code.Type)
	}
	items, ok := model.Properties.At("Items")
	if !ok {
		t.Fatal("missing items")
	}
	if "array" != *items.Type {
		t.Fatal("wrong items type:" + *items.Type)
	}
	items_items := items.Items
	if items_items == nil {
		t.Fatal("missing items->items")
	}
	ref := items_items.Ref
	if ref == nil {
		t.Fatal("missing $ref")
	}
	if *ref != "swagger.TestItem" {
		t.Fatal("wrong $ref:" + *ref)
	}
}
