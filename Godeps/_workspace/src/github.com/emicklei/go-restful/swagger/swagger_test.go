package swagger

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/emicklei/go-restful"
)

// go test -v -test.run TestApi ...swagger
func TestApi(t *testing.T) {
	value := Api{Path: "/", Description: "Some Path", Operations: []Operation{}}
	compareJson(t, true, value, `{"path":"/","description":"Some Path"}`)
}

// go test -v -test.run TestServiceToApi ...swagger
func TestServiceToApi(t *testing.T) {
	ws := new(restful.WebService)
	ws.Path("/tests")
	ws.Consumes(restful.MIME_JSON)
	ws.Produces(restful.MIME_XML)
	ws.Route(ws.GET("/all").To(dummy).Writes(sample{}))
	ws.ApiVersion("1.2.3")
	cfg := Config{
		WebServicesUrl: "http://here.com",
		ApiPath:        "/apipath",
		WebServices:    []*restful.WebService{ws}}
	sws := newSwaggerService(cfg)
	decl := sws.composeDeclaration(ws, "/tests")
	data, err := json.MarshalIndent(decl, " ", " ")
	if err != nil {
		t.Fatal(err.Error())
	}
	// for visual inspection only
	fmt.Println(string(data))
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
	decl.Models = map[string]Model{}
	msgs := composeResponseMessages(route, decl)
	if msgs[0].ResponseModel != "swagger.TestItem" {
		t.Errorf("got %s want swagger.TestItem", msgs[0].ResponseModel)
	}
}

// clear && go test -v -test.run TestComposeResponseMessageArray ...swagger
func TestComposeResponseMessageArray(t *testing.T) {
	responseErrors := map[int]restful.ResponseError{}
	responseErrors[400] = restful.ResponseError{Code: 400, Message: "Bad Request", Model: []TestItem{}}
	route := restful.Route{ResponseErrors: responseErrors}
	decl := new(ApiDeclaration)
	decl.Models = map[string]Model{}
	msgs := composeResponseMessages(route, decl)
	if msgs[0].ResponseModel != "array[swagger.TestItem]" {
		t.Errorf("got %s want swagger.TestItem", msgs[0].ResponseModel)
	}
}

func TestIssue78(t *testing.T) {
	sws := newSwaggerService(Config{})
	models := map[string]Model{}
	sws.addModelFromSampleTo(&Operation{}, true, Response{Items: &[]TestItem{}}, models)
	model, ok := models["swagger.Response"]
	if !ok {
		t.Fatal("missing response model")
	}
	if "swagger.Response" != model.Id {
		t.Fatal("wrong model id:" + model.Id)
	}
	code, ok := model.Properties["Code"]
	if !ok {
		t.Fatal("missing code")
	}
	if "integer" != *code.Type {
		t.Fatal("wrong code type:" + *code.Type)
	}
	items, ok := model.Properties["Items"]
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
