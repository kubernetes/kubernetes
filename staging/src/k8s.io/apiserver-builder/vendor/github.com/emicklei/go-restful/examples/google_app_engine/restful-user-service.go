package main

import (
	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"google.golang.org/appengine"
	"google.golang.org/appengine/memcache"
	"net/http"
)

// This example is functionally the same as ../restful-user-service.go
// but it`s supposed to run on Goole App Engine (GAE)
//
// contributed by ivanhawkes

type User struct {
	Id, Name string
}

type UserService struct {
	// normally one would use DAO (data access object)
	// but in this example we simple use memcache.
}

func (u UserService) Register() {
	ws := new(restful.WebService)

	ws.
		Path("/users").
		Consumes(restful.MIME_XML, restful.MIME_JSON).
		Produces(restful.MIME_JSON, restful.MIME_XML) // you can specify this per route as well

	ws.Route(ws.GET("/{user-id}").To(u.findUser).
		// docs
		Doc("get a user").
		Param(ws.PathParameter("user-id", "identifier of the user").DataType("string")).
		Writes(User{})) // on the response

	ws.Route(ws.PATCH("").To(u.updateUser).
		// docs
		Doc("update a user").
		Reads(User{})) // from the request

	ws.Route(ws.PUT("/{user-id}").To(u.createUser).
		// docs
		Doc("create a user").
		Param(ws.PathParameter("user-id", "identifier of the user").DataType("string")).
		Reads(User{})) // from the request

	ws.Route(ws.DELETE("/{user-id}").To(u.removeUser).
		// docs
		Doc("delete a user").
		Param(ws.PathParameter("user-id", "identifier of the user").DataType("string")))

	restful.Add(ws)
}

// GET http://localhost:8080/users/1
//
func (u UserService) findUser(request *restful.Request, response *restful.Response) {
	c := appengine.NewContext(request.Request)
	id := request.PathParameter("user-id")
	usr := new(User)
	_, err := memcache.Gob.Get(c, id, &usr)
	if err != nil || len(usr.Id) == 0 {
		response.WriteErrorString(http.StatusNotFound, "User could not be found.")
	} else {
		response.WriteEntity(usr)
	}
}

// PATCH http://localhost:8080/users
// <User><Id>1</Id><Name>Melissa Raspberry</Name></User>
//
func (u *UserService) updateUser(request *restful.Request, response *restful.Response) {
	c := appengine.NewContext(request.Request)
	usr := new(User)
	err := request.ReadEntity(&usr)
	if err == nil {
		item := &memcache.Item{
			Key:    usr.Id,
			Object: &usr,
		}
		err = memcache.Gob.Set(c, item)
		if err != nil {
			response.WriteError(http.StatusInternalServerError, err)
			return
		}
		response.WriteEntity(usr)
	} else {
		response.WriteError(http.StatusInternalServerError, err)
	}
}

// PUT http://localhost:8080/users/1
// <User><Id>1</Id><Name>Melissa</Name></User>
//
func (u *UserService) createUser(request *restful.Request, response *restful.Response) {
	c := appengine.NewContext(request.Request)
	usr := User{Id: request.PathParameter("user-id")}
	err := request.ReadEntity(&usr)
	if err == nil {
		item := &memcache.Item{
			Key:    usr.Id,
			Object: &usr,
		}
		err = memcache.Gob.Add(c, item)
		if err != nil {
			response.WriteError(http.StatusInternalServerError, err)
			return
		}
		response.WriteHeader(http.StatusCreated)
		response.WriteEntity(usr)
	} else {
		response.WriteError(http.StatusInternalServerError, err)
	}
}

// DELETE http://localhost:8080/users/1
//
func (u *UserService) removeUser(request *restful.Request, response *restful.Response) {
	c := appengine.NewContext(request.Request)
	id := request.PathParameter("user-id")
	err := memcache.Delete(c, id)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
	}
}

func getGaeURL() string {
	if appengine.IsDevAppServer() {
		return "http://localhost:8080"
	} else {
		/**
		 * Include your URL on App Engine here.
		 * I found no way to get AppID without appengine.Context and this always
		 * based on a http.Request.
		 */
		return "http://<your_app_id>.appspot.com"
	}
}

func init() {
	u := UserService{}
	u.Register()

	// Optionally, you can install the Swagger Service which provides a nice Web UI on your REST API
	// You need to download the Swagger HTML5 assets and change the FilePath location in the config below.
	// Open <your_app_id>.appspot.com/apidocs and enter http://<your_app_id>.appspot.com/apidocs.json in the api input field.
	config := swagger.Config{
		WebServices:    restful.RegisteredWebServices(), // you control what services are visible
		WebServicesUrl: getGaeURL(),
		ApiPath:        "/apidocs.json",

		// Optionally, specifiy where the UI is located
		SwaggerPath: "/apidocs/",
		// GAE support static content which is configured in your app.yaml.
		// This example expect the swagger-ui in static/swagger so you should place it there :)
		SwaggerFilePath: "static/swagger"}
	swagger.InstallSwaggerService(config)
}
