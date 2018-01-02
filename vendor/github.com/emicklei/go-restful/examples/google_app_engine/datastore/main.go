package main

import (
	"net/http"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful-swagger12"
	"google.golang.org/appengine"
	"google.golang.org/appengine/datastore"
	"google.golang.org/appengine/user"
)

// This example demonstrates a reasonably complete suite of RESTful operations backed
// by DataStore on Google App Engine.

// Our simple example struct.
type Profile struct {
	LastModified time.Time `json:"-" xml:"-"`
	Email        string    `json:"-" xml:"-"`
	FirstName    string    `json:"first_name" xml:"first-name"`
	NickName     string    `json:"nick_name" xml:"nick-name"`
	LastName     string    `json:"last_name" xml:"last-name"`
}

type ProfileApi struct {
	Path string
}

func gaeUrl() string {
	if appengine.IsDevAppServer() {
		return "http://localhost:8080"
	} else {
		// Include your URL on App Engine here.
		// I found no way to get AppID without appengine.Context and this always
		// based on a http.Request.
		return "http://federatedservices.appspot.com"
	}
}

func init() {
	u := ProfileApi{Path: "/profiles"}
	u.register()

	// Optionally, you can install the Swagger Service which provides a nice Web UI on your REST API
	// You need to download the Swagger HTML5 assets and change the FilePath location in the config below.
	// Open <your_app_id>.appspot.com/apidocs and enter
	// Place the Swagger UI files into a folder called static/swagger if you wish to use Swagger
	// http://<your_app_id>.appspot.com/apidocs.json in the api input field.
	// For testing, you can use http://localhost:8080/apidocs.json
	config := swagger.Config{
		// You control what services are visible
		WebServices:    restful.RegisteredWebServices(),
		WebServicesUrl: gaeUrl(),
		ApiPath:        "/apidocs.json",

		// Optionally, specifiy where the UI is located
		SwaggerPath: "/apidocs/",

		// GAE support static content which is configured in your app.yaml.
		// This example expect the swagger-ui in static/swagger so you should place it there :)
		SwaggerFilePath: "static/swagger"}
	swagger.InstallSwaggerService(config)
}

func (u ProfileApi) register() {
	ws := new(restful.WebService)

	ws.
		Path(u.Path).
		// You can specify consumes and produces per route as well.
		Consumes(restful.MIME_JSON, restful.MIME_XML).
		Produces(restful.MIME_JSON, restful.MIME_XML)

	ws.Route(ws.POST("").To(u.insert).
		// Swagger documentation.
		Doc("insert a new profile").
		Param(ws.BodyParameter("Profile", "representation of a profile").DataType("main.Profile")).
		Reads(Profile{}))

	ws.Route(ws.GET("/{profile-id}").To(u.read).
		// Swagger documentation.
		Doc("read a profile").
		Param(ws.PathParameter("profile-id", "identifier for a profile").DataType("string")).
		Writes(Profile{}))

	ws.Route(ws.PUT("/{profile-id}").To(u.update).
		// Swagger documentation.
		Doc("update an existing profile").
		Param(ws.PathParameter("profile-id", "identifier for a profile").DataType("string")).
		Param(ws.BodyParameter("Profile", "representation of a profile").DataType("main.Profile")).
		Reads(Profile{}))

	ws.Route(ws.DELETE("/{profile-id}").To(u.remove).
		// Swagger documentation.
		Doc("remove a profile").
		Param(ws.PathParameter("profile-id", "identifier for a profile").DataType("string")))

	restful.Add(ws)
}

// POST http://localhost:8080/profiles
// {"first_name": "Ivan", "nick_name": "Socks", "last_name": "Hawkes"}
//
func (u *ProfileApi) insert(r *restful.Request, w *restful.Response) {
	c := appengine.NewContext(r.Request)

	// Marshall the entity from the request into a struct.
	p := new(Profile)
	err := r.ReadEntity(&p)
	if err != nil {
		w.WriteError(http.StatusNotAcceptable, err)
		return
	}

	// Ensure we start with a sensible value for this field.
	p.LastModified = time.Now()

	// The profile belongs to this user.
	p.Email = user.Current(c).String()

	k, err := datastore.Put(c, datastore.NewIncompleteKey(c, "profiles", nil), p)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Let them know the location of the newly created resource.
	// TODO: Use a safe Url path append function.
	w.AddHeader("Location", u.Path+"/"+k.Encode())

	// Return the resultant entity.
	w.WriteHeader(http.StatusCreated)
	w.WriteEntity(p)
}

// GET http://localhost:8080/profiles/ahdkZXZ-ZmVkZXJhdGlvbi1zZXJ2aWNlc3IVCxIIcHJvZmlsZXMYgICAgICAgAoM
//
func (u ProfileApi) read(r *restful.Request, w *restful.Response) {
	c := appengine.NewContext(r.Request)

	// Decode the request parameter to determine the key for the entity.
	k, err := datastore.DecodeKey(r.PathParameter("profile-id"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Retrieve the entity from the datastore.
	p := Profile{}
	if err := datastore.Get(c, k, &p); err != nil {
		if err.Error() == "datastore: no such entity" {
			http.Error(w, err.Error(), http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}

	// Check we own the profile before allowing them to view it.
	// Optionally, return a 404 instead to help prevent guessing ids.
	// TODO: Allow admins access.
	if p.Email != user.Current(c).String() {
		http.Error(w, "You do not have access to this resource", http.StatusForbidden)
		return
	}

	w.WriteEntity(p)
}

// PUT http://localhost:8080/profiles/ahdkZXZ-ZmVkZXJhdGlvbi1zZXJ2aWNlc3IVCxIIcHJvZmlsZXMYgICAgICAgAoM
// {"first_name": "Ivan", "nick_name": "Socks", "last_name": "Hawkes"}
//
func (u *ProfileApi) update(r *restful.Request, w *restful.Response) {
	c := appengine.NewContext(r.Request)

	// Decode the request parameter to determine the key for the entity.
	k, err := datastore.DecodeKey(r.PathParameter("profile-id"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Marshall the entity from the request into a struct.
	p := new(Profile)
	err = r.ReadEntity(&p)
	if err != nil {
		w.WriteError(http.StatusNotAcceptable, err)
		return
	}

	// Retrieve the old entity from the datastore.
	old := Profile{}
	if err := datastore.Get(c, k, &old); err != nil {
		if err.Error() == "datastore: no such entity" {
			http.Error(w, err.Error(), http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}

	// Check we own the profile before allowing them to update it.
	// Optionally, return a 404 instead to help prevent guessing ids.
	// TODO: Allow admins access.
	if old.Email != user.Current(c).String() {
		http.Error(w, "You do not have access to this resource", http.StatusForbidden)
		return
	}

	// Since the whole entity is re-written, we need to assign any invariant fields again
	// e.g. the owner of the entity.
	p.Email = user.Current(c).String()

	// Keep track of the last modification date.
	p.LastModified = time.Now()

	// Attempt to overwrite the old entity.
	_, err = datastore.Put(c, k, p)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Let them know it succeeded.
	w.WriteHeader(http.StatusNoContent)
}

// DELETE http://localhost:8080/profiles/ahdkZXZ-ZmVkZXJhdGlvbi1zZXJ2aWNlc3IVCxIIcHJvZmlsZXMYgICAgICAgAoM
//
func (u *ProfileApi) remove(r *restful.Request, w *restful.Response) {
	c := appengine.NewContext(r.Request)

	// Decode the request parameter to determine the key for the entity.
	k, err := datastore.DecodeKey(r.PathParameter("profile-id"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Retrieve the old entity from the datastore.
	old := Profile{}
	if err := datastore.Get(c, k, &old); err != nil {
		if err.Error() == "datastore: no such entity" {
			http.Error(w, err.Error(), http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}

	// Check we own the profile before allowing them to delete it.
	// Optionally, return a 404 instead to help prevent guessing ids.
	// TODO: Allow admins access.
	if old.Email != user.Current(c).String() {
		http.Error(w, "You do not have access to this resource", http.StatusForbidden)
		return
	}

	// Delete the entity.
	if err := datastore.Delete(c, k); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}

	// Success notification.
	w.WriteHeader(http.StatusNoContent)
}
