package main

import (
	"fmt"
	"github.com/emicklei/go-restful"
	"github.com/gorilla/schema"
	"io"
	"net/http"
)

// This example shows how to handle a POST of a HTML form that uses the standard x-www-form-urlencoded content-type.
// It uses the gorilla web tool kit schema package to decode the form data into a struct.
//
// GET http://localhost:8080/profiles
//

type Profile struct {
	Name string
	Age  int
}

var decoder *schema.Decoder

func main() {
	decoder = schema.NewDecoder()
	ws := new(restful.WebService)
	ws.Route(ws.POST("/profiles").Consumes("application/x-www-form-urlencoded").To(postAdddress))
	ws.Route(ws.GET("/profiles").To(addresssForm))
	restful.Add(ws)
	http.ListenAndServe(":8080", nil)
}

func postAdddress(req *restful.Request, resp *restful.Response) {
	err := req.Request.ParseForm()
	if err != nil {
		resp.WriteErrorString(http.StatusBadRequest, err.Error())
		return
	}
	p := new(Profile)
	err = decoder.Decode(p, req.Request.PostForm)
	if err != nil {
		resp.WriteErrorString(http.StatusBadRequest, err.Error())
		return
	}
	io.WriteString(resp.ResponseWriter, fmt.Sprintf("<html><body>Name=%s, Age=%d</body></html>", p.Name, p.Age))
}

func addresssForm(req *restful.Request, resp *restful.Response) {
	io.WriteString(resp.ResponseWriter,
		`<html>
		<body>
		<h1>Enter Profile</h1>
		<form method="post">
		    <label>Name:</label>
			<input type="text" name="Name"/>
			<label>Age:</label>
		    <input type="text" name="Age"/>
			<input type="Submit" />
		</form>
		</body>
		</html>`)
}
