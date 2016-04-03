package main

import (
	"github.com/emicklei/go-restful"
	"io"
	"net/http"
)

// This example shows how to create a (Route) Filter that performs Basic Authentication on the Http request.
//
// GET http://localhost:8080/secret
// and use admin,admin for the credentials

func main() {
	ws := new(restful.WebService)
	ws.Route(ws.GET("/secret").Filter(basicAuthenticate).To(secret))
	restful.Add(ws)
	http.ListenAndServe(":8080", nil)
}

func basicAuthenticate(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	encoded := req.Request.Header.Get("Authorization")
	// usr/pwd = admin/admin
	// real code does some decoding
	if len(encoded) == 0 || "Basic YWRtaW46YWRtaW4=" != encoded {
		resp.AddHeader("WWW-Authenticate", "Basic realm=Protected Area")
		resp.WriteErrorString(401, "401: Not Authorized")
		return
	}
	chain.ProcessFilter(req, resp)
}

func secret(req *restful.Request, resp *restful.Response) {
	io.WriteString(resp, "42")
}
