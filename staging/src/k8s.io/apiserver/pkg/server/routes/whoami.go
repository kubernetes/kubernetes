/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package routes

import (
	"net/http"

	"github.com/emicklei/go-restful"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// UserIdentity provides a webservice with user identity information.
type UserIdentity struct{}

// Install registers the APIServer's `/whoami` handler.
func (u UserIdentity) Install(c *restful.Container) {
	versionWS := new(restful.WebService)
	versionWS.Path("/whoami")
	versionWS.Doc("get the information about requesting client")
	versionWS.Route(
		versionWS.GET("/").To(u.handleWhoAmI).
			Doc("get the user identity information").
			Operation("getUserIdentity").
			Produces(restful.MIME_JSON).
			Consumes(restful.MIME_JSON).
			Writes(user.DefaultInfo{}))
	c.Add(versionWS)
}

// handleWhoAmI writes the client's identity information.
func (u UserIdentity) handleWhoAmI(req *restful.Request, resp *restful.Response) {
	userInfo, ok := request.UserFrom(req.Request.Context())
	if !ok {
		userInfo = &user.DefaultInfo{}
	}
	responsewriters.WriteRawJSON(http.StatusOK, userInfo.(*user.DefaultInfo), resp.ResponseWriter)
}
