/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package basicauth

import (
	"errors"
	"github.com/golang/glog"
	"github.com/msteinert/pam"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
	"net/http"
	"os"
)

// Authenticator authenticates requests using basic auth
type Authenticator struct {
	auth authenticator.Password
}

// New returns a request authenticator that validates credentials using the provided password authenticator
func New(auth authenticator.Password) *Authenticator {
	return &Authenticator{auth}
}

type Credentials struct {
	User     string
	Password string
}

func (c Credentials) RespondPAM(s pam.Style, msg string) (string, error) {
	switch s {
	case pam.PromptEchoOn:
		return c.User, nil
	case pam.PromptEchoOff:
		return c.Password, nil
	}
	return "", errors.New("unexpected")
}

// AuthenticateRequest authenticates the request using the "Authorization: Basic" header in the request
func (a *Authenticator) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	username, password, found := req.BasicAuth()
	if !found {
		return nil, false, nil
	}

	if os.Getenv("BASIC_AUTH_PAM") == "1" {
		c := Credentials{
			Password: password,
		}

		module := os.Getenv("KUBE_PAM_MODULE_NAME")
		if module == "" {
			module = "kubepam"
		}

		tx, err := pam.Start(module, username, c)
		if err != nil {
			glog.Errorf("basicauth PAM start #error: %v", err)
			return nil, false, nil
		}

		err = tx.Authenticate(0)
		if err != nil {
			glog.Errorf("basicauth PAM #error: %v  failed auth for %s", err, username)
			return nil, false, nil
		}

		glog.Infof("basicauth PAM auth success for %s", username)
		return nil, true, nil
	}
	return a.auth.AuthenticatePassword(username, password)
}
