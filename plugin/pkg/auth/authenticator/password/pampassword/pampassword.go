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

package pampassword

import (
	"errors"
	"github.com/golang/glog"
	"github.com/msteinert/pam"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
	"os"
)

type pamPaswdAuthenticator struct{}

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

// NewAllow returns a password authenticator that allows any non-empty username
func NewPamPassword() authenticator.Password {
	return pamPaswdAuthenticator{}
}

// AuthenticatePassword implements authenticator.Password to allow any non-empty username,
// using the specified username as the name and UID
func (pamPaswdAuthenticator) AuthenticatePassword(username, password string) (user.Info, bool, error) {
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
	return &user.DefaultInfo{Name: username}, true, nil
}
