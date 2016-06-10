/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package ldap

import (
	"crypto/tls"
	"errors"
	"strings"

	"github.com/golang/glog"
	"gopkg.in/gcfg.v1"
	"gopkg.in/ldap.v2"

	"k8s.io/kubernetes/pkg/auth/user"
)

type config struct {
	Ldap struct {
		Host         string
		BindUser     string
		BindPassword string
		BaseDn       string
		Filter       string
		Secure       bool
	}
}

func (ldapconfig *config) AuthenticatePassword(username string, password string) (user.Info, bool, error) {
	ldapConn, err := ldap.Dial("tcp", ldapconfig.Ldap.Host)
	if err != nil {
		glog.Infof("Failed: Connecting ldap server :%s", err)
		return nil, false, errors.New("Failed to authenticate")
	}
	defer ldapConn.Close()

	if ldapconfig.Ldap.Secure {
		err = ldapConn.StartTLS(&tls.Config{InsecureSkipVerify: true})
		if err != nil {
			glog.Infof("Failed: To establish secure connection : %s", err)
			return nil, false, errors.New("Failed to authenticate")
		}
	}

	err = ldapConn.Bind(ldapconfig.Ldap.BindUser, ldapconfig.Ldap.BindPassword)
	if err != nil {
		glog.Infof("Failed to bind, check binduser/bindpassword in ldap config file : %s", err)
		return nil, false, errors.New("Failed to authenticate")
	}

	searchRequest := ldap.NewSearchRequest(
		ldapconfig.Ldap.BaseDn,
		ldap.ScopeWholeSubtree, ldap.NeverDerefAliases, 0, 0, false,
		strings.Replace(ldapconfig.Ldap.Filter, "{{username}}", username, -1),
		[]string{"dn"},
		nil,
	)

	sr, err := ldapConn.Search(searchRequest)
	if err != nil {
		glog.Infof("Failed to search user: %s and error: %s", username, err)
		return nil, false, errors.New("Failed to authenticate")
	}

	if len(sr.Entries) != 1 {
		glog.Info("User does not exist or too many entries returned")
		return nil, false, errors.New("Failed to authenticate")
	}

	userdn := sr.Entries[0].DN
	err = ldapConn.Bind(userdn, password)
	if err != nil {
		glog.Infof("Authentication failed, check the username/password. Error : %s", err)
		return nil, false, errors.New("Failed to authenticate")
	}

	return &user.DefaultInfo{Name: username}, true, nil
}

func NewLdapAuthenticator(ldapConfigFile string) (*config, error) {
	var cfg config
	err := gcfg.ReadFileInto(&cfg, ldapConfigFile)
	glog.Infof("config: %+v", cfg)
	glog.Info(err)
	if err != nil {
		return nil, err
	}

	return &cfg, nil
}
