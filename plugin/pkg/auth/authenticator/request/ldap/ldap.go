/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/go-ldap/ldap"
	"k8s.io/kubernetes/pkg/auth/user"
)

// LDAP authenticator binds to LDAP server to validate user's credentials passed in the request.
// The ldap url, base DN, user DN, and TLS is passed from apiserver options
type LDAPAuthenticator struct {
	AuthURL, BaseDN, UserDN string
	tls                     bool
}

// authenticate through LDAP server
func ldapBind(username, password, userdn, basedn, url string, useTLS bool) error {
	l, err := ldap.Dial("tcp", url)
	if err != nil {
		return err
	}
	defer l.Close()

	// set TLS
	if useTLS {
		err = l.StartTLS(&tls.Config{InsecureSkipVerify: true})
		if err != nil {
			return err
		}
	}
	binddn := fmt.Sprintf("uid=%s,%s,%s", username, userdn, basedn)
	err = l.Bind(binddn, password)
	return err
}

func (ldapAuthenticator *LDAPAuthenticator) AuthenticatePassword(username, password string) (user.Info, bool, error) {
	err := ldapBind(username, password, ldapAuthenticator.UserDN, ldapAuthenticator.BaseDN, ldapAuthenticator.AuthURL, ldapAuthenticator.tls)
	if err != nil {
		return nil, false, fmt.Errorf("Failed to authenticate with LDAP:%v", err)
	}

	return &user.DefaultInfo{Name: username}, true, nil
}

// New returns a request authenticator that validates credentials using ldap
func New(ldapConfigFile string) (*LDAPAuthenticator, error) {
	fp, err := os.Open(ldapConfigFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s: %v", ldapConfigFile, err)
	}
	defer fp.Close()

	decoder := json.NewDecoder(fp)
	var auth LDAPAuthenticator
	if err = decoder.Decode(&auth); err != nil {
		return nil, fmt.Errorf("LDAP: failed to decode %s: err: %v", ldapConfigFile, err)
	}
	if auth.AuthURL == "" {
		return nil, errors.New("LDAP URL is empty")
	}
	if auth.BaseDN == "" {
		return nil, errors.New("LDAP base DN is empty")
	}
	if auth.UserDN == "" {
		auth.UserDN = "ou=People"
	}
	auth.tls = false
	if strings.HasPrefix(auth.AuthURL, "ldaps") {
		auth.tls = true
	}
	return &auth, nil
}
