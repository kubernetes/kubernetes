/*
Copyright 2015 Google Inc. All rights reserved.

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

package saslauthd

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"net"
	"net/http"
	"os"
	"strings"
)

type saslAuthdAuthenticator struct {
	serviceName  string
	socketPath   string
	defaultRealm string
}

// NewSaslAuthd returns a password authenticator that checks passwords using
// saslauthd by connecting directly to its socket.  The request format matches
// the one expected by saslauthd versions 2.0.5 and later (through at least
// 2.1.26): login ID, password, service name, and realm, each preceded by a
// 16-bit network-order length, with no padding.
func NewSaslAuthd(defaultRealm, socketPath, serviceName string) (authenticator.Password, error) {
	if serviceName == "" {
		serviceName = os.Args[0]
	}
	index := strings.LastIndex(serviceName, "/")
	if index >= 0 {
		serviceName = serviceName[index+1:]
	}
	if serviceName == "" {
		return nil, errors.New("error determining SASL service name")
	}
	return saslAuthdAuthenticator{serviceName: serviceName, socketPath: socketPath, defaultRealm: defaultRealm}, nil
}

// GetRealm returns the realm name for use in constructing a challenge for
// Basic authentication.
func (s saslAuthdAuthenticator) GetRealm() string {
	return s.defaultRealm
}

// AuthenticatePassword implements authenticator.Password by handing it off to
// saslauthd, using the specified username as the login ID.  If successful, a
// user.DefaultInfo is returned.  That structure will contain the username, an
// empty UID, and no group information.
func (s saslAuthdAuthenticator) AuthenticatePassword(username, password string) (user.Info, http.Header, bool, error) {
	var replen uint16
	challenge := http.Header{"WWW-Authenticate": {"Basic " + "realm=\"" + s.defaultRealm + "\""}}
	realm := s.defaultRealm
	index := strings.LastIndex(username, "@")
	if index >= 0 {
		realm = username[index+1:]
		username = username[0:index]
	}
	service := s.serviceName
	if len(username) >= 256 {
		return nil, challenge, false, errors.New("User name would be too long for saslauthd")
	}
	if len(password) >= 256 {
		return nil, challenge, false, errors.New("Password would be too long for saslauthd")
	}
	if len(realm) >= 256 {
		return nil, challenge, false, errors.New("Realm name would be too long for saslauthd")
	}
	if len(service) >= 256 {
		return nil, challenge, false, errors.New("Service name would be too long for saslauthd")
	}
	path := s.socketPath
	if path == "" {
		path = "/var/run/saslauthd/mux"
	} else {
		fi, err := os.Stat(path)
		if err != nil {
			return nil, challenge, false, err
		}
		if fi.IsDir() {
			path = path + "/mux"
		}
	}
	conn, err := net.Dial("unix", path)
	if err != nil {
		return nil, challenge, false, err
	}
	defer conn.Close()
	req := new(bytes.Buffer)
	if req == nil {
		return nil, challenge, false, nil
	}
	binary.Write(req, binary.BigEndian, uint16(len(username)))
	req.WriteString(username)
	binary.Write(req, binary.BigEndian, uint16(len(password)))
	req.WriteString(password)
	binary.Write(req, binary.BigEndian, uint16(len(service)))
	req.WriteString(service)
	binary.Write(req, binary.BigEndian, uint16(len(realm)))
	req.WriteString(realm)
	rb := req.Bytes()
	n, err := conn.Write(rb)
	if n < len(rb) {
		return nil, challenge, false, errors.New("Error sending request to saslauthd")
	}
	binary.Read(conn, binary.BigEndian, &replen)
	if replen < 2 {
		return nil, challenge, false, errors.New("Response from saslauthd would be too short")
	}
	if replen > 1024 {
		return nil, challenge, false, errors.New("Response from saslauthd would be too long")
	}
	rep := make([]byte, replen)
	n, err = conn.Read(rep)
	b := make([]byte, 1)
	for n > 0 {
		n, err = conn.Read(b)
		if err != nil {
			break
		}
	}
	if rep[0] == 'O' && rep[1] == 'K' {
		return &user.DefaultInfo{Name: username + "@" + realm}, nil, true, nil
	}
	if rep[0] == 'N' && rep[1] == 'O' {
		return nil, challenge, false, errors.New(fmt.Sprintf("Authentication rejected by saslauthd: %s", bytes.NewBuffer(rep).String()))
	}
	return nil, challenge, false, errors.New("Unknown error authenticating with saslauthd")
}
