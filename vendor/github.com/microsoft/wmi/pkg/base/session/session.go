//go:build windows
// +build windows

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package session

import (
	"github.com/pkg/errors"
	"os"
	"strings"

	"github.com/microsoft/wmi/pkg/base/credential"
	"github.com/microsoft/wmi/pkg/base/host"
	wmi "github.com/microsoft/wmi/pkg/wmiinstance"
)

var (
	sessionManager *wmi.WmiSessionManager
	sessionsMap    map[string]*wmi.WmiSession
	localHostName  string
)

func init() {
	localHostName, _ = os.Hostname()
	sessionsMap = make(map[string]*wmi.WmiSession)
	sessionManager = wmi.NewWmiSessionManager()
}

// StopWMI
func StopWMI() {
	for key := range sessionsMap {
		if sessionsMap[key] != nil {
			sessionsMap[key].Dispose()
		}
		sessionsMap[key] = nil
	}

	if sessionManager != nil {
		sessionManager.Dispose()
		sessionManager = nil
	}
}

// GetHostSession
func GetHostSession(namespaceName string, whost *host.WmiHost) (*wmi.WmiSession, error) {
	cred := whost.GetCredential()
	return GetSession(namespaceName, whost.HostName, cred.Domain, cred.UserName, cred.Password)
}

func GetHostSessionWithCredentials(namespaceName string, whost *host.WmiHost, cred *credential.WmiCredential) (*wmi.WmiSession, error) {
	return GetSession(namespaceName, whost.HostName, cred.Domain, cred.UserName, cred.Password)
}

// GetSession
func GetSession(namespaceName string, serverName string, domain string, userName string, password string) (*wmi.WmiSession, error) {
	sessionsMapId := strings.Join([]string{namespaceName, serverName, domain}, "_")
	if sessionsMap[sessionsMapId] == nil {
		var err error
		sessionsMap[sessionsMapId], err = createSession(namespaceName, serverName, domain, userName, password)
		if err != nil {
			return nil, err
		}
	}

	return sessionsMap[sessionsMapId], nil
}

// //////////// Private functions ////////////////////////////
func createSession(sessionName string, serverName string, domain string, username string, password string) (*wmi.WmiSession, error) {
	// TODO: ideally, we should also compare the domain here.
	// that said, this is low priority as cross-domain WMI calls are rare
	if strings.EqualFold(localHostName, serverName) {
		// Optimization for local clusters: connecting to the local cluster through remote WMI results in a much longer
		// response than connecting directly. When providing the cluster name, the cluster has to go through a
		// long sequence of connection/authentication. Not providing the name allows the cluster to skip that
		// expensive sequence.
		serverName = ""
		domain = ""
	}

	session, err := sessionManager.GetSession(sessionName, serverName, domain, username, password)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed getting the WMI session for "+sessionName)
	}

	connected, err := session.Connect()

	if !connected || err != nil {
		return nil, errors.Wrapf(err, "Failed connecting to the WMI session for "+sessionName)
	}

	return session, nil
}
