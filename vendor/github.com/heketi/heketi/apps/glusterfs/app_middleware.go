//
// Copyright (c) 2017 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"net/http"
	"strings"

	jwt "github.com/dgrijalva/jwt-go"
	"github.com/gorilla/context"
	"github.com/urfave/negroni"

	"github.com/heketi/heketi/pkg/kubernetes"
)

var (
	kubeBackupDbToSecret = kubernetes.KubeBackupDbToSecret
)

// Authorization function
func (a *App) Auth(w http.ResponseWriter, r *http.Request, next http.HandlerFunc) {

	// Value saved by the JWT middleware.
	data := context.Get(r, "jwt")

	// Need to change from interface{} to the jwt.Token type
	token := data.(*jwt.Token)
	claims := token.Claims.(jwt.MapClaims)

	// Check access
	if "user" == claims["iss"] && r.URL.Path != "/volumes" {
		http.Error(w, "Administrator access required", http.StatusUnauthorized)
		return
	}

	// Everything is clean
	next(w, r)
}

// Backup database to a secret
func (a *App) BackupToKubernetesSecret(
	w http.ResponseWriter,
	r *http.Request,
	next http.HandlerFunc) {

	// Call the next middleware first
	// Wrap it in a negroni ResponseWriter because for some reason
	// the Golang http ResponseWriter does not provide access to
	// the HttpStatus.
	responsew := negroni.NewResponseWriter(w)
	next(responsew, r)

	// Backup for everything except GET methods which do not
	// provide information on asynchronous completion request
	if !a.isAsyncDone(responsew, r) && r.Method == http.MethodGet {
		return
	}

	// Backup database
	err := kubeBackupDbToSecret(a.db)
	if err != nil {
		logger.Err(err)
	} else {
		logger.Info("Backup successful")
	}
}

func (a *App) isAsyncDone(
	w negroni.ResponseWriter,
	r *http.Request) bool {

	return r.Method == http.MethodGet &&
		strings.HasPrefix(r.URL.Path, ASYNC_ROUTE) &&
		(w.Status() == http.StatusNoContent ||
			w.Status() == http.StatusSeeOther)
}
