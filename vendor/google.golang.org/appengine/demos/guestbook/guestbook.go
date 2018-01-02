// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// This example only works on App Engine "flexible environment".
// +build !appengine

package main

import (
	"html/template"
	"net/http"
	"time"

	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/datastore"
	"google.golang.org/appengine/log"
	"google.golang.org/appengine/user"
)

var initTime time.Time

type Greeting struct {
	Author  string
	Content string
	Date    time.Time
}

func main() {
	http.HandleFunc("/", handleMainPage)
	http.HandleFunc("/sign", handleSign)
	appengine.Main()
}

// guestbookKey returns the key used for all guestbook entries.
func guestbookKey(ctx context.Context) *datastore.Key {
	// The string "default_guestbook" here could be varied to have multiple guestbooks.
	return datastore.NewKey(ctx, "Guestbook", "default_guestbook", 0, nil)
}

var tpl = template.Must(template.ParseGlob("templates/*.html"))

func handleMainPage(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "GET requests only", http.StatusMethodNotAllowed)
		return
	}
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	ctx := appengine.NewContext(r)
	tic := time.Now()
	q := datastore.NewQuery("Greeting").Ancestor(guestbookKey(ctx)).Order("-Date").Limit(10)
	var gg []*Greeting
	if _, err := q.GetAll(ctx, &gg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		log.Errorf(ctx, "GetAll: %v", err)
		return
	}
	log.Infof(ctx, "Datastore lookup took %s", time.Since(tic).String())
	log.Infof(ctx, "Rendering %d greetings", len(gg))

	var email, logout, login string
	if u := user.Current(ctx); u != nil {
		logout, _ = user.LogoutURL(ctx, "/")
		email = u.Email
	} else {
		login, _ = user.LoginURL(ctx, "/")
	}
	data := struct {
		Greetings            []*Greeting
		Login, Logout, Email string
	}{
		Greetings: gg,
		Login:     login,
		Logout:    logout,
		Email:     email,
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := tpl.ExecuteTemplate(w, "guestbook.html", data); err != nil {
		log.Errorf(ctx, "%v", err)
	}
}

func handleSign(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST requests only", http.StatusMethodNotAllowed)
		return
	}
	ctx := appengine.NewContext(r)
	g := &Greeting{
		Content: r.FormValue("content"),
		Date:    time.Now(),
	}
	if u := user.Current(ctx); u != nil {
		g.Author = u.String()
	}
	key := datastore.NewIncompleteKey(ctx, "Greeting", guestbookKey(ctx))
	if _, err := datastore.Put(ctx, key, g); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	// Redirect with 303 which causes the subsequent request to use GET.
	http.Redirect(w, r, "/", http.StatusSeeOther)
}
