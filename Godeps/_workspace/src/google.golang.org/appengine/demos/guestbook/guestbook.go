// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package guestbook

import (
	"html/template"
	"net/http"
	"time"

	"google.golang.org/appengine"
	"google.golang.org/appengine/datastore"
	"google.golang.org/appengine/user"
)

var initTime time.Time

type Greeting struct {
	Author  string
	Content string
	Date    time.Time
}

func init() {
	http.HandleFunc("/", handleMainPage)
	http.HandleFunc("/sign", handleSign)
}

// guestbookKey returns the key used for all guestbook entries.
func guestbookKey(c appengine.Context) *datastore.Key {
	// The string "default_guestbook" here could be varied to have multiple guestbooks.
	return datastore.NewKey(c, "Guestbook", "default_guestbook", 0, nil)
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

	c := appengine.NewContext(r)
	tic := time.Now()
	q := datastore.NewQuery("Greeting").Ancestor(guestbookKey(c)).Order("-Date").Limit(10)
	var gg []*Greeting
	if _, err := q.GetAll(c, &gg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		c.Errorf("GetAll: %v", err)
		return
	}
	c.Infof("Datastore lookup took %s", time.Since(tic).String())
	c.Infof("Rendering %d greetings", len(gg))

	var email, logout, login string
	if u := user.Current(c); u != nil {
		logout, _ = user.LogoutURL(c, "/")
		email = u.Email
	} else {
		login, _ = user.LoginURL(c, "/")
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
		c.Errorf("%v", err)
	}
}

func handleSign(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST requests only", http.StatusMethodNotAllowed)
		return
	}
	c := appengine.NewContext(r)
	g := &Greeting{
		Content: r.FormValue("content"),
		Date:    time.Now(),
	}
	if u := user.Current(c); u != nil {
		g.Author = u.String()
	}
	key := datastore.NewIncompleteKey(c, "Greeting", guestbookKey(c))
	if _, err := datastore.Put(c, key, g); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	// Redirect with 303 which causes the subsequent request to use GET.
	http.Redirect(w, r, "/", http.StatusSeeOther)
}
