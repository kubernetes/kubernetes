// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package helloworld

import (
	"html/template"
	"net/http"
	"time"

	"google.golang.org/appengine"
)

var initTime = time.Now()

func init() {
	http.HandleFunc("/", handle)
}

func handle(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	c := appengine.NewContext(r)
	c.Infof("Serving the front page.")

	tmpl.Execute(w, time.Since(initTime))
}

var tmpl = template.Must(template.New("front").Parse(`
<html><body>

<p>
Hello, World! 세상아 안녕!
</p>

<p>
This instance has been running for <em>{{.}}</em>.
</p>

</body></html>
`))
