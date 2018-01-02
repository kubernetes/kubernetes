// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package discovery

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

type acMeta struct {
	name   string
	prefix string
	uri    string
}

type ACIEndpoint struct {
	ACI string
	ASC string
}

// A struct containing both discovered endpoints and keys. Used to avoid
// function duplication (one for endpoints and one for keys, so to avoid two
// doDiscover, two DiscoverWalkFunc)
type discoveryData struct {
	ACIEndpoints []ACIEndpoint
	PublicKeys   []string
}

type ACIEndpoints []ACIEndpoint

type PublicKeys []string

const (
	defaultVersion = "latest"
)

var (
	templateExpression = regexp.MustCompile(`{.*?}`)
	errEnough          = errors.New("enough discovery information found")
)

func appendMeta(meta []acMeta, attrs []html.Attribute) []acMeta {
	m := acMeta{}

	for _, a := range attrs {
		if a.Namespace != "" {
			continue
		}

		switch a.Key {
		case "name":
			m.name = a.Val

		case "content":
			parts := strings.SplitN(strings.TrimSpace(a.Val), " ", 2)
			if len(parts) < 2 {
				break
			}
			m.prefix = parts[0]
			m.uri = strings.TrimSpace(parts[1])
		}
	}

	// TODO(eyakubovich): should prefix be optional?
	if !strings.HasPrefix(m.name, "ac-") || m.prefix == "" || m.uri == "" {
		return meta
	}

	return append(meta, m)
}

func extractACMeta(r io.Reader) []acMeta {
	var meta []acMeta

	z := html.NewTokenizer(r)

	for {
		switch z.Next() {
		case html.ErrorToken:
			return meta

		case html.StartTagToken, html.SelfClosingTagToken:
			tok := z.Token()
			if tok.DataAtom == atom.Meta {
				meta = appendMeta(meta, tok.Attr)
			}
		}
	}
}

func renderTemplate(tpl string, kvs ...string) (string, bool) {
	for i := 0; i < len(kvs); i += 2 {
		k := kvs[i]
		v := kvs[i+1]
		tpl = strings.Replace(tpl, k, v, -1)
	}
	return tpl, !templateExpression.MatchString(tpl)
}

func createTemplateVars(app App) []string {
	tplVars := []string{"{name}", app.Name.String()}
	// If a label is called "name", it will be ignored as it appears after
	// in the slice
	for n, v := range app.Labels {
		tplVars = append(tplVars, fmt.Sprintf("{%s}", n), v)
	}
	return tplVars
}

func doDiscover(pre string, hostHeaders map[string]http.Header, app App, insecure InsecureOption, port uint) (*discoveryData, error) {
	app = *app.Copy()
	if app.Labels["version"] == "" {
		app.Labels["version"] = defaultVersion
	}

	_, body, err := httpsOrHTTP(pre, hostHeaders, insecure, port)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	meta := extractACMeta(body)

	tplVars := createTemplateVars(app)

	dd := &discoveryData{}

	for _, m := range meta {
		if !strings.HasPrefix(app.Name.String(), m.prefix) {
			continue
		}

		switch m.name {
		case "ac-discovery":
			// Ignore not handled variables as {ext} isn't already rendered.
			uri, _ := renderTemplate(m.uri, tplVars...)
			asc, ok := renderTemplate(uri, "{ext}", "aci.asc")
			if !ok {
				continue
			}
			aci, ok := renderTemplate(uri, "{ext}", "aci")
			if !ok {
				continue
			}
			dd.ACIEndpoints = append(dd.ACIEndpoints, ACIEndpoint{ACI: aci, ASC: asc})

		case "ac-discovery-pubkeys":
			dd.PublicKeys = append(dd.PublicKeys, m.uri)
		}
	}

	return dd, nil
}

// DiscoverWalk will make HTTPS requests to find discovery meta tags and
// optionally will use HTTP if insecure is set. hostHeaders specifies the
// header to apply depending on the host (e.g. authentication). Based on the
// response of the discoverFn it will continue to recurse up the tree. If port
// is 0, the default port will be used.
func DiscoverWalk(app App, hostHeaders map[string]http.Header, insecure InsecureOption, port uint, discoverFn DiscoverWalkFunc) (dd *discoveryData, err error) {
	parts := strings.Split(string(app.Name), "/")
	for i := range parts {
		end := len(parts) - i
		pre := strings.Join(parts[:end], "/")

		dd, err = doDiscover(pre, hostHeaders, app, insecure, port)
		if derr := discoverFn(pre, dd, err); derr != nil {
			return dd, derr
		}
	}

	return nil, fmt.Errorf("discovery failed")
}

// DiscoverWalkFunc can stop a DiscoverWalk by returning non-nil error.
type DiscoverWalkFunc func(prefix string, dd *discoveryData, err error) error

// FailedAttempt represents a failed discovery attempt. This is for debugging
// and user feedback.
type FailedAttempt struct {
	Prefix string
	Error  error
}

func walker(attempts *[]FailedAttempt, testFn DiscoverWalkFunc) DiscoverWalkFunc {
	return func(pre string, dd *discoveryData, err error) error {
		if err != nil {
			*attempts = append(*attempts, FailedAttempt{pre, err})
			return nil
		}
		if err := testFn(pre, dd, err); err != nil {
			return err
		}
		return nil
	}
}

// DiscoverACIEndpoints will make HTTPS requests to find the ac-discovery meta
// tags and optionally will use HTTP if insecure is set. hostHeaders
// specifies the header to apply depending on the host (e.g. authentication).
// It will not give up until it has exhausted the path or found an image
// discovery. If port is 0, the default port will be used.
func DiscoverACIEndpoints(app App, hostHeaders map[string]http.Header, insecure InsecureOption, port uint) (ACIEndpoints, []FailedAttempt, error) {
	testFn := func(pre string, dd *discoveryData, err error) error {
		if len(dd.ACIEndpoints) != 0 {
			return errEnough
		}
		return nil
	}

	attempts := []FailedAttempt{}
	dd, err := DiscoverWalk(app, hostHeaders, insecure, port, walker(&attempts, testFn))
	if err != nil && err != errEnough {
		return nil, attempts, err
	}

	return dd.ACIEndpoints, attempts, nil
}

// DiscoverPublicKey will make HTTPS requests to find the ac-public-keys meta
// tags and optionally will use HTTP if insecure is set. hostHeaders
// specifies the header to apply depending on the host (e.g. authentication).
// It will not give up until it has exhausted the path or found an public key.
// If port is 0, the default port will be used.
func DiscoverPublicKeys(app App, hostHeaders map[string]http.Header, insecure InsecureOption, port uint) (PublicKeys, []FailedAttempt, error) {
	testFn := func(pre string, dd *discoveryData, err error) error {
		if len(dd.PublicKeys) != 0 {
			return errEnough
		}
		return nil
	}

	attempts := []FailedAttempt{}
	dd, err := DiscoverWalk(app, hostHeaders, insecure, port, walker(&attempts, testFn))
	if err != nil && err != errEnough {
		return nil, attempts, err
	}

	return dd.PublicKeys, attempts, nil
}
