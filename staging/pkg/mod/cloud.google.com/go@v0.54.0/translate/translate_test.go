// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package translate

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"

	"cloud.google.com/go/internal/testutil"
	"golang.org/x/text/language"
	"google.golang.org/api/option"
)

var (
	once    sync.Once
	authOpt option.ClientOption
)

func initTest(ctx context.Context, t *testing.T) *Client {
	if testing.Short() {
		t.Skip("integration tests skipped in short mode")
	}
	once.Do(func() { authOpt = authOption() })
	if authOpt == nil {
		t.Skip("Integration tests skipped. See CONTRIBUTING.md for details")
	}
	client, err := NewClient(ctx, authOpt)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return client
}

func authOption() option.ClientOption {
	ts := testutil.TokenSource(context.Background(), Scope)
	if ts != nil {
		log.Println("authenticating via OAuth2")
		return option.WithTokenSource(ts)
	}
	apiKey := os.Getenv("GCLOUD_TESTS_API_KEY")
	if apiKey != "" {
		log.Println("authenticating with API key")
		return option.WithAPIKey(apiKey)
	}
	return nil
}

type fakeTransport struct {
	req *http.Request
}

func (t *fakeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.req = req
	return &http.Response{
		Status:     fmt.Sprintf("%d OK", http.StatusOK),
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(strings.NewReader("{}")),
	}, nil
}

func TestTranslateURL(t *testing.T) {
	// The translate API has all inputs in the URL.
	// Make sure we generate the right one.
	ctx := context.Background()
	ft := &fakeTransport{}
	c, err := NewClient(ctx, option.WithHTTPClient(&http.Client{Transport: ft}))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		target language.Tag
		inputs []string
		opts   *Options
		want   url.Values
	}{
		{language.Spanish, []string{"text"}, nil, url.Values{
			"q":           []string{"text"},
			"target":      []string{"es"},
			"prettyPrint": []string{"false"},
		}},
		{language.English, []string{"text"}, &Options{}, url.Values{
			"q":           []string{"text"},
			"target":      []string{"en"},
			"prettyPrint": []string{"false"},
		}},
		{language.Turkish, []string{"t1", "t2"}, nil, url.Values{
			"q":           []string{"t1", "t2"},
			"target":      []string{"tr"},
			"prettyPrint": []string{"false"},
		}},
		{language.English, []string{"text"}, &Options{Source: language.French},
			url.Values{
				"q":           []string{"text"},
				"source":      []string{"fr"},
				"target":      []string{"en"},
				"prettyPrint": []string{"false"},
			},
		},
		{language.English, []string{"text"}, &Options{Source: language.French, Format: HTML}, url.Values{
			"q":           []string{"text"},
			"source":      []string{"fr"},
			"format":      []string{"html"},
			"target":      []string{"en"},
			"prettyPrint": []string{"false"},
		}},
	} {
		_, err = c.Translate(ctx, test.inputs, test.target, test.opts)
		if err != nil {
			t.Fatal(err)
		}
		got := ft.req.URL.Query()
		test.want.Add("alt", "json")
		if !testutil.Equal(got, test.want) {
			t.Errorf("Translate(%s, %v, %+v):\ngot  %s\nwant %s",
				test.target, test.inputs, test.opts, got, test.want)
		}
	}
}

func TestTranslateOneInput(t *testing.T) {
	ctx := context.Background()
	c := initTest(ctx, t)
	defer c.Close()

	translate := func(input string, target language.Tag, opts *Options) Translation {
		ts, err := c.Translate(ctx, []string{input}, target, opts)
		if err != nil {
			t.Fatal(err)
		}
		if len(ts) != 1 {
			t.Fatalf("wanted one Translation, got %d", len(ts))
		}
		return ts[0]
	}

	for _, test := range []struct {
		input  string
		source language.Tag
		output string
		target language.Tag
	}{
		// https://www.youtube.com/watch?v=x1sQkEfAdfY
		{"Le singe est sur la branche", language.French,
			"The monkey is on the branch", language.English},
		{"The cat is on the chair", language.English,
			"Le chat est sur la chaise", language.French},
	} {
		// Provide source and format.
		tr := translate(test.input, test.target, &Options{Source: test.source, Format: Text})
		if got, want := tr.Source, language.Und; got != want {
			t.Errorf("source: got %q, wanted %q", got, want)
			continue
		}
		if got, want := tr.Text, test.output; got != want {
			t.Errorf("text: got %q, want %q", got, want)
		}
		// Omit source; it should be detected.
		tr = translate(test.input, test.target, &Options{Format: Text})
		if got, want := tr.Source, test.source; got != want {
			t.Errorf("source: got %q, wanted %q", got, want)
			continue
		}
		if got, want := tr.Text, test.output; got != want {
			t.Errorf("text: got %q, want %q", got, want)
		}

		// Omit format. Defaults to HTML. Still works with plain text.
		tr = translate(test.input, test.target, nil)
		if got, want := tr.Source, test.source; got != want {
			t.Errorf("source: got %q, wanted %q", got, want)
			continue
		}
		if got, want := tr.Text, test.output; got != want {
			t.Errorf("text: got %q, want %q", got, want)
		}

		// Add HTML tags to input. They should be in output.
		htmlify := func(s string) string {
			return "<b><i>" + s + "</i></b>"
		}
		tr = translate(htmlify(test.input), test.target, nil)
		if got, want := tr.Text, htmlify(test.output); got != want {
			t.Errorf("html: got %q, want %q", got, want)
		}
		// Using the HTML format behaves the same.
		tr = translate(htmlify(test.input), test.target, &Options{Format: HTML})
		if got, want := tr.Text, htmlify(test.output); got != want {
			t.Errorf("html: got %q, want %q", got, want)
		}
	}
}

// This tests the beta "nmt" model.
func TestTranslateModel(t *testing.T) {
	ctx := context.Background()
	c := initTest(ctx, t)
	defer c.Close()

	trs, err := c.Translate(ctx, []string{"Thanks"}, language.French, &Options{Model: "nmt"})
	if err != nil {
		t.Fatal(err)
	}
	if len(trs) != 1 {
		t.Fatalf("wanted one Translation, got %d", len(trs))
	}
	tr := trs[0]
	if got, want := tr.Text, "Merci"; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
	if got, want := tr.Model, "nmt"; got != want {
		t.Errorf("model: got %q, want %q", got, want)
	}
}

func TestTranslateMultipleInputs(t *testing.T) {
	ctx := context.Background()
	c := initTest(ctx, t)
	defer c.Close()

	inputs := []string{
		"When you're a Jet, you're a Jet all the way",
		"From your first cigarette to your last dying day",
		"When you're a Jet if the spit hits the fan",
		"You got brothers around, you're a family man",
	}
	ts, err := c.Translate(ctx, inputs, language.French, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(ts), len(inputs); got != want {
		t.Fatalf("got %d Translations, wanted %d", got, want)
	}
}

func TestTranslateErrors(t *testing.T) {
	ctx := context.Background()
	c := initTest(ctx, t)
	defer c.Close()

	for _, test := range []struct {
		ctx    context.Context
		target language.Tag
		inputs []string
		opts   *Options
	}{
		{ctx, language.English, nil, nil},
		{ctx, language.Und, []string{"input"}, nil},
		{ctx, language.English, []string{}, nil},
		{ctx, language.English, []string{"input"}, &Options{Format: "random"}},
	} {
		_, err := c.Translate(test.ctx, test.inputs, test.target, test.opts)
		if err == nil {
			t.Errorf("%+v: got nil, want error", test)
		}
	}
}

func TestDetectLanguage(t *testing.T) {
	ctx := context.Background()
	c := initTest(ctx, t)
	defer c.Close()
	ds, err := c.DetectLanguage(ctx, []string{
		"Today is Monday",
		"Aujourd'hui est lundi",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(ds) != 2 {
		t.Fatalf("got %d detection lists, want 2", len(ds))
	}
	checkDetections(t, ds[0], language.English)
	checkDetections(t, ds[1], language.French)
}

func checkDetections(t *testing.T, ds []Detection, want language.Tag) {
	for _, d := range ds {
		if d.Language == want {
			return
		}
	}
	t.Errorf("%v: missing %s", ds, want)
}

// A small subset of the supported languages.
var supportedLangs = []Language{
	{Name: "Danish", Tag: language.Danish},
	{Name: "English", Tag: language.English},
	{Name: "French", Tag: language.French},
	{Name: "German", Tag: language.German},
	{Name: "Greek", Tag: language.Greek},
	{Name: "Hindi", Tag: language.Hindi},
	{Name: "Hungarian", Tag: language.Hungarian},
	{Name: "Italian", Tag: language.Italian},
	{Name: "Russian", Tag: language.Russian},
	{Name: "Turkish", Tag: language.Turkish},
}

func TestSupportedLanguages(t *testing.T) {
	ctx := context.Background()
	c := initTest(ctx, t)
	defer c.Close()
	got, err := c.SupportedLanguages(ctx, language.English)
	if err != nil {
		t.Fatal(err)
	}
	want := map[language.Tag]Language{}
	for _, sl := range supportedLangs {
		want[sl.Tag] = sl
	}
	for _, g := range got {
		w, ok := want[g.Tag]
		if !ok {
			continue
		}
		if g != w {
			t.Errorf("got %+v, want %+v", g, w)
		}
		delete(want, g.Tag)
	}
	if len(want) > 0 {
		t.Errorf("missing: %+v", want)
	}
}
