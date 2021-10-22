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

// Package translate is the v2 client for the Google Translation API.
//
// PLEASE NOTE: We recommend using the new v3 client for new projects:
// https://cloud.google.com/go/translate/apiv3.
//
// See https://cloud.google.com/translation for details.
package translate

import (
	"context"
	"fmt"
	"net/http"

	"cloud.google.com/go/internal/version"
	"golang.org/x/text/language"
	"google.golang.org/api/option"
	raw "google.golang.org/api/translate/v2"
	htransport "google.golang.org/api/transport/http"
)

const userAgent = "gcloud-golang-translate/20161115"

// Scope is the OAuth2 scope required by the Google Cloud Vision API.
const Scope = raw.CloudPlatformScope

// Client is a client for the translate API.
type Client struct {
	raw *raw.Service
}

const prodAddr = "https://translation.googleapis.com/language/translate/"

// NewClient constructs a new Client that can perform Translation operations.
//
// You can find or create API key for your project from the Credentials page of
// the Developers Console (console.developers.google.com).
func NewClient(ctx context.Context, opts ...option.ClientOption) (*Client, error) {
	o := []option.ClientOption{
		option.WithEndpoint(prodAddr),
		option.WithScopes(Scope),
		option.WithUserAgent(userAgent),
	}
	o = append(o, opts...)
	httpClient, endpoint, err := htransport.NewClient(ctx, o...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}
	rawService, err := raw.New(httpClient)
	if err != nil {
		return nil, fmt.Errorf("translate client: %v", err)
	}
	rawService.BasePath = endpoint
	return &Client{raw: rawService}, nil
}

// Close closes any resources held by the client.
// Close should be called when the client is no longer needed.
// It need not be called at program exit.
func (c *Client) Close() error { return nil }

// Translate one or more strings of text from a source language to a target
// language. All inputs must be in the same language.
//
// The target parameter supplies the language to translate to. The supported
// languages are listed at
// https://cloud.google.com/translation/v2/translate-reference#supported_languages.
// You can also call the SupportedLanguages method.
//
// The returned Translations appear in the same order as the inputs.
func (c *Client) Translate(ctx context.Context, inputs []string, target language.Tag, opts *Options) ([]Translation, error) {
	call := c.raw.Translations.List(inputs, target.String()).Context(ctx)
	setClientHeader(call.Header())
	if opts != nil {
		if s := opts.Source; s != language.Und {
			call.Source(s.String())
		}
		if f := opts.Format; f != "" {
			call.Format(string(f))
		}
		if m := opts.Model; m != "" {
			call.Model(m)
		}
	}
	res, err := call.Do()
	if err != nil {
		return nil, err
	}
	var ts []Translation
	for _, t := range res.Translations {
		var source language.Tag
		if t.DetectedSourceLanguage != "" {
			source, err = language.Parse(t.DetectedSourceLanguage)
			if err != nil {
				return nil, err
			}
		}
		ts = append(ts, Translation{
			Text:   t.TranslatedText,
			Source: source,
			Model:  t.Model,
		})
	}
	return ts, nil
}

// Options contains options for Translate.
type Options struct {
	// Source is the language of the input strings. If empty, the service will
	// attempt to identify the source language automatically and return it within
	// the response.
	Source language.Tag

	// Format describes the format of the input texts. The choices are HTML or
	// Text. The default is HTML.
	Format Format

	// The model to use for translation. The choices are "nmt" or "base". The
	// default is "base".
	Model string
}

// Format is the format of the input text. Used in Options.Format.
type Format string

// Constants for Options.Format.
const (
	HTML Format = "html"
	Text Format = "text"
)

// Translation contains the results of translating a piece of text.
type Translation struct {
	// Text is the input text translated into the target language.
	Text string

	// Source is the detected language of the input text, if source was
	// not supplied to Client.Translate. If source was supplied, this field
	// will be empty.
	Source language.Tag

	// Model is the model that was used for translation.
	// It may not match the model provided as an option to Client.Translate.
	Model string
}

// DetectLanguage attempts to determine the language of the inputs. Each input
// string may be in a different language.
//
// Each slice of Detections in the return value corresponds with one input
// string. A slice of Detections holds multiple hypotheses for the language of
// a single input string.
func (c *Client) DetectLanguage(ctx context.Context, inputs []string) ([][]Detection, error) {
	call := c.raw.Detections.List(inputs).Context(ctx)
	setClientHeader(call.Header())
	res, err := call.Do()
	if err != nil {
		return nil, err
	}
	var result [][]Detection
	for _, raws := range res.Detections {
		var ds []Detection
		for _, rd := range raws {
			tag, err := language.Parse(rd.Language)
			if err != nil {
				return nil, err
			}
			ds = append(ds, Detection{
				Language:   tag,
				Confidence: rd.Confidence,
				IsReliable: rd.IsReliable,
			})
		}
		result = append(result, ds)
	}
	return result, nil
}

// Detection represents information about a language detected in an input.
type Detection struct {
	// Language is the code of the language detected.
	Language language.Tag

	// Confidence is a number from 0 to 1, with higher numbers indicating more
	// confidence in the detection.
	Confidence float64

	// IsReliable indicates whether the language detection result is reliable.
	IsReliable bool
}

// SupportedLanguages returns a list of supported languages for translation.
// The target parameter is the language to use to return localized, human
// readable names of supported languages.
func (c *Client) SupportedLanguages(ctx context.Context, target language.Tag) ([]Language, error) {
	call := c.raw.Languages.List().Context(ctx).Target(target.String())
	setClientHeader(call.Header())
	res, err := call.Do()
	if err != nil {
		return nil, err
	}
	var ls []Language
	for _, l := range res.Languages {
		tag, err := language.Parse(l.Language)
		if err != nil {
			return nil, err
		}
		ls = append(ls, Language{
			Name: l.Name,
			Tag:  tag,
		})
	}
	return ls, nil
}

// A Language describes a language supported for translation.
type Language struct {
	// Name is the human-readable name of the language.
	Name string

	// Tag is a standard code for the language.
	Tag language.Tag
}

func setClientHeader(headers http.Header) {
	headers.Set("x-goog-api-client", fmt.Sprintf("gl-go/%s gccl/%s", version.Go(), version.Repo))
}
