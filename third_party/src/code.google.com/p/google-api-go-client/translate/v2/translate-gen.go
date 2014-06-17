// Package translate provides access to the Translate API.
//
// See https://developers.google.com/translate/v2/using_rest
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/translate/v2"
//   ...
//   translateService, err := translate.New(oauthHttpClient)
package translate

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Always reference these packages, just in case the auto-generated code
// below doesn't.
var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace

const apiId = "translate:v2"
const apiName = "translate"
const apiVersion = "v2"
const basePath = "https://www.googleapis.com/language/translate/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Detections = NewDetectionsService(s)
	s.Languages = NewLanguagesService(s)
	s.Translations = NewTranslationsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Detections *DetectionsService

	Languages *LanguagesService

	Translations *TranslationsService
}

func NewDetectionsService(s *Service) *DetectionsService {
	rs := &DetectionsService{s: s}
	return rs
}

type DetectionsService struct {
	s *Service
}

func NewLanguagesService(s *Service) *LanguagesService {
	rs := &LanguagesService{s: s}
	return rs
}

type LanguagesService struct {
	s *Service
}

func NewTranslationsService(s *Service) *TranslationsService {
	rs := &TranslationsService{s: s}
	return rs
}

type TranslationsService struct {
	s *Service
}

type DetectionsListResponse struct {
	// Detections: A detections contains detection results of several text
	Detections [][]*DetectionsResourceItem `json:"detections,omitempty"`
}

type DetectionsResourceItem struct {
	// Confidence: The confidence of the detection resul of this language.
	Confidence float64 `json:"confidence,omitempty"`

	// IsReliable: A boolean to indicate is the language detection result
	// reliable.
	IsReliable bool `json:"isReliable,omitempty"`

	// Language: The language we detect
	Language string `json:"language,omitempty"`
}

type LanguagesListResponse struct {
	// Languages: List of source/target languages supported by the
	// translation API. If target parameter is unspecified, the list is
	// sorted by the ASCII code point order of the language code. If target
	// parameter is specified, the list is sorted by the collation order of
	// the language name in the target language.
	Languages []*LanguagesResource `json:"languages,omitempty"`
}

type LanguagesResource struct {
	// Language: The language code.
	Language string `json:"language,omitempty"`

	// Name: The localized name of the language if target parameter is
	// given.
	Name string `json:"name,omitempty"`
}

type TranslationsListResponse struct {
	// Translations: Translations contains list of translation results of
	// given text
	Translations []*TranslationsResource `json:"translations,omitempty"`
}

type TranslationsResource struct {
	// DetectedSourceLanguage: Detected source language if source parameter
	// is unspecified.
	DetectedSourceLanguage string `json:"detectedSourceLanguage,omitempty"`

	// TranslatedText: The translation.
	TranslatedText string `json:"translatedText,omitempty"`
}

// method id "language.detections.list":

type DetectionsListCall struct {
	s    *Service
	q    []string
	opt_ map[string]interface{}
}

// List: Detect the language of text.
func (r *DetectionsService) List(q []string) *DetectionsListCall {
	c := &DetectionsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.q = q
	return c
}

func (c *DetectionsListCall) Do() (*DetectionsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	for _, v := range c.q {
		params.Add("q", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2/detect")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(DetectionsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Detect the language of text.",
	//   "httpMethod": "GET",
	//   "id": "language.detections.list",
	//   "parameterOrder": [
	//     "q"
	//   ],
	//   "parameters": {
	//     "q": {
	//       "description": "The text to detect",
	//       "location": "query",
	//       "repeated": true,
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2/detect",
	//   "response": {
	//     "$ref": "DetectionsListResponse"
	//   }
	// }

}

// method id "language.languages.list":

type LanguagesListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List the source/target languages supported by the API
func (r *LanguagesService) List() *LanguagesListCall {
	c := &LanguagesListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Target sets the optional parameter "target": the language and
// collation in which the localized results should be returned
func (c *LanguagesListCall) Target(target string) *LanguagesListCall {
	c.opt_["target"] = target
	return c
}

func (c *LanguagesListCall) Do() (*LanguagesListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["target"]; ok {
		params.Set("target", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2/languages")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(LanguagesListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List the source/target languages supported by the API",
	//   "httpMethod": "GET",
	//   "id": "language.languages.list",
	//   "parameters": {
	//     "target": {
	//       "description": "the language and collation in which the localized results should be returned",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2/languages",
	//   "response": {
	//     "$ref": "LanguagesListResponse"
	//   }
	// }

}

// method id "language.translations.list":

type TranslationsListCall struct {
	s      *Service
	q      []string
	target string
	opt_   map[string]interface{}
}

// List: Returns text translations from one language to another.
func (r *TranslationsService) List(q []string, target string) *TranslationsListCall {
	c := &TranslationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.q = q
	c.target = target
	return c
}

// Cid sets the optional parameter "cid": The customization id for
// translate
func (c *TranslationsListCall) Cid(cid string) *TranslationsListCall {
	c.opt_["cid"] = cid
	return c
}

// Format sets the optional parameter "format": The format of the text
func (c *TranslationsListCall) Format(format string) *TranslationsListCall {
	c.opt_["format"] = format
	return c
}

// Source sets the optional parameter "source": The source language of
// the text
func (c *TranslationsListCall) Source(source string) *TranslationsListCall {
	c.opt_["source"] = source
	return c
}

func (c *TranslationsListCall) Do() (*TranslationsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("target", fmt.Sprintf("%v", c.target))
	for _, v := range c.q {
		params.Add("q", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["cid"]; ok {
		params.Set("cid", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["format"]; ok {
		params.Set("format", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TranslationsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns text translations from one language to another.",
	//   "httpMethod": "GET",
	//   "id": "language.translations.list",
	//   "parameterOrder": [
	//     "q",
	//     "target"
	//   ],
	//   "parameters": {
	//     "cid": {
	//       "description": "The customization id for translate",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "format": {
	//       "description": "The format of the text",
	//       "enum": [
	//         "html",
	//         "text"
	//       ],
	//       "enumDescriptions": [
	//         "Specifies the input is in HTML",
	//         "Specifies the input is in plain textual format"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "The text to translate",
	//       "location": "query",
	//       "repeated": true,
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "The source language of the text",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "target": {
	//       "description": "The target language into which the text should be translated",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2",
	//   "response": {
	//     "$ref": "TranslationsListResponse"
	//   }
	// }

}
