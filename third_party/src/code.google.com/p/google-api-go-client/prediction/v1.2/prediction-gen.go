// Package prediction provides access to the Prediction API.
//
// See https://developers.google.com/prediction/docs/developer-guide
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/prediction/v1.2"
//   ...
//   predictionService, err := prediction.New(oauthHttpClient)
package prediction

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

const apiId = "prediction:v1.2"
const apiName = "prediction"
const apiVersion = "v1.2"
const basePath = "https://www.googleapis.com/prediction/v1.2/"

// OAuth2 scopes used by this API.
const (
	// Manage your data and permissions in Google Cloud Storage
	DevstorageFull_controlScope = "https://www.googleapis.com/auth/devstorage.full_control"

	// View your data in Google Cloud Storage
	DevstorageRead_onlyScope = "https://www.googleapis.com/auth/devstorage.read_only"

	// Manage your data in Google Cloud Storage
	DevstorageRead_writeScope = "https://www.googleapis.com/auth/devstorage.read_write"

	// Manage your data in the Google Prediction API
	PredictionScope = "https://www.googleapis.com/auth/prediction"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Hostedmodels = NewHostedmodelsService(s)
	s.Training = NewTrainingService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Hostedmodels *HostedmodelsService

	Training *TrainingService
}

func NewHostedmodelsService(s *Service) *HostedmodelsService {
	rs := &HostedmodelsService{s: s}
	return rs
}

type HostedmodelsService struct {
	s *Service
}

func NewTrainingService(s *Service) *TrainingService {
	rs := &TrainingService{s: s}
	return rs
}

type TrainingService struct {
	s *Service
}

type Input struct {
	Input *InputInput `json:"input,omitempty"`
}

type InputInput struct {
	CsvInstance []interface{} `json:"csvInstance,omitempty"`
}

type Output struct {
	Id string `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	OutputLabel string `json:"outputLabel,omitempty"`

	OutputMulti []*OutputOutputMulti `json:"outputMulti,omitempty"`

	OutputValue float64 `json:"outputValue,omitempty"`

	SelfLink string `json:"selfLink,omitempty"`
}

type OutputOutputMulti struct {
	Label string `json:"label,omitempty"`

	Score float64 `json:"score,omitempty"`
}

type Training struct {
	Id string `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	ModelInfo *TrainingModelInfo `json:"modelInfo,omitempty"`

	SelfLink string `json:"selfLink,omitempty"`

	TrainingStatus string `json:"trainingStatus,omitempty"`
}

type TrainingModelInfo struct {
	ClassificationAccuracy float64 `json:"classificationAccuracy,omitempty"`

	MeanSquaredError float64 `json:"meanSquaredError,omitempty"`

	ModelType string `json:"modelType,omitempty"`
}

type Update struct {
	// ClassLabel: The true class label of this instance
	ClassLabel string `json:"classLabel,omitempty"`

	// CsvInstance: The input features for this instance
	CsvInstance []interface{} `json:"csvInstance,omitempty"`
}

// method id "prediction.predict":

type PredictCall struct {
	s     *Service
	data  string
	input *Input
	opt_  map[string]interface{}
}

// Predict: Submit data and request a prediction
func (s *Service) Predict(data string, input *Input) *PredictCall {
	c := &PredictCall{s: s, opt_: make(map[string]interface{})}
	c.data = data
	c.input = input
	return c
}

func (c *PredictCall) Do() (*Output, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.input)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "training/{data}/predict")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{data}", url.QueryEscape(c.data), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Output)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Submit data and request a prediction",
	//   "httpMethod": "POST",
	//   "id": "prediction.predict",
	//   "parameterOrder": [
	//     "data"
	//   ],
	//   "parameters": {
	//     "data": {
	//       "description": "mybucket%2Fmydata resource in Google Storage",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "training/{data}/predict",
	//   "request": {
	//     "$ref": "Input"
	//   },
	//   "response": {
	//     "$ref": "Output"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.hostedmodels.predict":

type HostedmodelsPredictCall struct {
	s               *Service
	hostedModelName string
	input           *Input
	opt_            map[string]interface{}
}

// Predict: Submit input and request an output against a hosted model
func (r *HostedmodelsService) Predict(hostedModelName string, input *Input) *HostedmodelsPredictCall {
	c := &HostedmodelsPredictCall{s: r.s, opt_: make(map[string]interface{})}
	c.hostedModelName = hostedModelName
	c.input = input
	return c
}

func (c *HostedmodelsPredictCall) Do() (*Output, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.input)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "hostedmodels/{hostedModelName}/predict")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{hostedModelName}", url.QueryEscape(c.hostedModelName), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Output)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Submit input and request an output against a hosted model",
	//   "httpMethod": "POST",
	//   "id": "prediction.hostedmodels.predict",
	//   "parameterOrder": [
	//     "hostedModelName"
	//   ],
	//   "parameters": {
	//     "hostedModelName": {
	//       "description": "The name of a hosted model",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "hostedmodels/{hostedModelName}/predict",
	//   "request": {
	//     "$ref": "Input"
	//   },
	//   "response": {
	//     "$ref": "Output"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.training.delete":

type TrainingDeleteCall struct {
	s    *Service
	data string
	opt_ map[string]interface{}
}

// Delete: Delete a trained model
func (r *TrainingService) Delete(data string) *TrainingDeleteCall {
	c := &TrainingDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.data = data
	return c
}

func (c *TrainingDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "training/{data}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{data}", url.QueryEscape(c.data), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Delete a trained model",
	//   "httpMethod": "DELETE",
	//   "id": "prediction.training.delete",
	//   "parameterOrder": [
	//     "data"
	//   ],
	//   "parameters": {
	//     "data": {
	//       "description": "mybucket/mydata resource in Google Storage",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "training/{data}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.training.get":

type TrainingGetCall struct {
	s    *Service
	data string
	opt_ map[string]interface{}
}

// Get: Check training status of your model
func (r *TrainingService) Get(data string) *TrainingGetCall {
	c := &TrainingGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.data = data
	return c
}

func (c *TrainingGetCall) Do() (*Training, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "training/{data}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{data}", url.QueryEscape(c.data), 1)
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
	ret := new(Training)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Check training status of your model",
	//   "httpMethod": "GET",
	//   "id": "prediction.training.get",
	//   "parameterOrder": [
	//     "data"
	//   ],
	//   "parameters": {
	//     "data": {
	//       "description": "mybucket/mydata resource in Google Storage",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "training/{data}",
	//   "response": {
	//     "$ref": "Training"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.training.insert":

type TrainingInsertCall struct {
	s        *Service
	training *Training
	opt_     map[string]interface{}
}

// Insert: Begin training your model
func (r *TrainingService) Insert(training *Training) *TrainingInsertCall {
	c := &TrainingInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.training = training
	return c
}

// Data sets the optional parameter "data": mybucket/mydata resource in
// Google Storage
func (c *TrainingInsertCall) Data(data string) *TrainingInsertCall {
	c.opt_["data"] = data
	return c
}

func (c *TrainingInsertCall) Do() (*Training, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.training)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["data"]; ok {
		params.Set("data", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "training")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Training)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Begin training your model",
	//   "httpMethod": "POST",
	//   "id": "prediction.training.insert",
	//   "parameters": {
	//     "data": {
	//       "description": "mybucket/mydata resource in Google Storage",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "training",
	//   "request": {
	//     "$ref": "Training"
	//   },
	//   "response": {
	//     "$ref": "Training"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/devstorage.full_control",
	//     "https://www.googleapis.com/auth/devstorage.read_only",
	//     "https://www.googleapis.com/auth/devstorage.read_write",
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.training.update":

type TrainingUpdateCall struct {
	s      *Service
	data   string
	update *Update
	opt_   map[string]interface{}
}

// Update: Add new data to a trained model
func (r *TrainingService) Update(data string, update *Update) *TrainingUpdateCall {
	c := &TrainingUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.data = data
	c.update = update
	return c
}

func (c *TrainingUpdateCall) Do() (*Training, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.update)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "training/{data}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{data}", url.QueryEscape(c.data), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Training)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Add new data to a trained model",
	//   "httpMethod": "PUT",
	//   "id": "prediction.training.update",
	//   "parameterOrder": [
	//     "data"
	//   ],
	//   "parameters": {
	//     "data": {
	//       "description": "mybucket/mydata resource in Google Storage",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "training/{data}",
	//   "request": {
	//     "$ref": "Update"
	//   },
	//   "response": {
	//     "$ref": "Training"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}
