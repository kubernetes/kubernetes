// Package prediction provides access to the Prediction API.
//
// See https://developers.google.com/prediction/docs/developer-guide
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/prediction/v1.5"
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

const apiId = "prediction:v1.5"
const apiName = "prediction"
const apiVersion = "v1.5"
const basePath = "https://www.googleapis.com/prediction/v1.5/"

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
	s.Trainedmodels = NewTrainedmodelsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Hostedmodels *HostedmodelsService

	Trainedmodels *TrainedmodelsService
}

func NewHostedmodelsService(s *Service) *HostedmodelsService {
	rs := &HostedmodelsService{s: s}
	return rs
}

type HostedmodelsService struct {
	s *Service
}

func NewTrainedmodelsService(s *Service) *TrainedmodelsService {
	rs := &TrainedmodelsService{s: s}
	return rs
}

type TrainedmodelsService struct {
	s *Service
}

type Analyze struct {
	// DataDescription: Description of the data the model was trained on.
	DataDescription *AnalyzeDataDescription `json:"dataDescription,omitempty"`

	// Errors: List of errors with the data.
	Errors []map[string]string `json:"errors,omitempty"`

	// Id: The unique name for the predictive model.
	Id string `json:"id,omitempty"`

	// Kind: What kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// ModelDescription: Description of the model.
	ModelDescription *AnalyzeModelDescription `json:"modelDescription,omitempty"`

	// SelfLink: A URL to re-request this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type AnalyzeDataDescription struct {
	// Features: Description of the input features in the data set.
	Features []*AnalyzeDataDescriptionFeatures `json:"features,omitempty"`

	// OutputFeature: Description of the output value or label.
	OutputFeature *AnalyzeDataDescriptionOutputFeature `json:"outputFeature,omitempty"`
}

type AnalyzeDataDescriptionFeatures struct {
	// Categorical: Description of the categorical values of this feature.
	Categorical *AnalyzeDataDescriptionFeaturesCategorical `json:"categorical,omitempty"`

	// Index: The feature index.
	Index int64 `json:"index,omitempty,string"`

	// Numeric: Description of the numeric values of this feature.
	Numeric *AnalyzeDataDescriptionFeaturesNumeric `json:"numeric,omitempty"`

	// Text: Description of multiple-word text values of this feature.
	Text *AnalyzeDataDescriptionFeaturesText `json:"text,omitempty"`
}

type AnalyzeDataDescriptionFeaturesCategorical struct {
	// Count: Number of categorical values for this feature in the data.
	Count int64 `json:"count,omitempty,string"`

	// Values: List of all the categories for this feature in the data set.
	Values []*AnalyzeDataDescriptionFeaturesCategoricalValues `json:"values,omitempty"`
}

type AnalyzeDataDescriptionFeaturesCategoricalValues struct {
	// Count: Number of times this feature had this value.
	Count int64 `json:"count,omitempty,string"`

	// Value: The category name.
	Value string `json:"value,omitempty"`
}

type AnalyzeDataDescriptionFeaturesNumeric struct {
	// Count: Number of numeric values for this feature in the data set.
	Count int64 `json:"count,omitempty,string"`

	// Mean: Mean of the numeric values of this feature in the data set.
	Mean float64 `json:"mean,omitempty"`

	// Variance: Variance of the numeric values of this feature in the data
	// set.
	Variance float64 `json:"variance,omitempty"`
}

type AnalyzeDataDescriptionFeaturesText struct {
	// Count: Number of multiple-word text values for this feature.
	Count int64 `json:"count,omitempty,string"`
}

type AnalyzeDataDescriptionOutputFeature struct {
	// Numeric: Description of the output values in the data set.
	Numeric *AnalyzeDataDescriptionOutputFeatureNumeric `json:"numeric,omitempty"`

	// Text: Description of the output labels in the data set.
	Text []*AnalyzeDataDescriptionOutputFeatureText `json:"text,omitempty"`
}

type AnalyzeDataDescriptionOutputFeatureNumeric struct {
	// Count: Number of numeric output values in the data set.
	Count int64 `json:"count,omitempty,string"`

	// Mean: Mean of the output values in the data set.
	Mean float64 `json:"mean,omitempty"`

	// Variance: Variance of the output values in the data set.
	Variance float64 `json:"variance,omitempty"`
}

type AnalyzeDataDescriptionOutputFeatureText struct {
	// Count: Number of times the output label occurred in the data set.
	Count int64 `json:"count,omitempty,string"`

	// Value: The output label.
	Value string `json:"value,omitempty"`
}

type AnalyzeModelDescription struct {
	// ConfusionMatrix: An output confusion matrix. This shows an estimate
	// for how this model will do in predictions. This is first indexed by
	// the true class label. For each true class label, this provides a pair
	// {predicted_label, count}, where count is the estimated number of
	// times the model will predict the predicted label given the true
	// label. Will not output if more then 100 classes [Categorical models
	// only].
	ConfusionMatrix *AnalyzeModelDescriptionConfusionMatrix `json:"confusionMatrix,omitempty"`

	// ConfusionMatrixRowTotals: A list of the confusion matrix row totals
	ConfusionMatrixRowTotals *AnalyzeModelDescriptionConfusionMatrixRowTotals `json:"confusionMatrixRowTotals,omitempty"`

	// Modelinfo: Basic information about the model.
	Modelinfo *Training `json:"modelinfo,omitempty"`
}

type AnalyzeModelDescriptionConfusionMatrix struct {
}

type AnalyzeModelDescriptionConfusionMatrixRowTotals struct {
}

type Input struct {
	// Input: Input to the model for a prediction
	Input *InputInput `json:"input,omitempty"`
}

type InputInput struct {
	// CsvInstance: A list of input features, these can be strings or
	// doubles.
	CsvInstance []interface{} `json:"csvInstance,omitempty"`
}

type List struct {
	// Items: List of models.
	Items []*Training `json:"items,omitempty"`

	// Kind: What kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Pagination token to fetch the next page, if one
	// exists.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: A URL to re-request this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type Output struct {
	// Id: The unique name for the predictive model.
	Id string `json:"id,omitempty"`

	// Kind: What kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// OutputLabel: The most likely class label [Categorical models only].
	OutputLabel string `json:"outputLabel,omitempty"`

	// OutputMulti: A list of class labels with their estimated
	// probabilities [Categorical models only].
	OutputMulti []*OutputOutputMulti `json:"outputMulti,omitempty"`

	// OutputValue: The estimated regression value [Regression models only].
	OutputValue float64 `json:"outputValue,omitempty"`

	// SelfLink: A URL to re-request this resource.
	SelfLink string `json:"selfLink,omitempty"`
}

type OutputOutputMulti struct {
	// Label: The class label.
	Label string `json:"label,omitempty"`

	// Score: The probability of the class label.
	Score float64 `json:"score,omitempty"`
}

type Training struct {
	// Created: Insert time of the model (as a RFC 3339 timestamp).
	Created string `json:"created,omitempty"`

	// Id: The unique name for the predictive model.
	Id string `json:"id,omitempty"`

	// Kind: What kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// ModelInfo: Model metadata.
	ModelInfo *TrainingModelInfo `json:"modelInfo,omitempty"`

	// ModelType: Type of predictive model (classification or regression)
	ModelType string `json:"modelType,omitempty"`

	// SelfLink: A URL to re-request this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// StorageDataLocation: Google storage location of the training data
	// file.
	StorageDataLocation string `json:"storageDataLocation,omitempty"`

	// StoragePMMLLocation: Google storage location of the preprocessing
	// pmml file.
	StoragePMMLLocation string `json:"storagePMMLLocation,omitempty"`

	// StoragePMMLModelLocation: Google storage location of the pmml model
	// file.
	StoragePMMLModelLocation string `json:"storagePMMLModelLocation,omitempty"`

	// TrainingComplete: Training completion time (as a RFC 3339 timestamp).
	TrainingComplete string `json:"trainingComplete,omitempty"`

	// TrainingInstances: Instances to train model on.
	TrainingInstances []*TrainingTrainingInstances `json:"trainingInstances,omitempty"`

	// TrainingStatus: The current status of the training job. This can be
	// one of following: RUNNING; DONE; ERROR; ERROR: TRAINING JOB NOT FOUND
	TrainingStatus string `json:"trainingStatus,omitempty"`

	// Utility: A class weighting function, which allows the importance
	// weights for class labels to be specified [Categorical models only].
	Utility []*TrainingUtility `json:"utility,omitempty"`
}

type TrainingModelInfo struct {
	// ClassWeightedAccuracy: Estimated accuracy of model taking utility
	// weights into account [Categorical models only].
	ClassWeightedAccuracy float64 `json:"classWeightedAccuracy,omitempty"`

	// ClassificationAccuracy: A number between 0.0 and 1.0, where 1.0 is
	// 100% accurate. This is an estimate, based on the amount and quality
	// of the training data, of the estimated prediction accuracy. You can
	// use this is a guide to decide whether the results are accurate enough
	// for your needs. This estimate will be more reliable if your real
	// input data is similar to your training data [Categorical models
	// only].
	ClassificationAccuracy float64 `json:"classificationAccuracy,omitempty"`

	// MeanSquaredError: An estimated mean squared error. The can be used to
	// measure the quality of the predicted model [Regression models only].
	MeanSquaredError float64 `json:"meanSquaredError,omitempty"`

	// ModelType: Type of predictive model (CLASSIFICATION or REGRESSION)
	ModelType string `json:"modelType,omitempty"`

	// NumberInstances: Number of valid data instances used in the trained
	// model.
	NumberInstances int64 `json:"numberInstances,omitempty,string"`

	// NumberLabels: Number of class labels in the trained model
	// [Categorical models only].
	NumberLabels int64 `json:"numberLabels,omitempty,string"`
}

type TrainingTrainingInstances struct {
	// CsvInstance: The input features for this instance
	CsvInstance []interface{} `json:"csvInstance,omitempty"`

	// Output: The generic output value - could be regression or class label
	Output string `json:"output,omitempty"`
}

type TrainingUtility struct {
}

type Update struct {
	// CsvInstance: The input features for this instance
	CsvInstance []interface{} `json:"csvInstance,omitempty"`

	// Label: The class label of this instance
	Label string `json:"label,omitempty"`

	// Output: The generic output value - could be regression value or class
	// label
	Output string `json:"output,omitempty"`
}

// method id "prediction.hostedmodels.predict":

type HostedmodelsPredictCall struct {
	s               *Service
	hostedModelName string
	input           *Input
	opt_            map[string]interface{}
}

// Predict: Submit input and request an output against a hosted model.
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
	//   "description": "Submit input and request an output against a hosted model.",
	//   "httpMethod": "POST",
	//   "id": "prediction.hostedmodels.predict",
	//   "parameterOrder": [
	//     "hostedModelName"
	//   ],
	//   "parameters": {
	//     "hostedModelName": {
	//       "description": "The name of a hosted model.",
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

// method id "prediction.trainedmodels.analyze":

type TrainedmodelsAnalyzeCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Analyze: Get analysis of the model and the data the model was trained
// on.
func (r *TrainedmodelsService) Analyze(id string) *TrainedmodelsAnalyzeCall {
	c := &TrainedmodelsAnalyzeCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *TrainedmodelsAnalyzeCall) Do() (*Analyze, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "trainedmodels/{id}/analyze")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(Analyze)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get analysis of the model and the data the model was trained on.",
	//   "httpMethod": "GET",
	//   "id": "prediction.trainedmodels.analyze",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "trainedmodels/{id}/analyze",
	//   "response": {
	//     "$ref": "Analyze"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.trainedmodels.delete":

type TrainedmodelsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Delete a trained model.
func (r *TrainedmodelsService) Delete(id string) *TrainedmodelsDeleteCall {
	c := &TrainedmodelsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *TrainedmodelsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "trainedmodels/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	//   "description": "Delete a trained model.",
	//   "httpMethod": "DELETE",
	//   "id": "prediction.trainedmodels.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "trainedmodels/{id}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.trainedmodels.get":

type TrainedmodelsGetCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Get: Check training status of your model.
func (r *TrainedmodelsService) Get(id string) *TrainedmodelsGetCall {
	c := &TrainedmodelsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *TrainedmodelsGetCall) Do() (*Training, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "trainedmodels/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	//   "description": "Check training status of your model.",
	//   "httpMethod": "GET",
	//   "id": "prediction.trainedmodels.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "trainedmodels/{id}",
	//   "response": {
	//     "$ref": "Training"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.trainedmodels.insert":

type TrainedmodelsInsertCall struct {
	s        *Service
	training *Training
	opt_     map[string]interface{}
}

// Insert: Begin training your model.
func (r *TrainedmodelsService) Insert(training *Training) *TrainedmodelsInsertCall {
	c := &TrainedmodelsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.training = training
	return c
}

func (c *TrainedmodelsInsertCall) Do() (*Training, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.training)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "trainedmodels")
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
	//   "description": "Begin training your model.",
	//   "httpMethod": "POST",
	//   "id": "prediction.trainedmodels.insert",
	//   "path": "trainedmodels",
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

// method id "prediction.trainedmodels.list":

type TrainedmodelsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List available models.
func (r *TrainedmodelsService) List() *TrainedmodelsListCall {
	c := &TrainedmodelsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *TrainedmodelsListCall) MaxResults(maxResults int64) *TrainedmodelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Pagination token
func (c *TrainedmodelsListCall) PageToken(pageToken string) *TrainedmodelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *TrainedmodelsListCall) Do() (*List, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "trainedmodels/list")
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
	ret := new(List)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List available models.",
	//   "httpMethod": "GET",
	//   "id": "prediction.trainedmodels.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Pagination token",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "trainedmodels/list",
	//   "response": {
	//     "$ref": "List"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.trainedmodels.predict":

type TrainedmodelsPredictCall struct {
	s     *Service
	id    string
	input *Input
	opt_  map[string]interface{}
}

// Predict: Submit model id and request a prediction.
func (r *TrainedmodelsService) Predict(id string, input *Input) *TrainedmodelsPredictCall {
	c := &TrainedmodelsPredictCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.input = input
	return c
}

func (c *TrainedmodelsPredictCall) Do() (*Output, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.input)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "trainedmodels/{id}/predict")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	//   "description": "Submit model id and request a prediction.",
	//   "httpMethod": "POST",
	//   "id": "prediction.trainedmodels.predict",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "trainedmodels/{id}/predict",
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

// method id "prediction.trainedmodels.update":

type TrainedmodelsUpdateCall struct {
	s      *Service
	id     string
	update *Update
	opt_   map[string]interface{}
}

// Update: Add new data to a trained model.
func (r *TrainedmodelsService) Update(id string, update *Update) *TrainedmodelsUpdateCall {
	c := &TrainedmodelsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.update = update
	return c
}

func (c *TrainedmodelsUpdateCall) Do() (*Training, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.update)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "trainedmodels/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	//   "description": "Add new data to a trained model.",
	//   "httpMethod": "PUT",
	//   "id": "prediction.trainedmodels.update",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "trainedmodels/{id}",
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
