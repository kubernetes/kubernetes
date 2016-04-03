// Package prediction provides access to the Prediction API.
//
// See https://developers.google.com/prediction/docs/developer-guide
//
// Usage example:
//
//   import "google.golang.org/api/prediction/v1.6"
//   ...
//   predictionService, err := prediction.New(oauthHttpClient)
package prediction // import "google.golang.org/api/prediction/v1.6"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	context "golang.org/x/net/context"
	ctxhttp "golang.org/x/net/context/ctxhttp"
	gensupport "google.golang.org/api/gensupport"
	googleapi "google.golang.org/api/googleapi"
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
var _ = gensupport.MarshalJSON
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace
var _ = context.Canceled
var _ = ctxhttp.Do

const apiId = "prediction:v1.6"
const apiName = "prediction"
const apiVersion = "v1.6"
const basePath = "https://www.googleapis.com/prediction/v1.6/projects/"

// OAuth2 scopes used by this API.
const (
	// Manage your data and permissions in Google Cloud Storage
	DevstorageFullControlScope = "https://www.googleapis.com/auth/devstorage.full_control"

	// View your data in Google Cloud Storage
	DevstorageReadOnlyScope = "https://www.googleapis.com/auth/devstorage.read_only"

	// Manage your data in Google Cloud Storage
	DevstorageReadWriteScope = "https://www.googleapis.com/auth/devstorage.read_write"

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
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Hostedmodels *HostedmodelsService

	Trainedmodels *TrainedmodelsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DataDescription") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Analyze) MarshalJSON() ([]byte, error) {
	type noMethod Analyze
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AnalyzeDataDescription: Description of the data the model was trained
// on.
type AnalyzeDataDescription struct {
	// Features: Description of the input features in the data set.
	Features []*AnalyzeDataDescriptionFeatures `json:"features,omitempty"`

	// OutputFeature: Description of the output value or label.
	OutputFeature *AnalyzeDataDescriptionOutputFeature `json:"outputFeature,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Features") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescription) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescription
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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

	// ForceSendFields is a list of field names (e.g. "Categorical") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescriptionFeatures) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescriptionFeatures
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AnalyzeDataDescriptionFeaturesCategorical: Description of the
// categorical values of this feature.
type AnalyzeDataDescriptionFeaturesCategorical struct {
	// Count: Number of categorical values for this feature in the data.
	Count int64 `json:"count,omitempty,string"`

	// Values: List of all the categories for this feature in the data set.
	Values []*AnalyzeDataDescriptionFeaturesCategoricalValues `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescriptionFeaturesCategorical) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescriptionFeaturesCategorical
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AnalyzeDataDescriptionFeaturesCategoricalValues struct {
	// Count: Number of times this feature had this value.
	Count int64 `json:"count,omitempty,string"`

	// Value: The category name.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescriptionFeaturesCategoricalValues) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescriptionFeaturesCategoricalValues
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AnalyzeDataDescriptionFeaturesNumeric: Description of the numeric
// values of this feature.
type AnalyzeDataDescriptionFeaturesNumeric struct {
	// Count: Number of numeric values for this feature in the data set.
	Count int64 `json:"count,omitempty,string"`

	// Mean: Mean of the numeric values of this feature in the data set.
	Mean string `json:"mean,omitempty"`

	// Variance: Variance of the numeric values of this feature in the data
	// set.
	Variance string `json:"variance,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescriptionFeaturesNumeric) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescriptionFeaturesNumeric
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AnalyzeDataDescriptionFeaturesText: Description of multiple-word text
// values of this feature.
type AnalyzeDataDescriptionFeaturesText struct {
	// Count: Number of multiple-word text values for this feature.
	Count int64 `json:"count,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescriptionFeaturesText) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescriptionFeaturesText
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AnalyzeDataDescriptionOutputFeature: Description of the output value
// or label.
type AnalyzeDataDescriptionOutputFeature struct {
	// Numeric: Description of the output values in the data set.
	Numeric *AnalyzeDataDescriptionOutputFeatureNumeric `json:"numeric,omitempty"`

	// Text: Description of the output labels in the data set.
	Text []*AnalyzeDataDescriptionOutputFeatureText `json:"text,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Numeric") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescriptionOutputFeature) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescriptionOutputFeature
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AnalyzeDataDescriptionOutputFeatureNumeric: Description of the output
// values in the data set.
type AnalyzeDataDescriptionOutputFeatureNumeric struct {
	// Count: Number of numeric output values in the data set.
	Count int64 `json:"count,omitempty,string"`

	// Mean: Mean of the output values in the data set.
	Mean string `json:"mean,omitempty"`

	// Variance: Variance of the output values in the data set.
	Variance string `json:"variance,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescriptionOutputFeatureNumeric) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescriptionOutputFeatureNumeric
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AnalyzeDataDescriptionOutputFeatureText struct {
	// Count: Number of times the output label occurred in the data set.
	Count int64 `json:"count,omitempty,string"`

	// Value: The output label.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Count") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeDataDescriptionOutputFeatureText) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeDataDescriptionOutputFeatureText
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AnalyzeModelDescription: Description of the model.
type AnalyzeModelDescription struct {
	// ConfusionMatrix: An output confusion matrix. This shows an estimate
	// for how this model will do in predictions. This is first indexed by
	// the true class label. For each true class label, this provides a pair
	// {predicted_label, count}, where count is the estimated number of
	// times the model will predict the predicted label given the true
	// label. Will not output if more then 100 classes (Categorical models
	// only).
	ConfusionMatrix *AnalyzeModelDescriptionConfusionMatrix `json:"confusionMatrix,omitempty"`

	// ConfusionMatrixRowTotals: A list of the confusion matrix row totals.
	ConfusionMatrixRowTotals map[string]string `json:"confusionMatrixRowTotals,omitempty"`

	// Modelinfo: Basic information about the model.
	Modelinfo *Insert2 `json:"modelinfo,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConfusionMatrix") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AnalyzeModelDescription) MarshalJSON() ([]byte, error) {
	type noMethod AnalyzeModelDescription
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AnalyzeModelDescriptionConfusionMatrix: An output confusion matrix.
// This shows an estimate for how this model will do in predictions.
// This is first indexed by the true class label. For each true class
// label, this provides a pair {predicted_label, count}, where count is
// the estimated number of times the model will predict the predicted
// label given the true label. Will not output if more then 100 classes
// (Categorical models only).
type AnalyzeModelDescriptionConfusionMatrix struct {
}

type Input struct {
	// Input: Input to the model for a prediction.
	Input *InputInput `json:"input,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Input") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Input) MarshalJSON() ([]byte, error) {
	type noMethod Input
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// InputInput: Input to the model for a prediction.
type InputInput struct {
	// CsvInstance: A list of input features, these can be strings or
	// doubles.
	CsvInstance []interface{} `json:"csvInstance,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CsvInstance") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InputInput) MarshalJSON() ([]byte, error) {
	type noMethod InputInput
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Insert struct {
	// Id: The unique name for the predictive model.
	Id string `json:"id,omitempty"`

	// ModelType: Type of predictive model (classification or regression).
	ModelType string `json:"modelType,omitempty"`

	// SourceModel: The Id of the model to be copied over.
	SourceModel string `json:"sourceModel,omitempty"`

	// StorageDataLocation: Google storage location of the training data
	// file.
	StorageDataLocation string `json:"storageDataLocation,omitempty"`

	// StoragePMMLLocation: Google storage location of the preprocessing
	// pmml file.
	StoragePMMLLocation string `json:"storagePMMLLocation,omitempty"`

	// StoragePMMLModelLocation: Google storage location of the pmml model
	// file.
	StoragePMMLModelLocation string `json:"storagePMMLModelLocation,omitempty"`

	// TrainingInstances: Instances to train model on.
	TrainingInstances []*InsertTrainingInstances `json:"trainingInstances,omitempty"`

	// Utility: A class weighting function, which allows the importance
	// weights for class labels to be specified (Categorical models only).
	Utility []*InsertUtility `json:"utility,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Insert) MarshalJSON() ([]byte, error) {
	type noMethod Insert
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type InsertTrainingInstances struct {
	// CsvInstance: The input features for this instance.
	CsvInstance []interface{} `json:"csvInstance,omitempty"`

	// Output: The generic output value - could be regression or class
	// label.
	Output string `json:"output,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CsvInstance") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InsertTrainingInstances) MarshalJSON() ([]byte, error) {
	type noMethod InsertTrainingInstances
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// InsertUtility: Class label (string).
type InsertUtility struct {
}

type Insert2 struct {
	// Created: Insert time of the model (as a RFC 3339 timestamp).
	Created string `json:"created,omitempty"`

	// Id: The unique name for the predictive model.
	Id string `json:"id,omitempty"`

	// Kind: What kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// ModelInfo: Model metadata.
	ModelInfo *Insert2ModelInfo `json:"modelInfo,omitempty"`

	// ModelType: Type of predictive model (CLASSIFICATION or REGRESSION).
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

	// TrainingStatus: The current status of the training job. This can be
	// one of following: RUNNING; DONE; ERROR; ERROR: TRAINING JOB NOT FOUND
	TrainingStatus string `json:"trainingStatus,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Created") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Insert2) MarshalJSON() ([]byte, error) {
	type noMethod Insert2
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Insert2ModelInfo: Model metadata.
type Insert2ModelInfo struct {
	// ClassWeightedAccuracy: Estimated accuracy of model taking utility
	// weights into account (Categorical models only).
	ClassWeightedAccuracy string `json:"classWeightedAccuracy,omitempty"`

	// ClassificationAccuracy: A number between 0.0 and 1.0, where 1.0 is
	// 100% accurate. This is an estimate, based on the amount and quality
	// of the training data, of the estimated prediction accuracy. You can
	// use this is a guide to decide whether the results are accurate enough
	// for your needs. This estimate will be more reliable if your real
	// input data is similar to your training data (Categorical models
	// only).
	ClassificationAccuracy string `json:"classificationAccuracy,omitempty"`

	// MeanSquaredError: An estimated mean squared error. The can be used to
	// measure the quality of the predicted model (Regression models only).
	MeanSquaredError string `json:"meanSquaredError,omitempty"`

	// ModelType: Type of predictive model (CLASSIFICATION or REGRESSION).
	ModelType string `json:"modelType,omitempty"`

	// NumberInstances: Number of valid data instances used in the trained
	// model.
	NumberInstances int64 `json:"numberInstances,omitempty,string"`

	// NumberLabels: Number of class labels in the trained model
	// (Categorical models only).
	NumberLabels int64 `json:"numberLabels,omitempty,string"`

	// ForceSendFields is a list of field names (e.g.
	// "ClassWeightedAccuracy") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Insert2ModelInfo) MarshalJSON() ([]byte, error) {
	type noMethod Insert2ModelInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type List struct {
	// Items: List of models.
	Items []*Insert2 `json:"items,omitempty"`

	// Kind: What kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Pagination token to fetch the next page, if one
	// exists.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: A URL to re-request this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *List) MarshalJSON() ([]byte, error) {
	type noMethod List
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Output struct {
	// Id: The unique name for the predictive model.
	Id string `json:"id,omitempty"`

	// Kind: What kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// OutputLabel: The most likely class label (Categorical models only).
	OutputLabel string `json:"outputLabel,omitempty"`

	// OutputMulti: A list of class labels with their estimated
	// probabilities (Categorical models only).
	OutputMulti []*OutputOutputMulti `json:"outputMulti,omitempty"`

	// OutputValue: The estimated regression value (Regression models only).
	OutputValue float64 `json:"outputValue,omitempty"`

	// SelfLink: A URL to re-request this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Output) MarshalJSON() ([]byte, error) {
	type noMethod Output
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OutputOutputMulti struct {
	// Label: The class label.
	Label string `json:"label,omitempty"`

	// Score: The probability of the class label.
	Score string `json:"score,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Label") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OutputOutputMulti) MarshalJSON() ([]byte, error) {
	type noMethod OutputOutputMulti
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Update struct {
	// CsvInstance: The input features for this instance.
	CsvInstance []interface{} `json:"csvInstance,omitempty"`

	// Output: The generic output value - could be regression or class
	// label.
	Output string `json:"output,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CsvInstance") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Update) MarshalJSON() ([]byte, error) {
	type noMethod Update
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "prediction.hostedmodels.predict":

type HostedmodelsPredictCall struct {
	s               *Service
	project         string
	hostedModelName string
	input           *Input
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// Predict: Submit input and request an output against a hosted model.
func (r *HostedmodelsService) Predict(project string, hostedModelName string, input *Input) *HostedmodelsPredictCall {
	c := &HostedmodelsPredictCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.hostedModelName = hostedModelName
	c.input = input
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *HostedmodelsPredictCall) QuotaUser(quotaUser string) *HostedmodelsPredictCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *HostedmodelsPredictCall) UserIP(userIP string) *HostedmodelsPredictCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *HostedmodelsPredictCall) Fields(s ...googleapi.Field) *HostedmodelsPredictCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *HostedmodelsPredictCall) Context(ctx context.Context) *HostedmodelsPredictCall {
	c.ctx_ = ctx
	return c
}

func (c *HostedmodelsPredictCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.input)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/hostedmodels/{hostedModelName}/predict")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"hostedModelName": c.hostedModelName,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "prediction.hostedmodels.predict" call.
// Exactly one of *Output or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Output.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *HostedmodelsPredictCall) Do() (*Output, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Output{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Submit input and request an output against a hosted model.",
	//   "httpMethod": "POST",
	//   "id": "prediction.hostedmodels.predict",
	//   "parameterOrder": [
	//     "project",
	//     "hostedModelName"
	//   ],
	//   "parameters": {
	//     "hostedModelName": {
	//       "description": "The name of a hosted model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project associated with the model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/hostedmodels/{hostedModelName}/predict",
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
	s            *Service
	project      string
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Analyze: Get analysis of the model and the data the model was trained
// on.
func (r *TrainedmodelsService) Analyze(project string, id string) *TrainedmodelsAnalyzeCall {
	c := &TrainedmodelsAnalyzeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TrainedmodelsAnalyzeCall) QuotaUser(quotaUser string) *TrainedmodelsAnalyzeCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TrainedmodelsAnalyzeCall) UserIP(userIP string) *TrainedmodelsAnalyzeCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TrainedmodelsAnalyzeCall) Fields(s ...googleapi.Field) *TrainedmodelsAnalyzeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TrainedmodelsAnalyzeCall) IfNoneMatch(entityTag string) *TrainedmodelsAnalyzeCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TrainedmodelsAnalyzeCall) Context(ctx context.Context) *TrainedmodelsAnalyzeCall {
	c.ctx_ = ctx
	return c
}

func (c *TrainedmodelsAnalyzeCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/trainedmodels/{id}/analyze")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"id":      c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "prediction.trainedmodels.analyze" call.
// Exactly one of *Analyze or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Analyze.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TrainedmodelsAnalyzeCall) Do() (*Analyze, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Analyze{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get analysis of the model and the data the model was trained on.",
	//   "httpMethod": "GET",
	//   "id": "prediction.trainedmodels.analyze",
	//   "parameterOrder": [
	//     "project",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project associated with the model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/trainedmodels/{id}/analyze",
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
	s          *Service
	project    string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Delete a trained model.
func (r *TrainedmodelsService) Delete(project string, id string) *TrainedmodelsDeleteCall {
	c := &TrainedmodelsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TrainedmodelsDeleteCall) QuotaUser(quotaUser string) *TrainedmodelsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TrainedmodelsDeleteCall) UserIP(userIP string) *TrainedmodelsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TrainedmodelsDeleteCall) Fields(s ...googleapi.Field) *TrainedmodelsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TrainedmodelsDeleteCall) Context(ctx context.Context) *TrainedmodelsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *TrainedmodelsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/trainedmodels/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"id":      c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "prediction.trainedmodels.delete" call.
func (c *TrainedmodelsDeleteCall) Do() error {
	res, err := c.doRequest("json")
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
	//     "project",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project associated with the model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/trainedmodels/{id}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.trainedmodels.get":

type TrainedmodelsGetCall struct {
	s            *Service
	project      string
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Check training status of your model.
func (r *TrainedmodelsService) Get(project string, id string) *TrainedmodelsGetCall {
	c := &TrainedmodelsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TrainedmodelsGetCall) QuotaUser(quotaUser string) *TrainedmodelsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TrainedmodelsGetCall) UserIP(userIP string) *TrainedmodelsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TrainedmodelsGetCall) Fields(s ...googleapi.Field) *TrainedmodelsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TrainedmodelsGetCall) IfNoneMatch(entityTag string) *TrainedmodelsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TrainedmodelsGetCall) Context(ctx context.Context) *TrainedmodelsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *TrainedmodelsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/trainedmodels/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"id":      c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "prediction.trainedmodels.get" call.
// Exactly one of *Insert2 or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Insert2.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TrainedmodelsGetCall) Do() (*Insert2, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Insert2{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Check training status of your model.",
	//   "httpMethod": "GET",
	//   "id": "prediction.trainedmodels.get",
	//   "parameterOrder": [
	//     "project",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project associated with the model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/trainedmodels/{id}",
	//   "response": {
	//     "$ref": "Insert2"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}

// method id "prediction.trainedmodels.insert":

type TrainedmodelsInsertCall struct {
	s          *Service
	project    string
	insert     *Insert
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Train a Prediction API model.
func (r *TrainedmodelsService) Insert(project string, insert *Insert) *TrainedmodelsInsertCall {
	c := &TrainedmodelsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.insert = insert
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TrainedmodelsInsertCall) QuotaUser(quotaUser string) *TrainedmodelsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TrainedmodelsInsertCall) UserIP(userIP string) *TrainedmodelsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TrainedmodelsInsertCall) Fields(s ...googleapi.Field) *TrainedmodelsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TrainedmodelsInsertCall) Context(ctx context.Context) *TrainedmodelsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *TrainedmodelsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.insert)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/trainedmodels")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "prediction.trainedmodels.insert" call.
// Exactly one of *Insert2 or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Insert2.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TrainedmodelsInsertCall) Do() (*Insert2, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Insert2{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Train a Prediction API model.",
	//   "httpMethod": "POST",
	//   "id": "prediction.trainedmodels.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project associated with the model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/trainedmodels",
	//   "request": {
	//     "$ref": "Insert"
	//   },
	//   "response": {
	//     "$ref": "Insert2"
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
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: List available models.
func (r *TrainedmodelsService) List(project string) *TrainedmodelsListCall {
	c := &TrainedmodelsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *TrainedmodelsListCall) MaxResults(maxResults int64) *TrainedmodelsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Pagination token.
func (c *TrainedmodelsListCall) PageToken(pageToken string) *TrainedmodelsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TrainedmodelsListCall) QuotaUser(quotaUser string) *TrainedmodelsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TrainedmodelsListCall) UserIP(userIP string) *TrainedmodelsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TrainedmodelsListCall) Fields(s ...googleapi.Field) *TrainedmodelsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TrainedmodelsListCall) IfNoneMatch(entityTag string) *TrainedmodelsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TrainedmodelsListCall) Context(ctx context.Context) *TrainedmodelsListCall {
	c.ctx_ = ctx
	return c
}

func (c *TrainedmodelsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/trainedmodels/list")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "prediction.trainedmodels.list" call.
// Exactly one of *List or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *List.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *TrainedmodelsListCall) Do() (*List, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &List{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List available models.",
	//   "httpMethod": "GET",
	//   "id": "prediction.trainedmodels.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Pagination token.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project associated with the model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/trainedmodels/list",
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
	s          *Service
	project    string
	id         string
	input      *Input
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Predict: Submit model id and request a prediction.
func (r *TrainedmodelsService) Predict(project string, id string, input *Input) *TrainedmodelsPredictCall {
	c := &TrainedmodelsPredictCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.id = id
	c.input = input
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TrainedmodelsPredictCall) QuotaUser(quotaUser string) *TrainedmodelsPredictCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TrainedmodelsPredictCall) UserIP(userIP string) *TrainedmodelsPredictCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TrainedmodelsPredictCall) Fields(s ...googleapi.Field) *TrainedmodelsPredictCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TrainedmodelsPredictCall) Context(ctx context.Context) *TrainedmodelsPredictCall {
	c.ctx_ = ctx
	return c
}

func (c *TrainedmodelsPredictCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.input)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/trainedmodels/{id}/predict")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"id":      c.id,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "prediction.trainedmodels.predict" call.
// Exactly one of *Output or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Output.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TrainedmodelsPredictCall) Do() (*Output, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Output{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Submit model id and request a prediction.",
	//   "httpMethod": "POST",
	//   "id": "prediction.trainedmodels.predict",
	//   "parameterOrder": [
	//     "project",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project associated with the model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/trainedmodels/{id}/predict",
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
	s          *Service
	project    string
	id         string
	update     *Update
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Add new data to a trained model.
func (r *TrainedmodelsService) Update(project string, id string, update *Update) *TrainedmodelsUpdateCall {
	c := &TrainedmodelsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.id = id
	c.update = update
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *TrainedmodelsUpdateCall) QuotaUser(quotaUser string) *TrainedmodelsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *TrainedmodelsUpdateCall) UserIP(userIP string) *TrainedmodelsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TrainedmodelsUpdateCall) Fields(s ...googleapi.Field) *TrainedmodelsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TrainedmodelsUpdateCall) Context(ctx context.Context) *TrainedmodelsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *TrainedmodelsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.update)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/trainedmodels/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
		"id":      c.id,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "prediction.trainedmodels.update" call.
// Exactly one of *Insert2 or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Insert2.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TrainedmodelsUpdateCall) Do() (*Insert2, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &Insert2{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Add new data to a trained model.",
	//   "httpMethod": "PUT",
	//   "id": "prediction.trainedmodels.update",
	//   "parameterOrder": [
	//     "project",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The unique name for the predictive model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The project associated with the model.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/trainedmodels/{id}",
	//   "request": {
	//     "$ref": "Update"
	//   },
	//   "response": {
	//     "$ref": "Insert2"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/prediction"
	//   ]
	// }

}
