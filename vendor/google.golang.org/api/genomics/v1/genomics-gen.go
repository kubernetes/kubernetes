// Package genomics provides access to the Genomics API.
//
// See https://cloud.google.com/genomics
//
// Usage example:
//
//   import "google.golang.org/api/genomics/v1"
//   ...
//   genomicsService, err := genomics.New(oauthHttpClient)
package genomics // import "google.golang.org/api/genomics/v1"

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

const apiId = "genomics:v1"
const apiName = "genomics"
const apiVersion = "v1"
const basePath = "https://genomics.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data in Google BigQuery
	BigqueryScope = "https://www.googleapis.com/auth/bigquery"

	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// Manage your data in Google Cloud Storage
	DevstorageReadWriteScope = "https://www.googleapis.com/auth/devstorage.read_write"

	// View and manage Genomics data
	GenomicsScope = "https://www.googleapis.com/auth/genomics"

	// View Genomics data
	GenomicsReadonlyScope = "https://www.googleapis.com/auth/genomics.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Annotations = NewAnnotationsService(s)
	s.Annotationsets = NewAnnotationsetsService(s)
	s.Callsets = NewCallsetsService(s)
	s.Datasets = NewDatasetsService(s)
	s.Operations = NewOperationsService(s)
	s.Readgroupsets = NewReadgroupsetsService(s)
	s.Reads = NewReadsService(s)
	s.References = NewReferencesService(s)
	s.Referencesets = NewReferencesetsService(s)
	s.Variants = NewVariantsService(s)
	s.Variantsets = NewVariantsetsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Annotations *AnnotationsService

	Annotationsets *AnnotationsetsService

	Callsets *CallsetsService

	Datasets *DatasetsService

	Operations *OperationsService

	Readgroupsets *ReadgroupsetsService

	Reads *ReadsService

	References *ReferencesService

	Referencesets *ReferencesetsService

	Variants *VariantsService

	Variantsets *VariantsetsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAnnotationsService(s *Service) *AnnotationsService {
	rs := &AnnotationsService{s: s}
	return rs
}

type AnnotationsService struct {
	s *Service
}

func NewAnnotationsetsService(s *Service) *AnnotationsetsService {
	rs := &AnnotationsetsService{s: s}
	return rs
}

type AnnotationsetsService struct {
	s *Service
}

func NewCallsetsService(s *Service) *CallsetsService {
	rs := &CallsetsService{s: s}
	return rs
}

type CallsetsService struct {
	s *Service
}

func NewDatasetsService(s *Service) *DatasetsService {
	rs := &DatasetsService{s: s}
	return rs
}

type DatasetsService struct {
	s *Service
}

func NewOperationsService(s *Service) *OperationsService {
	rs := &OperationsService{s: s}
	return rs
}

type OperationsService struct {
	s *Service
}

func NewReadgroupsetsService(s *Service) *ReadgroupsetsService {
	rs := &ReadgroupsetsService{s: s}
	rs.Coveragebuckets = NewReadgroupsetsCoveragebucketsService(s)
	return rs
}

type ReadgroupsetsService struct {
	s *Service

	Coveragebuckets *ReadgroupsetsCoveragebucketsService
}

func NewReadgroupsetsCoveragebucketsService(s *Service) *ReadgroupsetsCoveragebucketsService {
	rs := &ReadgroupsetsCoveragebucketsService{s: s}
	return rs
}

type ReadgroupsetsCoveragebucketsService struct {
	s *Service
}

func NewReadsService(s *Service) *ReadsService {
	rs := &ReadsService{s: s}
	return rs
}

type ReadsService struct {
	s *Service
}

func NewReferencesService(s *Service) *ReferencesService {
	rs := &ReferencesService{s: s}
	rs.Bases = NewReferencesBasesService(s)
	return rs
}

type ReferencesService struct {
	s *Service

	Bases *ReferencesBasesService
}

func NewReferencesBasesService(s *Service) *ReferencesBasesService {
	rs := &ReferencesBasesService{s: s}
	return rs
}

type ReferencesBasesService struct {
	s *Service
}

func NewReferencesetsService(s *Service) *ReferencesetsService {
	rs := &ReferencesetsService{s: s}
	return rs
}

type ReferencesetsService struct {
	s *Service
}

func NewVariantsService(s *Service) *VariantsService {
	rs := &VariantsService{s: s}
	return rs
}

type VariantsService struct {
	s *Service
}

func NewVariantsetsService(s *Service) *VariantsetsService {
	rs := &VariantsetsService{s: s}
	return rs
}

type VariantsetsService struct {
	s *Service
}

// Annotation: An annotation describes a region of reference genome. The
// value of an
// annotation may be one of several canonical types, supplemented by
// arbitrary
// info tags. An annotation is not inherently associated with a
// specific
// sample or individual (though a client could choose to use annotations
// in
// this way). Example canonical annotation types are `GENE`
// and
// `VARIANT`.
type Annotation struct {
	// AnnotationSetId: The annotation set to which this annotation belongs.
	AnnotationSetId string `json:"annotationSetId,omitempty"`

	// End: The end position of the range on the reference, 0-based
	// exclusive.
	End int64 `json:"end,omitempty,string"`

	// Id: The server-generated annotation ID, unique across all
	// annotations.
	Id string `json:"id,omitempty"`

	// Info: A map of additional read alignment information. This must be of
	// the form
	// map<string, string[]> (string key mapping to a list of string
	// values).
	Info map[string][]interface{} `json:"info,omitempty"`

	// Name: The display name of this annotation.
	Name string `json:"name,omitempty"`

	// ReferenceId: The ID of the Google Genomics reference associated with
	// this range.
	ReferenceId string `json:"referenceId,omitempty"`

	// ReferenceName: The display name corresponding to the reference
	// specified by
	// `referenceId`, for example `chr1`, `1`, or `chrX`.
	ReferenceName string `json:"referenceName,omitempty"`

	// ReverseStrand: Whether this range refers to the reverse strand, as
	// opposed to the forward
	// strand. Note that regardless of this field, the start/end position of
	// the
	// range always refer to the forward strand.
	ReverseStrand bool `json:"reverseStrand,omitempty"`

	// Start: The start position of the range on the reference, 0-based
	// inclusive.
	Start int64 `json:"start,omitempty,string"`

	// Transcript: A transcript value represents the assertion that a
	// particular region of
	// the reference genome may be transcribed as RNA. An alternative
	// splicing
	// pattern would be represented as a separate transcript object. This
	// field
	// is only set for annotations of type `TRANSCRIPT`.
	Transcript *Transcript `json:"transcript,omitempty"`

	// Type: The data type for this annotation. Must match the containing
	// annotation
	// set's type.
	//
	// Possible values:
	//   "ANNOTATION_TYPE_UNSPECIFIED"
	//   "GENERIC" - A `GENERIC` annotation type should be used when no
	// other annotation
	// type will suffice. This represents an untyped annotation of the
	// reference
	// genome.
	//   "VARIANT" - A `VARIANT` annotation type.
	//   "GENE" - A `GENE` annotation type represents the existence of a
	// gene at the
	// associated reference coordinates. The start coordinate is typically
	// the
	// gene's transcription start site and the end is typically the end of
	// the
	// gene's last exon.
	//   "TRANSCRIPT" - A `TRANSCRIPT` annotation type represents the
	// assertion that a
	// particular region of the reference genome may be transcribed as RNA.
	Type string `json:"type,omitempty"`

	// Variant: A variant annotation, which describes the effect of a
	// variant on the
	// genome, the coding sequence, and/or higher level consequences at
	// the
	// organism level e.g. pathogenicity. This field is only set for
	// annotations
	// of type `VARIANT`.
	Variant *VariantAnnotation `json:"variant,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AnnotationSetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationSetId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Annotation) MarshalJSON() ([]byte, error) {
	type noMethod Annotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AnnotationSet: An annotation set is a logical grouping of annotations
// that share consistent
// type information and provenance. Examples of annotation sets include
// 'all
// genes from refseq', and 'all variant annotations from ClinVar'.
type AnnotationSet struct {
	// DatasetId: The dataset to which this annotation set belongs.
	DatasetId string `json:"datasetId,omitempty"`

	// Id: The server-generated annotation set ID, unique across all
	// annotation sets.
	Id string `json:"id,omitempty"`

	// Info: A map of additional read alignment information. This must be of
	// the form
	// map<string, string[]> (string key mapping to a list of string
	// values).
	Info map[string][]interface{} `json:"info,omitempty"`

	// Name: The display name for this annotation set.
	Name string `json:"name,omitempty"`

	// ReferenceSetId: The ID of the reference set that defines the
	// coordinate space for this
	// set's annotations.
	ReferenceSetId string `json:"referenceSetId,omitempty"`

	// SourceUri: The source URI describing the file from which this
	// annotation set was
	// generated, if any.
	SourceUri string `json:"sourceUri,omitempty"`

	// Type: The type of annotations contained within this set.
	//
	// Possible values:
	//   "ANNOTATION_TYPE_UNSPECIFIED"
	//   "GENERIC" - A `GENERIC` annotation type should be used when no
	// other annotation
	// type will suffice. This represents an untyped annotation of the
	// reference
	// genome.
	//   "VARIANT" - A `VARIANT` annotation type.
	//   "GENE" - A `GENE` annotation type represents the existence of a
	// gene at the
	// associated reference coordinates. The start coordinate is typically
	// the
	// gene's transcription start site and the end is typically the end of
	// the
	// gene's last exon.
	//   "TRANSCRIPT" - A `TRANSCRIPT` annotation type represents the
	// assertion that a
	// particular region of the reference genome may be transcribed as RNA.
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DatasetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AnnotationSet) MarshalJSON() ([]byte, error) {
	type noMethod AnnotationSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BatchCreateAnnotationsRequest struct {
	// Annotations: The annotations to be created. At most 4096 can be
	// specified in a single
	// request.
	Annotations []*Annotation `json:"annotations,omitempty"`

	// RequestId: A unique request ID which enables the server to detect
	// duplicated requests.
	// If provided, duplicated requests will result in the same response; if
	// not
	// provided, duplicated requests may result in duplicated data. For a
	// given
	// annotation set, callers should not reuse `request_id`s when
	// writing
	// different batches of annotations - behavior in this case is
	// undefined.
	// A common approach is to use a UUID. For batch jobs where worker
	// crashes are
	// a possibility, consider using some unique variant of a worker or run
	// ID.
	RequestId string `json:"requestId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Annotations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Annotations") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchCreateAnnotationsRequest) MarshalJSON() ([]byte, error) {
	type noMethod BatchCreateAnnotationsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BatchCreateAnnotationsResponse struct {
	// Entries: The resulting per-annotation entries, ordered consistently
	// with the
	// original request.
	Entries []*Entry `json:"entries,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchCreateAnnotationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchCreateAnnotationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Binding: Associates `members` with a `role`.
type Binding struct {
	// Members: Specifies the identities requesting access for a Cloud
	// Platform resource.
	// `members` can have the following values:
	//
	// * `allUsers`: A special identifier that represents anyone who is
	//    on the internet; with or without a Google account.
	//
	// * `allAuthenticatedUsers`: A special identifier that represents
	// anyone
	//    who is authenticated with a Google account or a service
	// account.
	//
	// * `user:{emailid}`: An email address that represents a specific
	// Google
	//    account. For example, `alice@gmail.com` or `joe@example.com`.
	//
	//
	// * `serviceAccount:{emailid}`: An email address that represents a
	// service
	//    account. For example,
	// `my-other-app@appspot.gserviceaccount.com`.
	//
	// * `group:{emailid}`: An email address that represents a Google
	// group.
	//    For example, `admins@example.com`.
	//
	//
	// * `domain:{domain}`: A Google Apps domain name that represents all
	// the
	//    users of that domain. For example, `google.com` or
	// `example.com`.
	//
	//
	Members []string `json:"members,omitempty"`

	// Role: Role that is assigned to `members`.
	// For example, `roles/viewer`, `roles/editor`, or
	// `roles/owner`.
	// Required
	Role string `json:"role,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Members") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Members") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Binding) MarshalJSON() ([]byte, error) {
	type noMethod Binding
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CallSet: A call set is a collection of variant calls, typically for
// one sample. It
// belongs to a variant set.
//
// For more genomics resource definitions, see [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
type CallSet struct {
	// Created: The date this call set was created in milliseconds from the
	// epoch.
	Created int64 `json:"created,omitempty,string"`

	// Id: The server-generated call set ID, unique across all call sets.
	Id string `json:"id,omitempty"`

	// Info: A map of additional call set information. This must be of the
	// form
	// map<string, string[]> (string key mapping to a list of string
	// values).
	Info map[string][]interface{} `json:"info,omitempty"`

	// Name: The call set name.
	Name string `json:"name,omitempty"`

	// SampleId: The sample ID this call set corresponds to.
	SampleId string `json:"sampleId,omitempty"`

	// VariantSetIds: The IDs of the variant sets this call set belongs to.
	// This field must
	// have exactly length one, as a call set belongs to a single variant
	// set.
	// This field is repeated for compatibility with the
	// [GA4GH
	// 0.5.1
	// API](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resou
	// rces/avro/variants.avdl#L76).
	VariantSetIds []string `json:"variantSetIds,omitempty"`

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

	// NullFields is a list of field names (e.g. "Created") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CallSet) MarshalJSON() ([]byte, error) {
	type noMethod CallSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CancelOperationRequest: The request message for
// Operations.CancelOperation.
type CancelOperationRequest struct {
}

// CigarUnit: A single CIGAR operation.
type CigarUnit struct {
	// Possible values:
	//   "OPERATION_UNSPECIFIED"
	//   "ALIGNMENT_MATCH" - An alignment match indicates that a sequence
	// can be aligned to the
	// reference without evidence of an INDEL. Unlike the
	// `SEQUENCE_MATCH` and `SEQUENCE_MISMATCH` operators,
	// the `ALIGNMENT_MATCH` operator does not indicate whether
	// the
	// reference and read sequences are an exact match. This operator
	// is
	// equivalent to SAM's `M`.
	//   "INSERT" - The insert operator indicates that the read contains
	// evidence of bases
	// being inserted into the reference. This operator is equivalent to
	// SAM's
	// `I`.
	//   "DELETE" - The delete operator indicates that the read contains
	// evidence of bases
	// being deleted from the reference. This operator is equivalent to
	// SAM's
	// `D`.
	//   "SKIP" - The skip operator indicates that this read skips a long
	// segment of the
	// reference, but the bases have not been deleted. This operator is
	// commonly
	// used when working with RNA-seq data, where reads may skip long
	// segments
	// of the reference between exons. This operator is equivalent to
	// SAM's
	// `N`.
	//   "CLIP_SOFT" - The soft clip operator indicates that bases at the
	// start/end of a read
	// have not been considered during alignment. This may occur if the
	// majority
	// of a read maps, except for low quality bases at the start/end of a
	// read.
	// This operator is equivalent to SAM's `S`. Bases that are soft
	// clipped will still be stored in the read.
	//   "CLIP_HARD" - The hard clip operator indicates that bases at the
	// start/end of a read
	// have been omitted from this alignment. This may occur if this
	// linear
	// alignment is part of a chimeric alignment, or if the read has
	// been
	// trimmed (for example, during error correction or to trim poly-A tails
	// for
	// RNA-seq). This operator is equivalent to SAM's `H`.
	//   "PAD" - The pad operator indicates that there is padding in an
	// alignment. This
	// operator is equivalent to SAM's `P`.
	//   "SEQUENCE_MATCH" - This operator indicates that this portion of the
	// aligned sequence exactly
	// matches the reference. This operator is equivalent to SAM's `=`.
	//   "SEQUENCE_MISMATCH" - This operator indicates that this portion of
	// the aligned sequence is an
	// alignment match to the reference, but a sequence mismatch. This
	// can
	// indicate a SNP or a read error. This operator is equivalent to
	// SAM's
	// `X`.
	Operation string `json:"operation,omitempty"`

	// OperationLength: The number of genomic bases that the operation runs
	// for. Required.
	OperationLength int64 `json:"operationLength,omitempty,string"`

	// ReferenceSequence: `referenceSequence` is only used at
	// mismatches
	// (`SEQUENCE_MISMATCH`) and deletions (`DELETE`).
	// Filling this field replaces SAM's MD tag. If the relevant information
	// is
	// not available, this field is unset.
	ReferenceSequence string `json:"referenceSequence,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Operation") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Operation") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CigarUnit) MarshalJSON() ([]byte, error) {
	type noMethod CigarUnit
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ClinicalCondition struct {
	// ConceptId: The MedGen concept id associated with this gene.
	// Search for these IDs at http://www.ncbi.nlm.nih.gov/medgen/
	ConceptId string `json:"conceptId,omitempty"`

	// ExternalIds: The set of external IDs for this condition.
	ExternalIds []*ExternalId `json:"externalIds,omitempty"`

	// Names: A set of names for the condition.
	Names []string `json:"names,omitempty"`

	// OmimId: The OMIM id for this condition.
	// Search for these IDs at http://omim.org/
	OmimId string `json:"omimId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConceptId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConceptId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClinicalCondition) MarshalJSON() ([]byte, error) {
	type noMethod ClinicalCondition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type CodingSequence struct {
	// End: The end of the coding sequence on this annotation's reference
	// sequence,
	// 0-based exclusive. Note that this position is relative to the
	// reference
	// start, and *not* the containing annotation start.
	End int64 `json:"end,omitempty,string"`

	// Start: The start of the coding sequence on this annotation's
	// reference sequence,
	// 0-based inclusive. Note that this position is relative to the
	// reference
	// start, and *not* the containing annotation start.
	Start int64 `json:"start,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "End") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "End") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CodingSequence) MarshalJSON() ([]byte, error) {
	type noMethod CodingSequence
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ComputeEngine: Describes a Compute Engine resource that is being
// managed by a running
// pipeline.
type ComputeEngine struct {
	// DiskNames: The names of the disks that were created for this
	// pipeline.
	DiskNames []string `json:"diskNames,omitempty"`

	// InstanceName: The instance on which the operation is running.
	InstanceName string `json:"instanceName,omitempty"`

	// MachineType: The machine type of the instance.
	MachineType string `json:"machineType,omitempty"`

	// Zone: The availability zone in which the instance resides.
	Zone string `json:"zone,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DiskNames") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DiskNames") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ComputeEngine) MarshalJSON() ([]byte, error) {
	type noMethod ComputeEngine
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CoverageBucket: A bucket over which read coverage has been
// precomputed. A bucket corresponds
// to a specific range of the reference sequence.
type CoverageBucket struct {
	// MeanCoverage: The average number of reads which are aligned to each
	// individual
	// reference base in this bucket.
	MeanCoverage float64 `json:"meanCoverage,omitempty"`

	// Range: The genomic coordinate range spanned by this bucket.
	Range *Range `json:"range,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MeanCoverage") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MeanCoverage") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CoverageBucket) MarshalJSON() ([]byte, error) {
	type noMethod CoverageBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *CoverageBucket) UnmarshalJSON(data []byte) error {
	type noMethod CoverageBucket
	var s1 struct {
		MeanCoverage gensupport.JSONFloat64 `json:"meanCoverage"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.MeanCoverage = float64(s1.MeanCoverage)
	return nil
}

// Dataset: A Dataset is a collection of genomic data.
//
// For more genomics resource definitions, see [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
type Dataset struct {
	// CreateTime: The time this dataset was created, in seconds from the
	// epoch.
	CreateTime string `json:"createTime,omitempty"`

	// Id: The server-generated dataset ID, unique across all datasets.
	Id string `json:"id,omitempty"`

	// Name: The dataset name.
	Name string `json:"name,omitempty"`

	// ProjectId: The Google Cloud project ID that this dataset belongs to.
	ProjectId string `json:"projectId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreateTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CreateTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Dataset) MarshalJSON() ([]byte, error) {
	type noMethod Dataset
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated
// empty messages in your APIs. A typical example is to use it as the
// request
// or the response type of an API method. For instance:
//
//     service Foo {
//       rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty);
//     }
//
// The JSON representation for `Empty` is empty JSON object `{}`.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

type Entry struct {
	// Annotation: The created annotation, if creation was successful.
	Annotation *Annotation `json:"annotation,omitempty"`

	// Status: The creation status.
	Status *Status `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Annotation") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Annotation") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Entry) MarshalJSON() ([]byte, error) {
	type noMethod Entry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Exon struct {
	// End: The end position of the exon on this annotation's reference
	// sequence,
	// 0-based exclusive. Note that this is relative to the reference start,
	// and
	// *not* the containing annotation start.
	End int64 `json:"end,omitempty,string"`

	// Frame: The frame of this exon. Contains a value of 0, 1, or 2, which
	// indicates
	// the offset of the first coding base of the exon within the reading
	// frame
	// of the coding DNA sequence, if any. This field is dependent on
	// the
	// strandedness of this annotation (see
	// Annotation.reverse_strand).
	// For forward stranded annotations, this offset is relative to
	// the
	// exon.start. For reverse
	// strand annotations, this offset is relative to the
	// exon.end `- 1`.
	//
	// Unset if this exon does not intersect the coding sequence. Upon
	// creation
	// of a transcript, the frame must be populated for all or none of
	// the
	// coding exons.
	Frame int64 `json:"frame,omitempty"`

	// Start: The start position of the exon on this annotation's reference
	// sequence,
	// 0-based inclusive. Note that this is relative to the reference start,
	// and
	// **not** the containing annotation start.
	Start int64 `json:"start,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "End") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "End") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Exon) MarshalJSON() ([]byte, error) {
	type noMethod Exon
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Experiment struct {
	// InstrumentModel: The instrument model used as part of this
	// experiment. This maps to
	// sequencing technology in the SAM spec.
	InstrumentModel string `json:"instrumentModel,omitempty"`

	// LibraryId: A client-supplied library identifier; a library is a
	// collection of DNA
	// fragments which have been prepared for sequencing from a sample.
	// This
	// field is important for quality control as error or bias can be
	// introduced
	// during sample preparation.
	LibraryId string `json:"libraryId,omitempty"`

	// PlatformUnit: The platform unit used as part of this experiment, for
	// example
	// flowcell-barcode.lane for Illumina or slide for SOLiD. Corresponds to
	// the
	// @RG PU field in the SAM spec.
	PlatformUnit string `json:"platformUnit,omitempty"`

	// SequencingCenter: The sequencing center used as part of this
	// experiment.
	SequencingCenter string `json:"sequencingCenter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InstrumentModel") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InstrumentModel") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Experiment) MarshalJSON() ([]byte, error) {
	type noMethod Experiment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExportReadGroupSetRequest: The read group set export request.
type ExportReadGroupSetRequest struct {
	// ExportUri: Required. A Google Cloud Storage URI for the exported BAM
	// file.
	// The currently authenticated user must have write access to the new
	// file.
	// An error will be returned if the URI already contains data.
	ExportUri string `json:"exportUri,omitempty"`

	// ProjectId: Required. The Google Cloud project ID that owns
	// this
	// export. The caller must have WRITE access to this project.
	ProjectId string `json:"projectId,omitempty"`

	// ReferenceNames: The reference names to export. If this is not
	// specified, all reference
	// sequences, including unmapped reads, are exported.
	// Use `*` to export only unmapped reads.
	ReferenceNames []string `json:"referenceNames,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExportUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExportUri") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExportReadGroupSetRequest) MarshalJSON() ([]byte, error) {
	type noMethod ExportReadGroupSetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExportVariantSetRequest: The variant data export request.
type ExportVariantSetRequest struct {
	// BigqueryDataset: Required. The BigQuery dataset to export data to.
	// This dataset must already
	// exist. Note that this is distinct from the Genomics concept of
	// "dataset".
	BigqueryDataset string `json:"bigqueryDataset,omitempty"`

	// BigqueryTable: Required. The BigQuery table to export data to.
	// If the table doesn't exist, it will be created. If it already exists,
	// it
	// will be overwritten.
	BigqueryTable string `json:"bigqueryTable,omitempty"`

	// CallSetIds: If provided, only variant call information from the
	// specified call sets
	// will be exported. By default all variant calls are exported.
	CallSetIds []string `json:"callSetIds,omitempty"`

	// Format: The format for the exported data.
	//
	// Possible values:
	//   "FORMAT_UNSPECIFIED"
	//   "FORMAT_BIGQUERY" - Export the data to Google BigQuery.
	Format string `json:"format,omitempty"`

	// ProjectId: Required. The Google Cloud project ID that owns the
	// destination
	// BigQuery dataset. The caller must have WRITE access to this project.
	// This
	// project will also own the resulting export job.
	ProjectId string `json:"projectId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BigqueryDataset") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BigqueryDataset") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ExportVariantSetRequest) MarshalJSON() ([]byte, error) {
	type noMethod ExportVariantSetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ExternalId struct {
	// Id: The id used by the source of this data.
	Id string `json:"id,omitempty"`

	// SourceName: The name of the source of this data.
	SourceName string `json:"sourceName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExternalId) MarshalJSON() ([]byte, error) {
	type noMethod ExternalId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GetIamPolicyRequest: Request message for `GetIamPolicy` method.
type GetIamPolicyRequest struct {
}

// ImportReadGroupSetsRequest: The read group set import request.
type ImportReadGroupSetsRequest struct {
	// DatasetId: Required. The ID of the dataset these read group sets will
	// belong to. The
	// caller must have WRITE permissions to this dataset.
	DatasetId string `json:"datasetId,omitempty"`

	// PartitionStrategy: The partition strategy describes how read groups
	// are partitioned into read
	// group sets.
	//
	// Possible values:
	//   "PARTITION_STRATEGY_UNSPECIFIED"
	//   "PER_FILE_PER_SAMPLE" - In most cases, this strategy yields one
	// read group set per file. This is
	// the default behavior.
	//
	// Allocate one read group set per file per sample. For BAM files,
	// read
	// groups are considered to share a sample if they have identical
	// sample
	// names. Furthermore, all reads for each file which do not belong to a
	// read
	// group, if any, will be grouped into a single read group set per-file.
	//   "MERGE_ALL" - Includes all read groups in all imported files into a
	// single read group
	// set. Requires that the headers for all imported files are equivalent.
	// All
	// reads which do not belong to a read group, if any, will be grouped
	// into a
	// separate read group set.
	PartitionStrategy string `json:"partitionStrategy,omitempty"`

	// ReferenceSetId: The reference set to which the imported read group
	// sets are aligned to, if
	// any. The reference names of this reference set must be a superset of
	// those
	// found in the imported file headers. If no reference set id is
	// provided, a
	// best effort is made to associate with a matching reference set.
	ReferenceSetId string `json:"referenceSetId,omitempty"`

	// SourceUris: A list of URIs pointing at
	// [BAM
	// files](https://samtools.github.io/hts-specs/SAMv1.pdf)
	// in Google Cloud Storage.
	// Those URIs can include wildcards (*), but do not add or
	// remove
	// matching files before import has completed.
	//
	// Note that Google Cloud Storage object listing is only
	// eventually
	// consistent: files added may be not be immediately visible
	// to
	// everyone. Thus, if using a wildcard it is preferable not to start
	// the import immediately after the files are created.
	SourceUris []string `json:"sourceUris,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DatasetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ImportReadGroupSetsRequest) MarshalJSON() ([]byte, error) {
	type noMethod ImportReadGroupSetsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImportReadGroupSetsResponse: The read group set import response.
type ImportReadGroupSetsResponse struct {
	// ReadGroupSetIds: IDs of the read group sets that were created.
	ReadGroupSetIds []string `json:"readGroupSetIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReadGroupSetIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReadGroupSetIds") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ImportReadGroupSetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ImportReadGroupSetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImportVariantsRequest: The variant data import request.
type ImportVariantsRequest struct {
	// Format: The format of the variant data being imported. If
	// unspecified, defaults to
	// to `VCF`.
	//
	// Possible values:
	//   "FORMAT_UNSPECIFIED"
	//   "FORMAT_VCF" - VCF (Variant Call Format). The VCF files may be gzip
	// compressed. gVCF is
	// also supported. Disclaimer: gzip VCF imports are currently much
	// slower
	// than equivalent uncompressed VCF imports. For this reason,
	// uncompressed
	// VCF is currently recommended for imports with more than 1GB
	// combined
	// uncompressed size, or for time sensitive imports.
	//   "FORMAT_COMPLETE_GENOMICS" - Complete Genomics masterVarBeta
	// format. The masterVarBeta files may
	// be bzip2 compressed.
	Format string `json:"format,omitempty"`

	// InfoMergeConfig: A mapping between info field keys and the
	// InfoMergeOperations to
	// be performed on them. This is plumbed down to the
	// MergeVariantRequests
	// generated by the resulting import job.
	InfoMergeConfig map[string]string `json:"infoMergeConfig,omitempty"`

	// NormalizeReferenceNames: Convert reference names to the canonical
	// representation.
	// hg19 haploytypes (those reference names containing "_hap")
	// are not modified in any way.
	// All other reference names are modified according to the following
	// rules:
	// The reference name is capitalized.
	// The "chr" prefix is dropped for all autosomes and sex chromsomes.
	// For example "chr17" becomes "17" and "chrX" becomes "X".
	// All mitochondrial chromosomes ("chrM", "chrMT", etc) become "MT".
	NormalizeReferenceNames bool `json:"normalizeReferenceNames,omitempty"`

	// SourceUris: A list of URIs referencing variant files in Google Cloud
	// Storage. URIs can
	// include wildcards [as
	// described
	// here](https://cloud.google.com/storage/docs/gsutil/addlhelp/
	// WildcardNames).
	// Note that recursive wildcards ('**') are not supported.
	SourceUris []string `json:"sourceUris,omitempty"`

	// VariantSetId: Required. The variant set to which variant data should
	// be imported.
	VariantSetId string `json:"variantSetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Format") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Format") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ImportVariantsRequest) MarshalJSON() ([]byte, error) {
	type noMethod ImportVariantsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImportVariantsResponse: The variant data import response.
type ImportVariantsResponse struct {
	// CallSetIds: IDs of the call sets created during the import.
	CallSetIds []string `json:"callSetIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CallSetIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CallSetIds") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ImportVariantsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ImportVariantsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LinearAlignment: A linear alignment can be represented by one CIGAR
// string. Describes the
// mapped position and local alignment of the read to the reference.
type LinearAlignment struct {
	// Cigar: Represents the local alignment of this sequence (alignment
	// matches, indels,
	// etc) against the reference.
	Cigar []*CigarUnit `json:"cigar,omitempty"`

	// MappingQuality: The mapping quality of this alignment. Represents how
	// likely
	// the read maps to this position as opposed to other
	// locations.
	//
	// Specifically, this is -10 log10 Pr(mapping position is wrong),
	// rounded to
	// the nearest integer.
	MappingQuality int64 `json:"mappingQuality,omitempty"`

	// Position: The position of this alignment.
	Position *Position `json:"position,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cigar") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cigar") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LinearAlignment) MarshalJSON() ([]byte, error) {
	type noMethod LinearAlignment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListBasesResponse struct {
	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Offset: The offset position (0-based) of the given `sequence` from
	// the
	// start of this `Reference`. This value will differ for each page
	// in a paginated request.
	Offset int64 `json:"offset,omitempty,string"`

	// Sequence: A substring of the bases that make up this reference.
	Sequence string `json:"sequence,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListBasesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListBasesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListCoverageBucketsResponse struct {
	// BucketWidth: The length of each coverage bucket in base pairs. Note
	// that buckets at the
	// end of a reference sequence may be shorter. This value is omitted if
	// the
	// bucket width is infinity (the default behaviour, with no range
	// or
	// `targetBucketWidth`).
	BucketWidth int64 `json:"bucketWidth,omitempty,string"`

	// CoverageBuckets: The coverage buckets. The list of buckets is sparse;
	// a bucket with 0
	// overlapping reads is not returned. A bucket never crosses more than
	// one
	// reference sequence. Each bucket has width `bucketWidth`, unless
	// its end is the end of the reference sequence.
	CoverageBuckets []*CoverageBucket `json:"coverageBuckets,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BucketWidth") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketWidth") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListCoverageBucketsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListCoverageBucketsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListDatasetsResponse: The dataset list response.
type ListDatasetsResponse struct {
	// Datasets: The list of matching Datasets.
	Datasets []*Dataset `json:"datasets,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Datasets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Datasets") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListDatasetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListDatasetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListOperationsResponse: The response message for
// Operations.ListOperations.
type ListOperationsResponse struct {
	// NextPageToken: The standard List next-page token.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Operations: A list of operations that matches the specified filter in
	// the request.
	Operations []*Operation `json:"operations,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListOperationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListOperationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type MergeVariantsRequest struct {
	// InfoMergeConfig: A mapping between info field keys and the
	// InfoMergeOperations to
	// be performed on them.
	InfoMergeConfig map[string]string `json:"infoMergeConfig,omitempty"`

	// VariantSetId: The destination variant set.
	VariantSetId string `json:"variantSetId,omitempty"`

	// Variants: The variants to be merged with existing variants.
	Variants []*Variant `json:"variants,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InfoMergeConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InfoMergeConfig") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MergeVariantsRequest) MarshalJSON() ([]byte, error) {
	type noMethod MergeVariantsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: This resource represents a long-running operation that is
// the result of a
// network API call.
type Operation struct {
	// Done: If the value is `false`, it means the operation is still in
	// progress.
	// If `true`, the operation is completed, and either `error` or
	// `response` is
	// available.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure or
	// cancellation.
	Error *Status `json:"error,omitempty"`

	// Metadata: An OperationMetadata object. This will always be returned
	// with the Operation.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The server-assigned name, which is only unique within the same
	// service that originally returns it. For example&#58;
	// `operations/CJHU7Oi_ChDrveSpBRjfuL-qzoWAgEw`
	Name string `json:"name,omitempty"`

	// Response: If importing ReadGroupSets, an ImportReadGroupSetsResponse
	// is returned. If importing Variants, an ImportVariantsResponse is
	// returned. For pipelines and exports, an empty response is returned.
	Response googleapi.RawMessage `json:"response,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Done") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Done") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OperationEvent: An event that occurred during an Operation.
type OperationEvent struct {
	// Description: Required description of event.
	Description string `json:"description,omitempty"`

	// EndTime: Optional time of when event finished. An event can have a
	// start time and no
	// finish time. If an event has a finish time, there must be a start
	// time.
	EndTime string `json:"endTime,omitempty"`

	// StartTime: Optional time of when event started.
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Description") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Description") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationEvent) MarshalJSON() ([]byte, error) {
	type noMethod OperationEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OperationMetadata: Metadata describing an Operation.
type OperationMetadata struct {
	// ClientId: This field is deprecated. Use `labels` instead. Optionally
	// provided by the
	// caller when submitting the request that creates the operation.
	ClientId string `json:"clientId,omitempty"`

	// CreateTime: The time at which the job was submitted to the Genomics
	// service.
	CreateTime string `json:"createTime,omitempty"`

	// EndTime: The time at which the job stopped running.
	EndTime string `json:"endTime,omitempty"`

	// Events: Optional event messages that were generated during the job's
	// execution.
	// This also contains any warnings that were generated during import
	// or export.
	Events []*OperationEvent `json:"events,omitempty"`

	// Labels: Optionally provided by the caller when submitting the request
	// that creates
	// the operation.
	Labels map[string]string `json:"labels,omitempty"`

	// ProjectId: The Google Cloud Project in which the job is scoped.
	ProjectId string `json:"projectId,omitempty"`

	// Request: The original request that started the operation. Note that
	// this will be in
	// current version of the API. If the operation was started with v1beta2
	// API
	// and a GetOperation is performed on v1 API, a v1 request will be
	// returned.
	Request googleapi.RawMessage `json:"request,omitempty"`

	// RuntimeMetadata: Runtime metadata on this Operation.
	RuntimeMetadata googleapi.RawMessage `json:"runtimeMetadata,omitempty"`

	// StartTime: The time at which the job began to run.
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClientId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClientId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod OperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Policy: Defines an Identity and Access Management (IAM) policy. It is
// used to
// specify access control policies for Cloud Platform resources.
//
//
// A `Policy` consists of a list of `bindings`. A `Binding` binds a list
// of
// `members` to a `role`, where the members can be user accounts, Google
// groups,
// Google domains, and service accounts. A `role` is a named list of
// permissions
// defined by IAM.
//
// **Example**
//
//     {
//       "bindings": [
//         {
//           "role": "roles/owner",
//           "members": [
//             "user:mike@example.com",
//             "group:admins@example.com",
//             "domain:google.com",
//
// "serviceAccount:my-other-app@appspot.gserviceaccount.com",
//           ]
//         },
//         {
//           "role": "roles/viewer",
//           "members": ["user:sean@example.com"]
//         }
//       ]
//     }
//
// For a description of IAM and its features, see the
// [IAM developer's guide](https://cloud.google.com/iam).
type Policy struct {
	// Bindings: Associates a list of `members` to a `role`.
	// `bindings` with no members will result in an error.
	Bindings []*Binding `json:"bindings,omitempty"`

	// Etag: `etag` is used for optimistic concurrency control as a way to
	// help
	// prevent simultaneous updates of a policy from overwriting each
	// other.
	// It is strongly suggested that systems make use of the `etag` in
	// the
	// read-modify-write cycle to perform policy updates in order to avoid
	// race
	// conditions: An `etag` is returned in the response to `getIamPolicy`,
	// and
	// systems are expected to put that etag in the request to
	// `setIamPolicy` to
	// ensure that their change will be applied to the same version of the
	// policy.
	//
	// If no `etag` is provided in the call to `setIamPolicy`, then the
	// existing
	// policy is overwritten blindly.
	Etag string `json:"etag,omitempty"`

	// Version: Version of the `Policy`. The default version is 0.
	Version int64 `json:"version,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Bindings") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bindings") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Policy) MarshalJSON() ([]byte, error) {
	type noMethod Policy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Position: An abstraction for referring to a genomic position, in
// relation to some
// already known reference. For now, represents a genomic position as
// a
// reference name, a base number on that reference (0-based), and
// a
// determination of forward or reverse strand.
type Position struct {
	// Position: The 0-based offset from the start of the forward strand for
	// that reference.
	Position int64 `json:"position,omitempty,string"`

	// ReferenceName: The name of the reference in whatever reference set is
	// being used.
	ReferenceName string `json:"referenceName,omitempty"`

	// ReverseStrand: Whether this position is on the reverse strand, as
	// opposed to the forward
	// strand.
	ReverseStrand bool `json:"reverseStrand,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Position") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Position") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Position) MarshalJSON() ([]byte, error) {
	type noMethod Position
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Program struct {
	// CommandLine: The command line used to run this program.
	CommandLine string `json:"commandLine,omitempty"`

	// Id: The user specified locally unique ID of the program. Used along
	// with
	// `prevProgramId` to define an ordering between programs.
	Id string `json:"id,omitempty"`

	// Name: The display name of the program. This is typically the
	// colloquial name of
	// the tool used, for example 'bwa' or 'picard'.
	Name string `json:"name,omitempty"`

	// PrevProgramId: The ID of the program run before this one.
	PrevProgramId string `json:"prevProgramId,omitempty"`

	// Version: The version of the program run.
	Version string `json:"version,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CommandLine") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CommandLine") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Program) MarshalJSON() ([]byte, error) {
	type noMethod Program
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Range: A 0-based half-open genomic coordinate range for search
// requests.
type Range struct {
	// End: The end position of the range on the reference, 0-based
	// exclusive.
	End int64 `json:"end,omitempty,string"`

	// ReferenceName: The reference sequence name, for example `chr1`,
	// `1`, or `chrX`.
	ReferenceName string `json:"referenceName,omitempty"`

	// Start: The start position of the range on the reference, 0-based
	// inclusive.
	Start int64 `json:"start,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "End") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "End") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Range) MarshalJSON() ([]byte, error) {
	type noMethod Range
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Read: A read alignment describes a linear alignment of a string of
// DNA to a
// reference sequence, in addition to metadata
// about the fragment (the molecule of DNA sequenced) and the read (the
// bases
// which were read by the sequencer). A read is equivalent to a line in
// a SAM
// file. A read belongs to exactly one read group and exactly one
// read group set.
//
// For more genomics resource definitions, see [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// ### Reverse-stranded reads
//
// Mapped reads (reads having a non-null `alignment`) can be aligned to
// either
// the forward or the reverse strand of their associated reference.
// Strandedness
// of a mapped read is encoded by
// `alignment.position.reverseStrand`.
//
// If we consider the reference to be a forward-stranded coordinate
// space of
// `[0, reference.length)` with `0` as the left-most position
// and
// `reference.length` as the right-most position, reads are always
// aligned left
// to right. That is, `alignment.position.position` always refers to
// the
// left-most reference coordinate and `alignment.cigar` describes the
// alignment
// of this read to the reference from left to right. All per-base fields
// such as
// `alignedSequence` and `alignedQuality` share this same
// left-to-right
// orientation; this is true of reads which are aligned to either
// strand. For
// reverse-stranded reads, this means that `alignedSequence` is the
// reverse
// complement of the bases that were originally reported by the
// sequencing
// machine.
//
// ### Generating a reference-aligned sequence string
//
// When interacting with mapped reads, it's often useful to produce a
// string
// representing the local alignment of the read to reference. The
// following
// pseudocode demonstrates one way of doing this:
//
//     out = ""
//     offset = 0
//     for c in read.alignment.cigar {
//       switch c.operation {
//       case "ALIGNMENT_MATCH", "SEQUENCE_MATCH", "SEQUENCE_MISMATCH":
//         out += read.alignedSequence[offset:offset+c.operationLength]
//         offset += c.operationLength
//         break
//       case "CLIP_SOFT", "INSERT":
//         offset += c.operationLength
//         break
//       case "PAD":
//         out += repeat("*", c.operationLength)
//         break
//       case "DELETE":
//         out += repeat("-", c.operationLength)
//         break
//       case "SKIP":
//         out += repeat(" ", c.operationLength)
//         break
//       case "CLIP_HARD":
//         break
//       }
//     }
//     return out
//
// ### Converting to SAM's CIGAR string
//
// The following pseudocode generates a SAM CIGAR string from
// the
// `cigar` field. Note that this is a lossy
// conversion
// (`cigar.referenceSequence` is lost).
//
//     cigarMap = {
//       "ALIGNMENT_MATCH": "M",
//       "INSERT": "I",
//       "DELETE": "D",
//       "SKIP": "N",
//       "CLIP_SOFT": "S",
//       "CLIP_HARD": "H",
//       "PAD": "P",
//       "SEQUENCE_MATCH": "=",
//       "SEQUENCE_MISMATCH": "X",
//     }
//     cigarStr = ""
//     for c in read.alignment.cigar {
//       cigarStr += c.operationLength + cigarMap[c.operation]
//     }
//     return cigarStr
type Read struct {
	// AlignedQuality: The quality of the read sequence contained in this
	// alignment record
	// (equivalent to QUAL in SAM).
	// `alignedSequence` and `alignedQuality` may be shorter than the full
	// read
	// sequence and quality. This will occur if the alignment is part of
	// a
	// chimeric alignment, or if the read was trimmed. When this occurs, the
	// CIGAR
	// for this read will begin/end with a hard clip operator that will
	// indicate
	// the length of the excised sequence.
	AlignedQuality []int64 `json:"alignedQuality,omitempty"`

	// AlignedSequence: The bases of the read sequence contained in this
	// alignment record,
	// **without CIGAR operations applied** (equivalent to SEQ in
	// SAM).
	// `alignedSequence` and `alignedQuality` may be
	// shorter than the full read sequence and quality. This will occur if
	// the
	// alignment is part of a chimeric alignment, or if the read was
	// trimmed. When
	// this occurs, the CIGAR for this read will begin/end with a hard
	// clip
	// operator that will indicate the length of the excised sequence.
	AlignedSequence string `json:"alignedSequence,omitempty"`

	// Alignment: The linear alignment for this alignment record. This field
	// is null for
	// unmapped reads.
	Alignment *LinearAlignment `json:"alignment,omitempty"`

	// DuplicateFragment: The fragment is a PCR or optical duplicate (SAM
	// flag 0x400).
	DuplicateFragment bool `json:"duplicateFragment,omitempty"`

	// FailedVendorQualityChecks: Whether this read did not pass filters,
	// such as platform or vendor quality
	// controls (SAM flag 0x200).
	FailedVendorQualityChecks bool `json:"failedVendorQualityChecks,omitempty"`

	// FragmentLength: The observed length of the fragment, equivalent to
	// TLEN in SAM.
	FragmentLength int64 `json:"fragmentLength,omitempty"`

	// FragmentName: The fragment name. Equivalent to QNAME (query template
	// name) in SAM.
	FragmentName string `json:"fragmentName,omitempty"`

	// Id: The server-generated read ID, unique across all reads. This is
	// different
	// from the `fragmentName`.
	Id string `json:"id,omitempty"`

	// Info: A map of additional read alignment information. This must be of
	// the form
	// map<string, string[]> (string key mapping to a list of string
	// values).
	Info map[string][]interface{} `json:"info,omitempty"`

	// NextMatePosition: The mapping of the primary alignment of
	// the
	// `(readNumber+1)%numberReads` read in the fragment. It replaces
	// mate position and mate strand in SAM.
	NextMatePosition *Position `json:"nextMatePosition,omitempty"`

	// NumberReads: The number of reads in the fragment (extension to SAM
	// flag 0x1).
	NumberReads int64 `json:"numberReads,omitempty"`

	// ProperPlacement: The orientation and the distance between reads from
	// the fragment are
	// consistent with the sequencing protocol (SAM flag 0x2).
	ProperPlacement bool `json:"properPlacement,omitempty"`

	// ReadGroupId: The ID of the read group this read belongs to. A read
	// belongs to exactly
	// one read group. This is a server-generated ID which is distinct from
	// SAM's
	// RG tag (for that value, see
	// ReadGroup.name).
	ReadGroupId string `json:"readGroupId,omitempty"`

	// ReadGroupSetId: The ID of the read group set this read belongs to. A
	// read belongs to
	// exactly one read group set.
	ReadGroupSetId string `json:"readGroupSetId,omitempty"`

	// ReadNumber: The read number in sequencing. 0-based and less than
	// numberReads. This
	// field replaces SAM flag 0x40 and 0x80.
	ReadNumber int64 `json:"readNumber,omitempty"`

	// SecondaryAlignment: Whether this alignment is secondary. Equivalent
	// to SAM flag 0x100.
	// A secondary alignment represents an alternative to the primary
	// alignment
	// for this read. Aligners may return secondary alignments if a read can
	// map
	// ambiguously to multiple coordinates in the genome. By convention,
	// each read
	// has one and only one alignment where both `secondaryAlignment`
	// and `supplementaryAlignment` are false.
	SecondaryAlignment bool `json:"secondaryAlignment,omitempty"`

	// SupplementaryAlignment: Whether this alignment is supplementary.
	// Equivalent to SAM flag 0x800.
	// Supplementary alignments are used in the representation of a
	// chimeric
	// alignment. In a chimeric alignment, a read is split into
	// multiple
	// linear alignments that map to different reference contigs. The
	// first
	// linear alignment in the read will be designated as the
	// representative
	// alignment; the remaining linear alignments will be designated
	// as
	// supplementary alignments. These alignments may have different
	// mapping
	// quality scores. In each linear alignment in a chimeric alignment, the
	// read
	// will be hard clipped. The `alignedSequence` and
	// `alignedQuality` fields in the alignment record will only
	// represent the bases for its respective linear alignment.
	SupplementaryAlignment bool `json:"supplementaryAlignment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AlignedQuality") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AlignedQuality") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Read) MarshalJSON() ([]byte, error) {
	type noMethod Read
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReadGroup: A read group is all the data that's processed the same way
// by the sequencer.
type ReadGroup struct {
	// DatasetId: The dataset to which this read group belongs.
	DatasetId string `json:"datasetId,omitempty"`

	// Description: A free-form text description of this read group.
	Description string `json:"description,omitempty"`

	// Experiment: The experiment used to generate this read group.
	Experiment *Experiment `json:"experiment,omitempty"`

	// Id: The server-generated read group ID, unique for all read
	// groups.
	// Note: This is different than the @RG ID field in the SAM spec. For
	// that
	// value, see name.
	Id string `json:"id,omitempty"`

	// Info: A map of additional read group information. This must be of the
	// form
	// map<string, string[]> (string key mapping to a list of string
	// values).
	Info map[string][]interface{} `json:"info,omitempty"`

	// Name: The read group name. This corresponds to the @RG ID field in
	// the SAM spec.
	Name string `json:"name,omitempty"`

	// PredictedInsertSize: The predicted insert size of this read group.
	// The insert size is the length
	// the sequenced DNA fragment from end-to-end, not including the
	// adapters.
	PredictedInsertSize int64 `json:"predictedInsertSize,omitempty"`

	// Programs: The programs used to generate this read group. Programs are
	// always
	// identical for all read groups within a read group set. For this
	// reason,
	// only the first read group in a returned set will have this
	// field
	// populated.
	Programs []*Program `json:"programs,omitempty"`

	// ReferenceSetId: The reference set the reads in this read group are
	// aligned to.
	ReferenceSetId string `json:"referenceSetId,omitempty"`

	// SampleId: A client-supplied sample identifier for the reads in this
	// read group.
	SampleId string `json:"sampleId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DatasetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReadGroup) MarshalJSON() ([]byte, error) {
	type noMethod ReadGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReadGroupSet: A read group set is a logical collection of read
// groups, which are
// collections of reads produced by a sequencer. A read group set
// typically
// models reads corresponding to one sample, sequenced one way, and
// aligned one
// way.
//
// * A read group set belongs to one dataset.
// * A read group belongs to one read group set.
// * A read belongs to one read group.
//
// For more genomics resource definitions, see [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
type ReadGroupSet struct {
	// DatasetId: The dataset to which this read group set belongs.
	DatasetId string `json:"datasetId,omitempty"`

	// Filename: The filename of the original source file for this read
	// group set, if any.
	Filename string `json:"filename,omitempty"`

	// Id: The server-generated read group set ID, unique for all read group
	// sets.
	Id string `json:"id,omitempty"`

	// Info: A map of additional read group set information.
	Info map[string][]interface{} `json:"info,omitempty"`

	// Name: The read group set name. By default this will be initialized to
	// the sample
	// name of the sequenced data contained in this set.
	Name string `json:"name,omitempty"`

	// ReadGroups: The read groups in this set. There are typically 1-10
	// read groups in a read
	// group set.
	ReadGroups []*ReadGroup `json:"readGroups,omitempty"`

	// ReferenceSetId: The reference set to which the reads in this read
	// group set are aligned.
	ReferenceSetId string `json:"referenceSetId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DatasetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReadGroupSet) MarshalJSON() ([]byte, error) {
	type noMethod ReadGroupSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Reference: A reference is a canonical assembled DNA sequence,
// intended to act as a
// reference coordinate space for other genomic annotations. A single
// reference
// might represent the human chromosome 1 or mitochandrial DNA, for
// instance. A
// reference belongs to one or more reference sets.
//
// For more genomics resource definitions, see [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
type Reference struct {
	// Id: The server-generated reference ID, unique across all references.
	Id string `json:"id,omitempty"`

	// Length: The length of this reference's sequence.
	Length int64 `json:"length,omitempty,string"`

	// Md5checksum: MD5 of the upper-case sequence excluding all whitespace
	// characters (this
	// is equivalent to SQ:M5 in SAM). This value is represented in lower
	// case
	// hexadecimal format.
	Md5checksum string `json:"md5checksum,omitempty"`

	// Name: The name of this reference, for example `22`.
	Name string `json:"name,omitempty"`

	// NcbiTaxonId: ID from http://www.ncbi.nlm.nih.gov/taxonomy. For
	// example, 9606 for human.
	NcbiTaxonId int64 `json:"ncbiTaxonId,omitempty"`

	// SourceAccessions: All known corresponding accession IDs in INSDC
	// (GenBank/ENA/DDBJ) ideally
	// with a version number, for example `GCF_000001405.26`.
	SourceAccessions []string `json:"sourceAccessions,omitempty"`

	// SourceUri: The URI from which the sequence was obtained. Typically
	// specifies a FASTA
	// format file.
	SourceUri string `json:"sourceUri,omitempty"`

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

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Reference) MarshalJSON() ([]byte, error) {
	type noMethod Reference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReferenceBound: ReferenceBound records an upper bound for the
// starting coordinate of
// variants in a particular reference.
type ReferenceBound struct {
	// ReferenceName: The name of the reference associated with this
	// reference bound.
	ReferenceName string `json:"referenceName,omitempty"`

	// UpperBound: An upper bound (inclusive) on the starting coordinate of
	// any
	// variant in the reference sequence.
	UpperBound int64 `json:"upperBound,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "ReferenceName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ReferenceName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReferenceBound) MarshalJSON() ([]byte, error) {
	type noMethod ReferenceBound
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReferenceSet: A reference set is a set of references which typically
// comprise a reference
// assembly for a species, such as `GRCh38` which is representative
// of the human genome. A reference set defines a common coordinate
// space for
// comparing reference-aligned experimental data. A reference set
// contains 1 or
// more references.
//
// For more genomics resource definitions, see [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
type ReferenceSet struct {
	// AssemblyId: Public id of this reference set, such as `GRCh37`.
	AssemblyId string `json:"assemblyId,omitempty"`

	// Description: Free text description of this reference set.
	Description string `json:"description,omitempty"`

	// Id: The server-generated reference set ID, unique across all
	// reference sets.
	Id string `json:"id,omitempty"`

	// Md5checksum: Order-independent MD5 checksum which identifies this
	// reference set. The
	// checksum is computed by sorting all lower case hexidecimal
	// string
	// `reference.md5checksum` (for all reference in this set) in
	// ascending lexicographic order, concatenating, and taking the MD5 of
	// that
	// value. The resulting value is represented in lower case hexadecimal
	// format.
	Md5checksum string `json:"md5checksum,omitempty"`

	// NcbiTaxonId: ID from http://www.ncbi.nlm.nih.gov/taxonomy (for
	// example, 9606 for human)
	// indicating the species which this reference set is intended to model.
	// Note
	// that contained references may specify a different `ncbiTaxonId`,
	// as
	// assemblies may contain reference sequences which do not belong to
	// the
	// modeled species, for example EBV in a human reference genome.
	NcbiTaxonId int64 `json:"ncbiTaxonId,omitempty"`

	// ReferenceIds: The IDs of the reference objects that are part of this
	// set.
	// `Reference.md5checksum` must be unique within this set.
	ReferenceIds []string `json:"referenceIds,omitempty"`

	// SourceAccessions: All known corresponding accession IDs in INSDC
	// (GenBank/ENA/DDBJ) ideally
	// with a version number, for example `NC_000001.11`.
	SourceAccessions []string `json:"sourceAccessions,omitempty"`

	// SourceUri: The URI from which the references were obtained.
	SourceUri string `json:"sourceUri,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AssemblyId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AssemblyId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ReferenceSet) MarshalJSON() ([]byte, error) {
	type noMethod ReferenceSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RuntimeMetadata: Runtime metadata that will be populated in
// the
// runtimeMetadata
// field of the Operation associated with a RunPipeline execution.
type RuntimeMetadata struct {
	// ComputeEngine: Execution information specific to Google Compute
	// Engine.
	ComputeEngine *ComputeEngine `json:"computeEngine,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ComputeEngine") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ComputeEngine") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RuntimeMetadata) MarshalJSON() ([]byte, error) {
	type noMethod RuntimeMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type SearchAnnotationSetsRequest struct {
	// DatasetIds: Required. The dataset IDs to search within. Caller must
	// have `READ` access
	// to these datasets.
	DatasetIds []string `json:"datasetIds,omitempty"`

	// Name: Only return annotations sets for which a substring of the name
	// matches this
	// string (case insensitive).
	Name string `json:"name,omitempty"`

	// PageSize: The maximum number of results to return in a single page.
	// If unspecified,
	// defaults to 128. The maximum value is 1024.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// ReferenceSetId: If specified, only annotation sets associated with
	// the given reference set
	// are returned.
	ReferenceSetId string `json:"referenceSetId,omitempty"`

	// Types: If specified, only annotation sets that have any of these
	// types are
	// returned.
	//
	// Possible values:
	//   "ANNOTATION_TYPE_UNSPECIFIED"
	//   "GENERIC" - A `GENERIC` annotation type should be used when no
	// other annotation
	// type will suffice. This represents an untyped annotation of the
	// reference
	// genome.
	//   "VARIANT" - A `VARIANT` annotation type.
	//   "GENE" - A `GENE` annotation type represents the existence of a
	// gene at the
	// associated reference coordinates. The start coordinate is typically
	// the
	// gene's transcription start site and the end is typically the end of
	// the
	// gene's last exon.
	//   "TRANSCRIPT" - A `TRANSCRIPT` annotation type represents the
	// assertion that a
	// particular region of the reference genome may be transcribed as RNA.
	Types []string `json:"types,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DatasetIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetIds") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchAnnotationSetsRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchAnnotationSetsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type SearchAnnotationSetsResponse struct {
	// AnnotationSets: The matching annotation sets.
	AnnotationSets []*AnnotationSet `json:"annotationSets,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AnnotationSets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationSets") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SearchAnnotationSetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchAnnotationSetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type SearchAnnotationsRequest struct {
	// AnnotationSetIds: Required. The annotation sets to search within. The
	// caller must have
	// `READ` access to these annotation sets.
	// All queried annotation sets must have the same type.
	AnnotationSetIds []string `json:"annotationSetIds,omitempty"`

	// End: The end position of the range on the reference, 0-based
	// exclusive. If
	// referenceId or
	// referenceName
	// must be specified, Defaults to the length of the reference.
	End int64 `json:"end,omitempty,string"`

	// PageSize: The maximum number of results to return in a single page.
	// If unspecified,
	// defaults to 256. The maximum value is 2048.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// ReferenceId: The ID of the reference to query.
	ReferenceId string `json:"referenceId,omitempty"`

	// ReferenceName: The name of the reference to query, within the
	// reference set associated
	// with this query.
	ReferenceName string `json:"referenceName,omitempty"`

	// Start: The start position of the range on the reference, 0-based
	// inclusive. If
	// specified,
	// referenceId or
	// referenceName
	// must be specified. Defaults to 0.
	Start int64 `json:"start,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "AnnotationSetIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationSetIds") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SearchAnnotationsRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchAnnotationsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type SearchAnnotationsResponse struct {
	// Annotations: The matching annotations.
	Annotations []*Annotation `json:"annotations,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Annotations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Annotations") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchAnnotationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchAnnotationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchCallSetsRequest: The call set search request.
type SearchCallSetsRequest struct {
	// Name: Only return call sets for which a substring of the name matches
	// this
	// string.
	Name string `json:"name,omitempty"`

	// PageSize: The maximum number of results to return in a single page.
	// If unspecified,
	// defaults to 1024.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// VariantSetIds: Restrict the query to call sets within the given
	// variant sets. At least one
	// ID must be provided.
	VariantSetIds []string `json:"variantSetIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Name") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchCallSetsRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchCallSetsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchCallSetsResponse: The call set search response.
type SearchCallSetsResponse struct {
	// CallSets: The list of matching call sets.
	CallSets []*CallSet `json:"callSets,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CallSets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CallSets") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchCallSetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchCallSetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchReadGroupSetsRequest: The read group set search request.
type SearchReadGroupSetsRequest struct {
	// DatasetIds: Restricts this query to read group sets within the given
	// datasets. At least
	// one ID must be provided.
	DatasetIds []string `json:"datasetIds,omitempty"`

	// Name: Only return read group sets for which a substring of the name
	// matches this
	// string.
	Name string `json:"name,omitempty"`

	// PageSize: The maximum number of results to return in a single page.
	// If unspecified,
	// defaults to 256. The maximum value is 1024.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DatasetIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetIds") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchReadGroupSetsRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchReadGroupSetsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchReadGroupSetsResponse: The read group set search response.
type SearchReadGroupSetsResponse struct {
	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ReadGroupSets: The list of matching read group sets.
	ReadGroupSets []*ReadGroupSet `json:"readGroupSets,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchReadGroupSetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchReadGroupSetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchReadsRequest: The read search request.
type SearchReadsRequest struct {
	// End: The end position of the range on the reference, 0-based
	// exclusive. If
	// specified, `referenceName` must also be specified.
	End int64 `json:"end,omitempty,string"`

	// PageSize: The maximum number of results to return in a single page.
	// If unspecified,
	// defaults to 256. The maximum value is 2048.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// ReadGroupIds: The IDs of the read groups within which to search for
	// reads. All specified
	// read groups must belong to the same read group sets. Must specify one
	// of
	// `readGroupSetIds` or `readGroupIds`.
	ReadGroupIds []string `json:"readGroupIds,omitempty"`

	// ReadGroupSetIds: The IDs of the read groups sets within which to
	// search for reads. All
	// specified read group sets must be aligned against a common set of
	// reference
	// sequences; this defines the genomic coordinates for the query. Must
	// specify
	// one of `readGroupSetIds` or `readGroupIds`.
	ReadGroupSetIds []string `json:"readGroupSetIds,omitempty"`

	// ReferenceName: The reference sequence name, for example `chr1`, `1`,
	// or `chrX`. If set to
	// `*`, only unmapped reads are returned. If unspecified, all reads
	// (mapped
	// and unmapped) are returned.
	ReferenceName string `json:"referenceName,omitempty"`

	// Start: The start position of the range on the reference, 0-based
	// inclusive. If
	// specified, `referenceName` must also be specified.
	Start int64 `json:"start,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "End") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "End") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchReadsRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchReadsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchReadsResponse: The read search response.
type SearchReadsResponse struct {
	// Alignments: The list of matching alignments sorted by mapped genomic
	// coordinate,
	// if any, ascending in position within the same reference. Unmapped
	// reads,
	// which have no position, are returned contiguously and are sorted
	// in
	// ascending lexicographic order by fragment name.
	Alignments []*Read `json:"alignments,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Alignments") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Alignments") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchReadsResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchReadsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type SearchReferenceSetsRequest struct {
	// Accessions: If present, return reference sets for which a prefix of
	// any of
	// sourceAccessions
	// match any of these strings. Accession numbers typically have a main
	// number
	// and a version, for example `NC_000001.11`.
	Accessions []string `json:"accessions,omitempty"`

	// AssemblyId: If present, return reference sets for which a substring
	// of their
	// `assemblyId` matches this string (case insensitive).
	AssemblyId string `json:"assemblyId,omitempty"`

	// Md5checksums: If present, return reference sets for which
	// the
	// md5checksum matches exactly.
	Md5checksums []string `json:"md5checksums,omitempty"`

	// PageSize: The maximum number of results to return in a single page.
	// If unspecified,
	// defaults to 1024. The maximum value is 4096.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Accessions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Accessions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchReferenceSetsRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchReferenceSetsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type SearchReferenceSetsResponse struct {
	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ReferenceSets: The matching references sets.
	ReferenceSets []*ReferenceSet `json:"referenceSets,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchReferenceSetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchReferenceSetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type SearchReferencesRequest struct {
	// Accessions: If present, return references for which a prefix of any
	// of
	// sourceAccessions match
	// any of these strings. Accession numbers typically have a main number
	// and a
	// version, for example `GCF_000001405.26`.
	Accessions []string `json:"accessions,omitempty"`

	// Md5checksums: If present, return references for which the
	// md5checksum matches exactly.
	Md5checksums []string `json:"md5checksums,omitempty"`

	// PageSize: The maximum number of results to return in a single page.
	// If unspecified,
	// defaults to 1024. The maximum value is 4096.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// ReferenceSetId: If present, return only references which belong to
	// this reference set.
	ReferenceSetId string `json:"referenceSetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Accessions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Accessions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchReferencesRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchReferencesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type SearchReferencesResponse struct {
	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// References: The matching references.
	References []*Reference `json:"references,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchReferencesResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchReferencesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchVariantSetsRequest: The search variant sets request.
type SearchVariantSetsRequest struct {
	// DatasetIds: Exactly one dataset ID must be provided here. Only
	// variant sets which
	// belong to this dataset will be returned.
	DatasetIds []string `json:"datasetIds,omitempty"`

	// PageSize: The maximum number of results to return in a single page.
	// If unspecified,
	// defaults to 1024.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DatasetIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetIds") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchVariantSetsRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchVariantSetsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchVariantSetsResponse: The search variant sets response.
type SearchVariantSetsResponse struct {
	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// VariantSets: The variant sets belonging to the requested dataset.
	VariantSets []*VariantSet `json:"variantSets,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchVariantSetsResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchVariantSetsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchVariantsRequest: The variant search request.
type SearchVariantsRequest struct {
	// CallSetIds: Only return variant calls which belong to call sets with
	// these ids.
	// Leaving this blank returns all variant calls. If a variant has
	// no
	// calls belonging to any of these call sets, it won't be returned at
	// all.
	CallSetIds []string `json:"callSetIds,omitempty"`

	// End: The end of the window, 0-based exclusive. If unspecified or 0,
	// defaults to
	// the length of the reference.
	End int64 `json:"end,omitempty,string"`

	// MaxCalls: The maximum number of calls to return in a single page.
	// Note that this
	// limit may be exceeded in the event that a matching variant contains
	// more
	// calls than the requested maximum. If unspecified, defaults to 5000.
	// The
	// maximum value is 10000.
	MaxCalls int64 `json:"maxCalls,omitempty"`

	// PageSize: The maximum number of variants to return in a single page.
	// If unspecified,
	// defaults to 5000. The maximum value is 10000.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: The continuation token, which is used to page through
	// large result sets.
	// To get the next page of results, set this parameter to the value
	// of
	// `nextPageToken` from the previous response.
	PageToken string `json:"pageToken,omitempty"`

	// ReferenceName: Required. Only return variants in this reference
	// sequence.
	ReferenceName string `json:"referenceName,omitempty"`

	// Start: The beginning of the window (0-based, inclusive) for
	// which
	// overlapping variants should be returned. If unspecified, defaults to
	// 0.
	Start int64 `json:"start,omitempty,string"`

	// VariantName: Only return variants which have exactly this name.
	VariantName string `json:"variantName,omitempty"`

	// VariantSetIds: At most one variant set ID must be provided. Only
	// variants from this
	// variant set will be returned. If omitted, a call set id must be
	// included in
	// the request.
	VariantSetIds []string `json:"variantSetIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CallSetIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CallSetIds") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchVariantsRequest) MarshalJSON() ([]byte, error) {
	type noMethod SearchVariantsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SearchVariantsResponse: The variant search response.
type SearchVariantsResponse struct {
	// NextPageToken: The continuation token, which is used to page through
	// large result sets.
	// Provide this value in a subsequent request to return the next page
	// of
	// results. This field will be empty if there aren't any additional
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Variants: The list of matching Variants.
	Variants []*Variant `json:"variants,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SearchVariantsResponse) MarshalJSON() ([]byte, error) {
	type noMethod SearchVariantsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SetIamPolicyRequest: Request message for `SetIamPolicy` method.
type SetIamPolicyRequest struct {
	// Policy: REQUIRED: The complete policy to be applied to the
	// `resource`. The size of
	// the policy is limited to a few 10s of KB. An empty policy is a
	// valid policy but certain Cloud Platform services (such as
	// Projects)
	// might reject them.
	Policy *Policy `json:"policy,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Policy") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Policy") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SetIamPolicyRequest) MarshalJSON() ([]byte, error) {
	type noMethod SetIamPolicyRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Status: The `Status` type defines a logical error model that is
// suitable for different
// programming environments, including REST APIs and RPC APIs. It is
// used by
// [gRPC](https://github.com/grpc). The error model is designed to
// be:
//
// - Simple to use and understand for most users
// - Flexible enough to meet unexpected needs
//
// # Overview
//
// The `Status` message contains three pieces of data: error code, error
// message,
// and error details. The error code should be an enum value
// of
// google.rpc.Code, but it may accept additional error codes if needed.
// The
// error message should be a developer-facing English message that
// helps
// developers *understand* and *resolve* the error. If a localized
// user-facing
// error message is needed, put the localized message in the error
// details or
// localize it in the client. The optional error details may contain
// arbitrary
// information about the error. There is a predefined set of error
// detail types
// in the package `google.rpc` that can be used for common error
// conditions.
//
// # Language mapping
//
// The `Status` message is the logical representation of the error
// model, but it
// is not necessarily the actual wire format. When the `Status` message
// is
// exposed in different client libraries and different wire protocols,
// it can be
// mapped differently. For example, it will likely be mapped to some
// exceptions
// in Java, but more likely mapped to some error codes in C.
//
// # Other uses
//
// The error model and the `Status` message can be used in a variety
// of
// environments, either with or without APIs, to provide a
// consistent developer experience across different
// environments.
//
// Example uses of this error model include:
//
// - Partial errors. If a service needs to return partial errors to the
// client,
//     it may embed the `Status` in the normal response to indicate the
// partial
//     errors.
//
// - Workflow errors. A typical workflow has multiple steps. Each step
// may
//     have a `Status` message for error reporting.
//
// - Batch operations. If a client uses batch request and batch
// response, the
//     `Status` message should be used directly inside batch response,
// one for
//     each error sub-response.
//
// - Asynchronous operations. If an API call embeds asynchronous
// operation
//     results in its response, the status of those operations should
// be
//     represented directly using the `Status` message.
//
// - Logging. If some API errors are stored in logs, the message
// `Status` could
//     be used directly after any stripping needed for security/privacy
// reasons.
type Status struct {
	// Code: The status code, which should be an enum value of
	// google.rpc.Code.
	Code int64 `json:"code,omitempty"`

	// Details: A list of messages that carry the error details.  There is a
	// common set of
	// message types for APIs to use.
	Details []googleapi.RawMessage `json:"details,omitempty"`

	// Message: A developer-facing error message, which should be in
	// English. Any
	// user-facing error message should be localized and sent in
	// the
	// google.rpc.Status.details field, or localized by the client.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Status) MarshalJSON() ([]byte, error) {
	type noMethod Status
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestIamPermissionsRequest: Request message for `TestIamPermissions`
// method.
type TestIamPermissionsRequest struct {
	// Permissions: REQUIRED: The set of permissions to check for the
	// 'resource'.
	// Permissions with wildcards (such as '*' or 'storage.*') are not
	// allowed.
	// Allowed permissions are&#58;
	//
	// * `genomics.datasets.create`
	// * `genomics.datasets.delete`
	// * `genomics.datasets.get`
	// * `genomics.datasets.list`
	// * `genomics.datasets.update`
	// * `genomics.datasets.getIamPolicy`
	// * `genomics.datasets.setIamPolicy`
	Permissions []string `json:"permissions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Permissions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Permissions") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestIamPermissionsRequest) MarshalJSON() ([]byte, error) {
	type noMethod TestIamPermissionsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TestIamPermissionsResponse: Response message for `TestIamPermissions`
// method.
type TestIamPermissionsResponse struct {
	// Permissions: A subset of `TestPermissionsRequest.permissions` that
	// the caller is
	// allowed.
	Permissions []string `json:"permissions,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Permissions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Permissions") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TestIamPermissionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod TestIamPermissionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Transcript: A transcript represents the assertion that a particular
// region of the
// reference genome may be transcribed as RNA.
type Transcript struct {
	// CodingSequence: The range of the coding sequence for this transcript,
	// if any. To determine
	// the exact ranges of coding sequence, intersect this range with those
	// of the
	// exons, if any. If there are any
	// exons, the
	// codingSequence must start
	// and end within them.
	//
	// Note that in some cases, the reference genome will not exactly match
	// the
	// observed mRNA transcript e.g. due to variance in the source genome
	// from
	// reference. In these cases,
	// exon.frame will not necessarily
	// match the expected reference reading frame and coding exon reference
	// bases
	// cannot necessarily be concatenated to produce the original transcript
	// mRNA.
	CodingSequence *CodingSequence `json:"codingSequence,omitempty"`

	// Exons: The <a href="http://en.wikipedia.org/wiki/Exon">exons</a> that
	// compose
	// this transcript. This field should be unset for genomes where
	// transcript
	// splicing does not occur, for example prokaryotes.
	//
	// Introns are regions of the transcript that are not included in
	// the
	// spliced RNA product. Though not explicitly modeled here, intron
	// ranges can
	// be deduced; all regions of this transcript that are not exons are
	// introns.
	//
	// Exonic sequences do not necessarily code for a translational
	// product
	// (amino acids). Only the regions of exons bounded by
	// the
	// codingSequence correspond
	// to coding DNA sequence.
	//
	// Exons are ordered by start position and may not overlap.
	Exons []*Exon `json:"exons,omitempty"`

	// GeneId: The annotation ID of the gene from which this transcript is
	// transcribed.
	GeneId string `json:"geneId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CodingSequence") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CodingSequence") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Transcript) MarshalJSON() ([]byte, error) {
	type noMethod Transcript
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type UndeleteDatasetRequest struct {
}

// Variant: A variant represents a change in DNA sequence relative to a
// reference
// sequence. For example, a variant could represent a SNP or an
// insertion.
// Variants belong to a variant set.
//
// For more genomics resource definitions, see [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Each of the calls on a variant represent a determination of genotype
// with
// respect to that variant. For example, a call might assign probability
// of 0.32
// to the occurrence of a SNP named rs1234 in a sample named NA12345. A
// call
// belongs to a call set, which contains related calls typically from
// one
// sample.
type Variant struct {
	// AlternateBases: The bases that appear instead of the reference bases.
	AlternateBases []string `json:"alternateBases,omitempty"`

	// Calls: The variant calls for this particular variant. Each one
	// represents the
	// determination of genotype with respect to this variant.
	Calls []*VariantCall `json:"calls,omitempty"`

	// Created: The date this variant was created, in milliseconds from the
	// epoch.
	Created int64 `json:"created,omitempty,string"`

	// End: The end position (0-based) of this variant. This corresponds to
	// the first
	// base after the last base in the reference allele. So, the length
	// of
	// the reference allele is (end - start). This is useful for
	// variants
	// that don't explicitly give alternate bases, for example large
	// deletions.
	End int64 `json:"end,omitempty,string"`

	// Filter: A list of filters (normally quality filters) this variant has
	// failed.
	// `PASS` indicates this variant has passed all filters.
	Filter []string `json:"filter,omitempty"`

	// Id: The server-generated variant ID, unique across all variants.
	Id string `json:"id,omitempty"`

	// Info: A map of additional variant information. This must be of the
	// form
	// map<string, string[]> (string key mapping to a list of string
	// values).
	Info map[string][]interface{} `json:"info,omitempty"`

	// Names: Names for the variant, for example a RefSNP ID.
	Names []string `json:"names,omitempty"`

	// Quality: A measure of how likely this variant is to be real.
	// A higher value is better.
	Quality float64 `json:"quality,omitempty"`

	// ReferenceBases: The reference bases for this variant. They start at
	// the given
	// position.
	ReferenceBases string `json:"referenceBases,omitempty"`

	// ReferenceName: The reference on which this variant occurs.
	// (such as `chr20` or `X`)
	ReferenceName string `json:"referenceName,omitempty"`

	// Start: The position at which this variant occurs (0-based).
	// This corresponds to the first base of the string of reference bases.
	Start int64 `json:"start,omitempty,string"`

	// VariantSetId: The ID of the variant set this variant belongs to.
	VariantSetId string `json:"variantSetId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AlternateBases") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AlternateBases") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Variant) MarshalJSON() ([]byte, error) {
	type noMethod Variant
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Variant) UnmarshalJSON(data []byte) error {
	type noMethod Variant
	var s1 struct {
		Quality gensupport.JSONFloat64 `json:"quality"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Quality = float64(s1.Quality)
	return nil
}

type VariantAnnotation struct {
	// AlternateBases: The alternate allele for this variant. If multiple
	// alternate alleles
	// exist at this location, create a separate variant for each one, as
	// they
	// may represent distinct conditions.
	AlternateBases string `json:"alternateBases,omitempty"`

	// ClinicalSignificance: Describes the clinical significance of a
	// variant.
	// It is adapted from the ClinVar controlled vocabulary for
	// clinical
	// significance described
	// at:
	// http://www.ncbi.nlm.nih.gov/clinvar/docs/clinsig/
	//
	// Possible values:
	//   "CLINICAL_SIGNIFICANCE_UNSPECIFIED"
	//   "CLINICAL_SIGNIFICANCE_OTHER" - `OTHER` should be used when no
	// other clinical significance
	// value will suffice.
	//   "UNCERTAIN"
	//   "BENIGN"
	//   "LIKELY_BENIGN"
	//   "LIKELY_PATHOGENIC"
	//   "PATHOGENIC"
	//   "DRUG_RESPONSE"
	//   "HISTOCOMPATIBILITY"
	//   "CONFERS_SENSITIVITY"
	//   "RISK_FACTOR"
	//   "ASSOCIATION"
	//   "PROTECTIVE"
	//   "MULTIPLE_REPORTED" - `MULTIPLE_REPORTED` should be used when
	// multiple clinical
	// signficances are reported for a variant. The original
	// clinical
	// significance values may be provided in the `info` field.
	ClinicalSignificance string `json:"clinicalSignificance,omitempty"`

	// Conditions: The set of conditions associated with this variant.
	// A condition describes the way a variant influences human health.
	Conditions []*ClinicalCondition `json:"conditions,omitempty"`

	// Effect: Effect of the variant on the coding sequence.
	//
	// Possible values:
	//   "EFFECT_UNSPECIFIED"
	//   "EFFECT_OTHER" - `EFFECT_OTHER` should be used when no other
	// Effect
	// will suffice.
	//   "FRAMESHIFT" - `FRAMESHIFT` indicates a mutation in which the
	// insertion or
	// deletion of nucleotides resulted in a frameshift change.
	//   "FRAME_PRESERVING_INDEL" - `FRAME_PRESERVING_INDEL` indicates a
	// mutation in which a
	// multiple of three nucleotides has been inserted or deleted,
	// resulting
	// in no change to the reading frame of the coding sequence.
	//   "SYNONYMOUS_SNP" - `SYNONYMOUS_SNP` indicates a single nucleotide
	// polymorphism
	// mutation that results in no amino acid change.
	//   "NONSYNONYMOUS_SNP" - `NONSYNONYMOUS_SNP` indicates a single
	// nucleotide
	// polymorphism mutation that results in an amino acid change.
	//   "STOP_GAIN" - `STOP_GAIN` indicates a mutation that leads to the
	// creation
	// of a stop codon at the variant site. Frameshift mutations
	// creating
	// downstream stop codons do not count as `STOP_GAIN`.
	//   "STOP_LOSS" - `STOP_LOSS` indicates a mutation that eliminates
	// a
	// stop codon at the variant site.
	//   "SPLICE_SITE_DISRUPTION" - `SPLICE_SITE_DISRUPTION` indicates that
	// this variant is
	// found in a splice site for the associated transcript, and alters
	// the
	// normal splicing pattern.
	Effect string `json:"effect,omitempty"`

	// GeneId: Google annotation ID of the gene affected by this variant.
	// This should
	// be provided when the variant is created.
	GeneId string `json:"geneId,omitempty"`

	// TranscriptIds: Google annotation IDs of the transcripts affected by
	// this variant. These
	// should be provided when the variant is created.
	TranscriptIds []string `json:"transcriptIds,omitempty"`

	// Type: Type has been adapted from ClinVar's list of variant types.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED"
	//   "TYPE_OTHER" - `TYPE_OTHER` should be used when no other Type will
	// suffice.
	// Further explanation of the variant type may be included in the
	// info field.
	//   "INSERTION" - `INSERTION` indicates an insertion.
	//   "DELETION" - `DELETION` indicates a deletion.
	//   "SUBSTITUTION" - `SUBSTITUTION` indicates a block substitution
	// of
	// two or more nucleotides.
	//   "SNP" - `SNP` indicates a single nucleotide polymorphism.
	//   "STRUCTURAL" - `STRUCTURAL` indicates a large structural
	// variant,
	// including chromosomal fusions, inversions, etc.
	//   "CNV" - `CNV` indicates a variation in copy number.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AlternateBases") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AlternateBases") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *VariantAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod VariantAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// VariantCall: A call represents the determination of genotype with
// respect to a particular
// variant. It may include associated information such as quality and
// phasing.
// For example, a call might assign a probability of 0.32 to the
// occurrence of
// a SNP named rs1234 in a call set with the name NA12345.
type VariantCall struct {
	// CallSetId: The ID of the call set this variant call belongs to.
	CallSetId string `json:"callSetId,omitempty"`

	// CallSetName: The name of the call set this variant call belongs to.
	CallSetName string `json:"callSetName,omitempty"`

	// Genotype: The genotype of this variant call. Each value represents
	// either the value
	// of the `referenceBases` field or a 1-based index
	// into
	// `alternateBases`. If a variant had a `referenceBases`
	// value of `T` and an `alternateBases`
	// value of `["A", "C"]`, and the `genotype` was
	// `[2, 1]`, that would mean the call
	// represented the heterozygous value `CA` for this variant.
	// If the `genotype` was instead `[0, 1]`, the
	// represented value would be `TA`. Ordering of the
	// genotype values is important if the `phaseset` is present.
	// If a genotype is not called (that is, a `.` is present in the
	// GT string) -1 is returned.
	Genotype []int64 `json:"genotype,omitempty"`

	// GenotypeLikelihood: The genotype likelihoods for this variant call.
	// Each array entry
	// represents how likely a specific genotype is for this call. The
	// value
	// ordering is defined by the GL tag in the VCF spec.
	// If Phred-scaled genotype likelihood scores (PL) are available
	// and
	// log10(P) genotype likelihood scores (GL) are not, PL scores are
	// converted
	// to GL scores.  If both are available, PL scores are stored in `info`.
	GenotypeLikelihood []float64 `json:"genotypeLikelihood,omitempty"`

	// Info: A map of additional variant call information. This must be of
	// the form
	// map<string, string[]> (string key mapping to a list of string
	// values).
	Info map[string][]interface{} `json:"info,omitempty"`

	// Phaseset: If this field is present, this variant call's genotype
	// ordering implies
	// the phase of the bases and is consistent with any other variant calls
	// in
	// the same reference sequence which have the same phaseset value.
	// When importing data from VCF, if the genotype data was phased but
	// no
	// phase set was specified this field will be set to `*`.
	Phaseset string `json:"phaseset,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CallSetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CallSetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *VariantCall) MarshalJSON() ([]byte, error) {
	type noMethod VariantCall
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// VariantSet: A variant set is a collection of call sets and variants.
// It contains summary
// statistics of those contents. A variant set belongs to a
// dataset.
//
// For more genomics resource definitions, see [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
type VariantSet struct {
	// DatasetId: The dataset to which this variant set belongs.
	DatasetId string `json:"datasetId,omitempty"`

	// Description: A textual description of this variant set.
	Description string `json:"description,omitempty"`

	// Id: The server-generated variant set ID, unique across all variant
	// sets.
	Id string `json:"id,omitempty"`

	// Metadata: The metadata associated with this variant set.
	Metadata []*VariantSetMetadata `json:"metadata,omitempty"`

	// Name: User-specified, mutable name.
	Name string `json:"name,omitempty"`

	// ReferenceBounds: A list of all references used by the variants in a
	// variant set
	// with associated coordinate upper bounds for each one.
	ReferenceBounds []*ReferenceBound `json:"referenceBounds,omitempty"`

	// ReferenceSetId: The reference set to which the variant set is mapped.
	// The reference set
	// describes the alignment provenance of the variant set, while
	// the
	// `referenceBounds` describe the shape of the actual variant data.
	// The
	// reference set's reference names are a superset of those found in
	// the
	// `referenceBounds`.
	//
	// For example, given a variant set that is mapped to the GRCh38
	// reference set
	// and contains a single variant on reference 'X', `referenceBounds`
	// would
	// contain only an entry for 'X', while the associated reference
	// set
	// enumerates all possible references: '1', '2', 'X', 'Y', 'MT', etc.
	ReferenceSetId string `json:"referenceSetId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "DatasetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DatasetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *VariantSet) MarshalJSON() ([]byte, error) {
	type noMethod VariantSet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// VariantSetMetadata: Metadata describes a single piece of variant call
// metadata.
// These data include a top level key and either a single value string
// (value)
// or a list of key-value pairs (info.)
// Value and info are mutually exclusive.
type VariantSetMetadata struct {
	// Description: A textual description of this metadata.
	Description string `json:"description,omitempty"`

	// Id: User-provided ID field, not enforced by this API.
	// Two or more pieces of structured metadata with identical
	// id and key fields are considered equivalent.
	Id string `json:"id,omitempty"`

	// Info: Remaining structured metadata key-value pairs. This must be of
	// the form
	// map<string, string[]> (string key mapping to a list of string
	// values).
	Info map[string][]interface{} `json:"info,omitempty"`

	// Key: The top-level key.
	Key string `json:"key,omitempty"`

	// Number: The number of values that can be included in a field
	// described by this
	// metadata.
	Number string `json:"number,omitempty"`

	// Type: The type of data. Possible types include: Integer, Float,
	// Flag, Character, and String.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED"
	//   "INTEGER"
	//   "FLOAT"
	//   "FLAG"
	//   "CHARACTER"
	//   "STRING"
	Type string `json:"type,omitempty"`

	// Value: The value field for simple metadata
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Description") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Description") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *VariantSetMetadata) MarshalJSON() ([]byte, error) {
	type noMethod VariantSetMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "genomics.annotations.batchCreate":

type AnnotationsBatchCreateCall struct {
	s                             *Service
	batchcreateannotationsrequest *BatchCreateAnnotationsRequest
	urlParams_                    gensupport.URLParams
	ctx_                          context.Context
	header_                       http.Header
}

// BatchCreate: Creates one or more new annotations atomically. All
// annotations must
// belong to the same annotation set. Caller must have WRITE
// permission for this annotation set. For optimal performance,
// batch
// positionally adjacent annotations together.
//
// If the request has a systemic issue, such as an attempt to write
// to
// an inaccessible annotation set, the entire RPC will fail accordingly.
// For
// lesser data issues, when possible an error will be isolated to
// the
// corresponding batch entry in the response; the remaining well
// formed
// annotations will be created normally.
//
// For details on the requirements for each individual annotation
// resource,
// see
// CreateAnnotation.
func (r *AnnotationsService) BatchCreate(batchcreateannotationsrequest *BatchCreateAnnotationsRequest) *AnnotationsBatchCreateCall {
	c := &AnnotationsBatchCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.batchcreateannotationsrequest = batchcreateannotationsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsBatchCreateCall) Fields(s ...googleapi.Field) *AnnotationsBatchCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsBatchCreateCall) Context(ctx context.Context) *AnnotationsBatchCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsBatchCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsBatchCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.batchcreateannotationsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotations:batchCreate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotations.batchCreate" call.
// Exactly one of *BatchCreateAnnotationsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *BatchCreateAnnotationsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AnnotationsBatchCreateCall) Do(opts ...googleapi.CallOption) (*BatchCreateAnnotationsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &BatchCreateAnnotationsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates one or more new annotations atomically. All annotations must\nbelong to the same annotation set. Caller must have WRITE\npermission for this annotation set. For optimal performance, batch\npositionally adjacent annotations together.\n\nIf the request has a systemic issue, such as an attempt to write to\nan inaccessible annotation set, the entire RPC will fail accordingly. For\nlesser data issues, when possible an error will be isolated to the\ncorresponding batch entry in the response; the remaining well formed\nannotations will be created normally.\n\nFor details on the requirements for each individual annotation resource,\nsee\nCreateAnnotation.",
	//   "flatPath": "v1/annotations:batchCreate",
	//   "httpMethod": "POST",
	//   "id": "genomics.annotations.batchCreate",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/annotations:batchCreate",
	//   "request": {
	//     "$ref": "BatchCreateAnnotationsRequest"
	//   },
	//   "response": {
	//     "$ref": "BatchCreateAnnotationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.annotations.create":

type AnnotationsCreateCall struct {
	s          *Service
	annotation *Annotation
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new annotation. Caller must have WRITE
// permission
// for the associated annotation set.
//
// The following fields are required:
//
// * annotationSetId
// * referenceName or
//   referenceId
//
// ### Transcripts
//
// For annotations of type TRANSCRIPT, the following fields
// of
// transcript must be provided:
//
// * exons.start
// * exons.end
//
// All other fields may be optionally specified, unless documented as
// being
// server-generated (for example, the `id` field). The annotated
// range must be no longer than 100Mbp (mega base pairs). See
// the
// Annotation resource
// for additional restrictions on each field.
func (r *AnnotationsService) Create(annotation *Annotation) *AnnotationsCreateCall {
	c := &AnnotationsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.annotation = annotation
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsCreateCall) Fields(s ...googleapi.Field) *AnnotationsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsCreateCall) Context(ctx context.Context) *AnnotationsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.annotation)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotations.create" call.
// Exactly one of *Annotation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Annotation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AnnotationsCreateCall) Do(opts ...googleapi.CallOption) (*Annotation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Annotation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new annotation. Caller must have WRITE permission\nfor the associated annotation set.\n\nThe following fields are required:\n\n* annotationSetId\n* referenceName or\n  referenceId\n\n### Transcripts\n\nFor annotations of type TRANSCRIPT, the following fields of\ntranscript must be provided:\n\n* exons.start\n* exons.end\n\nAll other fields may be optionally specified, unless documented as being\nserver-generated (for example, the `id` field). The annotated\nrange must be no longer than 100Mbp (mega base pairs). See the\nAnnotation resource\nfor additional restrictions on each field.",
	//   "flatPath": "v1/annotations",
	//   "httpMethod": "POST",
	//   "id": "genomics.annotations.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/annotations",
	//   "request": {
	//     "$ref": "Annotation"
	//   },
	//   "response": {
	//     "$ref": "Annotation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.annotations.delete":

type AnnotationsDeleteCall struct {
	s            *Service
	annotationId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Deletes an annotation. Caller must have WRITE permission
// for
// the associated annotation set.
func (r *AnnotationsService) Delete(annotationId string) *AnnotationsDeleteCall {
	c := &AnnotationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.annotationId = annotationId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsDeleteCall) Fields(s ...googleapi.Field) *AnnotationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsDeleteCall) Context(ctx context.Context) *AnnotationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotations/{annotationId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"annotationId": c.annotationId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AnnotationsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes an annotation. Caller must have WRITE permission for\nthe associated annotation set.",
	//   "flatPath": "v1/annotations/{annotationId}",
	//   "httpMethod": "DELETE",
	//   "id": "genomics.annotations.delete",
	//   "parameterOrder": [
	//     "annotationId"
	//   ],
	//   "parameters": {
	//     "annotationId": {
	//       "description": "The ID of the annotation to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/annotations/{annotationId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.annotations.get":

type AnnotationsGetCall struct {
	s            *Service
	annotationId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets an annotation. Caller must have READ permission
// for the associated annotation set.
func (r *AnnotationsService) Get(annotationId string) *AnnotationsGetCall {
	c := &AnnotationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.annotationId = annotationId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsGetCall) Fields(s ...googleapi.Field) *AnnotationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AnnotationsGetCall) IfNoneMatch(entityTag string) *AnnotationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsGetCall) Context(ctx context.Context) *AnnotationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotations/{annotationId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"annotationId": c.annotationId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotations.get" call.
// Exactly one of *Annotation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Annotation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AnnotationsGetCall) Do(opts ...googleapi.CallOption) (*Annotation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Annotation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets an annotation. Caller must have READ permission\nfor the associated annotation set.",
	//   "flatPath": "v1/annotations/{annotationId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.annotations.get",
	//   "parameterOrder": [
	//     "annotationId"
	//   ],
	//   "parameters": {
	//     "annotationId": {
	//       "description": "The ID of the annotation to be retrieved.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/annotations/{annotationId}",
	//   "response": {
	//     "$ref": "Annotation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.annotations.search":

type AnnotationsSearchCall struct {
	s                        *Service
	searchannotationsrequest *SearchAnnotationsRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// Search: Searches for annotations that match the given criteria.
// Results are
// ordered by genomic coordinate (by reference sequence, then
// position).
// Annotations with equivalent genomic coordinates are returned in
// an
// unspecified order. This order is consistent, such that two queries
// for the
// same content (regardless of page size) yield annotations in the same
// order
// across their respective streams of paginated responses. Caller must
// have
// READ permission for the queried annotation sets.
func (r *AnnotationsService) Search(searchannotationsrequest *SearchAnnotationsRequest) *AnnotationsSearchCall {
	c := &AnnotationsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchannotationsrequest = searchannotationsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsSearchCall) Fields(s ...googleapi.Field) *AnnotationsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsSearchCall) Context(ctx context.Context) *AnnotationsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchannotationsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotations/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotations.search" call.
// Exactly one of *SearchAnnotationsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *SearchAnnotationsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AnnotationsSearchCall) Do(opts ...googleapi.CallOption) (*SearchAnnotationsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchAnnotationsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Searches for annotations that match the given criteria. Results are\nordered by genomic coordinate (by reference sequence, then position).\nAnnotations with equivalent genomic coordinates are returned in an\nunspecified order. This order is consistent, such that two queries for the\nsame content (regardless of page size) yield annotations in the same order\nacross their respective streams of paginated responses. Caller must have\nREAD permission for the queried annotation sets.",
	//   "flatPath": "v1/annotations/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.annotations.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/annotations/search",
	//   "request": {
	//     "$ref": "SearchAnnotationsRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchAnnotationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AnnotationsSearchCall) Pages(ctx context.Context, f func(*SearchAnnotationsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchannotationsrequest.PageToken = pt }(c.searchannotationsrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchannotationsrequest.PageToken = x.NextPageToken
	}
}

// method id "genomics.annotations.update":

type AnnotationsUpdateCall struct {
	s            *Service
	annotationId string
	annotation   *Annotation
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Update: Updates an annotation. Caller must have
// WRITE permission for the associated dataset.
func (r *AnnotationsService) Update(annotationId string, annotation *Annotation) *AnnotationsUpdateCall {
	c := &AnnotationsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.annotationId = annotationId
	c.annotation = annotation
	return c
}

// UpdateMask sets the optional parameter "updateMask": An optional mask
// specifying which fields to update. Mutable fields
// are
// name,
// variant,
// transcript, and
// info. If unspecified, all mutable
// fields will be updated.
func (c *AnnotationsUpdateCall) UpdateMask(updateMask string) *AnnotationsUpdateCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsUpdateCall) Fields(s ...googleapi.Field) *AnnotationsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsUpdateCall) Context(ctx context.Context) *AnnotationsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.annotation)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotations/{annotationId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"annotationId": c.annotationId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotations.update" call.
// Exactly one of *Annotation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Annotation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AnnotationsUpdateCall) Do(opts ...googleapi.CallOption) (*Annotation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Annotation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an annotation. Caller must have\nWRITE permission for the associated dataset.",
	//   "flatPath": "v1/annotations/{annotationId}",
	//   "httpMethod": "PUT",
	//   "id": "genomics.annotations.update",
	//   "parameterOrder": [
	//     "annotationId"
	//   ],
	//   "parameters": {
	//     "annotationId": {
	//       "description": "The ID of the annotation to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "An optional mask specifying which fields to update. Mutable fields are\nname,\nvariant,\ntranscript, and\ninfo. If unspecified, all mutable\nfields will be updated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/annotations/{annotationId}",
	//   "request": {
	//     "$ref": "Annotation"
	//   },
	//   "response": {
	//     "$ref": "Annotation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.annotationsets.create":

type AnnotationsetsCreateCall struct {
	s             *Service
	annotationset *AnnotationSet
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Create: Creates a new annotation set. Caller must have WRITE
// permission for the
// associated dataset.
//
// The following fields are required:
//
//   * datasetId
//   * referenceSetId
//
// All other fields may be optionally specified, unless documented as
// being
// server-generated (for example, the `id` field).
func (r *AnnotationsetsService) Create(annotationset *AnnotationSet) *AnnotationsetsCreateCall {
	c := &AnnotationsetsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.annotationset = annotationset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsetsCreateCall) Fields(s ...googleapi.Field) *AnnotationsetsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsetsCreateCall) Context(ctx context.Context) *AnnotationsetsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsetsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsetsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.annotationset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotationsets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotationsets.create" call.
// Exactly one of *AnnotationSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AnnotationSet.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AnnotationsetsCreateCall) Do(opts ...googleapi.CallOption) (*AnnotationSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &AnnotationSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new annotation set. Caller must have WRITE permission for the\nassociated dataset.\n\nThe following fields are required:\n\n  * datasetId\n  * referenceSetId\n\nAll other fields may be optionally specified, unless documented as being\nserver-generated (for example, the `id` field).",
	//   "flatPath": "v1/annotationsets",
	//   "httpMethod": "POST",
	//   "id": "genomics.annotationsets.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/annotationsets",
	//   "request": {
	//     "$ref": "AnnotationSet"
	//   },
	//   "response": {
	//     "$ref": "AnnotationSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.annotationsets.delete":

type AnnotationsetsDeleteCall struct {
	s               *Service
	annotationSetId string
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Delete: Deletes an annotation set. Caller must have WRITE
// permission
// for the associated annotation set.
func (r *AnnotationsetsService) Delete(annotationSetId string) *AnnotationsetsDeleteCall {
	c := &AnnotationsetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.annotationSetId = annotationSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsetsDeleteCall) Fields(s ...googleapi.Field) *AnnotationsetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsetsDeleteCall) Context(ctx context.Context) *AnnotationsetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsetsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotationsets/{annotationSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"annotationSetId": c.annotationSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotationsets.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AnnotationsetsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes an annotation set. Caller must have WRITE permission\nfor the associated annotation set.",
	//   "flatPath": "v1/annotationsets/{annotationSetId}",
	//   "httpMethod": "DELETE",
	//   "id": "genomics.annotationsets.delete",
	//   "parameterOrder": [
	//     "annotationSetId"
	//   ],
	//   "parameters": {
	//     "annotationSetId": {
	//       "description": "The ID of the annotation set to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/annotationsets/{annotationSetId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.annotationsets.get":

type AnnotationsetsGetCall struct {
	s               *Service
	annotationSetId string
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// Get: Gets an annotation set. Caller must have READ permission for
// the associated dataset.
func (r *AnnotationsetsService) Get(annotationSetId string) *AnnotationsetsGetCall {
	c := &AnnotationsetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.annotationSetId = annotationSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsetsGetCall) Fields(s ...googleapi.Field) *AnnotationsetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AnnotationsetsGetCall) IfNoneMatch(entityTag string) *AnnotationsetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsetsGetCall) Context(ctx context.Context) *AnnotationsetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsetsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotationsets/{annotationSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"annotationSetId": c.annotationSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotationsets.get" call.
// Exactly one of *AnnotationSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AnnotationSet.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AnnotationsetsGetCall) Do(opts ...googleapi.CallOption) (*AnnotationSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &AnnotationSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets an annotation set. Caller must have READ permission for\nthe associated dataset.",
	//   "flatPath": "v1/annotationsets/{annotationSetId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.annotationsets.get",
	//   "parameterOrder": [
	//     "annotationSetId"
	//   ],
	//   "parameters": {
	//     "annotationSetId": {
	//       "description": "The ID of the annotation set to be retrieved.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/annotationsets/{annotationSetId}",
	//   "response": {
	//     "$ref": "AnnotationSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.annotationsets.search":

type AnnotationsetsSearchCall struct {
	s                           *Service
	searchannotationsetsrequest *SearchAnnotationSetsRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// Search: Searches for annotation sets that match the given criteria.
// Annotation sets
// are returned in an unspecified order. This order is consistent, such
// that
// two queries for the same content (regardless of page size) yield
// annotation
// sets in the same order across their respective streams of
// paginated
// responses. Caller must have READ permission for the queried datasets.
func (r *AnnotationsetsService) Search(searchannotationsetsrequest *SearchAnnotationSetsRequest) *AnnotationsetsSearchCall {
	c := &AnnotationsetsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchannotationsetsrequest = searchannotationsetsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsetsSearchCall) Fields(s ...googleapi.Field) *AnnotationsetsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsetsSearchCall) Context(ctx context.Context) *AnnotationsetsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsetsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsetsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchannotationsetsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotationsets/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotationsets.search" call.
// Exactly one of *SearchAnnotationSetsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *SearchAnnotationSetsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AnnotationsetsSearchCall) Do(opts ...googleapi.CallOption) (*SearchAnnotationSetsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchAnnotationSetsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Searches for annotation sets that match the given criteria. Annotation sets\nare returned in an unspecified order. This order is consistent, such that\ntwo queries for the same content (regardless of page size) yield annotation\nsets in the same order across their respective streams of paginated\nresponses. Caller must have READ permission for the queried datasets.",
	//   "flatPath": "v1/annotationsets/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.annotationsets.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/annotationsets/search",
	//   "request": {
	//     "$ref": "SearchAnnotationSetsRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchAnnotationSetsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AnnotationsetsSearchCall) Pages(ctx context.Context, f func(*SearchAnnotationSetsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchannotationsetsrequest.PageToken = pt }(c.searchannotationsetsrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchannotationsetsrequest.PageToken = x.NextPageToken
	}
}

// method id "genomics.annotationsets.update":

type AnnotationsetsUpdateCall struct {
	s               *Service
	annotationSetId string
	annotationset   *AnnotationSet
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Update: Updates an annotation set. The update must respect all
// mutability
// restrictions and other invariants described on the annotation set
// resource.
// Caller must have WRITE permission for the associated dataset.
func (r *AnnotationsetsService) Update(annotationSetId string, annotationset *AnnotationSet) *AnnotationsetsUpdateCall {
	c := &AnnotationsetsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.annotationSetId = annotationSetId
	c.annotationset = annotationset
	return c
}

// UpdateMask sets the optional parameter "updateMask": An optional mask
// specifying which fields to update. Mutable fields
// are
// name,
// source_uri, and
// info. If unspecified, all
// mutable fields will be updated.
func (c *AnnotationsetsUpdateCall) UpdateMask(updateMask string) *AnnotationsetsUpdateCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AnnotationsetsUpdateCall) Fields(s ...googleapi.Field) *AnnotationsetsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AnnotationsetsUpdateCall) Context(ctx context.Context) *AnnotationsetsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AnnotationsetsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AnnotationsetsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.annotationset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/annotationsets/{annotationSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"annotationSetId": c.annotationSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.annotationsets.update" call.
// Exactly one of *AnnotationSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *AnnotationSet.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AnnotationsetsUpdateCall) Do(opts ...googleapi.CallOption) (*AnnotationSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &AnnotationSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an annotation set. The update must respect all mutability\nrestrictions and other invariants described on the annotation set resource.\nCaller must have WRITE permission for the associated dataset.",
	//   "flatPath": "v1/annotationsets/{annotationSetId}",
	//   "httpMethod": "PUT",
	//   "id": "genomics.annotationsets.update",
	//   "parameterOrder": [
	//     "annotationSetId"
	//   ],
	//   "parameters": {
	//     "annotationSetId": {
	//       "description": "The ID of the annotation set to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "An optional mask specifying which fields to update. Mutable fields are\nname,\nsource_uri, and\ninfo. If unspecified, all\nmutable fields will be updated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/annotationsets/{annotationSetId}",
	//   "request": {
	//     "$ref": "AnnotationSet"
	//   },
	//   "response": {
	//     "$ref": "AnnotationSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.callsets.create":

type CallsetsCreateCall struct {
	s          *Service
	callset    *CallSet
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new call set.
//
// For the definitions of call sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *CallsetsService) Create(callset *CallSet) *CallsetsCreateCall {
	c := &CallsetsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.callset = callset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CallsetsCreateCall) Fields(s ...googleapi.Field) *CallsetsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CallsetsCreateCall) Context(ctx context.Context) *CallsetsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CallsetsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CallsetsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.callset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/callsets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.callsets.create" call.
// Exactly one of *CallSet or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *CallSet.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CallsetsCreateCall) Do(opts ...googleapi.CallOption) (*CallSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &CallSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new call set.\n\nFor the definitions of call sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/callsets",
	//   "httpMethod": "POST",
	//   "id": "genomics.callsets.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/callsets",
	//   "request": {
	//     "$ref": "CallSet"
	//   },
	//   "response": {
	//     "$ref": "CallSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.callsets.delete":

type CallsetsDeleteCall struct {
	s          *Service
	callSetId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a call set.
//
// For the definitions of call sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *CallsetsService) Delete(callSetId string) *CallsetsDeleteCall {
	c := &CallsetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.callSetId = callSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CallsetsDeleteCall) Fields(s ...googleapi.Field) *CallsetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CallsetsDeleteCall) Context(ctx context.Context) *CallsetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CallsetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CallsetsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/callsets/{callSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"callSetId": c.callSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.callsets.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CallsetsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a call set.\n\nFor the definitions of call sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/callsets/{callSetId}",
	//   "httpMethod": "DELETE",
	//   "id": "genomics.callsets.delete",
	//   "parameterOrder": [
	//     "callSetId"
	//   ],
	//   "parameters": {
	//     "callSetId": {
	//       "description": "The ID of the call set to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/callsets/{callSetId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.callsets.get":

type CallsetsGetCall struct {
	s            *Service
	callSetId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a call set by ID.
//
// For the definitions of call sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *CallsetsService) Get(callSetId string) *CallsetsGetCall {
	c := &CallsetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.callSetId = callSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CallsetsGetCall) Fields(s ...googleapi.Field) *CallsetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CallsetsGetCall) IfNoneMatch(entityTag string) *CallsetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CallsetsGetCall) Context(ctx context.Context) *CallsetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CallsetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CallsetsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/callsets/{callSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"callSetId": c.callSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.callsets.get" call.
// Exactly one of *CallSet or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *CallSet.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CallsetsGetCall) Do(opts ...googleapi.CallOption) (*CallSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &CallSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a call set by ID.\n\nFor the definitions of call sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/callsets/{callSetId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.callsets.get",
	//   "parameterOrder": [
	//     "callSetId"
	//   ],
	//   "parameters": {
	//     "callSetId": {
	//       "description": "The ID of the call set.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/callsets/{callSetId}",
	//   "response": {
	//     "$ref": "CallSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.callsets.patch":

type CallsetsPatchCall struct {
	s          *Service
	callSetId  string
	callset    *CallSet
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a call set.
//
// For the definitions of call sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// This method supports patch semantics.
func (r *CallsetsService) Patch(callSetId string, callset *CallSet) *CallsetsPatchCall {
	c := &CallsetsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.callSetId = callSetId
	c.callset = callset
	return c
}

// UpdateMask sets the optional parameter "updateMask": An optional mask
// specifying which fields to update. At this time, the only
// mutable field is name. The only
// acceptable value is "name". If unspecified, all mutable fields will
// be
// updated.
func (c *CallsetsPatchCall) UpdateMask(updateMask string) *CallsetsPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CallsetsPatchCall) Fields(s ...googleapi.Field) *CallsetsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CallsetsPatchCall) Context(ctx context.Context) *CallsetsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CallsetsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CallsetsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.callset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/callsets/{callSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"callSetId": c.callSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.callsets.patch" call.
// Exactly one of *CallSet or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *CallSet.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CallsetsPatchCall) Do(opts ...googleapi.CallOption) (*CallSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &CallSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a call set.\n\nFor the definitions of call sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nThis method supports patch semantics.",
	//   "flatPath": "v1/callsets/{callSetId}",
	//   "httpMethod": "PATCH",
	//   "id": "genomics.callsets.patch",
	//   "parameterOrder": [
	//     "callSetId"
	//   ],
	//   "parameters": {
	//     "callSetId": {
	//       "description": "The ID of the call set to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "An optional mask specifying which fields to update. At this time, the only\nmutable field is name. The only\nacceptable value is \"name\". If unspecified, all mutable fields will be\nupdated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/callsets/{callSetId}",
	//   "request": {
	//     "$ref": "CallSet"
	//   },
	//   "response": {
	//     "$ref": "CallSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.callsets.search":

type CallsetsSearchCall struct {
	s                     *Service
	searchcallsetsrequest *SearchCallSetsRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Search: Gets a list of call sets matching the criteria.
//
// For the definitions of call sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.searchCallSets](https://g
// ithub.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/variantmet
// hods.avdl#L178).
func (r *CallsetsService) Search(searchcallsetsrequest *SearchCallSetsRequest) *CallsetsSearchCall {
	c := &CallsetsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchcallsetsrequest = searchcallsetsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CallsetsSearchCall) Fields(s ...googleapi.Field) *CallsetsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CallsetsSearchCall) Context(ctx context.Context) *CallsetsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CallsetsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CallsetsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchcallsetsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/callsets/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.callsets.search" call.
// Exactly one of *SearchCallSetsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SearchCallSetsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CallsetsSearchCall) Do(opts ...googleapi.CallOption) (*SearchCallSetsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchCallSetsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a list of call sets matching the criteria.\n\nFor the definitions of call sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.searchCallSets](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/variantmethods.avdl#L178).",
	//   "flatPath": "v1/callsets/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.callsets.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/callsets/search",
	//   "request": {
	//     "$ref": "SearchCallSetsRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchCallSetsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *CallsetsSearchCall) Pages(ctx context.Context, f func(*SearchCallSetsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchcallsetsrequest.PageToken = pt }(c.searchcallsetsrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchcallsetsrequest.PageToken = x.NextPageToken
	}
}

// method id "genomics.datasets.create":

type DatasetsCreateCall struct {
	s          *Service
	dataset    *Dataset
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new dataset.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *DatasetsService) Create(dataset *Dataset) *DatasetsCreateCall {
	c := &DatasetsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.dataset = dataset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsCreateCall) Fields(s ...googleapi.Field) *DatasetsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsCreateCall) Context(ctx context.Context) *DatasetsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.dataset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/datasets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.create" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsCreateCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Dataset{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new dataset.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/datasets",
	//   "httpMethod": "POST",
	//   "id": "genomics.datasets.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/datasets",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.datasets.delete":

type DatasetsDeleteCall struct {
	s          *Service
	datasetId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a dataset and all of its contents (all read group
// sets,
// reference sets, variant sets, call sets, annotation sets, etc.)
// This is reversible (up to one week after the deletion)
// via
// the
// datasets.undelete
// operation.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *DatasetsService) Delete(datasetId string) *DatasetsDeleteCall {
	c := &DatasetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsDeleteCall) Fields(s ...googleapi.Field) *DatasetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsDeleteCall) Context(ctx context.Context) *DatasetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a dataset and all of its contents (all read group sets,\nreference sets, variant sets, call sets, annotation sets, etc.)\nThis is reversible (up to one week after the deletion) via\nthe\ndatasets.undelete\noperation.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/datasets/{datasetId}",
	//   "httpMethod": "DELETE",
	//   "id": "genomics.datasets.delete",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "The ID of the dataset to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/datasets/{datasetId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.datasets.get":

type DatasetsGetCall struct {
	s            *Service
	datasetId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a dataset by ID.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *DatasetsService) Get(datasetId string) *DatasetsGetCall {
	c := &DatasetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsGetCall) Fields(s ...googleapi.Field) *DatasetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatasetsGetCall) IfNoneMatch(entityTag string) *DatasetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsGetCall) Context(ctx context.Context) *DatasetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.get" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsGetCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Dataset{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a dataset by ID.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/datasets/{datasetId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.datasets.get",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "The ID of the dataset.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/datasets/{datasetId}",
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.datasets.getIamPolicy":

type DatasetsGetIamPolicyCall struct {
	s                   *Service
	resource            string
	getiampolicyrequest *GetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// GetIamPolicy: Gets the access control policy for the dataset. This is
// empty if the
// policy or resource does not exist.
//
// See <a href="/iam/docs/managing-policies#getting_a_policy">Getting
// a
// Policy</a> for more information.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *DatasetsService) GetIamPolicy(resource string, getiampolicyrequest *GetIamPolicyRequest) *DatasetsGetIamPolicyCall {
	c := &DatasetsGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.getiampolicyrequest = getiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsGetIamPolicyCall) Fields(s ...googleapi.Field) *DatasetsGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsGetIamPolicyCall) Context(ctx context.Context) *DatasetsGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsGetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.getiampolicyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsGetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Policy{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the access control policy for the dataset. This is empty if the\npolicy or resource does not exist.\n\nSee \u003ca href=\"/iam/docs/managing-policies#getting_a_policy\"\u003eGetting a\nPolicy\u003c/a\u003e for more information.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/datasets/{datasetsId}:getIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "genomics.datasets.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy is being specified. Format is\n`datasets/\u003cdataset ID\u003e`.",
	//       "location": "path",
	//       "pattern": "^datasets/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:getIamPolicy",
	//   "request": {
	//     "$ref": "GetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.datasets.list":

type DatasetsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists datasets within a project.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *DatasetsService) List() *DatasetsListCall {
	c := &DatasetsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return in a single page. If unspecified,
// defaults to 50. The maximum value is 1024.
func (c *DatasetsListCall) PageSize(pageSize int64) *DatasetsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets.
// To get the next page of results, set this parameter to the value
// of
// `nextPageToken` from the previous response.
func (c *DatasetsListCall) PageToken(pageToken string) *DatasetsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// ProjectId sets the optional parameter "projectId": Required. The
// Google Cloud project ID to list datasets for.
func (c *DatasetsListCall) ProjectId(projectId string) *DatasetsListCall {
	c.urlParams_.Set("projectId", projectId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsListCall) Fields(s ...googleapi.Field) *DatasetsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatasetsListCall) IfNoneMatch(entityTag string) *DatasetsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsListCall) Context(ctx context.Context) *DatasetsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/datasets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.list" call.
// Exactly one of *ListDatasetsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListDatasetsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatasetsListCall) Do(opts ...googleapi.CallOption) (*ListDatasetsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &ListDatasetsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists datasets within a project.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/datasets",
	//   "httpMethod": "GET",
	//   "id": "genomics.datasets.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The maximum number of results to return in a single page. If unspecified,\ndefaults to 50. The maximum value is 1024.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets.\nTo get the next page of results, set this parameter to the value of\n`nextPageToken` from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required. The Google Cloud project ID to list datasets for.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/datasets",
	//   "response": {
	//     "$ref": "ListDatasetsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *DatasetsListCall) Pages(ctx context.Context, f func(*ListDatasetsResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "genomics.datasets.patch":

type DatasetsPatchCall struct {
	s          *Service
	datasetId  string
	dataset    *Dataset
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a dataset.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// This method supports patch semantics.
func (r *DatasetsService) Patch(datasetId string, dataset *Dataset) *DatasetsPatchCall {
	c := &DatasetsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	c.dataset = dataset
	return c
}

// UpdateMask sets the optional parameter "updateMask": An optional mask
// specifying which fields to update. At this time, the only
// mutable field is name. The only
// acceptable value is "name". If unspecified, all mutable fields will
// be
// updated.
func (c *DatasetsPatchCall) UpdateMask(updateMask string) *DatasetsPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsPatchCall) Fields(s ...googleapi.Field) *DatasetsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsPatchCall) Context(ctx context.Context) *DatasetsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.dataset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/datasets/{datasetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.patch" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsPatchCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Dataset{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a dataset.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nThis method supports patch semantics.",
	//   "flatPath": "v1/datasets/{datasetId}",
	//   "httpMethod": "PATCH",
	//   "id": "genomics.datasets.patch",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "The ID of the dataset to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "An optional mask specifying which fields to update. At this time, the only\nmutable field is name. The only\nacceptable value is \"name\". If unspecified, all mutable fields will be\nupdated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/datasets/{datasetId}",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.datasets.setIamPolicy":

type DatasetsSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// SetIamPolicy: Sets the access control policy on the specified
// dataset. Replaces any
// existing policy.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// See <a href="/iam/docs/managing-policies#setting_a_policy">Setting
// a
// Policy</a> for more information.
func (r *DatasetsService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *DatasetsSetIamPolicyCall {
	c := &DatasetsSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsSetIamPolicyCall) Fields(s ...googleapi.Field) *DatasetsSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsSetIamPolicyCall) Context(ctx context.Context) *DatasetsSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsSetIamPolicyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.setiampolicyrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsSetIamPolicyCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Policy{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Sets the access control policy on the specified dataset. Replaces any\nexisting policy.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nSee \u003ca href=\"/iam/docs/managing-policies#setting_a_policy\"\u003eSetting a\nPolicy\u003c/a\u003e for more information.",
	//   "flatPath": "v1/datasets/{datasetsId}:setIamPolicy",
	//   "httpMethod": "POST",
	//   "id": "genomics.datasets.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy is being specified. Format is\n`datasets/\u003cdataset ID\u003e`.",
	//       "location": "path",
	//       "pattern": "^datasets/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:setIamPolicy",
	//   "request": {
	//     "$ref": "SetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.datasets.testIamPermissions":

type DatasetsTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified resource.
// See <a
// href="/iam/docs/managing-policies#testing_permissions">Testing
// Permiss
// ions</a> for more information.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *DatasetsService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *DatasetsTestIamPermissionsCall {
	c := &DatasetsTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsTestIamPermissionsCall) Fields(s ...googleapi.Field) *DatasetsTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsTestIamPermissionsCall) Context(ctx context.Context) *DatasetsTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsTestIamPermissionsCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testiampermissionsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resource}:testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatasetsTestIamPermissionsCall) Do(opts ...googleapi.CallOption) (*TestIamPermissionsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &TestIamPermissionsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns permissions that a caller has on the specified resource.\nSee \u003ca href=\"/iam/docs/managing-policies#testing_permissions\"\u003eTesting\nPermissions\u003c/a\u003e for more information.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/datasets/{datasetsId}:testIamPermissions",
	//   "httpMethod": "POST",
	//   "id": "genomics.datasets.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy is being specified. Format is\n`datasets/\u003cdataset ID\u003e`.",
	//       "location": "path",
	//       "pattern": "^datasets/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resource}:testIamPermissions",
	//   "request": {
	//     "$ref": "TestIamPermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "TestIamPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.datasets.undelete":

type DatasetsUndeleteCall struct {
	s                      *Service
	datasetId              string
	undeletedatasetrequest *UndeleteDatasetRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Undelete: Undeletes a dataset by restoring a dataset which was
// deleted via this API.
//
// For the definitions of datasets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// This operation is only possible for a week after the deletion
// occurred.
func (r *DatasetsService) Undelete(datasetId string, undeletedatasetrequest *UndeleteDatasetRequest) *DatasetsUndeleteCall {
	c := &DatasetsUndeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.datasetId = datasetId
	c.undeletedatasetrequest = undeletedatasetrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatasetsUndeleteCall) Fields(s ...googleapi.Field) *DatasetsUndeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatasetsUndeleteCall) Context(ctx context.Context) *DatasetsUndeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatasetsUndeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatasetsUndeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.undeletedatasetrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/datasets/{datasetId}:undelete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"datasetId": c.datasetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.datasets.undelete" call.
// Exactly one of *Dataset or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Dataset.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *DatasetsUndeleteCall) Do(opts ...googleapi.CallOption) (*Dataset, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Dataset{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Undeletes a dataset by restoring a dataset which was deleted via this API.\n\nFor the definitions of datasets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nThis operation is only possible for a week after the deletion occurred.",
	//   "flatPath": "v1/datasets/{datasetId}:undelete",
	//   "httpMethod": "POST",
	//   "id": "genomics.datasets.undelete",
	//   "parameterOrder": [
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "The ID of the dataset to be undeleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/datasets/{datasetId}:undelete",
	//   "request": {
	//     "$ref": "UndeleteDatasetRequest"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.operations.cancel":

type OperationsCancelCall struct {
	s                      *Service
	name                   string
	canceloperationrequest *CancelOperationRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Cancel: Starts asynchronous cancellation on a long-running operation.
// The server makes a best effort to cancel the operation, but success
// is not guaranteed. Clients may use Operations.GetOperation or
// Operations.ListOperations to check whether the cancellation succeeded
// or the operation completed despite cancellation.
func (r *OperationsService) Cancel(name string, canceloperationrequest *CancelOperationRequest) *OperationsCancelCall {
	c := &OperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.canceloperationrequest = canceloperationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsCancelCall) Fields(s ...googleapi.Field) *OperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsCancelCall) Context(ctx context.Context) *OperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.canceloperationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.operations.cancel" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *OperationsCancelCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. Clients may use Operations.GetOperation or Operations.ListOperations to check whether the cancellation succeeded or the operation completed despite cancellation.",
	//   "flatPath": "v1/operations/{operationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "genomics.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^operations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:cancel",
	//   "request": {
	//     "$ref": "CancelOperationRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.operations.get":

type OperationsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the latest state of a long-running operation.  Clients can
// use this
// method to poll the operation result at intervals as recommended by
// the API
// service.
func (r *OperationsService) Get(name string) *OperationsGetCall {
	c := &OperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsGetCall) Fields(s ...googleapi.Field) *OperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OperationsGetCall) IfNoneMatch(entityTag string) *OperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsGetCall) Context(ctx context.Context) *OperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *OperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the latest state of a long-running operation.  Clients can use this\nmethod to poll the operation result at intervals as recommended by the API\nservice.",
	//   "flatPath": "v1/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^operations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.operations.list":

type OperationsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists operations that match the specified filter in the
// request.
func (r *OperationsService) List(name string) *OperationsListCall {
	c := &OperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": A string for filtering
// Operations.
// The following filter fields are supported&#58;
//
// * projectId&#58; Required. Corresponds to
//   OperationMetadata.projectId.
// * createTime&#58; The time this job was created, in seconds from the
//   [epoch](http://en.wikipedia.org/wiki/Unix_time). Can use `>=`
// and/or `<=`
//   operators.
// * status&#58; Can be `RUNNING`, `SUCCESS`, `FAILURE`, or `CANCELED`.
// Only
//   one status may be specified.
// * labels.key where key is a label key.
//
// Examples&#58;
//
// * `projectId = my-project AND createTime >= 1432140000`
// * `projectId = my-project AND createTime >= 1432140000 AND createTime
// <= 1432150000 AND status = RUNNING`
// * `projectId = my-project AND labels.color = *`
// * `projectId = my-project AND labels.color = red`
func (c *OperationsListCall) Filter(filter string) *OperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return. If unspecified, defaults to
// 256. The maximum value is 2048.
func (c *OperationsListCall) PageSize(pageSize int64) *OperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *OperationsListCall) PageToken(pageToken string) *OperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsListCall) Fields(s ...googleapi.Field) *OperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OperationsListCall) IfNoneMatch(entityTag string) *OperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsListCall) Context(ctx context.Context) *OperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OperationsListCall) Do(opts ...googleapi.CallOption) (*ListOperationsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &ListOperationsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists operations that match the specified filter in the request.",
	//   "flatPath": "v1/operations",
	//   "httpMethod": "GET",
	//   "id": "genomics.operations.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "A string for filtering Operations.\nThe following filter fields are supported\u0026#58;\n\n* projectId\u0026#58; Required. Corresponds to\n  OperationMetadata.projectId.\n* createTime\u0026#58; The time this job was created, in seconds from the\n  [epoch](http://en.wikipedia.org/wiki/Unix_time). Can use `\u003e=` and/or `\u003c=`\n  operators.\n* status\u0026#58; Can be `RUNNING`, `SUCCESS`, `FAILURE`, or `CANCELED`. Only\n  one status may be specified.\n* labels.key where key is a label key.\n\nExamples\u0026#58;\n\n* `projectId = my-project AND createTime \u003e= 1432140000`\n* `projectId = my-project AND createTime \u003e= 1432140000 AND createTime \u003c= 1432150000 AND status = RUNNING`\n* `projectId = my-project AND labels.color = *`\n* `projectId = my-project AND labels.color = red`",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The name of the operation's parent resource.",
	//       "location": "path",
	//       "pattern": "^operations$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The maximum number of results to return. If unspecified, defaults to\n256. The maximum value is 2048.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The standard list page token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *OperationsListCall) Pages(ctx context.Context, f func(*ListOperationsResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "genomics.readgroupsets.delete":

type ReadgroupsetsDeleteCall struct {
	s              *Service
	readGroupSetId string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
	header_        http.Header
}

// Delete: Deletes a read group set.
//
// For the definitions of read group sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *ReadgroupsetsService) Delete(readGroupSetId string) *ReadgroupsetsDeleteCall {
	c := &ReadgroupsetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.readGroupSetId = readGroupSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReadgroupsetsDeleteCall) Fields(s ...googleapi.Field) *ReadgroupsetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReadgroupsetsDeleteCall) Context(ctx context.Context) *ReadgroupsetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReadgroupsetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReadgroupsetsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/readgroupsets/{readGroupSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"readGroupSetId": c.readGroupSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.readgroupsets.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ReadgroupsetsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a read group set.\n\nFor the definitions of read group sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/readgroupsets/{readGroupSetId}",
	//   "httpMethod": "DELETE",
	//   "id": "genomics.readgroupsets.delete",
	//   "parameterOrder": [
	//     "readGroupSetId"
	//   ],
	//   "parameters": {
	//     "readGroupSetId": {
	//       "description": "The ID of the read group set to be deleted. The caller must have WRITE\npermissions to the dataset associated with this read group set.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/readgroupsets/{readGroupSetId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.readgroupsets.export":

type ReadgroupsetsExportCall struct {
	s                         *Service
	readGroupSetId            string
	exportreadgroupsetrequest *ExportReadGroupSetRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// Export: Exports a read group set to a BAM file in Google Cloud
// Storage.
//
// For the definitions of read group sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Note that currently there may be some differences between exported
// BAM
// files and the original BAM file at the time of import.
// See
// ImportReadGroupSets
// for caveats.
func (r *ReadgroupsetsService) Export(readGroupSetId string, exportreadgroupsetrequest *ExportReadGroupSetRequest) *ReadgroupsetsExportCall {
	c := &ReadgroupsetsExportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.readGroupSetId = readGroupSetId
	c.exportreadgroupsetrequest = exportreadgroupsetrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReadgroupsetsExportCall) Fields(s ...googleapi.Field) *ReadgroupsetsExportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReadgroupsetsExportCall) Context(ctx context.Context) *ReadgroupsetsExportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReadgroupsetsExportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReadgroupsetsExportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.exportreadgroupsetrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/readgroupsets/{readGroupSetId}:export")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"readGroupSetId": c.readGroupSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.readgroupsets.export" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReadgroupsetsExportCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Exports a read group set to a BAM file in Google Cloud Storage.\n\nFor the definitions of read group sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nNote that currently there may be some differences between exported BAM\nfiles and the original BAM file at the time of import. See\nImportReadGroupSets\nfor caveats.",
	//   "flatPath": "v1/readgroupsets/{readGroupSetId}:export",
	//   "httpMethod": "POST",
	//   "id": "genomics.readgroupsets.export",
	//   "parameterOrder": [
	//     "readGroupSetId"
	//   ],
	//   "parameters": {
	//     "readGroupSetId": {
	//       "description": "Required. The ID of the read group set to export. The caller must have\nREAD access to this read group set.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/readgroupsets/{readGroupSetId}:export",
	//   "request": {
	//     "$ref": "ExportReadGroupSetRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/devstorage.read_write",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.readgroupsets.get":

type ReadgroupsetsGetCall struct {
	s              *Service
	readGroupSetId string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
	header_        http.Header
}

// Get: Gets a read group set by ID.
//
// For the definitions of read group sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *ReadgroupsetsService) Get(readGroupSetId string) *ReadgroupsetsGetCall {
	c := &ReadgroupsetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.readGroupSetId = readGroupSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReadgroupsetsGetCall) Fields(s ...googleapi.Field) *ReadgroupsetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReadgroupsetsGetCall) IfNoneMatch(entityTag string) *ReadgroupsetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReadgroupsetsGetCall) Context(ctx context.Context) *ReadgroupsetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReadgroupsetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReadgroupsetsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/readgroupsets/{readGroupSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"readGroupSetId": c.readGroupSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.readgroupsets.get" call.
// Exactly one of *ReadGroupSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ReadGroupSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReadgroupsetsGetCall) Do(opts ...googleapi.CallOption) (*ReadGroupSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &ReadGroupSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a read group set by ID.\n\nFor the definitions of read group sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/readgroupsets/{readGroupSetId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.readgroupsets.get",
	//   "parameterOrder": [
	//     "readGroupSetId"
	//   ],
	//   "parameters": {
	//     "readGroupSetId": {
	//       "description": "The ID of the read group set.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/readgroupsets/{readGroupSetId}",
	//   "response": {
	//     "$ref": "ReadGroupSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.readgroupsets.import":

type ReadgroupsetsImportCall struct {
	s                          *Service
	importreadgroupsetsrequest *ImportReadGroupSetsRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// Import: Creates read group sets by asynchronously importing the
// provided
// information.
//
// For the definitions of read group sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// The caller must have WRITE permissions to the dataset.
//
// ## Notes on [BAM](https://samtools.github.io/hts-specs/SAMv1.pdf)
// import
//
// - Tags will be converted to strings - tag types are not preserved
// - Comments (`@CO`) in the input file header will not be preserved
// - Original header order of references (`@SQ`) will not be preserved
// - Any reverse stranded unmapped reads will be reverse complemented,
// and
// their qualities (also the "BQ" and "OQ" tags, if any) will be
// reversed
// - Unmapped reads will be stripped of positional information
// (reference name
// and position)
func (r *ReadgroupsetsService) Import(importreadgroupsetsrequest *ImportReadGroupSetsRequest) *ReadgroupsetsImportCall {
	c := &ReadgroupsetsImportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.importreadgroupsetsrequest = importreadgroupsetsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReadgroupsetsImportCall) Fields(s ...googleapi.Field) *ReadgroupsetsImportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReadgroupsetsImportCall) Context(ctx context.Context) *ReadgroupsetsImportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReadgroupsetsImportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReadgroupsetsImportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.importreadgroupsetsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/readgroupsets:import")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.readgroupsets.import" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReadgroupsetsImportCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates read group sets by asynchronously importing the provided\ninformation.\n\nFor the definitions of read group sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nThe caller must have WRITE permissions to the dataset.\n\n## Notes on [BAM](https://samtools.github.io/hts-specs/SAMv1.pdf) import\n\n- Tags will be converted to strings - tag types are not preserved\n- Comments (`@CO`) in the input file header will not be preserved\n- Original header order of references (`@SQ`) will not be preserved\n- Any reverse stranded unmapped reads will be reverse complemented, and\ntheir qualities (also the \"BQ\" and \"OQ\" tags, if any) will be reversed\n- Unmapped reads will be stripped of positional information (reference name\nand position)",
	//   "flatPath": "v1/readgroupsets:import",
	//   "httpMethod": "POST",
	//   "id": "genomics.readgroupsets.import",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/readgroupsets:import",
	//   "request": {
	//     "$ref": "ImportReadGroupSetsRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/devstorage.read_write",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.readgroupsets.patch":

type ReadgroupsetsPatchCall struct {
	s              *Service
	readGroupSetId string
	readgroupset   *ReadGroupSet
	urlParams_     gensupport.URLParams
	ctx_           context.Context
	header_        http.Header
}

// Patch: Updates a read group set.
//
// For the definitions of read group sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// This method supports patch semantics.
func (r *ReadgroupsetsService) Patch(readGroupSetId string, readgroupset *ReadGroupSet) *ReadgroupsetsPatchCall {
	c := &ReadgroupsetsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.readGroupSetId = readGroupSetId
	c.readgroupset = readgroupset
	return c
}

// UpdateMask sets the optional parameter "updateMask": An optional mask
// specifying which fields to update. Supported fields:
//
// * name.
// * referenceSetId.
//
// Leaving `updateMask` unset is equivalent to specifying all
// mutable
// fields.
func (c *ReadgroupsetsPatchCall) UpdateMask(updateMask string) *ReadgroupsetsPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReadgroupsetsPatchCall) Fields(s ...googleapi.Field) *ReadgroupsetsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReadgroupsetsPatchCall) Context(ctx context.Context) *ReadgroupsetsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReadgroupsetsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReadgroupsetsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.readgroupset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/readgroupsets/{readGroupSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"readGroupSetId": c.readGroupSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.readgroupsets.patch" call.
// Exactly one of *ReadGroupSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ReadGroupSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReadgroupsetsPatchCall) Do(opts ...googleapi.CallOption) (*ReadGroupSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &ReadGroupSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a read group set.\n\nFor the definitions of read group sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nThis method supports patch semantics.",
	//   "flatPath": "v1/readgroupsets/{readGroupSetId}",
	//   "httpMethod": "PATCH",
	//   "id": "genomics.readgroupsets.patch",
	//   "parameterOrder": [
	//     "readGroupSetId"
	//   ],
	//   "parameters": {
	//     "readGroupSetId": {
	//       "description": "The ID of the read group set to be updated. The caller must have WRITE\npermissions to the dataset associated with this read group set.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "An optional mask specifying which fields to update. Supported fields:\n\n* name.\n* referenceSetId.\n\nLeaving `updateMask` unset is equivalent to specifying all mutable\nfields.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/readgroupsets/{readGroupSetId}",
	//   "request": {
	//     "$ref": "ReadGroupSet"
	//   },
	//   "response": {
	//     "$ref": "ReadGroupSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.readgroupsets.search":

type ReadgroupsetsSearchCall struct {
	s                          *Service
	searchreadgroupsetsrequest *SearchReadGroupSetsRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// Search: Searches for read group sets matching the criteria.
//
// For the definitions of read group sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.searchReadGroupSets](http
// s://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/readm
// ethods.avdl#L135).
func (r *ReadgroupsetsService) Search(searchreadgroupsetsrequest *SearchReadGroupSetsRequest) *ReadgroupsetsSearchCall {
	c := &ReadgroupsetsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchreadgroupsetsrequest = searchreadgroupsetsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReadgroupsetsSearchCall) Fields(s ...googleapi.Field) *ReadgroupsetsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReadgroupsetsSearchCall) Context(ctx context.Context) *ReadgroupsetsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReadgroupsetsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReadgroupsetsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchreadgroupsetsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/readgroupsets/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.readgroupsets.search" call.
// Exactly one of *SearchReadGroupSetsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *SearchReadGroupSetsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ReadgroupsetsSearchCall) Do(opts ...googleapi.CallOption) (*SearchReadGroupSetsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchReadGroupSetsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Searches for read group sets matching the criteria.\n\nFor the definitions of read group sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.searchReadGroupSets](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/readmethods.avdl#L135).",
	//   "flatPath": "v1/readgroupsets/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.readgroupsets.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/readgroupsets/search",
	//   "request": {
	//     "$ref": "SearchReadGroupSetsRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchReadGroupSetsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ReadgroupsetsSearchCall) Pages(ctx context.Context, f func(*SearchReadGroupSetsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchreadgroupsetsrequest.PageToken = pt }(c.searchreadgroupsetsrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchreadgroupsetsrequest.PageToken = x.NextPageToken
	}
}

// method id "genomics.readgroupsets.coveragebuckets.list":

type ReadgroupsetsCoveragebucketsListCall struct {
	s              *Service
	readGroupSetId string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
	header_        http.Header
}

// List: Lists fixed width coverage buckets for a read group set, each
// of which
// correspond to a range of a reference sequence. Each bucket
// summarizes
// coverage information across its corresponding genomic range.
//
// For the definitions of read group sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Coverage is defined as the number of reads which are aligned to a
// given
// base in the reference sequence. Coverage buckets are available at
// several
// precomputed bucket widths, enabling retrieval of various coverage
// 'zoom
// levels'. The caller must have READ permissions for the target read
// group
// set.
func (r *ReadgroupsetsCoveragebucketsService) List(readGroupSetId string) *ReadgroupsetsCoveragebucketsListCall {
	c := &ReadgroupsetsCoveragebucketsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.readGroupSetId = readGroupSetId
	return c
}

// End sets the optional parameter "end": The end position of the range
// on the reference, 0-based exclusive. If
// specified, `referenceName` must also be specified. If unset or 0,
// defaults
// to the length of the reference.
func (c *ReadgroupsetsCoveragebucketsListCall) End(end int64) *ReadgroupsetsCoveragebucketsListCall {
	c.urlParams_.Set("end", fmt.Sprint(end))
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of results to return in a single page. If unspecified,
// defaults to 1024. The maximum value is 2048.
func (c *ReadgroupsetsCoveragebucketsListCall) PageSize(pageSize int64) *ReadgroupsetsCoveragebucketsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets.
// To get the next page of results, set this parameter to the value
// of
// `nextPageToken` from the previous response.
func (c *ReadgroupsetsCoveragebucketsListCall) PageToken(pageToken string) *ReadgroupsetsCoveragebucketsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// ReferenceName sets the optional parameter "referenceName": The name
// of the reference to query, within the reference set associated
// with this query.
func (c *ReadgroupsetsCoveragebucketsListCall) ReferenceName(referenceName string) *ReadgroupsetsCoveragebucketsListCall {
	c.urlParams_.Set("referenceName", referenceName)
	return c
}

// Start sets the optional parameter "start": The start position of the
// range on the reference, 0-based inclusive. If
// specified, `referenceName` must also be specified. Defaults to 0.
func (c *ReadgroupsetsCoveragebucketsListCall) Start(start int64) *ReadgroupsetsCoveragebucketsListCall {
	c.urlParams_.Set("start", fmt.Sprint(start))
	return c
}

// TargetBucketWidth sets the optional parameter "targetBucketWidth":
// The desired width of each reported coverage bucket in base pairs.
// This
// will be rounded down to the nearest precomputed bucket width; the
// value
// of which is returned as `bucketWidth` in the response. Defaults
// to infinity (each bucket spans an entire reference sequence) or the
// length
// of the target range, if specified. The smallest
// precomputed
// `bucketWidth` is currently 2048 base pairs; this is subject
// to
// change.
func (c *ReadgroupsetsCoveragebucketsListCall) TargetBucketWidth(targetBucketWidth int64) *ReadgroupsetsCoveragebucketsListCall {
	c.urlParams_.Set("targetBucketWidth", fmt.Sprint(targetBucketWidth))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReadgroupsetsCoveragebucketsListCall) Fields(s ...googleapi.Field) *ReadgroupsetsCoveragebucketsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReadgroupsetsCoveragebucketsListCall) IfNoneMatch(entityTag string) *ReadgroupsetsCoveragebucketsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReadgroupsetsCoveragebucketsListCall) Context(ctx context.Context) *ReadgroupsetsCoveragebucketsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReadgroupsetsCoveragebucketsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReadgroupsetsCoveragebucketsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/readgroupsets/{readGroupSetId}/coveragebuckets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"readGroupSetId": c.readGroupSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.readgroupsets.coveragebuckets.list" call.
// Exactly one of *ListCoverageBucketsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListCoverageBucketsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ReadgroupsetsCoveragebucketsListCall) Do(opts ...googleapi.CallOption) (*ListCoverageBucketsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &ListCoverageBucketsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists fixed width coverage buckets for a read group set, each of which\ncorrespond to a range of a reference sequence. Each bucket summarizes\ncoverage information across its corresponding genomic range.\n\nFor the definitions of read group sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nCoverage is defined as the number of reads which are aligned to a given\nbase in the reference sequence. Coverage buckets are available at several\nprecomputed bucket widths, enabling retrieval of various coverage 'zoom\nlevels'. The caller must have READ permissions for the target read group\nset.",
	//   "flatPath": "v1/readgroupsets/{readGroupSetId}/coveragebuckets",
	//   "httpMethod": "GET",
	//   "id": "genomics.readgroupsets.coveragebuckets.list",
	//   "parameterOrder": [
	//     "readGroupSetId"
	//   ],
	//   "parameters": {
	//     "end": {
	//       "description": "The end position of the range on the reference, 0-based exclusive. If\nspecified, `referenceName` must also be specified. If unset or 0, defaults\nto the length of the reference.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The maximum number of results to return in a single page. If unspecified,\ndefaults to 1024. The maximum value is 2048.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets.\nTo get the next page of results, set this parameter to the value of\n`nextPageToken` from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "readGroupSetId": {
	//       "description": "Required. The ID of the read group set over which coverage is requested.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "referenceName": {
	//       "description": "The name of the reference to query, within the reference set associated\nwith this query. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "start": {
	//       "description": "The start position of the range on the reference, 0-based inclusive. If\nspecified, `referenceName` must also be specified. Defaults to 0.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "targetBucketWidth": {
	//       "description": "The desired width of each reported coverage bucket in base pairs. This\nwill be rounded down to the nearest precomputed bucket width; the value\nof which is returned as `bucketWidth` in the response. Defaults\nto infinity (each bucket spans an entire reference sequence) or the length\nof the target range, if specified. The smallest precomputed\n`bucketWidth` is currently 2048 base pairs; this is subject to\nchange.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/readgroupsets/{readGroupSetId}/coveragebuckets",
	//   "response": {
	//     "$ref": "ListCoverageBucketsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ReadgroupsetsCoveragebucketsListCall) Pages(ctx context.Context, f func(*ListCoverageBucketsResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "genomics.reads.search":

type ReadsSearchCall struct {
	s                  *Service
	searchreadsrequest *SearchReadsRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
	header_            http.Header
}

// Search: Gets a list of reads for one or more read group sets.
//
// For the definitions of read group sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Reads search operates over a genomic coordinate space of reference
// sequence
// & position defined over the reference sequences to which the
// requested
// read group sets are aligned.
//
// If a target positional range is specified, search returns all reads
// whose
// alignment to the reference genome overlap the range. A query
// which
// specifies only read group set IDs yields all reads in those read
// group
// sets, including unmapped reads.
//
// All reads returned (including reads on subsequent pages) are ordered
// by
// genomic coordinate (by reference sequence, then position). Reads
// with
// equivalent genomic coordinates are returned in an unspecified order.
// This
// order is consistent, such that two queries for the same content
// (regardless
// of page size) yield reads in the same order across their respective
// streams
// of paginated
// responses.
//
// Implements
// [GlobalAllianceApi.searchReads](https://github.
// com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/readmethods.avdl
// #L85).
func (r *ReadsService) Search(searchreadsrequest *SearchReadsRequest) *ReadsSearchCall {
	c := &ReadsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchreadsrequest = searchreadsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReadsSearchCall) Fields(s ...googleapi.Field) *ReadsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReadsSearchCall) Context(ctx context.Context) *ReadsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReadsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReadsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchreadsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/reads/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.reads.search" call.
// Exactly one of *SearchReadsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SearchReadsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ReadsSearchCall) Do(opts ...googleapi.CallOption) (*SearchReadsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchReadsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a list of reads for one or more read group sets.\n\nFor the definitions of read group sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nReads search operates over a genomic coordinate space of reference sequence\n\u0026 position defined over the reference sequences to which the requested\nread group sets are aligned.\n\nIf a target positional range is specified, search returns all reads whose\nalignment to the reference genome overlap the range. A query which\nspecifies only read group set IDs yields all reads in those read group\nsets, including unmapped reads.\n\nAll reads returned (including reads on subsequent pages) are ordered by\ngenomic coordinate (by reference sequence, then position). Reads with\nequivalent genomic coordinates are returned in an unspecified order. This\norder is consistent, such that two queries for the same content (regardless\nof page size) yield reads in the same order across their respective streams\nof paginated responses.\n\nImplements\n[GlobalAllianceApi.searchReads](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/readmethods.avdl#L85).",
	//   "flatPath": "v1/reads/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.reads.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/reads/search",
	//   "request": {
	//     "$ref": "SearchReadsRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchReadsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ReadsSearchCall) Pages(ctx context.Context, f func(*SearchReadsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchreadsrequest.PageToken = pt }(c.searchreadsrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchreadsrequest.PageToken = x.NextPageToken
	}
}

// method id "genomics.references.get":

type ReferencesGetCall struct {
	s            *Service
	referenceId  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a reference.
//
// For the definitions of references and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.getReference](https://git
// hub.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/referencemet
// hods.avdl#L158).
func (r *ReferencesService) Get(referenceId string) *ReferencesGetCall {
	c := &ReferencesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.referenceId = referenceId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReferencesGetCall) Fields(s ...googleapi.Field) *ReferencesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReferencesGetCall) IfNoneMatch(entityTag string) *ReferencesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReferencesGetCall) Context(ctx context.Context) *ReferencesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReferencesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReferencesGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/references/{referenceId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"referenceId": c.referenceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.references.get" call.
// Exactly one of *Reference or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Reference.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReferencesGetCall) Do(opts ...googleapi.CallOption) (*Reference, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Reference{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a reference.\n\nFor the definitions of references and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.getReference](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/referencemethods.avdl#L158).",
	//   "flatPath": "v1/references/{referenceId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.references.get",
	//   "parameterOrder": [
	//     "referenceId"
	//   ],
	//   "parameters": {
	//     "referenceId": {
	//       "description": "The ID of the reference.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/references/{referenceId}",
	//   "response": {
	//     "$ref": "Reference"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.references.search":

type ReferencesSearchCall struct {
	s                       *Service
	searchreferencesrequest *SearchReferencesRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
	header_                 http.Header
}

// Search: Searches for references which match the given criteria.
//
// For the definitions of references and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.searchReferences](https:/
// /github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/referenc
// emethods.avdl#L146).
func (r *ReferencesService) Search(searchreferencesrequest *SearchReferencesRequest) *ReferencesSearchCall {
	c := &ReferencesSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchreferencesrequest = searchreferencesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReferencesSearchCall) Fields(s ...googleapi.Field) *ReferencesSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReferencesSearchCall) Context(ctx context.Context) *ReferencesSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReferencesSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReferencesSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchreferencesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/references/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.references.search" call.
// Exactly one of *SearchReferencesResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *SearchReferencesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ReferencesSearchCall) Do(opts ...googleapi.CallOption) (*SearchReferencesResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchReferencesResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Searches for references which match the given criteria.\n\nFor the definitions of references and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.searchReferences](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/referencemethods.avdl#L146).",
	//   "flatPath": "v1/references/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.references.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/references/search",
	//   "request": {
	//     "$ref": "SearchReferencesRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchReferencesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ReferencesSearchCall) Pages(ctx context.Context, f func(*SearchReferencesResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchreferencesrequest.PageToken = pt }(c.searchreferencesrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchreferencesrequest.PageToken = x.NextPageToken
	}
}

// method id "genomics.references.bases.list":

type ReferencesBasesListCall struct {
	s            *Service
	referenceId  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists the bases in a reference, optionally restricted to a
// range.
//
// For the definitions of references and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.getReferenceBases](https:
// //github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/referen
// cemethods.avdl#L221).
func (r *ReferencesBasesService) List(referenceId string) *ReferencesBasesListCall {
	c := &ReferencesBasesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.referenceId = referenceId
	return c
}

// End sets the optional parameter "end": The end position (0-based,
// exclusive) of this query. Defaults to the length
// of this reference.
func (c *ReferencesBasesListCall) End(end int64) *ReferencesBasesListCall {
	c.urlParams_.Set("end", fmt.Sprint(end))
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of bases to return in a single page. If unspecified,
// defaults to 200Kbp (kilo base pairs). The maximum value is 10Mbp
// (mega base
// pairs).
func (c *ReferencesBasesListCall) PageSize(pageSize int64) *ReferencesBasesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets.
// To get the next page of results, set this parameter to the value
// of
// `nextPageToken` from the previous response.
func (c *ReferencesBasesListCall) PageToken(pageToken string) *ReferencesBasesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Start sets the optional parameter "start": The start position
// (0-based) of this query. Defaults to 0.
func (c *ReferencesBasesListCall) Start(start int64) *ReferencesBasesListCall {
	c.urlParams_.Set("start", fmt.Sprint(start))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReferencesBasesListCall) Fields(s ...googleapi.Field) *ReferencesBasesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReferencesBasesListCall) IfNoneMatch(entityTag string) *ReferencesBasesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReferencesBasesListCall) Context(ctx context.Context) *ReferencesBasesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReferencesBasesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReferencesBasesListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/references/{referenceId}/bases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"referenceId": c.referenceId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.references.bases.list" call.
// Exactly one of *ListBasesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListBasesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ReferencesBasesListCall) Do(opts ...googleapi.CallOption) (*ListBasesResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &ListBasesResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the bases in a reference, optionally restricted to a range.\n\nFor the definitions of references and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.getReferenceBases](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/referencemethods.avdl#L221).",
	//   "flatPath": "v1/references/{referenceId}/bases",
	//   "httpMethod": "GET",
	//   "id": "genomics.references.bases.list",
	//   "parameterOrder": [
	//     "referenceId"
	//   ],
	//   "parameters": {
	//     "end": {
	//       "description": "The end position (0-based, exclusive) of this query. Defaults to the length\nof this reference.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The maximum number of bases to return in a single page. If unspecified,\ndefaults to 200Kbp (kilo base pairs). The maximum value is 10Mbp (mega base\npairs).",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets.\nTo get the next page of results, set this parameter to the value of\n`nextPageToken` from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "referenceId": {
	//       "description": "The ID of the reference.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "start": {
	//       "description": "The start position (0-based) of this query. Defaults to 0.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/references/{referenceId}/bases",
	//   "response": {
	//     "$ref": "ListBasesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ReferencesBasesListCall) Pages(ctx context.Context, f func(*ListBasesResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "genomics.referencesets.get":

type ReferencesetsGetCall struct {
	s              *Service
	referenceSetId string
	urlParams_     gensupport.URLParams
	ifNoneMatch_   string
	ctx_           context.Context
	header_        http.Header
}

// Get: Gets a reference set.
//
// For the definitions of references and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.getReferenceSet](https://
// github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/reference
// methods.avdl#L83).
func (r *ReferencesetsService) Get(referenceSetId string) *ReferencesetsGetCall {
	c := &ReferencesetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.referenceSetId = referenceSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReferencesetsGetCall) Fields(s ...googleapi.Field) *ReferencesetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReferencesetsGetCall) IfNoneMatch(entityTag string) *ReferencesetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReferencesetsGetCall) Context(ctx context.Context) *ReferencesetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReferencesetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReferencesetsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/referencesets/{referenceSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"referenceSetId": c.referenceSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.referencesets.get" call.
// Exactly one of *ReferenceSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ReferenceSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReferencesetsGetCall) Do(opts ...googleapi.CallOption) (*ReferenceSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &ReferenceSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a reference set.\n\nFor the definitions of references and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.getReferenceSet](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/referencemethods.avdl#L83).",
	//   "flatPath": "v1/referencesets/{referenceSetId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.referencesets.get",
	//   "parameterOrder": [
	//     "referenceSetId"
	//   ],
	//   "parameters": {
	//     "referenceSetId": {
	//       "description": "The ID of the reference set.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/referencesets/{referenceSetId}",
	//   "response": {
	//     "$ref": "ReferenceSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.referencesets.search":

type ReferencesetsSearchCall struct {
	s                          *Service
	searchreferencesetsrequest *SearchReferenceSetsRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// Search: Searches for reference sets which match the given
// criteria.
//
// For the definitions of references and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.searchReferenceSets](http
// s://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/refer
// encemethods.avdl#L71)
func (r *ReferencesetsService) Search(searchreferencesetsrequest *SearchReferenceSetsRequest) *ReferencesetsSearchCall {
	c := &ReferencesetsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchreferencesetsrequest = searchreferencesetsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReferencesetsSearchCall) Fields(s ...googleapi.Field) *ReferencesetsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReferencesetsSearchCall) Context(ctx context.Context) *ReferencesetsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ReferencesetsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ReferencesetsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchreferencesetsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/referencesets/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.referencesets.search" call.
// Exactly one of *SearchReferenceSetsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *SearchReferenceSetsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ReferencesetsSearchCall) Do(opts ...googleapi.CallOption) (*SearchReferenceSetsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchReferenceSetsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Searches for reference sets which match the given criteria.\n\nFor the definitions of references and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.searchReferenceSets](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/referencemethods.avdl#L71)",
	//   "flatPath": "v1/referencesets/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.referencesets.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/referencesets/search",
	//   "request": {
	//     "$ref": "SearchReferenceSetsRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchReferenceSetsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ReferencesetsSearchCall) Pages(ctx context.Context, f func(*SearchReferenceSetsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchreferencesetsrequest.PageToken = pt }(c.searchreferencesetsrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchreferencesetsrequest.PageToken = x.NextPageToken
	}
}

// method id "genomics.variants.create":

type VariantsCreateCall struct {
	s          *Service
	variant    *Variant
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new variant.
//
// For the definitions of variants and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *VariantsService) Create(variant *Variant) *VariantsCreateCall {
	c := &VariantsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variant = variant
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsCreateCall) Fields(s ...googleapi.Field) *VariantsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsCreateCall) Context(ctx context.Context) *VariantsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.variant)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variants")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variants.create" call.
// Exactly one of *Variant or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Variant.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *VariantsCreateCall) Do(opts ...googleapi.CallOption) (*Variant, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Variant{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new variant.\n\nFor the definitions of variants and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/variants",
	//   "httpMethod": "POST",
	//   "id": "genomics.variants.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/variants",
	//   "request": {
	//     "$ref": "Variant"
	//   },
	//   "response": {
	//     "$ref": "Variant"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variants.delete":

type VariantsDeleteCall struct {
	s          *Service
	variantId  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a variant.
//
// For the definitions of variants and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *VariantsService) Delete(variantId string) *VariantsDeleteCall {
	c := &VariantsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variantId = variantId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsDeleteCall) Fields(s ...googleapi.Field) *VariantsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsDeleteCall) Context(ctx context.Context) *VariantsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variants/{variantId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"variantId": c.variantId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variants.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *VariantsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a variant.\n\nFor the definitions of variants and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/variants/{variantId}",
	//   "httpMethod": "DELETE",
	//   "id": "genomics.variants.delete",
	//   "parameterOrder": [
	//     "variantId"
	//   ],
	//   "parameters": {
	//     "variantId": {
	//       "description": "The ID of the variant to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/variants/{variantId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variants.get":

type VariantsGetCall struct {
	s            *Service
	variantId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a variant by ID.
//
// For the definitions of variants and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *VariantsService) Get(variantId string) *VariantsGetCall {
	c := &VariantsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variantId = variantId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsGetCall) Fields(s ...googleapi.Field) *VariantsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *VariantsGetCall) IfNoneMatch(entityTag string) *VariantsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsGetCall) Context(ctx context.Context) *VariantsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variants/{variantId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"variantId": c.variantId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variants.get" call.
// Exactly one of *Variant or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Variant.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *VariantsGetCall) Do(opts ...googleapi.CallOption) (*Variant, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Variant{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a variant by ID.\n\nFor the definitions of variants and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/variants/{variantId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.variants.get",
	//   "parameterOrder": [
	//     "variantId"
	//   ],
	//   "parameters": {
	//     "variantId": {
	//       "description": "The ID of the variant.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/variants/{variantId}",
	//   "response": {
	//     "$ref": "Variant"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.variants.import":

type VariantsImportCall struct {
	s                     *Service
	importvariantsrequest *ImportVariantsRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Import: Creates variant data by asynchronously importing the provided
// information.
//
// For the definitions of variant sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// The variants for import will be merged with any existing variant
// that
// matches its reference sequence, start, end, reference bases,
// and
// alternative bases. If no such variant exists, a new one will be
// created.
//
// When variants are merged, the call information from the new
// variant
// is added to the existing variant, and Variant info fields are
// merged
// as specified in
// infoMergeConfig.
// As a special case, for single-sample VCF files, QUAL and FILTER
// fields will
// be moved to the call level; these are sometimes interpreted in
// a
// call-specific context.
// Imported VCF headers are appended to the metadata already in a
// variant set.
func (r *VariantsService) Import(importvariantsrequest *ImportVariantsRequest) *VariantsImportCall {
	c := &VariantsImportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.importvariantsrequest = importvariantsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsImportCall) Fields(s ...googleapi.Field) *VariantsImportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsImportCall) Context(ctx context.Context) *VariantsImportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsImportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsImportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.importvariantsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variants:import")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variants.import" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *VariantsImportCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates variant data by asynchronously importing the provided information.\n\nFor the definitions of variant sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nThe variants for import will be merged with any existing variant that\nmatches its reference sequence, start, end, reference bases, and\nalternative bases. If no such variant exists, a new one will be created.\n\nWhen variants are merged, the call information from the new variant\nis added to the existing variant, and Variant info fields are merged\nas specified in\ninfoMergeConfig.\nAs a special case, for single-sample VCF files, QUAL and FILTER fields will\nbe moved to the call level; these are sometimes interpreted in a\ncall-specific context.\nImported VCF headers are appended to the metadata already in a variant set.",
	//   "flatPath": "v1/variants:import",
	//   "httpMethod": "POST",
	//   "id": "genomics.variants.import",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/variants:import",
	//   "request": {
	//     "$ref": "ImportVariantsRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/devstorage.read_write",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variants.merge":

type VariantsMergeCall struct {
	s                    *Service
	mergevariantsrequest *MergeVariantsRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Merge: Merges the given variants with existing variants.
//
// For the definitions of variants and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Each variant will be
// merged with an existing variant that matches its reference
// sequence,
// start, end, reference bases, and alternative bases. If no such
// variant
// exists, a new one will be created.
//
// When variants are merged, the call information from the new
// variant
// is added to the existing variant. Variant info fields are merged
// as
// specified in the
// infoMergeConfig
// field of the MergeVariantsRequest.
//
// Please exercise caution when using this method!  It is easy to
// introduce
// mistakes in existing variants and difficult to back out of them.
// For
// example,
// suppose you were trying to merge a new variant with an existing one
// and
// both
// variants contain calls that belong to callsets with the same callset
// ID.
//
//     // Existing variant - irrelevant fields trimmed for clarity
//     {
//         "variantSetId": "10473108253681171589",
//         "referenceName": "1",
//         "start": "10582",
//         "referenceBases": "G",
//         "alternateBases": [
//             "A"
//         ],
//         "calls": [
//             {
//                 "callSetId": "10473108253681171589-0",
//                 "callSetName": "CALLSET0",
//                 "genotype": [
//                     0,
//                     1
//                 ],
//             }
//         ]
//     }
//
//     // New variant with conflicting call information
//     {
//         "variantSetId": "10473108253681171589",
//         "referenceName": "1",
//         "start": "10582",
//         "referenceBases": "G",
//         "alternateBases": [
//             "A"
//         ],
//         "calls": [
//             {
//                 "callSetId": "10473108253681171589-0",
//                 "callSetName": "CALLSET0",
//                 "genotype": [
//                     1,
//                     1
//                 ],
//             }
//         ]
//     }
//
// The resulting merged variant would overwrite the existing calls with
// those
// from the new variant:
//
//     {
//         "variantSetId": "10473108253681171589",
//         "referenceName": "1",
//         "start": "10582",
//         "referenceBases": "G",
//         "alternateBases": [
//             "A"
//         ],
//         "calls": [
//             {
//                 "callSetId": "10473108253681171589-0",
//                 "callSetName": "CALLSET0",
//                 "genotype": [
//                     1,
//                     1
//                 ],
//             }
//         ]
//     }
//
// This may be the desired outcome, but it is up to the user to
// determine if
// if that is indeed the case.
func (r *VariantsService) Merge(mergevariantsrequest *MergeVariantsRequest) *VariantsMergeCall {
	c := &VariantsMergeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.mergevariantsrequest = mergevariantsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsMergeCall) Fields(s ...googleapi.Field) *VariantsMergeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsMergeCall) Context(ctx context.Context) *VariantsMergeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsMergeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsMergeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.mergevariantsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variants:merge")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variants.merge" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *VariantsMergeCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Merges the given variants with existing variants.\n\nFor the definitions of variants and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nEach variant will be\nmerged with an existing variant that matches its reference sequence,\nstart, end, reference bases, and alternative bases. If no such variant\nexists, a new one will be created.\n\nWhen variants are merged, the call information from the new variant\nis added to the existing variant. Variant info fields are merged as\nspecified in the\ninfoMergeConfig\nfield of the MergeVariantsRequest.\n\nPlease exercise caution when using this method!  It is easy to introduce\nmistakes in existing variants and difficult to back out of them.  For\nexample,\nsuppose you were trying to merge a new variant with an existing one and\nboth\nvariants contain calls that belong to callsets with the same callset ID.\n\n    // Existing variant - irrelevant fields trimmed for clarity\n    {\n        \"variantSetId\": \"10473108253681171589\",\n        \"referenceName\": \"1\",\n        \"start\": \"10582\",\n        \"referenceBases\": \"G\",\n        \"alternateBases\": [\n            \"A\"\n        ],\n        \"calls\": [\n            {\n                \"callSetId\": \"10473108253681171589-0\",\n                \"callSetName\": \"CALLSET0\",\n                \"genotype\": [\n                    0,\n                    1\n                ],\n            }\n        ]\n    }\n\n    // New variant with conflicting call information\n    {\n        \"variantSetId\": \"10473108253681171589\",\n        \"referenceName\": \"1\",\n        \"start\": \"10582\",\n        \"referenceBases\": \"G\",\n        \"alternateBases\": [\n            \"A\"\n        ],\n        \"calls\": [\n            {\n                \"callSetId\": \"10473108253681171589-0\",\n                \"callSetName\": \"CALLSET0\",\n                \"genotype\": [\n                    1,\n                    1\n                ],\n            }\n        ]\n    }\n\nThe resulting merged variant would overwrite the existing calls with those\nfrom the new variant:\n\n    {\n        \"variantSetId\": \"10473108253681171589\",\n        \"referenceName\": \"1\",\n        \"start\": \"10582\",\n        \"referenceBases\": \"G\",\n        \"alternateBases\": [\n            \"A\"\n        ],\n        \"calls\": [\n            {\n                \"callSetId\": \"10473108253681171589-0\",\n                \"callSetName\": \"CALLSET0\",\n                \"genotype\": [\n                    1,\n                    1\n                ],\n            }\n        ]\n    }\n\nThis may be the desired outcome, but it is up to the user to determine if\nif that is indeed the case.",
	//   "flatPath": "v1/variants:merge",
	//   "httpMethod": "POST",
	//   "id": "genomics.variants.merge",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/variants:merge",
	//   "request": {
	//     "$ref": "MergeVariantsRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variants.patch":

type VariantsPatchCall struct {
	s          *Service
	variantId  string
	variant    *Variant
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a variant.
//
// For the definitions of variants and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// This method supports patch semantics. Returns the modified variant
// without
// its calls.
func (r *VariantsService) Patch(variantId string, variant *Variant) *VariantsPatchCall {
	c := &VariantsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variantId = variantId
	c.variant = variant
	return c
}

// UpdateMask sets the optional parameter "updateMask": An optional mask
// specifying which fields to update. At this time, mutable
// fields are names and
// info. Acceptable values are "names" and
// "info". If unspecified, all mutable fields will be updated.
func (c *VariantsPatchCall) UpdateMask(updateMask string) *VariantsPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsPatchCall) Fields(s ...googleapi.Field) *VariantsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsPatchCall) Context(ctx context.Context) *VariantsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.variant)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variants/{variantId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"variantId": c.variantId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variants.patch" call.
// Exactly one of *Variant or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Variant.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *VariantsPatchCall) Do(opts ...googleapi.CallOption) (*Variant, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Variant{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a variant.\n\nFor the definitions of variants and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nThis method supports patch semantics. Returns the modified variant without\nits calls.",
	//   "flatPath": "v1/variants/{variantId}",
	//   "httpMethod": "PATCH",
	//   "id": "genomics.variants.patch",
	//   "parameterOrder": [
	//     "variantId"
	//   ],
	//   "parameters": {
	//     "updateMask": {
	//       "description": "An optional mask specifying which fields to update. At this time, mutable\nfields are names and\ninfo. Acceptable values are \"names\" and\n\"info\". If unspecified, all mutable fields will be updated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "variantId": {
	//       "description": "The ID of the variant to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/variants/{variantId}",
	//   "request": {
	//     "$ref": "Variant"
	//   },
	//   "response": {
	//     "$ref": "Variant"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variants.search":

type VariantsSearchCall struct {
	s                     *Service
	searchvariantsrequest *SearchVariantsRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Search: Gets a list of variants matching the criteria.
//
// For the definitions of variants and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.searchVariants](https://g
// ithub.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/variantmet
// hods.avdl#L126).
func (r *VariantsService) Search(searchvariantsrequest *SearchVariantsRequest) *VariantsSearchCall {
	c := &VariantsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchvariantsrequest = searchvariantsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsSearchCall) Fields(s ...googleapi.Field) *VariantsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsSearchCall) Context(ctx context.Context) *VariantsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchvariantsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variants/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variants.search" call.
// Exactly one of *SearchVariantsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SearchVariantsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *VariantsSearchCall) Do(opts ...googleapi.CallOption) (*SearchVariantsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchVariantsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a list of variants matching the criteria.\n\nFor the definitions of variants and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.searchVariants](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/variantmethods.avdl#L126).",
	//   "flatPath": "v1/variants/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.variants.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/variants/search",
	//   "request": {
	//     "$ref": "SearchVariantsRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchVariantsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *VariantsSearchCall) Pages(ctx context.Context, f func(*SearchVariantsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchvariantsrequest.PageToken = pt }(c.searchvariantsrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchvariantsrequest.PageToken = x.NextPageToken
	}
}

// method id "genomics.variantsets.create":

type VariantsetsCreateCall struct {
	s          *Service
	variantset *VariantSet
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a new variant set.
//
// For the definitions of variant sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// The provided variant set must have a valid `datasetId` set - all
// other
// fields are optional. Note that the `id` field will be ignored, as
// this is
// assigned by the server.
func (r *VariantsetsService) Create(variantset *VariantSet) *VariantsetsCreateCall {
	c := &VariantsetsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variantset = variantset
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsetsCreateCall) Fields(s ...googleapi.Field) *VariantsetsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsetsCreateCall) Context(ctx context.Context) *VariantsetsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsetsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsetsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.variantset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variantsets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variantsets.create" call.
// Exactly one of *VariantSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *VariantSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *VariantsetsCreateCall) Do(opts ...googleapi.CallOption) (*VariantSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &VariantSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new variant set.\n\nFor the definitions of variant sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nThe provided variant set must have a valid `datasetId` set - all other\nfields are optional. Note that the `id` field will be ignored, as this is\nassigned by the server.",
	//   "flatPath": "v1/variantsets",
	//   "httpMethod": "POST",
	//   "id": "genomics.variantsets.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/variantsets",
	//   "request": {
	//     "$ref": "VariantSet"
	//   },
	//   "response": {
	//     "$ref": "VariantSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variantsets.delete":

type VariantsetsDeleteCall struct {
	s            *Service
	variantSetId string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Deletes a variant set including all variants, call sets, and
// calls within.
// This is not reversible.
//
// For the definitions of variant sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *VariantsetsService) Delete(variantSetId string) *VariantsetsDeleteCall {
	c := &VariantsetsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variantSetId = variantSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsetsDeleteCall) Fields(s ...googleapi.Field) *VariantsetsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsetsDeleteCall) Context(ctx context.Context) *VariantsetsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsetsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsetsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variantsets/{variantSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"variantSetId": c.variantSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variantsets.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *VariantsetsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Empty{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a variant set including all variants, call sets, and calls within.\nThis is not reversible.\n\nFor the definitions of variant sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/variantsets/{variantSetId}",
	//   "httpMethod": "DELETE",
	//   "id": "genomics.variantsets.delete",
	//   "parameterOrder": [
	//     "variantSetId"
	//   ],
	//   "parameters": {
	//     "variantSetId": {
	//       "description": "The ID of the variant set to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/variantsets/{variantSetId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variantsets.export":

type VariantsetsExportCall struct {
	s                       *Service
	variantSetId            string
	exportvariantsetrequest *ExportVariantSetRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
	header_                 http.Header
}

// Export: Exports variant set data to an external destination.
//
// For the definitions of variant sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *VariantsetsService) Export(variantSetId string, exportvariantsetrequest *ExportVariantSetRequest) *VariantsetsExportCall {
	c := &VariantsetsExportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variantSetId = variantSetId
	c.exportvariantsetrequest = exportvariantsetrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsetsExportCall) Fields(s ...googleapi.Field) *VariantsetsExportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsetsExportCall) Context(ctx context.Context) *VariantsetsExportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsetsExportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsetsExportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.exportvariantsetrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variantsets/{variantSetId}:export")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"variantSetId": c.variantSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variantsets.export" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *VariantsetsExportCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Exports variant set data to an external destination.\n\nFor the definitions of variant sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/variantsets/{variantSetId}:export",
	//   "httpMethod": "POST",
	//   "id": "genomics.variantsets.export",
	//   "parameterOrder": [
	//     "variantSetId"
	//   ],
	//   "parameters": {
	//     "variantSetId": {
	//       "description": "Required. The ID of the variant set that contains variant data which\nshould be exported. The caller must have READ access to this variant set.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/variantsets/{variantSetId}:export",
	//   "request": {
	//     "$ref": "ExportVariantSetRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery",
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variantsets.get":

type VariantsetsGetCall struct {
	s            *Service
	variantSetId string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a variant set by ID.
//
// For the definitions of variant sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *VariantsetsService) Get(variantSetId string) *VariantsetsGetCall {
	c := &VariantsetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variantSetId = variantSetId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsetsGetCall) Fields(s ...googleapi.Field) *VariantsetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *VariantsetsGetCall) IfNoneMatch(entityTag string) *VariantsetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsetsGetCall) Context(ctx context.Context) *VariantsetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsetsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variantsets/{variantSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"variantSetId": c.variantSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variantsets.get" call.
// Exactly one of *VariantSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *VariantSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *VariantsetsGetCall) Do(opts ...googleapi.CallOption) (*VariantSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &VariantSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a variant set by ID.\n\nFor the definitions of variant sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/variantsets/{variantSetId}",
	//   "httpMethod": "GET",
	//   "id": "genomics.variantsets.get",
	//   "parameterOrder": [
	//     "variantSetId"
	//   ],
	//   "parameters": {
	//     "variantSetId": {
	//       "description": "Required. The ID of the variant set.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/variantsets/{variantSetId}",
	//   "response": {
	//     "$ref": "VariantSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// method id "genomics.variantsets.patch":

type VariantsetsPatchCall struct {
	s            *Service
	variantSetId string
	variantset   *VariantSet
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Patch: Updates a variant set using patch semantics.
//
// For the definitions of variant sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
func (r *VariantsetsService) Patch(variantSetId string, variantset *VariantSet) *VariantsetsPatchCall {
	c := &VariantsetsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.variantSetId = variantSetId
	c.variantset = variantset
	return c
}

// UpdateMask sets the optional parameter "updateMask": An optional mask
// specifying which fields to update. Supported fields:
//
// * metadata.
// * name.
// * description.
//
// Leaving `updateMask` unset is equivalent to specifying all
// mutable
// fields.
func (c *VariantsetsPatchCall) UpdateMask(updateMask string) *VariantsetsPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsetsPatchCall) Fields(s ...googleapi.Field) *VariantsetsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsetsPatchCall) Context(ctx context.Context) *VariantsetsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsetsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsetsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.variantset)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variantsets/{variantSetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"variantSetId": c.variantSetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variantsets.patch" call.
// Exactly one of *VariantSet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *VariantSet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *VariantsetsPatchCall) Do(opts ...googleapi.CallOption) (*VariantSet, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &VariantSet{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a variant set using patch semantics.\n\nFor the definitions of variant sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)",
	//   "flatPath": "v1/variantsets/{variantSetId}",
	//   "httpMethod": "PATCH",
	//   "id": "genomics.variantsets.patch",
	//   "parameterOrder": [
	//     "variantSetId"
	//   ],
	//   "parameters": {
	//     "updateMask": {
	//       "description": "An optional mask specifying which fields to update. Supported fields:\n\n* metadata.\n* name.\n* description.\n\nLeaving `updateMask` unset is equivalent to specifying all mutable\nfields.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "variantSetId": {
	//       "description": "The ID of the variant to be updated (must already exist).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/variantsets/{variantSetId}",
	//   "request": {
	//     "$ref": "VariantSet"
	//   },
	//   "response": {
	//     "$ref": "VariantSet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics"
	//   ]
	// }

}

// method id "genomics.variantsets.search":

type VariantsetsSearchCall struct {
	s                        *Service
	searchvariantsetsrequest *SearchVariantSetsRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// Search: Returns a list of all variant sets matching search
// criteria.
//
// For the definitions of variant sets and other genomics resources,
// see
// [Fundamentals of
// Google
// Genomics](https://cloud.google.com/genomics/fundamentals-of-goo
// gle-genomics)
//
// Implements
// [GlobalAllianceApi.searchVariantSets](https:
// //github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/variant
// methods.avdl#L49).
func (r *VariantsetsService) Search(searchvariantsetsrequest *SearchVariantSetsRequest) *VariantsetsSearchCall {
	c := &VariantsetsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.searchvariantsetsrequest = searchvariantsetsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VariantsetsSearchCall) Fields(s ...googleapi.Field) *VariantsetsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VariantsetsSearchCall) Context(ctx context.Context) *VariantsetsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VariantsetsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VariantsetsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.searchvariantsetsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/variantsets/search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "genomics.variantsets.search" call.
// Exactly one of *SearchVariantSetsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *SearchVariantSetsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *VariantsetsSearchCall) Do(opts ...googleapi.CallOption) (*SearchVariantSetsResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &SearchVariantSetsResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of all variant sets matching search criteria.\n\nFor the definitions of variant sets and other genomics resources, see\n[Fundamentals of Google\nGenomics](https://cloud.google.com/genomics/fundamentals-of-google-genomics)\n\nImplements\n[GlobalAllianceApi.searchVariantSets](https://github.com/ga4gh/schemas/blob/v0.5.1/src/main/resources/avro/variantmethods.avdl#L49).",
	//   "flatPath": "v1/variantsets/search",
	//   "httpMethod": "POST",
	//   "id": "genomics.variantsets.search",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/variantsets/search",
	//   "request": {
	//     "$ref": "SearchVariantSetsRequest"
	//   },
	//   "response": {
	//     "$ref": "SearchVariantSetsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/genomics",
	//     "https://www.googleapis.com/auth/genomics.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *VariantsetsSearchCall) Pages(ctx context.Context, f func(*SearchVariantSetsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.searchvariantsetsrequest.PageToken = pt }(c.searchvariantsetsrequest.PageToken) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.searchvariantsetsrequest.PageToken = x.NextPageToken
	}
}
