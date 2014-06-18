// Package books provides access to the Books API.
//
// See https://developers.google.com/books/docs/v1/getting_started
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/books/v1"
//   ...
//   booksService, err := books.New(oauthHttpClient)
package books

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

const apiId = "books:v1"
const apiName = "books"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/books/v1/"

// OAuth2 scopes used by this API.
const (
	// Manage your books
	BooksScope = "https://www.googleapis.com/auth/books"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Bookshelves = NewBookshelvesService(s)
	s.Cloudloading = NewCloudloadingService(s)
	s.Layers = NewLayersService(s)
	s.Myconfig = NewMyconfigService(s)
	s.Mylibrary = NewMylibraryService(s)
	s.Volumes = NewVolumesService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Bookshelves *BookshelvesService

	Cloudloading *CloudloadingService

	Layers *LayersService

	Myconfig *MyconfigService

	Mylibrary *MylibraryService

	Volumes *VolumesService
}

func NewBookshelvesService(s *Service) *BookshelvesService {
	rs := &BookshelvesService{s: s}
	rs.Volumes = NewBookshelvesVolumesService(s)
	return rs
}

type BookshelvesService struct {
	s *Service

	Volumes *BookshelvesVolumesService
}

func NewBookshelvesVolumesService(s *Service) *BookshelvesVolumesService {
	rs := &BookshelvesVolumesService{s: s}
	return rs
}

type BookshelvesVolumesService struct {
	s *Service
}

func NewCloudloadingService(s *Service) *CloudloadingService {
	rs := &CloudloadingService{s: s}
	return rs
}

type CloudloadingService struct {
	s *Service
}

func NewLayersService(s *Service) *LayersService {
	rs := &LayersService{s: s}
	rs.AnnotationData = NewLayersAnnotationDataService(s)
	rs.VolumeAnnotations = NewLayersVolumeAnnotationsService(s)
	return rs
}

type LayersService struct {
	s *Service

	AnnotationData *LayersAnnotationDataService

	VolumeAnnotations *LayersVolumeAnnotationsService
}

func NewLayersAnnotationDataService(s *Service) *LayersAnnotationDataService {
	rs := &LayersAnnotationDataService{s: s}
	return rs
}

type LayersAnnotationDataService struct {
	s *Service
}

func NewLayersVolumeAnnotationsService(s *Service) *LayersVolumeAnnotationsService {
	rs := &LayersVolumeAnnotationsService{s: s}
	return rs
}

type LayersVolumeAnnotationsService struct {
	s *Service
}

func NewMyconfigService(s *Service) *MyconfigService {
	rs := &MyconfigService{s: s}
	return rs
}

type MyconfigService struct {
	s *Service
}

func NewMylibraryService(s *Service) *MylibraryService {
	rs := &MylibraryService{s: s}
	rs.Annotations = NewMylibraryAnnotationsService(s)
	rs.Bookshelves = NewMylibraryBookshelvesService(s)
	rs.Readingpositions = NewMylibraryReadingpositionsService(s)
	return rs
}

type MylibraryService struct {
	s *Service

	Annotations *MylibraryAnnotationsService

	Bookshelves *MylibraryBookshelvesService

	Readingpositions *MylibraryReadingpositionsService
}

func NewMylibraryAnnotationsService(s *Service) *MylibraryAnnotationsService {
	rs := &MylibraryAnnotationsService{s: s}
	return rs
}

type MylibraryAnnotationsService struct {
	s *Service
}

func NewMylibraryBookshelvesService(s *Service) *MylibraryBookshelvesService {
	rs := &MylibraryBookshelvesService{s: s}
	rs.Volumes = NewMylibraryBookshelvesVolumesService(s)
	return rs
}

type MylibraryBookshelvesService struct {
	s *Service

	Volumes *MylibraryBookshelvesVolumesService
}

func NewMylibraryBookshelvesVolumesService(s *Service) *MylibraryBookshelvesVolumesService {
	rs := &MylibraryBookshelvesVolumesService{s: s}
	return rs
}

type MylibraryBookshelvesVolumesService struct {
	s *Service
}

func NewMylibraryReadingpositionsService(s *Service) *MylibraryReadingpositionsService {
	rs := &MylibraryReadingpositionsService{s: s}
	return rs
}

type MylibraryReadingpositionsService struct {
	s *Service
}

func NewVolumesService(s *Service) *VolumesService {
	rs := &VolumesService{s: s}
	rs.Associated = NewVolumesAssociatedService(s)
	rs.Mybooks = NewVolumesMybooksService(s)
	rs.Recommended = NewVolumesRecommendedService(s)
	rs.Useruploaded = NewVolumesUseruploadedService(s)
	return rs
}

type VolumesService struct {
	s *Service

	Associated *VolumesAssociatedService

	Mybooks *VolumesMybooksService

	Recommended *VolumesRecommendedService

	Useruploaded *VolumesUseruploadedService
}

func NewVolumesAssociatedService(s *Service) *VolumesAssociatedService {
	rs := &VolumesAssociatedService{s: s}
	return rs
}

type VolumesAssociatedService struct {
	s *Service
}

func NewVolumesMybooksService(s *Service) *VolumesMybooksService {
	rs := &VolumesMybooksService{s: s}
	return rs
}

type VolumesMybooksService struct {
	s *Service
}

func NewVolumesRecommendedService(s *Service) *VolumesRecommendedService {
	rs := &VolumesRecommendedService{s: s}
	return rs
}

type VolumesRecommendedService struct {
	s *Service
}

func NewVolumesUseruploadedService(s *Service) *VolumesUseruploadedService {
	rs := &VolumesUseruploadedService{s: s}
	return rs
}

type VolumesUseruploadedService struct {
	s *Service
}

type Annotation struct {
	// AfterSelectedText: Anchor text after excerpt. For requests, if the
	// user bookmarked a screen that has no flowing text on it, then this
	// field should be empty.
	AfterSelectedText string `json:"afterSelectedText,omitempty"`

	// BeforeSelectedText: Anchor text before excerpt. For requests, if the
	// user bookmarked a screen that has no flowing text on it, then this
	// field should be empty.
	BeforeSelectedText string `json:"beforeSelectedText,omitempty"`

	// ClientVersionRanges: Selection ranges sent from the client.
	ClientVersionRanges *AnnotationClientVersionRanges `json:"clientVersionRanges,omitempty"`

	// Created: Timestamp for the created time of this annotation.
	Created string `json:"created,omitempty"`

	// CurrentVersionRanges: Selection ranges for the most recent content
	// version.
	CurrentVersionRanges *AnnotationCurrentVersionRanges `json:"currentVersionRanges,omitempty"`

	// Data: User-created data for this annotation.
	Data string `json:"data,omitempty"`

	// Deleted: Indicates that this annotation is deleted.
	Deleted bool `json:"deleted,omitempty"`

	// HighlightStyle: The highlight style for this annotation.
	HighlightStyle string `json:"highlightStyle,omitempty"`

	// Id: Id of this annotation, in the form of a GUID.
	Id string `json:"id,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// LayerId: The layer this annotation is for.
	LayerId string `json:"layerId,omitempty"`

	LayerSummary *AnnotationLayerSummary `json:"layerSummary,omitempty"`

	// PageIds: Pages that this annotation spans.
	PageIds []string `json:"pageIds,omitempty"`

	// SelectedText: Excerpt from the volume.
	SelectedText string `json:"selectedText,omitempty"`

	// SelfLink: URL to this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Updated: Timestamp for the last time this annotation was modified.
	Updated string `json:"updated,omitempty"`

	// VolumeId: The volume that this annotation belongs to.
	VolumeId string `json:"volumeId,omitempty"`
}

type AnnotationClientVersionRanges struct {
	// CfiRange: Range in CFI format for this annotation sent by client.
	CfiRange *BooksAnnotationsRange `json:"cfiRange,omitempty"`

	// ContentVersion: Content version the client sent in.
	ContentVersion string `json:"contentVersion,omitempty"`

	// GbImageRange: Range in GB image format for this annotation sent by
	// client.
	GbImageRange *BooksAnnotationsRange `json:"gbImageRange,omitempty"`

	// GbTextRange: Range in GB text format for this annotation sent by
	// client.
	GbTextRange *BooksAnnotationsRange `json:"gbTextRange,omitempty"`

	// ImageCfiRange: Range in image CFI format for this annotation sent by
	// client.
	ImageCfiRange *BooksAnnotationsRange `json:"imageCfiRange,omitempty"`
}

type AnnotationCurrentVersionRanges struct {
	// CfiRange: Range in CFI format for this annotation for version above.
	CfiRange *BooksAnnotationsRange `json:"cfiRange,omitempty"`

	// ContentVersion: Content version applicable to ranges below.
	ContentVersion string `json:"contentVersion,omitempty"`

	// GbImageRange: Range in GB image format for this annotation for
	// version above.
	GbImageRange *BooksAnnotationsRange `json:"gbImageRange,omitempty"`

	// GbTextRange: Range in GB text format for this annotation for version
	// above.
	GbTextRange *BooksAnnotationsRange `json:"gbTextRange,omitempty"`

	// ImageCfiRange: Range in image CFI format for this annotation for
	// version above.
	ImageCfiRange *BooksAnnotationsRange `json:"imageCfiRange,omitempty"`
}

type AnnotationLayerSummary struct {
	// AllowedCharacterCount: Maximum allowed characters on this layer,
	// especially for the "copy" layer.
	AllowedCharacterCount int64 `json:"allowedCharacterCount,omitempty"`

	// LimitType: Type of limitation on this layer. "limited" or "unlimited"
	// for the "copy" layer.
	LimitType string `json:"limitType,omitempty"`

	// RemainingCharacterCount: Remaining allowed characters on this layer,
	// especially for the "copy" layer.
	RemainingCharacterCount int64 `json:"remainingCharacterCount,omitempty"`
}

type Annotationdata struct {
	// AnnotationType: The type of annotation this data is for.
	AnnotationType string `json:"annotationType,omitempty"`

	Data interface{} `json:"data,omitempty"`

	// Encoded_data: Base64 encoded data for this annotation data.
	Encoded_data string `json:"encoded_data,omitempty"`

	// Id: Unique id for this annotation data.
	Id string `json:"id,omitempty"`

	// Kind: Resource Type
	Kind string `json:"kind,omitempty"`

	// LayerId: The Layer id for this data. *
	LayerId string `json:"layerId,omitempty"`

	// SelfLink: URL for this resource. *
	SelfLink string `json:"selfLink,omitempty"`

	// Updated: Timestamp for the last time this data was updated. (RFC 3339
	// UTC date-time format).
	Updated string `json:"updated,omitempty"`

	// VolumeId: The volume id for this data. *
	VolumeId string `json:"volumeId,omitempty"`
}

type Annotations struct {
	// Items: A list of annotations.
	Items []*Annotation `json:"items,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Token to pass in for pagination for the next page.
	// This will not be present if this request does not have more results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TotalItems: Total number of annotations found. This may be greater
	// than the number of notes returned in this response if results have
	// been paginated.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type AnnotationsSummary struct {
	Kind string `json:"kind,omitempty"`

	Layers []*AnnotationsSummaryLayers `json:"layers,omitempty"`
}

type AnnotationsSummaryLayers struct {
	AllowedCharacterCount int64 `json:"allowedCharacterCount,omitempty"`

	LayerId string `json:"layerId,omitempty"`

	LimitType string `json:"limitType,omitempty"`

	RemainingCharacterCount int64 `json:"remainingCharacterCount,omitempty"`

	Updated string `json:"updated,omitempty"`
}

type Annotationsdata struct {
	// Items: A list of Annotation Data.
	Items []*Annotationdata `json:"items,omitempty"`

	// Kind: Resource type
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Token to pass in for pagination for the next page.
	// This will not be present if this request does not have more results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TotalItems: The total number of volume annotations found.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type BooksAnnotationsRange struct {
	// EndOffset: The offset from the ending position.
	EndOffset string `json:"endOffset,omitempty"`

	// EndPosition: The ending position for the range.
	EndPosition string `json:"endPosition,omitempty"`

	// StartOffset: The offset from the starting position.
	StartOffset string `json:"startOffset,omitempty"`

	// StartPosition: The starting position for the range.
	StartPosition string `json:"startPosition,omitempty"`
}

type BooksCloudloadingResource struct {
	Author string `json:"author,omitempty"`

	ProcessingState string `json:"processingState,omitempty"`

	Title string `json:"title,omitempty"`

	VolumeId string `json:"volumeId,omitempty"`
}

type BooksVolumesRecommendedRateResponse struct {
	Consistency_token string `json:"consistency_token,omitempty"`
}

type Bookshelf struct {
	// Access: Whether this bookshelf is PUBLIC or PRIVATE.
	Access string `json:"access,omitempty"`

	// Created: Created time for this bookshelf (formatted UTC timestamp
	// with millisecond resolution).
	Created string `json:"created,omitempty"`

	// Description: Description of this bookshelf.
	Description string `json:"description,omitempty"`

	// Id: Id of this bookshelf, only unique by user.
	Id int64 `json:"id,omitempty"`

	// Kind: Resource type for bookshelf metadata.
	Kind string `json:"kind,omitempty"`

	// SelfLink: URL to this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Title: Title of this bookshelf.
	Title string `json:"title,omitempty"`

	// Updated: Last modified time of this bookshelf (formatted UTC
	// timestamp with millisecond resolution).
	Updated string `json:"updated,omitempty"`

	// VolumeCount: Number of volumes in this bookshelf.
	VolumeCount int64 `json:"volumeCount,omitempty"`

	// VolumesLastUpdated: Last time a volume was added or removed from this
	// bookshelf (formatted UTC timestamp with millisecond resolution).
	VolumesLastUpdated string `json:"volumesLastUpdated,omitempty"`
}

type Bookshelves struct {
	// Items: A list of bookshelves.
	Items []*Bookshelf `json:"items,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`
}

type ConcurrentAccessRestriction struct {
	// DeviceAllowed: Whether access is granted for this (user, device,
	// volume).
	DeviceAllowed bool `json:"deviceAllowed,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// MaxConcurrentDevices: The maximum number of concurrent access
	// licenses for this volume.
	MaxConcurrentDevices int64 `json:"maxConcurrentDevices,omitempty"`

	// Message: Error/warning message.
	Message string `json:"message,omitempty"`

	// Nonce: Client nonce for verification. Download access and
	// client-validation only.
	Nonce string `json:"nonce,omitempty"`

	// ReasonCode: Error/warning reason code.
	ReasonCode string `json:"reasonCode,omitempty"`

	// Restricted: Whether this volume has any concurrent access
	// restrictions.
	Restricted bool `json:"restricted,omitempty"`

	// Signature: Response signature.
	Signature string `json:"signature,omitempty"`

	// Source: Client app identifier for verification. Download access and
	// client-validation only.
	Source string `json:"source,omitempty"`

	// TimeWindowSeconds: Time in seconds for license auto-expiration.
	TimeWindowSeconds int64 `json:"timeWindowSeconds,omitempty"`

	// VolumeId: Identifies the volume for which this entry applies.
	VolumeId string `json:"volumeId,omitempty"`
}

type Dictlayerdata struct {
	Common *DictlayerdataCommon `json:"common,omitempty"`

	Dict *DictlayerdataDict `json:"dict,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type DictlayerdataCommon struct {
	// Title: The display title and localized canonical name to use when
	// searching for this entity on Google search.
	Title string `json:"title,omitempty"`
}

type DictlayerdataDict struct {
	// Source: The source, url and attribution for this dictionary data.
	Source *DictlayerdataDictSource `json:"source,omitempty"`

	Words []*DictlayerdataDictWords `json:"words,omitempty"`
}

type DictlayerdataDictSource struct {
	Attribution string `json:"attribution,omitempty"`

	Url string `json:"url,omitempty"`
}

type DictlayerdataDictWords struct {
	Derivatives []*DictlayerdataDictWordsDerivatives `json:"derivatives,omitempty"`

	Examples []*DictlayerdataDictWordsExamples `json:"examples,omitempty"`

	Senses []*DictlayerdataDictWordsSenses `json:"senses,omitempty"`

	// Source: The words with different meanings but not related words, e.g.
	// "go" (game) and "go" (verb).
	Source *DictlayerdataDictWordsSource `json:"source,omitempty"`
}

type DictlayerdataDictWordsDerivatives struct {
	Source *DictlayerdataDictWordsDerivativesSource `json:"source,omitempty"`

	Text string `json:"text,omitempty"`
}

type DictlayerdataDictWordsDerivativesSource struct {
	Attribution string `json:"attribution,omitempty"`

	Url string `json:"url,omitempty"`
}

type DictlayerdataDictWordsExamples struct {
	Source *DictlayerdataDictWordsExamplesSource `json:"source,omitempty"`

	Text string `json:"text,omitempty"`
}

type DictlayerdataDictWordsExamplesSource struct {
	Attribution string `json:"attribution,omitempty"`

	Url string `json:"url,omitempty"`
}

type DictlayerdataDictWordsSenses struct {
	Conjugations []*DictlayerdataDictWordsSensesConjugations `json:"conjugations,omitempty"`

	Definitions []*DictlayerdataDictWordsSensesDefinitions `json:"definitions,omitempty"`

	PartOfSpeech string `json:"partOfSpeech,omitempty"`

	Pronunciation string `json:"pronunciation,omitempty"`

	PronunciationUrl string `json:"pronunciationUrl,omitempty"`

	Source *DictlayerdataDictWordsSensesSource `json:"source,omitempty"`

	Syllabification string `json:"syllabification,omitempty"`

	Synonyms []*DictlayerdataDictWordsSensesSynonyms `json:"synonyms,omitempty"`
}

type DictlayerdataDictWordsSensesConjugations struct {
	Type string `json:"type,omitempty"`

	Value string `json:"value,omitempty"`
}

type DictlayerdataDictWordsSensesDefinitions struct {
	Definition string `json:"definition,omitempty"`

	Examples []*DictlayerdataDictWordsSensesDefinitionsExamples `json:"examples,omitempty"`
}

type DictlayerdataDictWordsSensesDefinitionsExamples struct {
	Source *DictlayerdataDictWordsSensesDefinitionsExamplesSource `json:"source,omitempty"`

	Text string `json:"text,omitempty"`
}

type DictlayerdataDictWordsSensesDefinitionsExamplesSource struct {
	Attribution string `json:"attribution,omitempty"`

	Url string `json:"url,omitempty"`
}

type DictlayerdataDictWordsSensesSource struct {
	Attribution string `json:"attribution,omitempty"`

	Url string `json:"url,omitempty"`
}

type DictlayerdataDictWordsSensesSynonyms struct {
	Source *DictlayerdataDictWordsSensesSynonymsSource `json:"source,omitempty"`

	Text string `json:"text,omitempty"`
}

type DictlayerdataDictWordsSensesSynonymsSource struct {
	Attribution string `json:"attribution,omitempty"`

	Url string `json:"url,omitempty"`
}

type DictlayerdataDictWordsSource struct {
	Attribution string `json:"attribution,omitempty"`

	Url string `json:"url,omitempty"`
}

type DownloadAccessRestriction struct {
	// DeviceAllowed: If restricted, whether access is granted for this
	// (user, device, volume).
	DeviceAllowed bool `json:"deviceAllowed,omitempty"`

	// DownloadsAcquired: If restricted, the number of content download
	// licenses already acquired (including the requesting client, if
	// licensed).
	DownloadsAcquired int64 `json:"downloadsAcquired,omitempty"`

	// JustAcquired: If deviceAllowed, whether access was just acquired with
	// this request.
	JustAcquired bool `json:"justAcquired,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// MaxDownloadDevices: If restricted, the maximum number of content
	// download licenses for this volume.
	MaxDownloadDevices int64 `json:"maxDownloadDevices,omitempty"`

	// Message: Error/warning message.
	Message string `json:"message,omitempty"`

	// Nonce: Client nonce for verification. Download access and
	// client-validation only.
	Nonce string `json:"nonce,omitempty"`

	// ReasonCode: Error/warning reason code. Additional codes may be added
	// in the future. 0 OK 100 ACCESS_DENIED_PUBLISHER_LIMIT 101
	// ACCESS_DENIED_LIMIT 200 WARNING_USED_LAST_ACCESS
	ReasonCode string `json:"reasonCode,omitempty"`

	// Restricted: Whether this volume has any download access restrictions.
	Restricted bool `json:"restricted,omitempty"`

	// Signature: Response signature.
	Signature string `json:"signature,omitempty"`

	// Source: Client app identifier for verification. Download access and
	// client-validation only.
	Source string `json:"source,omitempty"`

	// VolumeId: Identifies the volume for which this entry applies.
	VolumeId string `json:"volumeId,omitempty"`
}

type DownloadAccesses struct {
	// DownloadAccessList: A list of download access responses.
	DownloadAccessList []*DownloadAccessRestriction `json:"downloadAccessList,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`
}

type Geolayerdata struct {
	Common *GeolayerdataCommon `json:"common,omitempty"`

	Geo *GeolayerdataGeo `json:"geo,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type GeolayerdataCommon struct {
	// Lang: The language of the information url and description.
	Lang string `json:"lang,omitempty"`

	// PreviewImageUrl: The URL for the preview image information.
	PreviewImageUrl string `json:"previewImageUrl,omitempty"`

	// Snippet: The description for this location.
	Snippet string `json:"snippet,omitempty"`

	// SnippetUrl: The URL for information for this location. Ex: wikipedia
	// link.
	SnippetUrl string `json:"snippetUrl,omitempty"`

	// Title: The display title and localized canonical name to use when
	// searching for this entity on Google search.
	Title string `json:"title,omitempty"`
}

type GeolayerdataGeo struct {
	// Boundary: The boundary of the location as a set of loops containing
	// pairs of latitude, longitude coordinates.
	Boundary [][]*GeolayerdataGeoBoundaryItem `json:"boundary,omitempty"`

	// CachePolicy: The cache policy active for this data. EX: UNRESTRICTED,
	// RESTRICTED, NEVER
	CachePolicy string `json:"cachePolicy,omitempty"`

	// CountryCode: The country code of the location.
	CountryCode string `json:"countryCode,omitempty"`

	// Latitude: The latitude of the location.
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: The longitude of the location.
	Longitude float64 `json:"longitude,omitempty"`

	// MapType: The type of map that should be used for this location. EX:
	// HYBRID, ROADMAP, SATELLITE, TERRAIN
	MapType string `json:"mapType,omitempty"`

	// Viewport: The viewport for showing this location. This is a latitude,
	// longitude rectangle.
	Viewport *GeolayerdataGeoViewport `json:"viewport,omitempty"`

	// Zoom: The Zoom level to use for the map. Zoom levels between 0 (the
	// lowest zoom level, in which the entire world can be seen on one map)
	// to 21+ (down to individual buildings). See:
	// https://developers.google.com/maps/documentation/staticmaps/#Zoomlevel
	// s
	Zoom int64 `json:"zoom,omitempty"`
}

type GeolayerdataGeoBoundaryItem struct {
	Latitude int64 `json:"latitude,omitempty"`

	Longitude int64 `json:"longitude,omitempty"`
}

type GeolayerdataGeoViewport struct {
	Hi *GeolayerdataGeoViewportHi `json:"hi,omitempty"`

	Lo *GeolayerdataGeoViewportLo `json:"lo,omitempty"`
}

type GeolayerdataGeoViewportHi struct {
	Latitude float64 `json:"latitude,omitempty"`

	Longitude float64 `json:"longitude,omitempty"`
}

type GeolayerdataGeoViewportLo struct {
	Latitude float64 `json:"latitude,omitempty"`

	Longitude float64 `json:"longitude,omitempty"`
}

type Layersummaries struct {
	// Items: A list of layer summary items.
	Items []*Layersummary `json:"items,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// TotalItems: The total number of layer summaries found.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type Layersummary struct {
	// AnnotationCount: The number of annotations for this layer.
	AnnotationCount int64 `json:"annotationCount,omitempty"`

	// AnnotationTypes: The list of annotation types contained for this
	// layer.
	AnnotationTypes []string `json:"annotationTypes,omitempty"`

	// AnnotationsDataLink: Link to get data for this annotation.
	AnnotationsDataLink string `json:"annotationsDataLink,omitempty"`

	// AnnotationsLink: The link to get the annotations for this layer.
	AnnotationsLink string `json:"annotationsLink,omitempty"`

	// ContentVersion: The content version this resource is for.
	ContentVersion string `json:"contentVersion,omitempty"`

	// DataCount: The number of data items for this layer.
	DataCount int64 `json:"dataCount,omitempty"`

	// Id: Unique id of this layer summary.
	Id string `json:"id,omitempty"`

	// Kind: Resource Type
	Kind string `json:"kind,omitempty"`

	// LayerId: The layer id for this summary.
	LayerId string `json:"layerId,omitempty"`

	// SelfLink: URL to this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Updated: Timestamp for the last time an item in this layer was
	// updated. (RFC 3339 UTC date-time format).
	Updated string `json:"updated,omitempty"`

	// VolumeAnnotationsVersion: The current version of this layer's volume
	// annotations. Note that this version applies only to the data in the
	// books.layers.volumeAnnotations.* responses. The actual annotation
	// data is versioned separately.
	VolumeAnnotationsVersion string `json:"volumeAnnotationsVersion,omitempty"`

	// VolumeId: The volume id this resource is for.
	VolumeId string `json:"volumeId,omitempty"`
}

type ReadingPosition struct {
	// EpubCfiPosition: Position in an EPUB as a CFI.
	EpubCfiPosition string `json:"epubCfiPosition,omitempty"`

	// GbImagePosition: Position in a volume for image-based content.
	GbImagePosition string `json:"gbImagePosition,omitempty"`

	// GbTextPosition: Position in a volume for text-based content.
	GbTextPosition string `json:"gbTextPosition,omitempty"`

	// Kind: Resource type for a reading position.
	Kind string `json:"kind,omitempty"`

	// PdfPosition: Position in a PDF file.
	PdfPosition string `json:"pdfPosition,omitempty"`

	// Updated: Timestamp when this reading position was last updated
	// (formatted UTC timestamp with millisecond resolution).
	Updated string `json:"updated,omitempty"`

	// VolumeId: Volume id associated with this reading position.
	VolumeId string `json:"volumeId,omitempty"`
}

type RequestAccess struct {
	// ConcurrentAccess: A concurrent access response.
	ConcurrentAccess *ConcurrentAccessRestriction `json:"concurrentAccess,omitempty"`

	// DownloadAccess: A download access response.
	DownloadAccess *DownloadAccessRestriction `json:"downloadAccess,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`
}

type Review struct {
	// Author: Author of this review.
	Author *ReviewAuthor `json:"author,omitempty"`

	// Content: Review text.
	Content string `json:"content,omitempty"`

	// Date: Date of this review.
	Date string `json:"date,omitempty"`

	// FullTextUrl: URL for the full review text, for reviews gathered from
	// the web.
	FullTextUrl string `json:"fullTextUrl,omitempty"`

	// Kind: Resource type for a review.
	Kind string `json:"kind,omitempty"`

	// Rating: Star rating for this review. Possible values are ONE, TWO,
	// THREE, FOUR, FIVE or NOT_RATED.
	Rating string `json:"rating,omitempty"`

	// Source: Information regarding the source of this review, when the
	// review is not from a Google Books user.
	Source *ReviewSource `json:"source,omitempty"`

	// Title: Title for this review.
	Title string `json:"title,omitempty"`

	// Type: Source type for this review. Possible values are EDITORIAL,
	// WEB_USER or GOOGLE_USER.
	Type string `json:"type,omitempty"`

	// VolumeId: Volume that this review is for.
	VolumeId string `json:"volumeId,omitempty"`
}

type ReviewAuthor struct {
	// DisplayName: Name of this person.
	DisplayName string `json:"displayName,omitempty"`
}

type ReviewSource struct {
	// Description: Name of the source.
	Description string `json:"description,omitempty"`

	// ExtraDescription: Extra text about the source of the review.
	ExtraDescription string `json:"extraDescription,omitempty"`

	// Url: URL of the source of the review.
	Url string `json:"url,omitempty"`
}

type Volume struct {
	// AccessInfo: Any information about a volume related to reading or
	// obtaining that volume text. This information can depend on country
	// (books may be public domain in one country but not in another, e.g.).
	AccessInfo *VolumeAccessInfo `json:"accessInfo,omitempty"`

	// Etag: Opaque identifier for a specific version of a volume resource.
	// (In LITE projection)
	Etag string `json:"etag,omitempty"`

	// Id: Unique identifier for a volume. (In LITE projection.)
	Id string `json:"id,omitempty"`

	// Kind: Resource type for a volume. (In LITE projection.)
	Kind string `json:"kind,omitempty"`

	// LayerInfo: What layers exist in this volume and high level
	// information about them.
	LayerInfo *VolumeLayerInfo `json:"layerInfo,omitempty"`

	// RecommendedInfo: Recommendation related information for this volume.
	RecommendedInfo *VolumeRecommendedInfo `json:"recommendedInfo,omitempty"`

	// SaleInfo: Any information about a volume related to the eBookstore
	// and/or purchaseability. This information can depend on the country
	// where the request originates from (i.e. books may not be for sale in
	// certain countries).
	SaleInfo *VolumeSaleInfo `json:"saleInfo,omitempty"`

	// SearchInfo: Search result information related to this volume.
	SearchInfo *VolumeSearchInfo `json:"searchInfo,omitempty"`

	// SelfLink: URL to this resource. (In LITE projection.)
	SelfLink string `json:"selfLink,omitempty"`

	// UserInfo: User specific information related to this volume. (e.g.
	// page this user last read or whether they purchased this book)
	UserInfo *VolumeUserInfo `json:"userInfo,omitempty"`

	// VolumeInfo: General volume information.
	VolumeInfo *VolumeVolumeInfo `json:"volumeInfo,omitempty"`
}

type VolumeAccessInfo struct {
	// AccessViewStatus: Combines the access and viewability of this volume
	// into a single status field for this user. Values can be
	// FULL_PURCHASED, FULL_PUBLIC_DOMAIN, SAMPLE or NONE. (In LITE
	// projection.)
	AccessViewStatus string `json:"accessViewStatus,omitempty"`

	// Country: The two-letter ISO_3166-1 country code for which this access
	// information is valid. (In LITE projection.)
	Country string `json:"country,omitempty"`

	// DownloadAccess: Information about a volume's download license access
	// restrictions.
	DownloadAccess *DownloadAccessRestriction `json:"downloadAccess,omitempty"`

	// DriveImportedContentLink: URL to the Google Drive viewer if this
	// volume is uploaded by the user by selecting the file from Google
	// Drive.
	DriveImportedContentLink string `json:"driveImportedContentLink,omitempty"`

	// Embeddable: Whether this volume can be embedded in a viewport using
	// the Embedded Viewer API.
	Embeddable bool `json:"embeddable,omitempty"`

	// Epub: Information about epub content. (In LITE projection.)
	Epub *VolumeAccessInfoEpub `json:"epub,omitempty"`

	// ExplicitOfflineLicenseManagement: Whether this volume requires that
	// the client explicitly request offline download license rather than
	// have it done automatically when loading the content, if the client
	// supports it.
	ExplicitOfflineLicenseManagement bool `json:"explicitOfflineLicenseManagement,omitempty"`

	// Pdf: Information about pdf content. (In LITE projection.)
	Pdf *VolumeAccessInfoPdf `json:"pdf,omitempty"`

	// PublicDomain: Whether or not this book is public domain in the
	// country listed above.
	PublicDomain bool `json:"publicDomain,omitempty"`

	// QuoteSharingAllowed: Whether quote sharing is allowed for this
	// volume.
	QuoteSharingAllowed bool `json:"quoteSharingAllowed,omitempty"`

	// TextToSpeechPermission: Whether text-to-speech is permitted for this
	// volume. Values can be ALLOWED, ALLOWED_FOR_ACCESSIBILITY, or
	// NOT_ALLOWED.
	TextToSpeechPermission string `json:"textToSpeechPermission,omitempty"`

	// ViewOrderUrl: For ordered but not yet processed orders, we give a URL
	// that can be used to go to the appropriate Google Wallet page.
	ViewOrderUrl string `json:"viewOrderUrl,omitempty"`

	// Viewability: The read access of a volume. Possible values are
	// PARTIAL, ALL_PAGES, NO_PAGES or UNKNOWN. This value depends on the
	// country listed above. A value of PARTIAL means that the publisher has
	// allowed some portion of the volume to be viewed publicly, without
	// purchase. This can apply to eBooks as well as non-eBooks. Public
	// domain books will always have a value of ALL_PAGES.
	Viewability string `json:"viewability,omitempty"`

	// WebReaderLink: URL to read this volume on the Google Books site. Link
	// will not allow users to read non-viewable volumes.
	WebReaderLink string `json:"webReaderLink,omitempty"`
}

type VolumeAccessInfoEpub struct {
	// AcsTokenLink: URL to retrieve ACS token for epub download. (In LITE
	// projection.)
	AcsTokenLink string `json:"acsTokenLink,omitempty"`

	// DownloadLink: URL to download epub. (In LITE projection.)
	DownloadLink string `json:"downloadLink,omitempty"`

	// IsAvailable: Is a flowing text epub available either as public domain
	// or for purchase. (In LITE projection.)
	IsAvailable bool `json:"isAvailable,omitempty"`
}

type VolumeAccessInfoPdf struct {
	// AcsTokenLink: URL to retrieve ACS token for pdf download. (In LITE
	// projection.)
	AcsTokenLink string `json:"acsTokenLink,omitempty"`

	// DownloadLink: URL to download pdf. (In LITE projection.)
	DownloadLink string `json:"downloadLink,omitempty"`

	// IsAvailable: Is a scanned image pdf available either as public domain
	// or for purchase. (In LITE projection.)
	IsAvailable bool `json:"isAvailable,omitempty"`
}

type VolumeLayerInfo struct {
	// Layers: A layer should appear here if and only if the layer exists
	// for this book.
	Layers []*VolumeLayerInfoLayers `json:"layers,omitempty"`
}

type VolumeLayerInfoLayers struct {
	// LayerId: The layer id of this layer (e.g. "geo").
	LayerId string `json:"layerId,omitempty"`

	// VolumeAnnotationsVersion: The current version of this layer's volume
	// annotations. Note that this version applies only to the data in the
	// books.layers.volumeAnnotations.* responses. The actual annotation
	// data is versioned separately.
	VolumeAnnotationsVersion string `json:"volumeAnnotationsVersion,omitempty"`
}

type VolumeRecommendedInfo struct {
	// Explanation: A text explaining why this volume is recommended.
	Explanation string `json:"explanation,omitempty"`
}

type VolumeSaleInfo struct {
	// BuyLink: URL to purchase this volume on the Google Books site. (In
	// LITE projection)
	BuyLink string `json:"buyLink,omitempty"`

	// Country: The two-letter ISO_3166-1 country code for which this sale
	// information is valid. (In LITE projection.)
	Country string `json:"country,omitempty"`

	// IsEbook: Whether or not this volume is an eBook (can be added to the
	// My eBooks shelf).
	IsEbook bool `json:"isEbook,omitempty"`

	// ListPrice: Suggested retail price. (In LITE projection.)
	ListPrice *VolumeSaleInfoListPrice `json:"listPrice,omitempty"`

	// Offers: Offers available for this volume (sales and rentals).
	Offers []*VolumeSaleInfoOffers `json:"offers,omitempty"`

	// OnSaleDate: The date on which this book is available for sale.
	OnSaleDate string `json:"onSaleDate,omitempty"`

	// RetailPrice: The actual selling price of the book. This is the same
	// as the suggested retail or list price unless there are offers or
	// discounts on this volume. (In LITE projection.)
	RetailPrice *VolumeSaleInfoRetailPrice `json:"retailPrice,omitempty"`

	// Saleability: Whether or not this book is available for sale or
	// offered for free in the Google eBookstore for the country listed
	// above. Possible values are FOR_SALE, FOR_RENTAL_ONLY,
	// FOR_SALE_AND_RENTAL, FREE, NOT_FOR_SALE, or FOR_PREORDER.
	Saleability string `json:"saleability,omitempty"`
}

type VolumeSaleInfoListPrice struct {
	// Amount: Amount in the currency listed below. (In LITE projection.)
	Amount float64 `json:"amount,omitempty"`

	// CurrencyCode: An ISO 4217, three-letter currency code. (In LITE
	// projection.)
	CurrencyCode string `json:"currencyCode,omitempty"`
}

type VolumeSaleInfoOffers struct {
	// FinskyOfferType: The finsky offer type (e.g., PURCHASE=0 RENTAL=3)
	FinskyOfferType int64 `json:"finskyOfferType,omitempty"`

	// ListPrice: Offer list (=undiscounted) price in Micros.
	ListPrice *VolumeSaleInfoOffersListPrice `json:"listPrice,omitempty"`

	// RentalDuration: The rental duration (for rental offers only).
	RentalDuration *VolumeSaleInfoOffersRentalDuration `json:"rentalDuration,omitempty"`

	// RetailPrice: Offer retail (=discounted) price in Micros
	RetailPrice *VolumeSaleInfoOffersRetailPrice `json:"retailPrice,omitempty"`
}

type VolumeSaleInfoOffersListPrice struct {
	AmountInMicros float64 `json:"amountInMicros,omitempty"`

	CurrencyCode string `json:"currencyCode,omitempty"`
}

type VolumeSaleInfoOffersRentalDuration struct {
	Count float64 `json:"count,omitempty"`

	Unit string `json:"unit,omitempty"`
}

type VolumeSaleInfoOffersRetailPrice struct {
	AmountInMicros float64 `json:"amountInMicros,omitempty"`

	CurrencyCode string `json:"currencyCode,omitempty"`
}

type VolumeSaleInfoRetailPrice struct {
	// Amount: Amount in the currency listed below. (In LITE projection.)
	Amount float64 `json:"amount,omitempty"`

	// CurrencyCode: An ISO 4217, three-letter currency code. (In LITE
	// projection.)
	CurrencyCode string `json:"currencyCode,omitempty"`
}

type VolumeSearchInfo struct {
	// TextSnippet: A text snippet containing the search query.
	TextSnippet string `json:"textSnippet,omitempty"`
}

type VolumeUserInfo struct {
	// Copy: Copy/Paste accounting information.
	Copy *VolumeUserInfoCopy `json:"copy,omitempty"`

	// IsInMyBooks: Whether or not this volume is currently in "my books."
	IsInMyBooks bool `json:"isInMyBooks,omitempty"`

	// IsPreordered: Whether or not this volume was pre-ordered by the
	// authenticated user making the request. (In LITE projection.)
	IsPreordered bool `json:"isPreordered,omitempty"`

	// IsPurchased: Whether or not this volume was purchased by the
	// authenticated user making the request. (In LITE projection.)
	IsPurchased bool `json:"isPurchased,omitempty"`

	// IsUploaded: Whether or not this volume was user uploaded.
	IsUploaded bool `json:"isUploaded,omitempty"`

	// ReadingPosition: The user's current reading position in the volume,
	// if one is available. (In LITE projection.)
	ReadingPosition *ReadingPosition `json:"readingPosition,omitempty"`

	// RentalPeriod: Period during this book is/was a valid rental.
	RentalPeriod *VolumeUserInfoRentalPeriod `json:"rentalPeriod,omitempty"`

	// RentalState: Whether this book is an active or an expired rental.
	RentalState string `json:"rentalState,omitempty"`

	// Review: This user's review of this volume, if one exists.
	Review *Review `json:"review,omitempty"`

	// Updated: Timestamp when this volume was last modified by a user
	// action, such as a reading position update, volume purchase or writing
	// a review. (RFC 3339 UTC date-time format).
	Updated string `json:"updated,omitempty"`

	UserUploadedVolumeInfo *VolumeUserInfoUserUploadedVolumeInfo `json:"userUploadedVolumeInfo,omitempty"`
}

type VolumeUserInfoCopy struct {
	AllowedCharacterCount int64 `json:"allowedCharacterCount,omitempty"`

	LimitType string `json:"limitType,omitempty"`

	RemainingCharacterCount int64 `json:"remainingCharacterCount,omitempty"`

	Updated string `json:"updated,omitempty"`
}

type VolumeUserInfoRentalPeriod struct {
	EndUtcSec int64 `json:"endUtcSec,omitempty,string"`

	StartUtcSec int64 `json:"startUtcSec,omitempty,string"`
}

type VolumeUserInfoUserUploadedVolumeInfo struct {
	ProcessingState string `json:"processingState,omitempty"`
}

type VolumeVolumeInfo struct {
	// Authors: The names of the authors and/or editors for this volume. (In
	// LITE projection)
	Authors []string `json:"authors,omitempty"`

	// AverageRating: The mean review rating for this volume. (min = 1.0,
	// max = 5.0)
	AverageRating float64 `json:"averageRating,omitempty"`

	// CanonicalVolumeLink: Canonical URL for a volume. (In LITE
	// projection.)
	CanonicalVolumeLink string `json:"canonicalVolumeLink,omitempty"`

	// Categories: A list of subject categories, such as "Fiction",
	// "Suspense", etc.
	Categories []string `json:"categories,omitempty"`

	// ContentVersion: An identifier for the version of the volume content
	// (text & images). (In LITE projection)
	ContentVersion string `json:"contentVersion,omitempty"`

	// Description: A synopsis of the volume. The text of the description is
	// formatted in HTML and includes simple formatting elements, such as b,
	// i, and br tags. (In LITE projection.)
	Description string `json:"description,omitempty"`

	// Dimensions: Physical dimensions of this volume.
	Dimensions *VolumeVolumeInfoDimensions `json:"dimensions,omitempty"`

	// ImageLinks: A list of image links for all the sizes that are
	// available. (In LITE projection.)
	ImageLinks *VolumeVolumeInfoImageLinks `json:"imageLinks,omitempty"`

	// IndustryIdentifiers: Industry standard identifiers for this volume.
	IndustryIdentifiers []*VolumeVolumeInfoIndustryIdentifiers `json:"industryIdentifiers,omitempty"`

	// InfoLink: URL to view information about this volume on the Google
	// Books site. (In LITE projection)
	InfoLink string `json:"infoLink,omitempty"`

	// Language: Best language for this volume (based on content). It is the
	// two-letter ISO 639-1 code such as 'fr', 'en', etc.
	Language string `json:"language,omitempty"`

	// MainCategory: The main category to which this volume belongs. It will
	// be the category from the categories list returned below that has the
	// highest weight.
	MainCategory string `json:"mainCategory,omitempty"`

	// PageCount: Total number of pages as per publisher metadata.
	PageCount int64 `json:"pageCount,omitempty"`

	// PreviewLink: URL to preview this volume on the Google Books site.
	PreviewLink string `json:"previewLink,omitempty"`

	// PrintType: Type of publication of this volume. Possible values are
	// BOOK or MAGAZINE.
	PrintType string `json:"printType,omitempty"`

	// PrintedPageCount: Total number of printed pages in generated pdf
	// representation.
	PrintedPageCount int64 `json:"printedPageCount,omitempty"`

	// PublishedDate: Date of publication. (In LITE projection.)
	PublishedDate string `json:"publishedDate,omitempty"`

	// Publisher: Publisher of this volume. (In LITE projection.)
	Publisher string `json:"publisher,omitempty"`

	// RatingsCount: The number of review ratings for this volume.
	RatingsCount int64 `json:"ratingsCount,omitempty"`

	// Subtitle: Volume subtitle. (In LITE projection.)
	Subtitle string `json:"subtitle,omitempty"`

	// Title: Volume title. (In LITE projection.)
	Title string `json:"title,omitempty"`
}

type VolumeVolumeInfoDimensions struct {
	// Height: Height or length of this volume (in cm).
	Height string `json:"height,omitempty"`

	// Thickness: Thickness of this volume (in cm).
	Thickness string `json:"thickness,omitempty"`

	// Width: Width of this volume (in cm).
	Width string `json:"width,omitempty"`
}

type VolumeVolumeInfoImageLinks struct {
	// ExtraLarge: Image link for extra large size (width of ~1280 pixels).
	// (In LITE projection)
	ExtraLarge string `json:"extraLarge,omitempty"`

	// Large: Image link for large size (width of ~800 pixels). (In LITE
	// projection)
	Large string `json:"large,omitempty"`

	// Medium: Image link for medium size (width of ~575 pixels). (In LITE
	// projection)
	Medium string `json:"medium,omitempty"`

	// Small: Image link for small size (width of ~300 pixels). (In LITE
	// projection)
	Small string `json:"small,omitempty"`

	// SmallThumbnail: Image link for small thumbnail size (width of ~80
	// pixels). (In LITE projection)
	SmallThumbnail string `json:"smallThumbnail,omitempty"`

	// Thumbnail: Image link for thumbnail size (width of ~128 pixels). (In
	// LITE projection)
	Thumbnail string `json:"thumbnail,omitempty"`
}

type VolumeVolumeInfoIndustryIdentifiers struct {
	// Identifier: Industry specific volume identifier.
	Identifier string `json:"identifier,omitempty"`

	// Type: Identifier type. Possible values are ISBN_10, ISBN_13, ISSN and
	// OTHER.
	Type string `json:"type,omitempty"`
}

type Volumeannotation struct {
	// AnnotationDataId: The annotation data id for this volume annotation.
	AnnotationDataId string `json:"annotationDataId,omitempty"`

	// AnnotationDataLink: Link to get data for this annotation.
	AnnotationDataLink string `json:"annotationDataLink,omitempty"`

	// AnnotationType: The type of annotation this is.
	AnnotationType string `json:"annotationType,omitempty"`

	// ContentRanges: The content ranges to identify the selected text.
	ContentRanges *VolumeannotationContentRanges `json:"contentRanges,omitempty"`

	// Data: Data for this annotation.
	Data string `json:"data,omitempty"`

	// Deleted: Indicates that this annotation is deleted.
	Deleted bool `json:"deleted,omitempty"`

	// Id: Unique id of this volume annotation.
	Id string `json:"id,omitempty"`

	// Kind: Resource Type
	Kind string `json:"kind,omitempty"`

	// LayerId: The Layer this annotation is for.
	LayerId string `json:"layerId,omitempty"`

	// PageIds: Pages the annotation spans.
	PageIds []string `json:"pageIds,omitempty"`

	// SelectedText: Excerpt from the volume.
	SelectedText string `json:"selectedText,omitempty"`

	// SelfLink: URL to this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Updated: Timestamp for the last time this anntoation was updated.
	// (RFC 3339 UTC date-time format).
	Updated string `json:"updated,omitempty"`

	// VolumeId: The Volume this annotation is for.
	VolumeId string `json:"volumeId,omitempty"`
}

type VolumeannotationContentRanges struct {
	// CfiRange: Range in CFI format for this annotation for version above.
	CfiRange *BooksAnnotationsRange `json:"cfiRange,omitempty"`

	// ContentVersion: Content version applicable to ranges below.
	ContentVersion string `json:"contentVersion,omitempty"`

	// GbImageRange: Range in GB image format for this annotation for
	// version above.
	GbImageRange *BooksAnnotationsRange `json:"gbImageRange,omitempty"`

	// GbTextRange: Range in GB text format for this annotation for version
	// above.
	GbTextRange *BooksAnnotationsRange `json:"gbTextRange,omitempty"`
}

type Volumeannotations struct {
	// Items: A list of volume annotations.
	Items []*Volumeannotation `json:"items,omitempty"`

	// Kind: Resource type
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Token to pass in for pagination for the next page.
	// This will not be present if this request does not have more results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TotalItems: The total number of volume annotations found.
	TotalItems int64 `json:"totalItems,omitempty"`

	// Version: The version string for all of the volume annotations in this
	// layer (not just the ones in this response). Note: the version string
	// doesn't apply to the annotation data, just the information in this
	// response (e.g. the location of annotations in the book).
	Version string `json:"version,omitempty"`
}

type Volumes struct {
	// Items: A list of volumes.
	Items []*Volume `json:"items,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// TotalItems: Total number of volumes found. This might be greater than
	// the number of volumes returned in this response if results have been
	// paginated.
	TotalItems int64 `json:"totalItems,omitempty"`
}

// method id "books.bookshelves.get":

type BookshelvesGetCall struct {
	s      *Service
	userId string
	shelf  string
	opt_   map[string]interface{}
}

// Get: Retrieves metadata for a specific bookshelf for the specified
// user.
func (r *BookshelvesService) Get(userId string, shelf string) *BookshelvesGetCall {
	c := &BookshelvesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.shelf = shelf
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *BookshelvesGetCall) Source(source string) *BookshelvesGetCall {
	c.opt_["source"] = source
	return c
}

func (c *BookshelvesGetCall) Do() (*Bookshelf, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}/bookshelves/{shelf}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{shelf}", url.QueryEscape(c.shelf), 1)
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
	ret := new(Bookshelf)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves metadata for a specific bookshelf for the specified user.",
	//   "httpMethod": "GET",
	//   "id": "books.bookshelves.get",
	//   "parameterOrder": [
	//     "userId",
	//     "shelf"
	//   ],
	//   "parameters": {
	//     "shelf": {
	//       "description": "ID of bookshelf to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "ID of user for whom to retrieve bookshelves.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}/bookshelves/{shelf}",
	//   "response": {
	//     "$ref": "Bookshelf"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.bookshelves.list":

type BookshelvesListCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// List: Retrieves a list of public bookshelves for the specified user.
func (r *BookshelvesService) List(userId string) *BookshelvesListCall {
	c := &BookshelvesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *BookshelvesListCall) Source(source string) *BookshelvesListCall {
	c.opt_["source"] = source
	return c
}

func (c *BookshelvesListCall) Do() (*Bookshelves, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}/bookshelves")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(Bookshelves)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of public bookshelves for the specified user.",
	//   "httpMethod": "GET",
	//   "id": "books.bookshelves.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "ID of user for whom to retrieve bookshelves.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}/bookshelves",
	//   "response": {
	//     "$ref": "Bookshelves"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.bookshelves.volumes.list":

type BookshelvesVolumesListCall struct {
	s      *Service
	userId string
	shelf  string
	opt_   map[string]interface{}
}

// List: Retrieves volumes in a specific bookshelf for the specified
// user.
func (r *BookshelvesVolumesService) List(userId string, shelf string) *BookshelvesVolumesListCall {
	c := &BookshelvesVolumesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.shelf = shelf
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *BookshelvesVolumesListCall) MaxResults(maxResults int64) *BookshelvesVolumesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// ShowPreorders sets the optional parameter "showPreorders": Set to
// true to show pre-ordered books. Defaults to false.
func (c *BookshelvesVolumesListCall) ShowPreorders(showPreorders bool) *BookshelvesVolumesListCall {
	c.opt_["showPreorders"] = showPreorders
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *BookshelvesVolumesListCall) Source(source string) *BookshelvesVolumesListCall {
	c.opt_["source"] = source
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first element to return (starts at 0)
func (c *BookshelvesVolumesListCall) StartIndex(startIndex int64) *BookshelvesVolumesListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *BookshelvesVolumesListCall) Do() (*Volumes, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showPreorders"]; ok {
		params.Set("showPreorders", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}/bookshelves/{shelf}/volumes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{shelf}", url.QueryEscape(c.shelf), 1)
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
	ret := new(Volumes)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves volumes in a specific bookshelf for the specified user.",
	//   "httpMethod": "GET",
	//   "id": "books.bookshelves.volumes.list",
	//   "parameterOrder": [
	//     "userId",
	//     "shelf"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "shelf": {
	//       "description": "ID of bookshelf to retrieve volumes.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "showPreorders": {
	//       "description": "Set to true to show pre-ordered books. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first element to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "userId": {
	//       "description": "ID of user for whom to retrieve bookshelf volumes.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}/bookshelves/{shelf}/volumes",
	//   "response": {
	//     "$ref": "Volumes"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.cloudloading.addBook":

type CloudloadingAddBookCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// AddBook:
func (r *CloudloadingService) AddBook() *CloudloadingAddBookCall {
	c := &CloudloadingAddBookCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Drive_document_id sets the optional parameter "drive_document_id": A
// drive document id. The upload_client_token must not be set.
func (c *CloudloadingAddBookCall) Drive_document_id(drive_document_id string) *CloudloadingAddBookCall {
	c.opt_["drive_document_id"] = drive_document_id
	return c
}

// Mime_type sets the optional parameter "mime_type": The document MIME
// type. It can be set only if the drive_document_id is set.
func (c *CloudloadingAddBookCall) Mime_type(mime_type string) *CloudloadingAddBookCall {
	c.opt_["mime_type"] = mime_type
	return c
}

// Name sets the optional parameter "name": The document name. It can be
// set only if the drive_document_id is set.
func (c *CloudloadingAddBookCall) Name(name string) *CloudloadingAddBookCall {
	c.opt_["name"] = name
	return c
}

// Upload_client_token sets the optional parameter
// "upload_client_token":
func (c *CloudloadingAddBookCall) Upload_client_token(upload_client_token string) *CloudloadingAddBookCall {
	c.opt_["upload_client_token"] = upload_client_token
	return c
}

func (c *CloudloadingAddBookCall) Do() (*BooksCloudloadingResource, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["drive_document_id"]; ok {
		params.Set("drive_document_id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["mime_type"]; ok {
		params.Set("mime_type", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["name"]; ok {
		params.Set("name", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["upload_client_token"]; ok {
		params.Set("upload_client_token", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "cloudloading/addBook")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(BooksCloudloadingResource)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "",
	//   "httpMethod": "POST",
	//   "id": "books.cloudloading.addBook",
	//   "parameters": {
	//     "drive_document_id": {
	//       "description": "A drive document id. The upload_client_token must not be set.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "mime_type": {
	//       "description": "The document MIME type. It can be set only if the drive_document_id is set.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The document name. It can be set only if the drive_document_id is set.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "upload_client_token": {
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "cloudloading/addBook",
	//   "response": {
	//     "$ref": "BooksCloudloadingResource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.cloudloading.deleteBook":

type CloudloadingDeleteBookCall struct {
	s        *Service
	volumeId string
	opt_     map[string]interface{}
}

// DeleteBook: Remove the book and its contents
func (r *CloudloadingService) DeleteBook(volumeId string) *CloudloadingDeleteBookCall {
	c := &CloudloadingDeleteBookCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	return c
}

func (c *CloudloadingDeleteBookCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("volumeId", fmt.Sprintf("%v", c.volumeId))
	urls := googleapi.ResolveRelative(c.s.BasePath, "cloudloading/deleteBook")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	//   "description": "Remove the book and its contents",
	//   "httpMethod": "POST",
	//   "id": "books.cloudloading.deleteBook",
	//   "parameterOrder": [
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "volumeId": {
	//       "description": "The id of the book to be removed.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "cloudloading/deleteBook",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.cloudloading.updateBook":

type CloudloadingUpdateBookCall struct {
	s                         *Service
	bookscloudloadingresource *BooksCloudloadingResource
	opt_                      map[string]interface{}
}

// UpdateBook:
func (r *CloudloadingService) UpdateBook(bookscloudloadingresource *BooksCloudloadingResource) *CloudloadingUpdateBookCall {
	c := &CloudloadingUpdateBookCall{s: r.s, opt_: make(map[string]interface{})}
	c.bookscloudloadingresource = bookscloudloadingresource
	return c
}

func (c *CloudloadingUpdateBookCall) Do() (*BooksCloudloadingResource, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.bookscloudloadingresource)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "cloudloading/updateBook")
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
	ret := new(BooksCloudloadingResource)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "",
	//   "httpMethod": "POST",
	//   "id": "books.cloudloading.updateBook",
	//   "path": "cloudloading/updateBook",
	//   "request": {
	//     "$ref": "BooksCloudloadingResource"
	//   },
	//   "response": {
	//     "$ref": "BooksCloudloadingResource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.layers.get":

type LayersGetCall struct {
	s         *Service
	volumeId  string
	summaryId string
	opt_      map[string]interface{}
}

// Get: Gets the layer summary for a volume.
func (r *LayersService) Get(volumeId string, summaryId string) *LayersGetCall {
	c := &LayersGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	c.summaryId = summaryId
	return c
}

// ContentVersion sets the optional parameter "contentVersion": The
// content version for the requested volume.
func (c *LayersGetCall) ContentVersion(contentVersion string) *LayersGetCall {
	c.opt_["contentVersion"] = contentVersion
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *LayersGetCall) Source(source string) *LayersGetCall {
	c.opt_["source"] = source
	return c
}

func (c *LayersGetCall) Do() (*Layersummary, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["contentVersion"]; ok {
		params.Set("contentVersion", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/{volumeId}/layersummary/{summaryId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{summaryId}", url.QueryEscape(c.summaryId), 1)
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
	ret := new(Layersummary)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the layer summary for a volume.",
	//   "httpMethod": "GET",
	//   "id": "books.layers.get",
	//   "parameterOrder": [
	//     "volumeId",
	//     "summaryId"
	//   ],
	//   "parameters": {
	//     "contentVersion": {
	//       "description": "The content version for the requested volume.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "summaryId": {
	//       "description": "The ID for the layer to get the summary for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "The volume to retrieve layers for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/{volumeId}/layersummary/{summaryId}",
	//   "response": {
	//     "$ref": "Layersummary"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.layers.list":

type LayersListCall struct {
	s        *Service
	volumeId string
	opt_     map[string]interface{}
}

// List: List the layer summaries for a volume.
func (r *LayersService) List(volumeId string) *LayersListCall {
	c := &LayersListCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	return c
}

// ContentVersion sets the optional parameter "contentVersion": The
// content version for the requested volume.
func (c *LayersListCall) ContentVersion(contentVersion string) *LayersListCall {
	c.opt_["contentVersion"] = contentVersion
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *LayersListCall) MaxResults(maxResults int64) *LayersListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous page.
func (c *LayersListCall) PageToken(pageToken string) *LayersListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *LayersListCall) Source(source string) *LayersListCall {
	c.opt_["source"] = source
	return c
}

func (c *LayersListCall) Do() (*Layersummaries, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["contentVersion"]; ok {
		params.Set("contentVersion", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/{volumeId}/layersummary")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
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
	ret := new(Layersummaries)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List the layer summaries for a volume.",
	//   "httpMethod": "GET",
	//   "id": "books.layers.list",
	//   "parameterOrder": [
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "contentVersion": {
	//       "description": "The content version for the requested volume.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "200",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value of the nextToken from the previous page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "The volume to retrieve layers for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/{volumeId}/layersummary",
	//   "response": {
	//     "$ref": "Layersummaries"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.layers.annotationData.get":

type LayersAnnotationDataGetCall struct {
	s                *Service
	volumeId         string
	layerId          string
	annotationDataId string
	contentVersion   string
	opt_             map[string]interface{}
}

// Get: Gets the annotation data.
func (r *LayersAnnotationDataService) Get(volumeId string, layerId string, annotationDataId string, contentVersion string) *LayersAnnotationDataGetCall {
	c := &LayersAnnotationDataGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	c.layerId = layerId
	c.annotationDataId = annotationDataId
	c.contentVersion = contentVersion
	return c
}

// AllowWebDefinitions sets the optional parameter
// "allowWebDefinitions": For the dictionary layer. Whether or not to
// allow web definitions.
func (c *LayersAnnotationDataGetCall) AllowWebDefinitions(allowWebDefinitions bool) *LayersAnnotationDataGetCall {
	c.opt_["allowWebDefinitions"] = allowWebDefinitions
	return c
}

// H sets the optional parameter "h": The requested pixel height for any
// images. If height is provided width must also be provided.
func (c *LayersAnnotationDataGetCall) H(h int64) *LayersAnnotationDataGetCall {
	c.opt_["h"] = h
	return c
}

// Locale sets the optional parameter "locale": The locale information
// for the data. ISO-639-1 language and ISO-3166-1 country code. Ex:
// 'en_US'.
func (c *LayersAnnotationDataGetCall) Locale(locale string) *LayersAnnotationDataGetCall {
	c.opt_["locale"] = locale
	return c
}

// Scale sets the optional parameter "scale": The requested scale for
// the image.
func (c *LayersAnnotationDataGetCall) Scale(scale int64) *LayersAnnotationDataGetCall {
	c.opt_["scale"] = scale
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *LayersAnnotationDataGetCall) Source(source string) *LayersAnnotationDataGetCall {
	c.opt_["source"] = source
	return c
}

// W sets the optional parameter "w": The requested pixel width for any
// images. If width is provided height must also be provided.
func (c *LayersAnnotationDataGetCall) W(w int64) *LayersAnnotationDataGetCall {
	c.opt_["w"] = w
	return c
}

func (c *LayersAnnotationDataGetCall) Do() (*Annotationdata, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("contentVersion", fmt.Sprintf("%v", c.contentVersion))
	if v, ok := c.opt_["allowWebDefinitions"]; ok {
		params.Set("allowWebDefinitions", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["h"]; ok {
		params.Set("h", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["scale"]; ok {
		params.Set("scale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["w"]; ok {
		params.Set("w", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/{volumeId}/layers/{layerId}/data/{annotationDataId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{layerId}", url.QueryEscape(c.layerId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{annotationDataId}", url.QueryEscape(c.annotationDataId), 1)
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
	ret := new(Annotationdata)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the annotation data.",
	//   "httpMethod": "GET",
	//   "id": "books.layers.annotationData.get",
	//   "parameterOrder": [
	//     "volumeId",
	//     "layerId",
	//     "annotationDataId",
	//     "contentVersion"
	//   ],
	//   "parameters": {
	//     "allowWebDefinitions": {
	//       "description": "For the dictionary layer. Whether or not to allow web definitions.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "annotationDataId": {
	//       "description": "The ID of the annotation data to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "contentVersion": {
	//       "description": "The content version for the volume you are trying to retrieve.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "h": {
	//       "description": "The requested pixel height for any images. If height is provided width must also be provided.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "layerId": {
	//       "description": "The ID for the layer to get the annotations.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "The locale information for the data. ISO-639-1 language and ISO-3166-1 country code. Ex: 'en_US'.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "scale": {
	//       "description": "The requested scale for the image.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "The volume to retrieve annotations for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "w": {
	//       "description": "The requested pixel width for any images. If width is provided height must also be provided.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "volumes/{volumeId}/layers/{layerId}/data/{annotationDataId}",
	//   "response": {
	//     "$ref": "Annotationdata"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.layers.annotationData.list":

type LayersAnnotationDataListCall struct {
	s              *Service
	volumeId       string
	layerId        string
	contentVersion string
	opt_           map[string]interface{}
}

// List: Gets the annotation data for a volume and layer.
func (r *LayersAnnotationDataService) List(volumeId string, layerId string, contentVersion string) *LayersAnnotationDataListCall {
	c := &LayersAnnotationDataListCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	c.layerId = layerId
	c.contentVersion = contentVersion
	return c
}

// AnnotationDataId sets the optional parameter "annotationDataId": The
// list of Annotation Data Ids to retrieve. Pagination is ignored if
// this is set.
func (c *LayersAnnotationDataListCall) AnnotationDataId(annotationDataId string) *LayersAnnotationDataListCall {
	c.opt_["annotationDataId"] = annotationDataId
	return c
}

// H sets the optional parameter "h": The requested pixel height for any
// images. If height is provided width must also be provided.
func (c *LayersAnnotationDataListCall) H(h int64) *LayersAnnotationDataListCall {
	c.opt_["h"] = h
	return c
}

// Locale sets the optional parameter "locale": The locale information
// for the data. ISO-639-1 language and ISO-3166-1 country code. Ex:
// 'en_US'.
func (c *LayersAnnotationDataListCall) Locale(locale string) *LayersAnnotationDataListCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *LayersAnnotationDataListCall) MaxResults(maxResults int64) *LayersAnnotationDataListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous page.
func (c *LayersAnnotationDataListCall) PageToken(pageToken string) *LayersAnnotationDataListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Scale sets the optional parameter "scale": The requested scale for
// the image.
func (c *LayersAnnotationDataListCall) Scale(scale int64) *LayersAnnotationDataListCall {
	c.opt_["scale"] = scale
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *LayersAnnotationDataListCall) Source(source string) *LayersAnnotationDataListCall {
	c.opt_["source"] = source
	return c
}

// UpdatedMax sets the optional parameter "updatedMax": RFC 3339
// timestamp to restrict to items updated prior to this timestamp
// (exclusive).
func (c *LayersAnnotationDataListCall) UpdatedMax(updatedMax string) *LayersAnnotationDataListCall {
	c.opt_["updatedMax"] = updatedMax
	return c
}

// UpdatedMin sets the optional parameter "updatedMin": RFC 3339
// timestamp to restrict to items updated since this timestamp
// (inclusive).
func (c *LayersAnnotationDataListCall) UpdatedMin(updatedMin string) *LayersAnnotationDataListCall {
	c.opt_["updatedMin"] = updatedMin
	return c
}

// W sets the optional parameter "w": The requested pixel width for any
// images. If width is provided height must also be provided.
func (c *LayersAnnotationDataListCall) W(w int64) *LayersAnnotationDataListCall {
	c.opt_["w"] = w
	return c
}

func (c *LayersAnnotationDataListCall) Do() (*Annotationsdata, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("contentVersion", fmt.Sprintf("%v", c.contentVersion))
	if v, ok := c.opt_["annotationDataId"]; ok {
		params.Set("annotationDataId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["h"]; ok {
		params.Set("h", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["scale"]; ok {
		params.Set("scale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updatedMax"]; ok {
		params.Set("updatedMax", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updatedMin"]; ok {
		params.Set("updatedMin", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["w"]; ok {
		params.Set("w", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/{volumeId}/layers/{layerId}/data")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{layerId}", url.QueryEscape(c.layerId), 1)
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
	ret := new(Annotationsdata)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the annotation data for a volume and layer.",
	//   "httpMethod": "GET",
	//   "id": "books.layers.annotationData.list",
	//   "parameterOrder": [
	//     "volumeId",
	//     "layerId",
	//     "contentVersion"
	//   ],
	//   "parameters": {
	//     "annotationDataId": {
	//       "description": "The list of Annotation Data Ids to retrieve. Pagination is ignored if this is set.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "contentVersion": {
	//       "description": "The content version for the requested volume.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "h": {
	//       "description": "The requested pixel height for any images. If height is provided width must also be provided.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "layerId": {
	//       "description": "The ID for the layer to get the annotation data.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "The locale information for the data. ISO-639-1 language and ISO-3166-1 country code. Ex: 'en_US'.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "200",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value of the nextToken from the previous page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "scale": {
	//       "description": "The requested scale for the image.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "updatedMax": {
	//       "description": "RFC 3339 timestamp to restrict to items updated prior to this timestamp (exclusive).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "updatedMin": {
	//       "description": "RFC 3339 timestamp to restrict to items updated since this timestamp (inclusive).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "The volume to retrieve annotation data for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "w": {
	//       "description": "The requested pixel width for any images. If width is provided height must also be provided.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "volumes/{volumeId}/layers/{layerId}/data",
	//   "response": {
	//     "$ref": "Annotationsdata"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.layers.volumeAnnotations.get":

type LayersVolumeAnnotationsGetCall struct {
	s            *Service
	volumeId     string
	layerId      string
	annotationId string
	opt_         map[string]interface{}
}

// Get: Gets the volume annotation.
func (r *LayersVolumeAnnotationsService) Get(volumeId string, layerId string, annotationId string) *LayersVolumeAnnotationsGetCall {
	c := &LayersVolumeAnnotationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	c.layerId = layerId
	c.annotationId = annotationId
	return c
}

// Locale sets the optional parameter "locale": The locale information
// for the data. ISO-639-1 language and ISO-3166-1 country code. Ex:
// 'en_US'.
func (c *LayersVolumeAnnotationsGetCall) Locale(locale string) *LayersVolumeAnnotationsGetCall {
	c.opt_["locale"] = locale
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *LayersVolumeAnnotationsGetCall) Source(source string) *LayersVolumeAnnotationsGetCall {
	c.opt_["source"] = source
	return c
}

func (c *LayersVolumeAnnotationsGetCall) Do() (*Volumeannotation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/{volumeId}/layers/{layerId}/annotations/{annotationId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{layerId}", url.QueryEscape(c.layerId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{annotationId}", url.QueryEscape(c.annotationId), 1)
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
	ret := new(Volumeannotation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the volume annotation.",
	//   "httpMethod": "GET",
	//   "id": "books.layers.volumeAnnotations.get",
	//   "parameterOrder": [
	//     "volumeId",
	//     "layerId",
	//     "annotationId"
	//   ],
	//   "parameters": {
	//     "annotationId": {
	//       "description": "The ID of the volume annotation to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "layerId": {
	//       "description": "The ID for the layer to get the annotations.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "The locale information for the data. ISO-639-1 language and ISO-3166-1 country code. Ex: 'en_US'.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "The volume to retrieve annotations for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/{volumeId}/layers/{layerId}/annotations/{annotationId}",
	//   "response": {
	//     "$ref": "Volumeannotation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.layers.volumeAnnotations.list":

type LayersVolumeAnnotationsListCall struct {
	s              *Service
	volumeId       string
	layerId        string
	contentVersion string
	opt_           map[string]interface{}
}

// List: Gets the volume annotations for a volume and layer.
func (r *LayersVolumeAnnotationsService) List(volumeId string, layerId string, contentVersion string) *LayersVolumeAnnotationsListCall {
	c := &LayersVolumeAnnotationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	c.layerId = layerId
	c.contentVersion = contentVersion
	return c
}

// EndOffset sets the optional parameter "endOffset": The end offset to
// end retrieving data from.
func (c *LayersVolumeAnnotationsListCall) EndOffset(endOffset string) *LayersVolumeAnnotationsListCall {
	c.opt_["endOffset"] = endOffset
	return c
}

// EndPosition sets the optional parameter "endPosition": The end
// position to end retrieving data from.
func (c *LayersVolumeAnnotationsListCall) EndPosition(endPosition string) *LayersVolumeAnnotationsListCall {
	c.opt_["endPosition"] = endPosition
	return c
}

// Locale sets the optional parameter "locale": The locale information
// for the data. ISO-639-1 language and ISO-3166-1 country code. Ex:
// 'en_US'.
func (c *LayersVolumeAnnotationsListCall) Locale(locale string) *LayersVolumeAnnotationsListCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *LayersVolumeAnnotationsListCall) MaxResults(maxResults int64) *LayersVolumeAnnotationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous page.
func (c *LayersVolumeAnnotationsListCall) PageToken(pageToken string) *LayersVolumeAnnotationsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// ShowDeleted sets the optional parameter "showDeleted": Set to true to
// return deleted annotations. updatedMin must be in the request to use
// this. Defaults to false.
func (c *LayersVolumeAnnotationsListCall) ShowDeleted(showDeleted bool) *LayersVolumeAnnotationsListCall {
	c.opt_["showDeleted"] = showDeleted
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *LayersVolumeAnnotationsListCall) Source(source string) *LayersVolumeAnnotationsListCall {
	c.opt_["source"] = source
	return c
}

// StartOffset sets the optional parameter "startOffset": The start
// offset to start retrieving data from.
func (c *LayersVolumeAnnotationsListCall) StartOffset(startOffset string) *LayersVolumeAnnotationsListCall {
	c.opt_["startOffset"] = startOffset
	return c
}

// StartPosition sets the optional parameter "startPosition": The start
// position to start retrieving data from.
func (c *LayersVolumeAnnotationsListCall) StartPosition(startPosition string) *LayersVolumeAnnotationsListCall {
	c.opt_["startPosition"] = startPosition
	return c
}

// UpdatedMax sets the optional parameter "updatedMax": RFC 3339
// timestamp to restrict to items updated prior to this timestamp
// (exclusive).
func (c *LayersVolumeAnnotationsListCall) UpdatedMax(updatedMax string) *LayersVolumeAnnotationsListCall {
	c.opt_["updatedMax"] = updatedMax
	return c
}

// UpdatedMin sets the optional parameter "updatedMin": RFC 3339
// timestamp to restrict to items updated since this timestamp
// (inclusive).
func (c *LayersVolumeAnnotationsListCall) UpdatedMin(updatedMin string) *LayersVolumeAnnotationsListCall {
	c.opt_["updatedMin"] = updatedMin
	return c
}

// VolumeAnnotationsVersion sets the optional parameter
// "volumeAnnotationsVersion": The version of the volume annotations
// that you are requesting.
func (c *LayersVolumeAnnotationsListCall) VolumeAnnotationsVersion(volumeAnnotationsVersion string) *LayersVolumeAnnotationsListCall {
	c.opt_["volumeAnnotationsVersion"] = volumeAnnotationsVersion
	return c
}

func (c *LayersVolumeAnnotationsListCall) Do() (*Volumeannotations, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("contentVersion", fmt.Sprintf("%v", c.contentVersion))
	if v, ok := c.opt_["endOffset"]; ok {
		params.Set("endOffset", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["endPosition"]; ok {
		params.Set("endPosition", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showDeleted"]; ok {
		params.Set("showDeleted", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startOffset"]; ok {
		params.Set("startOffset", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startPosition"]; ok {
		params.Set("startPosition", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updatedMax"]; ok {
		params.Set("updatedMax", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updatedMin"]; ok {
		params.Set("updatedMin", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["volumeAnnotationsVersion"]; ok {
		params.Set("volumeAnnotationsVersion", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/{volumeId}/layers/{layerId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{layerId}", url.QueryEscape(c.layerId), 1)
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
	ret := new(Volumeannotations)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the volume annotations for a volume and layer.",
	//   "httpMethod": "GET",
	//   "id": "books.layers.volumeAnnotations.list",
	//   "parameterOrder": [
	//     "volumeId",
	//     "layerId",
	//     "contentVersion"
	//   ],
	//   "parameters": {
	//     "contentVersion": {
	//       "description": "The content version for the requested volume.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "endOffset": {
	//       "description": "The end offset to end retrieving data from.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "endPosition": {
	//       "description": "The end position to end retrieving data from.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "layerId": {
	//       "description": "The ID for the layer to get the annotations.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "The locale information for the data. ISO-639-1 language and ISO-3166-1 country code. Ex: 'en_US'.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "200",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value of the nextToken from the previous page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "showDeleted": {
	//       "description": "Set to true to return deleted annotations. updatedMin must be in the request to use this. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startOffset": {
	//       "description": "The start offset to start retrieving data from.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startPosition": {
	//       "description": "The start position to start retrieving data from.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "updatedMax": {
	//       "description": "RFC 3339 timestamp to restrict to items updated prior to this timestamp (exclusive).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "updatedMin": {
	//       "description": "RFC 3339 timestamp to restrict to items updated since this timestamp (inclusive).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeAnnotationsVersion": {
	//       "description": "The version of the volume annotations that you are requesting.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "The volume to retrieve annotations for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/{volumeId}/layers/{layerId}",
	//   "response": {
	//     "$ref": "Volumeannotations"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.myconfig.releaseDownloadAccess":

type MyconfigReleaseDownloadAccessCall struct {
	s         *Service
	volumeIds []string
	cpksver   string
	opt_      map[string]interface{}
}

// ReleaseDownloadAccess: Release downloaded content access restriction.
func (r *MyconfigService) ReleaseDownloadAccess(volumeIds []string, cpksver string) *MyconfigReleaseDownloadAccessCall {
	c := &MyconfigReleaseDownloadAccessCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeIds = volumeIds
	c.cpksver = cpksver
	return c
}

// Locale sets the optional parameter "locale": ISO-639-1, ISO-3166-1
// codes for message localization, i.e. en_US.
func (c *MyconfigReleaseDownloadAccessCall) Locale(locale string) *MyconfigReleaseDownloadAccessCall {
	c.opt_["locale"] = locale
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MyconfigReleaseDownloadAccessCall) Source(source string) *MyconfigReleaseDownloadAccessCall {
	c.opt_["source"] = source
	return c
}

func (c *MyconfigReleaseDownloadAccessCall) Do() (*DownloadAccesses, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("cpksver", fmt.Sprintf("%v", c.cpksver))
	for _, v := range c.volumeIds {
		params.Add("volumeIds", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "myconfig/releaseDownloadAccess")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(DownloadAccesses)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Release downloaded content access restriction.",
	//   "httpMethod": "POST",
	//   "id": "books.myconfig.releaseDownloadAccess",
	//   "parameterOrder": [
	//     "volumeIds",
	//     "cpksver"
	//   ],
	//   "parameters": {
	//     "cpksver": {
	//       "description": "The device/version ID from which to release the restriction.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "ISO-639-1, ISO-3166-1 codes for message localization, i.e. en_US.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeIds": {
	//       "description": "The volume(s) to release restrictions for.",
	//       "location": "query",
	//       "repeated": true,
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "myconfig/releaseDownloadAccess",
	//   "response": {
	//     "$ref": "DownloadAccesses"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.myconfig.requestAccess":

type MyconfigRequestAccessCall struct {
	s        *Service
	source   string
	volumeId string
	nonce    string
	cpksver  string
	opt_     map[string]interface{}
}

// RequestAccess: Request concurrent and download access restrictions.
func (r *MyconfigService) RequestAccess(source string, volumeId string, nonce string, cpksver string) *MyconfigRequestAccessCall {
	c := &MyconfigRequestAccessCall{s: r.s, opt_: make(map[string]interface{})}
	c.source = source
	c.volumeId = volumeId
	c.nonce = nonce
	c.cpksver = cpksver
	return c
}

// LicenseTypes sets the optional parameter "licenseTypes": The type of
// access license to request. If not specified, the default is BOTH.
func (c *MyconfigRequestAccessCall) LicenseTypes(licenseTypes string) *MyconfigRequestAccessCall {
	c.opt_["licenseTypes"] = licenseTypes
	return c
}

// Locale sets the optional parameter "locale": ISO-639-1, ISO-3166-1
// codes for message localization, i.e. en_US.
func (c *MyconfigRequestAccessCall) Locale(locale string) *MyconfigRequestAccessCall {
	c.opt_["locale"] = locale
	return c
}

func (c *MyconfigRequestAccessCall) Do() (*RequestAccess, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("cpksver", fmt.Sprintf("%v", c.cpksver))
	params.Set("nonce", fmt.Sprintf("%v", c.nonce))
	params.Set("source", fmt.Sprintf("%v", c.source))
	params.Set("volumeId", fmt.Sprintf("%v", c.volumeId))
	if v, ok := c.opt_["licenseTypes"]; ok {
		params.Set("licenseTypes", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "myconfig/requestAccess")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(RequestAccess)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Request concurrent and download access restrictions.",
	//   "httpMethod": "POST",
	//   "id": "books.myconfig.requestAccess",
	//   "parameterOrder": [
	//     "source",
	//     "volumeId",
	//     "nonce",
	//     "cpksver"
	//   ],
	//   "parameters": {
	//     "cpksver": {
	//       "description": "The device/version ID from which to request the restrictions.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "licenseTypes": {
	//       "description": "The type of access license to request. If not specified, the default is BOTH.",
	//       "enum": [
	//         "BOTH",
	//         "CONCURRENT",
	//         "DOWNLOAD"
	//       ],
	//       "enumDescriptions": [
	//         "Both concurrent and download licenses.",
	//         "Concurrent access license.",
	//         "Offline download access license."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "ISO-639-1, ISO-3166-1 codes for message localization, i.e. en_US.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "nonce": {
	//       "description": "The client nonce value.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "The volume to request concurrent/download restrictions for.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "myconfig/requestAccess",
	//   "response": {
	//     "$ref": "RequestAccess"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.myconfig.syncVolumeLicenses":

type MyconfigSyncVolumeLicensesCall struct {
	s       *Service
	source  string
	nonce   string
	cpksver string
	opt_    map[string]interface{}
}

// SyncVolumeLicenses: Request downloaded content access for specified
// volumes on the My eBooks shelf.
func (r *MyconfigService) SyncVolumeLicenses(source string, nonce string, cpksver string) *MyconfigSyncVolumeLicensesCall {
	c := &MyconfigSyncVolumeLicensesCall{s: r.s, opt_: make(map[string]interface{})}
	c.source = source
	c.nonce = nonce
	c.cpksver = cpksver
	return c
}

// Features sets the optional parameter "features": List of features
// supported by the client, i.e., 'RENTALS'
func (c *MyconfigSyncVolumeLicensesCall) Features(features string) *MyconfigSyncVolumeLicensesCall {
	c.opt_["features"] = features
	return c
}

// Locale sets the optional parameter "locale": ISO-639-1, ISO-3166-1
// codes for message localization, i.e. en_US.
func (c *MyconfigSyncVolumeLicensesCall) Locale(locale string) *MyconfigSyncVolumeLicensesCall {
	c.opt_["locale"] = locale
	return c
}

// ShowPreorders sets the optional parameter "showPreorders": Set to
// true to show pre-ordered books. Defaults to false.
func (c *MyconfigSyncVolumeLicensesCall) ShowPreorders(showPreorders bool) *MyconfigSyncVolumeLicensesCall {
	c.opt_["showPreorders"] = showPreorders
	return c
}

// VolumeIds sets the optional parameter "volumeIds": The volume(s) to
// request download restrictions for.
func (c *MyconfigSyncVolumeLicensesCall) VolumeIds(volumeIds string) *MyconfigSyncVolumeLicensesCall {
	c.opt_["volumeIds"] = volumeIds
	return c
}

func (c *MyconfigSyncVolumeLicensesCall) Do() (*Volumes, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("cpksver", fmt.Sprintf("%v", c.cpksver))
	params.Set("nonce", fmt.Sprintf("%v", c.nonce))
	params.Set("source", fmt.Sprintf("%v", c.source))
	if v, ok := c.opt_["features"]; ok {
		params.Set("features", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showPreorders"]; ok {
		params.Set("showPreorders", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["volumeIds"]; ok {
		params.Set("volumeIds", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "myconfig/syncVolumeLicenses")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(Volumes)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Request downloaded content access for specified volumes on the My eBooks shelf.",
	//   "httpMethod": "POST",
	//   "id": "books.myconfig.syncVolumeLicenses",
	//   "parameterOrder": [
	//     "source",
	//     "nonce",
	//     "cpksver"
	//   ],
	//   "parameters": {
	//     "cpksver": {
	//       "description": "The device/version ID from which to release the restriction.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "features": {
	//       "description": "List of features supported by the client, i.e., 'RENTALS'",
	//       "enum": [
	//         "RENTALS"
	//       ],
	//       "enumDescriptions": [
	//         "Client supports rentals."
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "ISO-639-1, ISO-3166-1 codes for message localization, i.e. en_US.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "nonce": {
	//       "description": "The client nonce value.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "showPreorders": {
	//       "description": "Set to true to show pre-ordered books. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "volumeIds": {
	//       "description": "The volume(s) to request download restrictions for.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "myconfig/syncVolumeLicenses",
	//   "response": {
	//     "$ref": "Volumes"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.annotations.delete":

type MylibraryAnnotationsDeleteCall struct {
	s            *Service
	annotationId string
	opt_         map[string]interface{}
}

// Delete: Deletes an annotation.
func (r *MylibraryAnnotationsService) Delete(annotationId string) *MylibraryAnnotationsDeleteCall {
	c := &MylibraryAnnotationsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.annotationId = annotationId
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryAnnotationsDeleteCall) Source(source string) *MylibraryAnnotationsDeleteCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryAnnotationsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/annotations/{annotationId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{annotationId}", url.QueryEscape(c.annotationId), 1)
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
	//   "description": "Deletes an annotation.",
	//   "httpMethod": "DELETE",
	//   "id": "books.mylibrary.annotations.delete",
	//   "parameterOrder": [
	//     "annotationId"
	//   ],
	//   "parameters": {
	//     "annotationId": {
	//       "description": "The ID for the annotation to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/annotations/{annotationId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.annotations.get":

type MylibraryAnnotationsGetCall struct {
	s            *Service
	annotationId string
	opt_         map[string]interface{}
}

// Get: Gets an annotation by its ID.
func (r *MylibraryAnnotationsService) Get(annotationId string) *MylibraryAnnotationsGetCall {
	c := &MylibraryAnnotationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.annotationId = annotationId
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryAnnotationsGetCall) Source(source string) *MylibraryAnnotationsGetCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryAnnotationsGetCall) Do() (*Annotation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/annotations/{annotationId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{annotationId}", url.QueryEscape(c.annotationId), 1)
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
	ret := new(Annotation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets an annotation by its ID.",
	//   "httpMethod": "GET",
	//   "id": "books.mylibrary.annotations.get",
	//   "parameterOrder": [
	//     "annotationId"
	//   ],
	//   "parameters": {
	//     "annotationId": {
	//       "description": "The ID for the annotation to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/annotations/{annotationId}",
	//   "response": {
	//     "$ref": "Annotation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.annotations.insert":

type MylibraryAnnotationsInsertCall struct {
	s          *Service
	annotation *Annotation
	opt_       map[string]interface{}
}

// Insert: Inserts a new annotation.
func (r *MylibraryAnnotationsService) Insert(annotation *Annotation) *MylibraryAnnotationsInsertCall {
	c := &MylibraryAnnotationsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.annotation = annotation
	return c
}

// ShowOnlySummaryInResponse sets the optional parameter
// "showOnlySummaryInResponse": Requests that only the summary of the
// specified layer be provided in the response.
func (c *MylibraryAnnotationsInsertCall) ShowOnlySummaryInResponse(showOnlySummaryInResponse bool) *MylibraryAnnotationsInsertCall {
	c.opt_["showOnlySummaryInResponse"] = showOnlySummaryInResponse
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryAnnotationsInsertCall) Source(source string) *MylibraryAnnotationsInsertCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryAnnotationsInsertCall) Do() (*Annotation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.annotation)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["showOnlySummaryInResponse"]; ok {
		params.Set("showOnlySummaryInResponse", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/annotations")
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
	ret := new(Annotation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new annotation.",
	//   "httpMethod": "POST",
	//   "id": "books.mylibrary.annotations.insert",
	//   "parameters": {
	//     "showOnlySummaryInResponse": {
	//       "description": "Requests that only the summary of the specified layer be provided in the response.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/annotations",
	//   "request": {
	//     "$ref": "Annotation"
	//   },
	//   "response": {
	//     "$ref": "Annotation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.annotations.list":

type MylibraryAnnotationsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves a list of annotations, possibly filtered.
func (r *MylibraryAnnotationsService) List() *MylibraryAnnotationsListCall {
	c := &MylibraryAnnotationsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// ContentVersion sets the optional parameter "contentVersion": The
// content version for the requested volume.
func (c *MylibraryAnnotationsListCall) ContentVersion(contentVersion string) *MylibraryAnnotationsListCall {
	c.opt_["contentVersion"] = contentVersion
	return c
}

// LayerId sets the optional parameter "layerId": The layer ID to limit
// annotation by.
func (c *MylibraryAnnotationsListCall) LayerId(layerId string) *MylibraryAnnotationsListCall {
	c.opt_["layerId"] = layerId
	return c
}

// LayerIds sets the optional parameter "layerIds": The layer ID(s) to
// limit annotation by.
func (c *MylibraryAnnotationsListCall) LayerIds(layerIds string) *MylibraryAnnotationsListCall {
	c.opt_["layerIds"] = layerIds
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *MylibraryAnnotationsListCall) MaxResults(maxResults int64) *MylibraryAnnotationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageIds sets the optional parameter "pageIds": The page ID(s) for the
// volume that is being queried.
func (c *MylibraryAnnotationsListCall) PageIds(pageIds string) *MylibraryAnnotationsListCall {
	c.opt_["pageIds"] = pageIds
	return c
}

// PageToken sets the optional parameter "pageToken": The value of the
// nextToken from the previous page.
func (c *MylibraryAnnotationsListCall) PageToken(pageToken string) *MylibraryAnnotationsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// ShowDeleted sets the optional parameter "showDeleted": Set to true to
// return deleted annotations. updatedMin must be in the request to use
// this. Defaults to false.
func (c *MylibraryAnnotationsListCall) ShowDeleted(showDeleted bool) *MylibraryAnnotationsListCall {
	c.opt_["showDeleted"] = showDeleted
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryAnnotationsListCall) Source(source string) *MylibraryAnnotationsListCall {
	c.opt_["source"] = source
	return c
}

// UpdatedMax sets the optional parameter "updatedMax": RFC 3339
// timestamp to restrict to items updated prior to this timestamp
// (exclusive).
func (c *MylibraryAnnotationsListCall) UpdatedMax(updatedMax string) *MylibraryAnnotationsListCall {
	c.opt_["updatedMax"] = updatedMax
	return c
}

// UpdatedMin sets the optional parameter "updatedMin": RFC 3339
// timestamp to restrict to items updated since this timestamp
// (inclusive).
func (c *MylibraryAnnotationsListCall) UpdatedMin(updatedMin string) *MylibraryAnnotationsListCall {
	c.opt_["updatedMin"] = updatedMin
	return c
}

// VolumeId sets the optional parameter "volumeId": The volume to
// restrict annotations to.
func (c *MylibraryAnnotationsListCall) VolumeId(volumeId string) *MylibraryAnnotationsListCall {
	c.opt_["volumeId"] = volumeId
	return c
}

func (c *MylibraryAnnotationsListCall) Do() (*Annotations, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["contentVersion"]; ok {
		params.Set("contentVersion", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["layerId"]; ok {
		params.Set("layerId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["layerIds"]; ok {
		params.Set("layerIds", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageIds"]; ok {
		params.Set("pageIds", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showDeleted"]; ok {
		params.Set("showDeleted", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updatedMax"]; ok {
		params.Set("updatedMax", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updatedMin"]; ok {
		params.Set("updatedMin", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["volumeId"]; ok {
		params.Set("volumeId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/annotations")
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
	ret := new(Annotations)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of annotations, possibly filtered.",
	//   "httpMethod": "GET",
	//   "id": "books.mylibrary.annotations.list",
	//   "parameters": {
	//     "contentVersion": {
	//       "description": "The content version for the requested volume.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "layerId": {
	//       "description": "The layer ID to limit annotation by.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "layerIds": {
	//       "description": "The layer ID(s) to limit annotation by.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "40",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageIds": {
	//       "description": "The page ID(s) for the volume that is being queried.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The value of the nextToken from the previous page.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "showDeleted": {
	//       "description": "Set to true to return deleted annotations. updatedMin must be in the request to use this. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "updatedMax": {
	//       "description": "RFC 3339 timestamp to restrict to items updated prior to this timestamp (exclusive).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "updatedMin": {
	//       "description": "RFC 3339 timestamp to restrict to items updated since this timestamp (inclusive).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "The volume to restrict annotations to.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/annotations",
	//   "response": {
	//     "$ref": "Annotations"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.annotations.summary":

type MylibraryAnnotationsSummaryCall struct {
	s        *Service
	layerIds []string
	volumeId string
	opt_     map[string]interface{}
}

// Summary: Gets the summary of specified layers.
func (r *MylibraryAnnotationsService) Summary(layerIds []string, volumeId string) *MylibraryAnnotationsSummaryCall {
	c := &MylibraryAnnotationsSummaryCall{s: r.s, opt_: make(map[string]interface{})}
	c.layerIds = layerIds
	c.volumeId = volumeId
	return c
}

func (c *MylibraryAnnotationsSummaryCall) Do() (*AnnotationsSummary, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("volumeId", fmt.Sprintf("%v", c.volumeId))
	for _, v := range c.layerIds {
		params.Add("layerIds", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/annotations/summary")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(AnnotationsSummary)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the summary of specified layers.",
	//   "httpMethod": "POST",
	//   "id": "books.mylibrary.annotations.summary",
	//   "parameterOrder": [
	//     "layerIds",
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "layerIds": {
	//       "description": "Array of layer IDs to get the summary for.",
	//       "location": "query",
	//       "repeated": true,
	//       "required": true,
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "Volume id to get the summary for.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/annotations/summary",
	//   "response": {
	//     "$ref": "AnnotationsSummary"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.annotations.update":

type MylibraryAnnotationsUpdateCall struct {
	s            *Service
	annotationId string
	annotation   *Annotation
	opt_         map[string]interface{}
}

// Update: Updates an existing annotation.
func (r *MylibraryAnnotationsService) Update(annotationId string, annotation *Annotation) *MylibraryAnnotationsUpdateCall {
	c := &MylibraryAnnotationsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.annotationId = annotationId
	c.annotation = annotation
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryAnnotationsUpdateCall) Source(source string) *MylibraryAnnotationsUpdateCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryAnnotationsUpdateCall) Do() (*Annotation, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.annotation)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/annotations/{annotationId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{annotationId}", url.QueryEscape(c.annotationId), 1)
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
	ret := new(Annotation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing annotation.",
	//   "httpMethod": "PUT",
	//   "id": "books.mylibrary.annotations.update",
	//   "parameterOrder": [
	//     "annotationId"
	//   ],
	//   "parameters": {
	//     "annotationId": {
	//       "description": "The ID for the annotation to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/annotations/{annotationId}",
	//   "request": {
	//     "$ref": "Annotation"
	//   },
	//   "response": {
	//     "$ref": "Annotation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.bookshelves.addVolume":

type MylibraryBookshelvesAddVolumeCall struct {
	s        *Service
	shelf    string
	volumeId string
	opt_     map[string]interface{}
}

// AddVolume: Adds a volume to a bookshelf.
func (r *MylibraryBookshelvesService) AddVolume(shelf string, volumeId string) *MylibraryBookshelvesAddVolumeCall {
	c := &MylibraryBookshelvesAddVolumeCall{s: r.s, opt_: make(map[string]interface{})}
	c.shelf = shelf
	c.volumeId = volumeId
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryBookshelvesAddVolumeCall) Source(source string) *MylibraryBookshelvesAddVolumeCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryBookshelvesAddVolumeCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("volumeId", fmt.Sprintf("%v", c.volumeId))
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/bookshelves/{shelf}/addVolume")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{shelf}", url.QueryEscape(c.shelf), 1)
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
	//   "description": "Adds a volume to a bookshelf.",
	//   "httpMethod": "POST",
	//   "id": "books.mylibrary.bookshelves.addVolume",
	//   "parameterOrder": [
	//     "shelf",
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "shelf": {
	//       "description": "ID of bookshelf to which to add a volume.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "ID of volume to add.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/bookshelves/{shelf}/addVolume",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.bookshelves.clearVolumes":

type MylibraryBookshelvesClearVolumesCall struct {
	s     *Service
	shelf string
	opt_  map[string]interface{}
}

// ClearVolumes: Clears all volumes from a bookshelf.
func (r *MylibraryBookshelvesService) ClearVolumes(shelf string) *MylibraryBookshelvesClearVolumesCall {
	c := &MylibraryBookshelvesClearVolumesCall{s: r.s, opt_: make(map[string]interface{})}
	c.shelf = shelf
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryBookshelvesClearVolumesCall) Source(source string) *MylibraryBookshelvesClearVolumesCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryBookshelvesClearVolumesCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/bookshelves/{shelf}/clearVolumes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{shelf}", url.QueryEscape(c.shelf), 1)
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
	//   "description": "Clears all volumes from a bookshelf.",
	//   "httpMethod": "POST",
	//   "id": "books.mylibrary.bookshelves.clearVolumes",
	//   "parameterOrder": [
	//     "shelf"
	//   ],
	//   "parameters": {
	//     "shelf": {
	//       "description": "ID of bookshelf from which to remove a volume.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/bookshelves/{shelf}/clearVolumes",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.bookshelves.get":

type MylibraryBookshelvesGetCall struct {
	s     *Service
	shelf string
	opt_  map[string]interface{}
}

// Get: Retrieves metadata for a specific bookshelf belonging to the
// authenticated user.
func (r *MylibraryBookshelvesService) Get(shelf string) *MylibraryBookshelvesGetCall {
	c := &MylibraryBookshelvesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.shelf = shelf
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryBookshelvesGetCall) Source(source string) *MylibraryBookshelvesGetCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryBookshelvesGetCall) Do() (*Bookshelf, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/bookshelves/{shelf}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{shelf}", url.QueryEscape(c.shelf), 1)
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
	ret := new(Bookshelf)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves metadata for a specific bookshelf belonging to the authenticated user.",
	//   "httpMethod": "GET",
	//   "id": "books.mylibrary.bookshelves.get",
	//   "parameterOrder": [
	//     "shelf"
	//   ],
	//   "parameters": {
	//     "shelf": {
	//       "description": "ID of bookshelf to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/bookshelves/{shelf}",
	//   "response": {
	//     "$ref": "Bookshelf"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.bookshelves.list":

type MylibraryBookshelvesListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves a list of bookshelves belonging to the authenticated
// user.
func (r *MylibraryBookshelvesService) List() *MylibraryBookshelvesListCall {
	c := &MylibraryBookshelvesListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryBookshelvesListCall) Source(source string) *MylibraryBookshelvesListCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryBookshelvesListCall) Do() (*Bookshelves, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/bookshelves")
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
	ret := new(Bookshelves)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of bookshelves belonging to the authenticated user.",
	//   "httpMethod": "GET",
	//   "id": "books.mylibrary.bookshelves.list",
	//   "parameters": {
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/bookshelves",
	//   "response": {
	//     "$ref": "Bookshelves"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.bookshelves.moveVolume":

type MylibraryBookshelvesMoveVolumeCall struct {
	s              *Service
	shelf          string
	volumeId       string
	volumePosition int64
	opt_           map[string]interface{}
}

// MoveVolume: Moves a volume within a bookshelf.
func (r *MylibraryBookshelvesService) MoveVolume(shelf string, volumeId string, volumePosition int64) *MylibraryBookshelvesMoveVolumeCall {
	c := &MylibraryBookshelvesMoveVolumeCall{s: r.s, opt_: make(map[string]interface{})}
	c.shelf = shelf
	c.volumeId = volumeId
	c.volumePosition = volumePosition
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryBookshelvesMoveVolumeCall) Source(source string) *MylibraryBookshelvesMoveVolumeCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryBookshelvesMoveVolumeCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("volumeId", fmt.Sprintf("%v", c.volumeId))
	params.Set("volumePosition", fmt.Sprintf("%v", c.volumePosition))
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/bookshelves/{shelf}/moveVolume")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{shelf}", url.QueryEscape(c.shelf), 1)
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
	//   "description": "Moves a volume within a bookshelf.",
	//   "httpMethod": "POST",
	//   "id": "books.mylibrary.bookshelves.moveVolume",
	//   "parameterOrder": [
	//     "shelf",
	//     "volumeId",
	//     "volumePosition"
	//   ],
	//   "parameters": {
	//     "shelf": {
	//       "description": "ID of bookshelf with the volume.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "ID of volume to move.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "volumePosition": {
	//       "description": "Position on shelf to move the item (0 puts the item before the current first item, 1 puts it between the first and the second and so on.)",
	//       "format": "int32",
	//       "location": "query",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "mylibrary/bookshelves/{shelf}/moveVolume",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.bookshelves.removeVolume":

type MylibraryBookshelvesRemoveVolumeCall struct {
	s        *Service
	shelf    string
	volumeId string
	opt_     map[string]interface{}
}

// RemoveVolume: Removes a volume from a bookshelf.
func (r *MylibraryBookshelvesService) RemoveVolume(shelf string, volumeId string) *MylibraryBookshelvesRemoveVolumeCall {
	c := &MylibraryBookshelvesRemoveVolumeCall{s: r.s, opt_: make(map[string]interface{})}
	c.shelf = shelf
	c.volumeId = volumeId
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryBookshelvesRemoveVolumeCall) Source(source string) *MylibraryBookshelvesRemoveVolumeCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryBookshelvesRemoveVolumeCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("volumeId", fmt.Sprintf("%v", c.volumeId))
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/bookshelves/{shelf}/removeVolume")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{shelf}", url.QueryEscape(c.shelf), 1)
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
	//   "description": "Removes a volume from a bookshelf.",
	//   "httpMethod": "POST",
	//   "id": "books.mylibrary.bookshelves.removeVolume",
	//   "parameterOrder": [
	//     "shelf",
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "shelf": {
	//       "description": "ID of bookshelf from which to remove a volume.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "ID of volume to remove.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/bookshelves/{shelf}/removeVolume",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.bookshelves.volumes.list":

type MylibraryBookshelvesVolumesListCall struct {
	s     *Service
	shelf string
	opt_  map[string]interface{}
}

// List: Gets volume information for volumes on a bookshelf.
func (r *MylibraryBookshelvesVolumesService) List(shelf string) *MylibraryBookshelvesVolumesListCall {
	c := &MylibraryBookshelvesVolumesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.shelf = shelf
	return c
}

// Country sets the optional parameter "country": ISO-3166-1 code to
// override the IP-based location.
func (c *MylibraryBookshelvesVolumesListCall) Country(country string) *MylibraryBookshelvesVolumesListCall {
	c.opt_["country"] = country
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *MylibraryBookshelvesVolumesListCall) MaxResults(maxResults int64) *MylibraryBookshelvesVolumesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Projection sets the optional parameter "projection": Restrict
// information returned to a set of selected fields.
func (c *MylibraryBookshelvesVolumesListCall) Projection(projection string) *MylibraryBookshelvesVolumesListCall {
	c.opt_["projection"] = projection
	return c
}

// Q sets the optional parameter "q": Full-text search query string in
// this bookshelf.
func (c *MylibraryBookshelvesVolumesListCall) Q(q string) *MylibraryBookshelvesVolumesListCall {
	c.opt_["q"] = q
	return c
}

// ShowPreorders sets the optional parameter "showPreorders": Set to
// true to show pre-ordered books. Defaults to false.
func (c *MylibraryBookshelvesVolumesListCall) ShowPreorders(showPreorders bool) *MylibraryBookshelvesVolumesListCall {
	c.opt_["showPreorders"] = showPreorders
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryBookshelvesVolumesListCall) Source(source string) *MylibraryBookshelvesVolumesListCall {
	c.opt_["source"] = source
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first element to return (starts at 0)
func (c *MylibraryBookshelvesVolumesListCall) StartIndex(startIndex int64) *MylibraryBookshelvesVolumesListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *MylibraryBookshelvesVolumesListCall) Do() (*Volumes, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["country"]; ok {
		params.Set("country", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["q"]; ok {
		params.Set("q", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showPreorders"]; ok {
		params.Set("showPreorders", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/bookshelves/{shelf}/volumes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{shelf}", url.QueryEscape(c.shelf), 1)
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
	ret := new(Volumes)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets volume information for volumes on a bookshelf.",
	//   "httpMethod": "GET",
	//   "id": "books.mylibrary.bookshelves.volumes.list",
	//   "parameterOrder": [
	//     "shelf"
	//   ],
	//   "parameters": {
	//     "country": {
	//       "description": "ISO-3166-1 code to override the IP-based location.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "projection": {
	//       "description": "Restrict information returned to a set of selected fields.",
	//       "enum": [
	//         "full",
	//         "lite"
	//       ],
	//       "enumDescriptions": [
	//         "Includes all volume data.",
	//         "Includes a subset of fields in volumeInfo and accessInfo."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "Full-text search query string in this bookshelf.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "shelf": {
	//       "description": "The bookshelf ID or name retrieve volumes for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "showPreorders": {
	//       "description": "Set to true to show pre-ordered books. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first element to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "mylibrary/bookshelves/{shelf}/volumes",
	//   "response": {
	//     "$ref": "Volumes"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.readingpositions.get":

type MylibraryReadingpositionsGetCall struct {
	s        *Service
	volumeId string
	opt_     map[string]interface{}
}

// Get: Retrieves my reading position information for a volume.
func (r *MylibraryReadingpositionsService) Get(volumeId string) *MylibraryReadingpositionsGetCall {
	c := &MylibraryReadingpositionsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	return c
}

// ContentVersion sets the optional parameter "contentVersion": Volume
// content version for which this reading position is requested.
func (c *MylibraryReadingpositionsGetCall) ContentVersion(contentVersion string) *MylibraryReadingpositionsGetCall {
	c.opt_["contentVersion"] = contentVersion
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryReadingpositionsGetCall) Source(source string) *MylibraryReadingpositionsGetCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryReadingpositionsGetCall) Do() (*ReadingPosition, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["contentVersion"]; ok {
		params.Set("contentVersion", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/readingpositions/{volumeId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
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
	ret := new(ReadingPosition)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves my reading position information for a volume.",
	//   "httpMethod": "GET",
	//   "id": "books.mylibrary.readingpositions.get",
	//   "parameterOrder": [
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "contentVersion": {
	//       "description": "Volume content version for which this reading position is requested.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "ID of volume for which to retrieve a reading position.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/readingpositions/{volumeId}",
	//   "response": {
	//     "$ref": "ReadingPosition"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.mylibrary.readingpositions.setPosition":

type MylibraryReadingpositionsSetPositionCall struct {
	s         *Service
	volumeId  string
	timestamp string
	position  string
	opt_      map[string]interface{}
}

// SetPosition: Sets my reading position information for a volume.
func (r *MylibraryReadingpositionsService) SetPosition(volumeId string, timestamp string, position string) *MylibraryReadingpositionsSetPositionCall {
	c := &MylibraryReadingpositionsSetPositionCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	c.timestamp = timestamp
	c.position = position
	return c
}

// Action sets the optional parameter "action": Action that caused this
// reading position to be set.
func (c *MylibraryReadingpositionsSetPositionCall) Action(action string) *MylibraryReadingpositionsSetPositionCall {
	c.opt_["action"] = action
	return c
}

// ContentVersion sets the optional parameter "contentVersion": Volume
// content version for which this reading position applies.
func (c *MylibraryReadingpositionsSetPositionCall) ContentVersion(contentVersion string) *MylibraryReadingpositionsSetPositionCall {
	c.opt_["contentVersion"] = contentVersion
	return c
}

// DeviceCookie sets the optional parameter "deviceCookie": Random
// persistent device cookie optional on set position.
func (c *MylibraryReadingpositionsSetPositionCall) DeviceCookie(deviceCookie string) *MylibraryReadingpositionsSetPositionCall {
	c.opt_["deviceCookie"] = deviceCookie
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *MylibraryReadingpositionsSetPositionCall) Source(source string) *MylibraryReadingpositionsSetPositionCall {
	c.opt_["source"] = source
	return c
}

func (c *MylibraryReadingpositionsSetPositionCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("position", fmt.Sprintf("%v", c.position))
	params.Set("timestamp", fmt.Sprintf("%v", c.timestamp))
	if v, ok := c.opt_["action"]; ok {
		params.Set("action", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["contentVersion"]; ok {
		params.Set("contentVersion", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["deviceCookie"]; ok {
		params.Set("deviceCookie", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "mylibrary/readingpositions/{volumeId}/setPosition")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
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
	//   "description": "Sets my reading position information for a volume.",
	//   "httpMethod": "POST",
	//   "id": "books.mylibrary.readingpositions.setPosition",
	//   "parameterOrder": [
	//     "volumeId",
	//     "timestamp",
	//     "position"
	//   ],
	//   "parameters": {
	//     "action": {
	//       "description": "Action that caused this reading position to be set.",
	//       "enum": [
	//         "bookmark",
	//         "chapter",
	//         "next-page",
	//         "prev-page",
	//         "scroll",
	//         "search"
	//       ],
	//       "enumDescriptions": [
	//         "User chose bookmark within volume.",
	//         "User selected chapter from list.",
	//         "Next page event.",
	//         "Previous page event.",
	//         "User navigated to page.",
	//         "User chose search results within volume."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "contentVersion": {
	//       "description": "Volume content version for which this reading position applies.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "deviceCookie": {
	//       "description": "Random persistent device cookie optional on set position.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "position": {
	//       "description": "Position string for the new volume reading position.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "timestamp": {
	//       "description": "RFC 3339 UTC format timestamp associated with this reading position.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "ID of volume for which to update the reading position.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "mylibrary/readingpositions/{volumeId}/setPosition",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.volumes.get":

type VolumesGetCall struct {
	s        *Service
	volumeId string
	opt_     map[string]interface{}
}

// Get: Gets volume information for a single volume.
func (r *VolumesService) Get(volumeId string) *VolumesGetCall {
	c := &VolumesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	return c
}

// Country sets the optional parameter "country": ISO-3166-1 code to
// override the IP-based location.
func (c *VolumesGetCall) Country(country string) *VolumesGetCall {
	c.opt_["country"] = country
	return c
}

// Partner sets the optional parameter "partner": Brand results for
// partner ID.
func (c *VolumesGetCall) Partner(partner string) *VolumesGetCall {
	c.opt_["partner"] = partner
	return c
}

// Projection sets the optional parameter "projection": Restrict
// information returned to a set of selected fields.
func (c *VolumesGetCall) Projection(projection string) *VolumesGetCall {
	c.opt_["projection"] = projection
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *VolumesGetCall) Source(source string) *VolumesGetCall {
	c.opt_["source"] = source
	return c
}

func (c *VolumesGetCall) Do() (*Volume, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["country"]; ok {
		params.Set("country", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["partner"]; ok {
		params.Set("partner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/{volumeId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
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
	ret := new(Volume)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets volume information for a single volume.",
	//   "httpMethod": "GET",
	//   "id": "books.volumes.get",
	//   "parameterOrder": [
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "country": {
	//       "description": "ISO-3166-1 code to override the IP-based location.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "partner": {
	//       "description": "Brand results for partner ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Restrict information returned to a set of selected fields.",
	//       "enum": [
	//         "full",
	//         "lite"
	//       ],
	//       "enumDescriptions": [
	//         "Includes all volume data.",
	//         "Includes a subset of fields in volumeInfo and accessInfo."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "ID of volume to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/{volumeId}",
	//   "response": {
	//     "$ref": "Volume"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.volumes.list":

type VolumesListCall struct {
	s    *Service
	q    string
	opt_ map[string]interface{}
}

// List: Performs a book search.
func (r *VolumesService) List(q string) *VolumesListCall {
	c := &VolumesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.q = q
	return c
}

// Download sets the optional parameter "download": Restrict to volumes
// by download availability.
func (c *VolumesListCall) Download(download string) *VolumesListCall {
	c.opt_["download"] = download
	return c
}

// Filter sets the optional parameter "filter": Filter search results.
func (c *VolumesListCall) Filter(filter string) *VolumesListCall {
	c.opt_["filter"] = filter
	return c
}

// LangRestrict sets the optional parameter "langRestrict": Restrict
// results to books with this language code.
func (c *VolumesListCall) LangRestrict(langRestrict string) *VolumesListCall {
	c.opt_["langRestrict"] = langRestrict
	return c
}

// LibraryRestrict sets the optional parameter "libraryRestrict":
// Restrict search to this user's library.
func (c *VolumesListCall) LibraryRestrict(libraryRestrict string) *VolumesListCall {
	c.opt_["libraryRestrict"] = libraryRestrict
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *VolumesListCall) MaxResults(maxResults int64) *VolumesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OrderBy sets the optional parameter "orderBy": Sort search results.
func (c *VolumesListCall) OrderBy(orderBy string) *VolumesListCall {
	c.opt_["orderBy"] = orderBy
	return c
}

// Partner sets the optional parameter "partner": Restrict and brand
// results for partner ID.
func (c *VolumesListCall) Partner(partner string) *VolumesListCall {
	c.opt_["partner"] = partner
	return c
}

// PrintType sets the optional parameter "printType": Restrict to books
// or magazines.
func (c *VolumesListCall) PrintType(printType string) *VolumesListCall {
	c.opt_["printType"] = printType
	return c
}

// Projection sets the optional parameter "projection": Restrict
// information returned to a set of selected fields.
func (c *VolumesListCall) Projection(projection string) *VolumesListCall {
	c.opt_["projection"] = projection
	return c
}

// ShowPreorders sets the optional parameter "showPreorders": Set to
// true to show books available for preorder. Defaults to false.
func (c *VolumesListCall) ShowPreorders(showPreorders bool) *VolumesListCall {
	c.opt_["showPreorders"] = showPreorders
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *VolumesListCall) Source(source string) *VolumesListCall {
	c.opt_["source"] = source
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first result to return (starts at 0)
func (c *VolumesListCall) StartIndex(startIndex int64) *VolumesListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *VolumesListCall) Do() (*Volumes, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("q", fmt.Sprintf("%v", c.q))
	if v, ok := c.opt_["download"]; ok {
		params.Set("download", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["langRestrict"]; ok {
		params.Set("langRestrict", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["libraryRestrict"]; ok {
		params.Set("libraryRestrict", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["orderBy"]; ok {
		params.Set("orderBy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["partner"]; ok {
		params.Set("partner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["printType"]; ok {
		params.Set("printType", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showPreorders"]; ok {
		params.Set("showPreorders", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes")
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
	ret := new(Volumes)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Performs a book search.",
	//   "httpMethod": "GET",
	//   "id": "books.volumes.list",
	//   "parameterOrder": [
	//     "q"
	//   ],
	//   "parameters": {
	//     "download": {
	//       "description": "Restrict to volumes by download availability.",
	//       "enum": [
	//         "epub"
	//       ],
	//       "enumDescriptions": [
	//         "All volumes with epub."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "filter": {
	//       "description": "Filter search results.",
	//       "enum": [
	//         "ebooks",
	//         "free-ebooks",
	//         "full",
	//         "paid-ebooks",
	//         "partial"
	//       ],
	//       "enumDescriptions": [
	//         "All Google eBooks.",
	//         "Google eBook with full volume text viewability.",
	//         "Public can view entire volume text.",
	//         "Google eBook with a price.",
	//         "Public able to see parts of text."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "langRestrict": {
	//       "description": "Restrict results to books with this language code.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "libraryRestrict": {
	//       "description": "Restrict search to this user's library.",
	//       "enum": [
	//         "my-library",
	//         "no-restrict"
	//       ],
	//       "enumDescriptions": [
	//         "Restrict to the user's library, any shelf.",
	//         "Do not restrict based on user's library."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "40",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "description": "Sort search results.",
	//       "enum": [
	//         "newest",
	//         "relevance"
	//       ],
	//       "enumDescriptions": [
	//         "Most recently published.",
	//         "Relevance to search terms."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "partner": {
	//       "description": "Restrict and brand results for partner ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "printType": {
	//       "description": "Restrict to books or magazines.",
	//       "enum": [
	//         "all",
	//         "books",
	//         "magazines"
	//       ],
	//       "enumDescriptions": [
	//         "All volume content types.",
	//         "Just books.",
	//         "Just magazines."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Restrict information returned to a set of selected fields.",
	//       "enum": [
	//         "full",
	//         "lite"
	//       ],
	//       "enumDescriptions": [
	//         "Includes all volume data.",
	//         "Includes a subset of fields in volumeInfo and accessInfo."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "Full-text search query string.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "showPreorders": {
	//       "description": "Set to true to show books available for preorder. Defaults to false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first result to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "volumes",
	//   "response": {
	//     "$ref": "Volumes"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.volumes.associated.list":

type VolumesAssociatedListCall struct {
	s        *Service
	volumeId string
	opt_     map[string]interface{}
}

// List: Return a list of associated books.
func (r *VolumesAssociatedService) List(volumeId string) *VolumesAssociatedListCall {
	c := &VolumesAssociatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.volumeId = volumeId
	return c
}

// Association sets the optional parameter "association": Association
// type.
func (c *VolumesAssociatedListCall) Association(association string) *VolumesAssociatedListCall {
	c.opt_["association"] = association
	return c
}

// Locale sets the optional parameter "locale": ISO-639-1 language and
// ISO-3166-1 country code. Ex: 'en_US'. Used for generating
// recommendations.
func (c *VolumesAssociatedListCall) Locale(locale string) *VolumesAssociatedListCall {
	c.opt_["locale"] = locale
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *VolumesAssociatedListCall) Source(source string) *VolumesAssociatedListCall {
	c.opt_["source"] = source
	return c
}

func (c *VolumesAssociatedListCall) Do() (*Volumes, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["association"]; ok {
		params.Set("association", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/{volumeId}/associated")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{volumeId}", url.QueryEscape(c.volumeId), 1)
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
	ret := new(Volumes)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Return a list of associated books.",
	//   "httpMethod": "GET",
	//   "id": "books.volumes.associated.list",
	//   "parameterOrder": [
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "association": {
	//       "description": "Association type.",
	//       "enum": [
	//         "end-of-sample",
	//         "end-of-volume"
	//       ],
	//       "enumDescriptions": [
	//         "Recommendations for display end-of-sample.",
	//         "Recommendations for display end-of-volume."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "ISO-639-1 language and ISO-3166-1 country code. Ex: 'en_US'. Used for generating recommendations.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "ID of the source volume.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/{volumeId}/associated",
	//   "response": {
	//     "$ref": "Volumes"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.volumes.mybooks.list":

type VolumesMybooksListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Return a list of books in My Library.
func (r *VolumesMybooksService) List() *VolumesMybooksListCall {
	c := &VolumesMybooksListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// AcquireMethod sets the optional parameter "acquireMethod": How the
// book was aquired
func (c *VolumesMybooksListCall) AcquireMethod(acquireMethod string) *VolumesMybooksListCall {
	c.opt_["acquireMethod"] = acquireMethod
	return c
}

// Locale sets the optional parameter "locale": ISO-639-1 language and
// ISO-3166-1 country code. Ex:'en_US'. Used for generating
// recommendations.
func (c *VolumesMybooksListCall) Locale(locale string) *VolumesMybooksListCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *VolumesMybooksListCall) MaxResults(maxResults int64) *VolumesMybooksListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// ProcessingState sets the optional parameter "processingState": The
// processing state of the user uploaded volumes to be returned.
// Applicable only if the UPLOADED is specified in the acquireMethod.
func (c *VolumesMybooksListCall) ProcessingState(processingState string) *VolumesMybooksListCall {
	c.opt_["processingState"] = processingState
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *VolumesMybooksListCall) Source(source string) *VolumesMybooksListCall {
	c.opt_["source"] = source
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first result to return (starts at 0)
func (c *VolumesMybooksListCall) StartIndex(startIndex int64) *VolumesMybooksListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *VolumesMybooksListCall) Do() (*Volumes, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["acquireMethod"]; ok {
		params.Set("acquireMethod", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["processingState"]; ok {
		params.Set("processingState", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/mybooks")
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
	ret := new(Volumes)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Return a list of books in My Library.",
	//   "httpMethod": "GET",
	//   "id": "books.volumes.mybooks.list",
	//   "parameters": {
	//     "acquireMethod": {
	//       "description": "How the book was aquired",
	//       "enum": [
	//         "PREORDERED",
	//         "PREVIOUSLY_RENTED",
	//         "PUBLIC_DOMAIN",
	//         "PURCHASED",
	//         "RENTED",
	//         "SAMPLE",
	//         "UPLOADED"
	//       ],
	//       "enumDescriptions": [
	//         "Preordered books (not yet available)",
	//         "User-rented books past their expiration time",
	//         "Public domain books",
	//         "Purchased books",
	//         "User-rented books",
	//         "Sample books",
	//         "User uploaded books"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "ISO-639-1 language and ISO-3166-1 country code. Ex:'en_US'. Used for generating recommendations.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "processingState": {
	//       "description": "The processing state of the user uploaded volumes to be returned. Applicable only if the UPLOADED is specified in the acquireMethod.",
	//       "enum": [
	//         "COMPLETED_FAILED",
	//         "COMPLETED_SUCCESS",
	//         "RUNNING"
	//       ],
	//       "enumDescriptions": [
	//         "The volume processing hase failed.",
	//         "The volume processing was completed.",
	//         "The volume processing is not completed."
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first result to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "volumes/mybooks",
	//   "response": {
	//     "$ref": "Volumes"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.volumes.recommended.list":

type VolumesRecommendedListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Return a list of recommended books for the current user.
func (r *VolumesRecommendedService) List() *VolumesRecommendedListCall {
	c := &VolumesRecommendedListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Locale sets the optional parameter "locale": ISO-639-1 language and
// ISO-3166-1 country code. Ex: 'en_US'. Used for generating
// recommendations.
func (c *VolumesRecommendedListCall) Locale(locale string) *VolumesRecommendedListCall {
	c.opt_["locale"] = locale
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *VolumesRecommendedListCall) Source(source string) *VolumesRecommendedListCall {
	c.opt_["source"] = source
	return c
}

func (c *VolumesRecommendedListCall) Do() (*Volumes, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/recommended")
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
	ret := new(Volumes)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Return a list of recommended books for the current user.",
	//   "httpMethod": "GET",
	//   "id": "books.volumes.recommended.list",
	//   "parameters": {
	//     "locale": {
	//       "description": "ISO-639-1 language and ISO-3166-1 country code. Ex: 'en_US'. Used for generating recommendations.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/recommended",
	//   "response": {
	//     "$ref": "Volumes"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.volumes.recommended.rate":

type VolumesRecommendedRateCall struct {
	s        *Service
	rating   string
	volumeId string
	opt_     map[string]interface{}
}

// Rate: Rate a recommended book for the current user.
func (r *VolumesRecommendedService) Rate(rating string, volumeId string) *VolumesRecommendedRateCall {
	c := &VolumesRecommendedRateCall{s: r.s, opt_: make(map[string]interface{})}
	c.rating = rating
	c.volumeId = volumeId
	return c
}

// Locale sets the optional parameter "locale": ISO-639-1 language and
// ISO-3166-1 country code. Ex: 'en_US'. Used for generating
// recommendations.
func (c *VolumesRecommendedRateCall) Locale(locale string) *VolumesRecommendedRateCall {
	c.opt_["locale"] = locale
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *VolumesRecommendedRateCall) Source(source string) *VolumesRecommendedRateCall {
	c.opt_["source"] = source
	return c
}

func (c *VolumesRecommendedRateCall) Do() (*BooksVolumesRecommendedRateResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("rating", fmt.Sprintf("%v", c.rating))
	params.Set("volumeId", fmt.Sprintf("%v", c.volumeId))
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/recommended/rate")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(BooksVolumesRecommendedRateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Rate a recommended book for the current user.",
	//   "httpMethod": "POST",
	//   "id": "books.volumes.recommended.rate",
	//   "parameterOrder": [
	//     "rating",
	//     "volumeId"
	//   ],
	//   "parameters": {
	//     "locale": {
	//       "description": "ISO-639-1 language and ISO-3166-1 country code. Ex: 'en_US'. Used for generating recommendations.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "rating": {
	//       "description": "Rating to be given to the volume.",
	//       "enum": [
	//         "HAVE_IT",
	//         "NOT_INTERESTED"
	//       ],
	//       "enumDescriptions": [
	//         "Rating indicating a dismissal due to ownership.",
	//         "Rating indicating a negative dismissal of a volume."
	//       ],
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "volumeId": {
	//       "description": "ID of the source volume.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/recommended/rate",
	//   "response": {
	//     "$ref": "BooksVolumesRecommendedRateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}

// method id "books.volumes.useruploaded.list":

type VolumesUseruploadedListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Return a list of books uploaded by the current user.
func (r *VolumesUseruploadedService) List() *VolumesUseruploadedListCall {
	c := &VolumesUseruploadedListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Locale sets the optional parameter "locale": ISO-639-1 language and
// ISO-3166-1 country code. Ex: 'en_US'. Used for generating
// recommendations.
func (c *VolumesUseruploadedListCall) Locale(locale string) *VolumesUseruploadedListCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return.
func (c *VolumesUseruploadedListCall) MaxResults(maxResults int64) *VolumesUseruploadedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// ProcessingState sets the optional parameter "processingState": The
// processing state of the user uploaded volumes to be returned.
func (c *VolumesUseruploadedListCall) ProcessingState(processingState string) *VolumesUseruploadedListCall {
	c.opt_["processingState"] = processingState
	return c
}

// Source sets the optional parameter "source": String to identify the
// originator of this request.
func (c *VolumesUseruploadedListCall) Source(source string) *VolumesUseruploadedListCall {
	c.opt_["source"] = source
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first result to return (starts at 0)
func (c *VolumesUseruploadedListCall) StartIndex(startIndex int64) *VolumesUseruploadedListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

// VolumeId sets the optional parameter "volumeId": The ids of the
// volumes to be returned. If not specified all that match the
// processingState are returned.
func (c *VolumesUseruploadedListCall) VolumeId(volumeId string) *VolumesUseruploadedListCall {
	c.opt_["volumeId"] = volumeId
	return c
}

func (c *VolumesUseruploadedListCall) Do() (*Volumes, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["processingState"]; ok {
		params.Set("processingState", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["source"]; ok {
		params.Set("source", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["volumeId"]; ok {
		params.Set("volumeId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "volumes/useruploaded")
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
	ret := new(Volumes)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Return a list of books uploaded by the current user.",
	//   "httpMethod": "GET",
	//   "id": "books.volumes.useruploaded.list",
	//   "parameters": {
	//     "locale": {
	//       "description": "ISO-639-1 language and ISO-3166-1 country code. Ex: 'en_US'. Used for generating recommendations.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "40",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "processingState": {
	//       "description": "The processing state of the user uploaded volumes to be returned.",
	//       "enum": [
	//         "COMPLETED_FAILED",
	//         "COMPLETED_SUCCESS",
	//         "RUNNING"
	//       ],
	//       "enumDescriptions": [
	//         "The volume processing hase failed.",
	//         "The volume processing was completed.",
	//         "The volume processing is not completed."
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "String to identify the originator of this request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first result to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "volumeId": {
	//       "description": "The ids of the volumes to be returned. If not specified all that match the processingState are returned.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "volumes/useruploaded",
	//   "response": {
	//     "$ref": "Volumes"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/books"
	//   ]
	// }

}
