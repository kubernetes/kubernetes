// Package vision provides access to the Google Cloud Vision API.
//
// See https://cloud.google.com/vision/
//
// Usage example:
//
//   import "google.golang.org/api/vision/v1"
//   ...
//   visionService, err := vision.New(oauthHttpClient)
package vision // import "google.golang.org/api/vision/v1"

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

const apiId = "vision:v1"
const apiName = "vision"
const apiVersion = "v1"
const basePath = "https://vision.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// Apply machine learning models to understand and label images
	CloudVisionScope = "https://www.googleapis.com/auth/cloud-vision"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Images = NewImagesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Images *ImagesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewImagesService(s *Service) *ImagesService {
	rs := &ImagesService{s: s}
	return rs
}

type ImagesService struct {
	s *Service
}

// AnnotateImageRequest: Request for performing Google Cloud Vision API
// tasks over a user-provided
// image, with user-requested features.
type AnnotateImageRequest struct {
	// Features: Requested features.
	Features []*Feature `json:"features,omitempty"`

	// Image: The image to be processed.
	Image *Image `json:"image,omitempty"`

	// ImageContext: Additional context that may accompany the image.
	ImageContext *ImageContext `json:"imageContext,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Features") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Features") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AnnotateImageRequest) MarshalJSON() ([]byte, error) {
	type noMethod AnnotateImageRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AnnotateImageResponse: Response to an image annotation request.
type AnnotateImageResponse struct {
	// CropHintsAnnotation: If present, crop hints have completed
	// successfully.
	CropHintsAnnotation *CropHintsAnnotation `json:"cropHintsAnnotation,omitempty"`

	// Error: If set, represents the error message for the operation.
	// Note that filled-in image annotations are guaranteed to be
	// correct, even when `error` is set.
	Error *Status `json:"error,omitempty"`

	// FaceAnnotations: If present, face detection has completed
	// successfully.
	FaceAnnotations []*FaceAnnotation `json:"faceAnnotations,omitempty"`

	// FullTextAnnotation: If present, text (OCR) detection or document
	// (OCR) text detection has
	// completed successfully.
	// This annotation provides the structural hierarchy for the OCR
	// detected
	// text.
	FullTextAnnotation *TextAnnotation `json:"fullTextAnnotation,omitempty"`

	// ImagePropertiesAnnotation: If present, image properties were
	// extracted successfully.
	ImagePropertiesAnnotation *ImageProperties `json:"imagePropertiesAnnotation,omitempty"`

	// LabelAnnotations: If present, label detection has completed
	// successfully.
	LabelAnnotations []*EntityAnnotation `json:"labelAnnotations,omitempty"`

	// LandmarkAnnotations: If present, landmark detection has completed
	// successfully.
	LandmarkAnnotations []*EntityAnnotation `json:"landmarkAnnotations,omitempty"`

	// LogoAnnotations: If present, logo detection has completed
	// successfully.
	LogoAnnotations []*EntityAnnotation `json:"logoAnnotations,omitempty"`

	// SafeSearchAnnotation: If present, safe-search annotation has
	// completed successfully.
	SafeSearchAnnotation *SafeSearchAnnotation `json:"safeSearchAnnotation,omitempty"`

	// TextAnnotations: If present, text (OCR) detection has completed
	// successfully.
	TextAnnotations []*EntityAnnotation `json:"textAnnotations,omitempty"`

	// WebDetection: If present, web detection has completed successfully.
	WebDetection *WebDetection `json:"webDetection,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CropHintsAnnotation")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CropHintsAnnotation") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AnnotateImageResponse) MarshalJSON() ([]byte, error) {
	type noMethod AnnotateImageResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchAnnotateImagesRequest: Multiple image annotation requests are
// batched into a single service call.
type BatchAnnotateImagesRequest struct {
	// Requests: Individual image annotation requests for this batch.
	Requests []*AnnotateImageRequest `json:"requests,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Requests") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Requests") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchAnnotateImagesRequest) MarshalJSON() ([]byte, error) {
	type noMethod BatchAnnotateImagesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchAnnotateImagesResponse: Response to a batch image annotation
// request.
type BatchAnnotateImagesResponse struct {
	// Responses: Individual responses to image annotation requests within
	// the batch.
	Responses []*AnnotateImageResponse `json:"responses,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Responses") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Responses") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchAnnotateImagesResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchAnnotateImagesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Block: Logical element on the page.
type Block struct {
	// BlockType: Detected block type (text, image etc) for this block.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown block type.
	//   "TEXT" - Regular text block.
	//   "TABLE" - Table block.
	//   "PICTURE" - Image block.
	//   "RULER" - Horizontal/vertical line box.
	//   "BARCODE" - Barcode block.
	BlockType string `json:"blockType,omitempty"`

	// BoundingBox: The bounding box for the block.
	// The vertices are in the order of top-left, top-right,
	// bottom-right,
	// bottom-left. When a rotation of the bounding box is detected the
	// rotation
	// is represented as around the top-left corner as defined when the text
	// is
	// read in the 'natural' orientation.
	// For example:
	//   * when the text is horizontal it might look like:
	//      0----1
	//      |    |
	//      3----2
	//   * when it's rotated 180 degrees around the top-left corner it
	// becomes:
	//      2----3
	//      |    |
	//      1----0
	//   and the vertice order will still be (0, 1, 2, 3).
	BoundingBox *BoundingPoly `json:"boundingBox,omitempty"`

	// Paragraphs: List of paragraphs in this block (if this blocks is of
	// type text).
	Paragraphs []*Paragraph `json:"paragraphs,omitempty"`

	// Property: Additional information detected for the block.
	Property *TextProperty `json:"property,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BlockType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BlockType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Block) MarshalJSON() ([]byte, error) {
	type noMethod Block
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BoundingPoly: A bounding polygon for the detected image annotation.
type BoundingPoly struct {
	// Vertices: The bounding polygon vertices.
	Vertices []*Vertex `json:"vertices,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Vertices") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Vertices") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BoundingPoly) MarshalJSON() ([]byte, error) {
	type noMethod BoundingPoly
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Color: Represents a color in the RGBA color space. This
// representation is designed
// for simplicity of conversion to/from color representations in
// various
// languages over compactness; for example, the fields of this
// representation
// can be trivially provided to the constructor of "java.awt.Color" in
// Java; it
// can also be trivially provided to UIColor's
// "+colorWithRed:green:blue:alpha"
// method in iOS; and, with just a little work, it can be easily
// formatted into
// a CSS "rgba()" string in JavaScript, as well. Here are some
// examples:
//
// Example (Java):
//
//      import com.google.type.Color;
//
//      // ...
//      public static java.awt.Color fromProto(Color protocolor) {
//        float alpha = protocolor.hasAlpha()
//            ? protocolor.getAlpha().getValue()
//            : 1.0;
//
//        return new java.awt.Color(
//            protocolor.getRed(),
//            protocolor.getGreen(),
//            protocolor.getBlue(),
//            alpha);
//      }
//
//      public static Color toProto(java.awt.Color color) {
//        float red = (float) color.getRed();
//        float green = (float) color.getGreen();
//        float blue = (float) color.getBlue();
//        float denominator = 255.0;
//        Color.Builder resultBuilder =
//            Color
//                .newBuilder()
//                .setRed(red / denominator)
//                .setGreen(green / denominator)
//                .setBlue(blue / denominator);
//        int alpha = color.getAlpha();
//        if (alpha != 255) {
//          result.setAlpha(
//              FloatValue
//                  .newBuilder()
//                  .setValue(((float) alpha) / denominator)
//                  .build());
//        }
//        return resultBuilder.build();
//      }
//      // ...
//
// Example (iOS / Obj-C):
//
//      // ...
//      static UIColor* fromProto(Color* protocolor) {
//         float red = [protocolor red];
//         float green = [protocolor green];
//         float blue = [protocolor blue];
//         FloatValue* alpha_wrapper = [protocolor alpha];
//         float alpha = 1.0;
//         if (alpha_wrapper != nil) {
//           alpha = [alpha_wrapper value];
//         }
//         return [UIColor colorWithRed:red green:green blue:blue
// alpha:alpha];
//      }
//
//      static Color* toProto(UIColor* color) {
//          CGFloat red, green, blue, alpha;
//          if (![color getRed:&red green:&green blue:&blue
// alpha:&alpha]) {
//            return nil;
//          }
//          Color* result = [Color alloc] init];
//          [result setRed:red];
//          [result setGreen:green];
//          [result setBlue:blue];
//          if (alpha <= 0.9999) {
//            [result setAlpha:floatWrapperWithValue(alpha)];
//          }
//          [result autorelease];
//          return result;
//     }
//     // ...
//
//  Example (JavaScript):
//
//     // ...
//
//     var protoToCssColor = function(rgb_color) {
//        var redFrac = rgb_color.red || 0.0;
//        var greenFrac = rgb_color.green || 0.0;
//        var blueFrac = rgb_color.blue || 0.0;
//        var red = Math.floor(redFrac * 255);
//        var green = Math.floor(greenFrac * 255);
//        var blue = Math.floor(blueFrac * 255);
//
//        if (!('alpha' in rgb_color)) {
//           return rgbToCssColor_(red, green, blue);
//        }
//
//        var alphaFrac = rgb_color.alpha.value || 0.0;
//        var rgbParams = [red, green, blue].join(',');
//        return ['rgba(', rgbParams, ',', alphaFrac, ')'].join('');
//     };
//
//     var rgbToCssColor_ = function(red, green, blue) {
//       var rgbNumber = new Number((red << 16) | (green << 8) | blue);
//       var hexString = rgbNumber.toString(16);
//       var missingZeros = 6 - hexString.length;
//       var resultBuilder = ['#'];
//       for (var i = 0; i < missingZeros; i++) {
//          resultBuilder.push('0');
//       }
//       resultBuilder.push(hexString);
//       return resultBuilder.join('');
//     };
//
//     // ...
type Color struct {
	// Alpha: The fraction of this color that should be applied to the
	// pixel. That is,
	// the final pixel color is defined by the equation:
	//
	//   pixel color = alpha * (this color) + (1.0 - alpha) * (background
	// color)
	//
	// This means that a value of 1.0 corresponds to a solid color,
	// whereas
	// a value of 0.0 corresponds to a completely transparent color.
	// This
	// uses a wrapper message rather than a simple float scalar so that it
	// is
	// possible to distinguish between a default value and the value being
	// unset.
	// If omitted, this color object is to be rendered as a solid color
	// (as if the alpha value had been explicitly given with a value of
	// 1.0).
	Alpha float64 `json:"alpha,omitempty"`

	// Blue: The amount of blue in the color as a value in the interval [0,
	// 1].
	Blue float64 `json:"blue,omitempty"`

	// Green: The amount of green in the color as a value in the interval
	// [0, 1].
	Green float64 `json:"green,omitempty"`

	// Red: The amount of red in the color as a value in the interval [0,
	// 1].
	Red float64 `json:"red,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Alpha") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Alpha") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Color) MarshalJSON() ([]byte, error) {
	type noMethod Color
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Color) UnmarshalJSON(data []byte) error {
	type noMethod Color
	var s1 struct {
		Alpha gensupport.JSONFloat64 `json:"alpha"`
		Blue  gensupport.JSONFloat64 `json:"blue"`
		Green gensupport.JSONFloat64 `json:"green"`
		Red   gensupport.JSONFloat64 `json:"red"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Alpha = float64(s1.Alpha)
	s.Blue = float64(s1.Blue)
	s.Green = float64(s1.Green)
	s.Red = float64(s1.Red)
	return nil
}

// ColorInfo: Color information consists of RGB channels, score, and the
// fraction of
// the image that the color occupies in the image.
type ColorInfo struct {
	// Color: RGB components of the color.
	Color *Color `json:"color,omitempty"`

	// PixelFraction: The fraction of pixels the color occupies in the
	// image.
	// Value in range [0, 1].
	PixelFraction float64 `json:"pixelFraction,omitempty"`

	// Score: Image-specific score for this color. Value in range [0, 1].
	Score float64 `json:"score,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Color") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Color") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ColorInfo) MarshalJSON() ([]byte, error) {
	type noMethod ColorInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ColorInfo) UnmarshalJSON(data []byte) error {
	type noMethod ColorInfo
	var s1 struct {
		PixelFraction gensupport.JSONFloat64 `json:"pixelFraction"`
		Score         gensupport.JSONFloat64 `json:"score"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.PixelFraction = float64(s1.PixelFraction)
	s.Score = float64(s1.Score)
	return nil
}

// CropHint: Single crop hint that is used to generate a new crop when
// serving an image.
type CropHint struct {
	// BoundingPoly: The bounding polygon for the crop region. The
	// coordinates of the bounding
	// box are in the original image's scale, as returned in `ImageParams`.
	BoundingPoly *BoundingPoly `json:"boundingPoly,omitempty"`

	// Confidence: Confidence of this being a salient region.  Range [0, 1].
	Confidence float64 `json:"confidence,omitempty"`

	// ImportanceFraction: Fraction of importance of this salient region
	// with respect to the original
	// image.
	ImportanceFraction float64 `json:"importanceFraction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoundingPoly") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoundingPoly") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CropHint) MarshalJSON() ([]byte, error) {
	type noMethod CropHint
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *CropHint) UnmarshalJSON(data []byte) error {
	type noMethod CropHint
	var s1 struct {
		Confidence         gensupport.JSONFloat64 `json:"confidence"`
		ImportanceFraction gensupport.JSONFloat64 `json:"importanceFraction"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Confidence = float64(s1.Confidence)
	s.ImportanceFraction = float64(s1.ImportanceFraction)
	return nil
}

// CropHintsAnnotation: Set of crop hints that are used to generate new
// crops when serving images.
type CropHintsAnnotation struct {
	// CropHints: Crop hint results.
	CropHints []*CropHint `json:"cropHints,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CropHints") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CropHints") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CropHintsAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod CropHintsAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CropHintsParams: Parameters for crop hints annotation request.
type CropHintsParams struct {
	// AspectRatios: Aspect ratios in floats, representing the ratio of the
	// width to the height
	// of the image. For example, if the desired aspect ratio is 4/3,
	// the
	// corresponding float value should be 1.33333.  If not specified,
	// the
	// best possible crop is returned. The number of provided aspect ratios
	// is
	// limited to a maximum of 16; any aspect ratios provided after the 16th
	// are
	// ignored.
	AspectRatios []float64 `json:"aspectRatios,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AspectRatios") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AspectRatios") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CropHintsParams) MarshalJSON() ([]byte, error) {
	type noMethod CropHintsParams
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DetectedBreak: Detected start or end of a structural component.
type DetectedBreak struct {
	// IsPrefix: True if break prepends the element.
	IsPrefix bool `json:"isPrefix,omitempty"`

	// Type: Detected break type.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown break label type.
	//   "SPACE" - Regular space.
	//   "SURE_SPACE" - Sure space (very wide).
	//   "EOL_SURE_SPACE" - Line-wrapping break.
	//   "HYPHEN" - End-line hyphen that is not present in text; does not
	// co-occur with
	// `SPACE`, `LEADER_SPACE`, or `LINE_BREAK`.
	//   "LINE_BREAK" - Line break that ends a paragraph.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IsPrefix") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "IsPrefix") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DetectedBreak) MarshalJSON() ([]byte, error) {
	type noMethod DetectedBreak
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DetectedLanguage: Detected language for a structural component.
type DetectedLanguage struct {
	// Confidence: Confidence of detected language. Range [0, 1].
	Confidence float64 `json:"confidence,omitempty"`

	// LanguageCode: The BCP-47 language code, such as "en-US" or "sr-Latn".
	// For more
	// information,
	// see
	// http://www.unicode.org/reports/tr35/#Unicode_locale_identifier.
	LanguageCode string `json:"languageCode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Confidence") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Confidence") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DetectedLanguage) MarshalJSON() ([]byte, error) {
	type noMethod DetectedLanguage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *DetectedLanguage) UnmarshalJSON(data []byte) error {
	type noMethod DetectedLanguage
	var s1 struct {
		Confidence gensupport.JSONFloat64 `json:"confidence"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Confidence = float64(s1.Confidence)
	return nil
}

// DominantColorsAnnotation: Set of dominant colors and their
// corresponding scores.
type DominantColorsAnnotation struct {
	// Colors: RGB color values with their score and pixel fraction.
	Colors []*ColorInfo `json:"colors,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Colors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Colors") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DominantColorsAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod DominantColorsAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EntityAnnotation: Set of detected entity features.
type EntityAnnotation struct {
	// BoundingPoly: Image region to which this entity belongs. Not
	// produced
	// for `LABEL_DETECTION` features.
	BoundingPoly *BoundingPoly `json:"boundingPoly,omitempty"`

	// Confidence: The accuracy of the entity detection in an image.
	// For example, for an image in which the "Eiffel Tower" entity is
	// detected,
	// this field represents the confidence that there is a tower in the
	// query
	// image. Range [0, 1].
	Confidence float64 `json:"confidence,omitempty"`

	// Description: Entity textual description, expressed in its `locale`
	// language.
	Description string `json:"description,omitempty"`

	// Locale: The language code for the locale in which the entity
	// textual
	// `description` is expressed.
	Locale string `json:"locale,omitempty"`

	// Locations: The location information for the detected entity.
	// Multiple
	// `LocationInfo` elements can be present because one location
	// may
	// indicate the location of the scene in the image, and another
	// location
	// may indicate the location of the place where the image was
	// taken.
	// Location information is usually present for landmarks.
	Locations []*LocationInfo `json:"locations,omitempty"`

	// Mid: Opaque entity ID. Some IDs may be available in
	// [Google Knowledge Graph Search
	// API](https://developers.google.com/knowledge-graph/).
	Mid string `json:"mid,omitempty"`

	// Properties: Some entities may have optional user-supplied `Property`
	// (name/value)
	// fields, such a score or string that qualifies the entity.
	Properties []*Property `json:"properties,omitempty"`

	// Score: Overall score of the result. Range [0, 1].
	Score float64 `json:"score,omitempty"`

	// Topicality: The relevancy of the ICA (Image Content Annotation) label
	// to the
	// image. For example, the relevancy of "tower" is likely higher to an
	// image
	// containing the detected "Eiffel Tower" than to an image containing
	// a
	// detected distant towering building, even though the confidence
	// that
	// there is a tower in each image may be the same. Range [0, 1].
	Topicality float64 `json:"topicality,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoundingPoly") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoundingPoly") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EntityAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod EntityAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *EntityAnnotation) UnmarshalJSON(data []byte) error {
	type noMethod EntityAnnotation
	var s1 struct {
		Confidence gensupport.JSONFloat64 `json:"confidence"`
		Score      gensupport.JSONFloat64 `json:"score"`
		Topicality gensupport.JSONFloat64 `json:"topicality"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Confidence = float64(s1.Confidence)
	s.Score = float64(s1.Score)
	s.Topicality = float64(s1.Topicality)
	return nil
}

// FaceAnnotation: A face annotation object contains the results of face
// detection.
type FaceAnnotation struct {
	// AngerLikelihood: Anger likelihood.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	AngerLikelihood string `json:"angerLikelihood,omitempty"`

	// BlurredLikelihood: Blurred likelihood.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	BlurredLikelihood string `json:"blurredLikelihood,omitempty"`

	// BoundingPoly: The bounding polygon around the face. The coordinates
	// of the bounding box
	// are in the original image's scale, as returned in `ImageParams`.
	// The bounding box is computed to "frame" the face in accordance with
	// human
	// expectations. It is based on the landmarker results.
	// Note that one or more x and/or y coordinates may not be generated in
	// the
	// `BoundingPoly` (the polygon will be unbounded) if only a partial
	// face
	// appears in the image to be annotated.
	BoundingPoly *BoundingPoly `json:"boundingPoly,omitempty"`

	// DetectionConfidence: Detection confidence. Range [0, 1].
	DetectionConfidence float64 `json:"detectionConfidence,omitempty"`

	// FdBoundingPoly: The `fd_bounding_poly` bounding polygon is tighter
	// than the
	// `boundingPoly`, and encloses only the skin part of the face.
	// Typically, it
	// is used to eliminate the face from any image analysis that detects
	// the
	// "amount of skin" visible in an image. It is not based on
	// the
	// landmarker results, only on the initial face detection, hence
	// the <code>fd</code> (face detection) prefix.
	FdBoundingPoly *BoundingPoly `json:"fdBoundingPoly,omitempty"`

	// HeadwearLikelihood: Headwear likelihood.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	HeadwearLikelihood string `json:"headwearLikelihood,omitempty"`

	// JoyLikelihood: Joy likelihood.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	JoyLikelihood string `json:"joyLikelihood,omitempty"`

	// LandmarkingConfidence: Face landmarking confidence. Range [0, 1].
	LandmarkingConfidence float64 `json:"landmarkingConfidence,omitempty"`

	// Landmarks: Detected face landmarks.
	Landmarks []*Landmark `json:"landmarks,omitempty"`

	// PanAngle: Yaw angle, which indicates the leftward/rightward angle
	// that the face is
	// pointing relative to the vertical plane perpendicular to the image.
	// Range
	// [-180,180].
	PanAngle float64 `json:"panAngle,omitempty"`

	// RollAngle: Roll angle, which indicates the amount of
	// clockwise/anti-clockwise rotation
	// of the face relative to the image vertical about the axis
	// perpendicular to
	// the face. Range [-180,180].
	RollAngle float64 `json:"rollAngle,omitempty"`

	// SorrowLikelihood: Sorrow likelihood.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	SorrowLikelihood string `json:"sorrowLikelihood,omitempty"`

	// SurpriseLikelihood: Surprise likelihood.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	SurpriseLikelihood string `json:"surpriseLikelihood,omitempty"`

	// TiltAngle: Pitch angle, which indicates the upwards/downwards angle
	// that the face is
	// pointing relative to the image's horizontal plane. Range [-180,180].
	TiltAngle float64 `json:"tiltAngle,omitempty"`

	// UnderExposedLikelihood: Under-exposed likelihood.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	UnderExposedLikelihood string `json:"underExposedLikelihood,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AngerLikelihood") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AngerLikelihood") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FaceAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod FaceAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *FaceAnnotation) UnmarshalJSON(data []byte) error {
	type noMethod FaceAnnotation
	var s1 struct {
		DetectionConfidence   gensupport.JSONFloat64 `json:"detectionConfidence"`
		LandmarkingConfidence gensupport.JSONFloat64 `json:"landmarkingConfidence"`
		PanAngle              gensupport.JSONFloat64 `json:"panAngle"`
		RollAngle             gensupport.JSONFloat64 `json:"rollAngle"`
		TiltAngle             gensupport.JSONFloat64 `json:"tiltAngle"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.DetectionConfidence = float64(s1.DetectionConfidence)
	s.LandmarkingConfidence = float64(s1.LandmarkingConfidence)
	s.PanAngle = float64(s1.PanAngle)
	s.RollAngle = float64(s1.RollAngle)
	s.TiltAngle = float64(s1.TiltAngle)
	return nil
}

// Feature: Users describe the type of Google Cloud Vision API tasks to
// perform over
// images by using *Feature*s. Each Feature indicates a type of
// image
// detection task to perform. Features encode the Cloud Vision
// API
// vertical to operate on and the number of top-scoring results to
// return.
type Feature struct {
	// MaxResults: Maximum number of results of this type.
	MaxResults int64 `json:"maxResults,omitempty"`

	// Type: The feature type.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED" - Unspecified feature type.
	//   "FACE_DETECTION" - Run face detection.
	//   "LANDMARK_DETECTION" - Run landmark detection.
	//   "LOGO_DETECTION" - Run logo detection.
	//   "LABEL_DETECTION" - Run label detection.
	//   "TEXT_DETECTION" - Run OCR.
	//   "DOCUMENT_TEXT_DETECTION" - Run dense text document OCR. Takes
	// precedence when both
	// DOCUMENT_TEXT_DETECTION and TEXT_DETECTION are present.
	//   "SAFE_SEARCH_DETECTION" - Run computer vision models to compute
	// image safe-search properties.
	//   "IMAGE_PROPERTIES" - Compute a set of image properties, such as the
	// image's dominant colors.
	//   "CROP_HINTS" - Run crop hints.
	//   "WEB_DETECTION" - Run web detection.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxResults") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxResults") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Feature) MarshalJSON() ([]byte, error) {
	type noMethod Feature
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Image: Client image to perform Google Cloud Vision API tasks over.
type Image struct {
	// Content: Image content, represented as a stream of bytes.
	// Note: as with all `bytes` fields, protobuffers use a pure
	// binary
	// representation, whereas JSON representations use base64.
	Content string `json:"content,omitempty"`

	// Source: Google Cloud Storage image location. If both `content` and
	// `source`
	// are provided for an image, `content` takes precedence and is
	// used to perform the image annotation request.
	Source *ImageSource `json:"source,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Content") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Content") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Image) MarshalJSON() ([]byte, error) {
	type noMethod Image
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImageContext: Image context and/or feature-specific parameters.
type ImageContext struct {
	// CropHintsParams: Parameters for crop hints annotation request.
	CropHintsParams *CropHintsParams `json:"cropHintsParams,omitempty"`

	// LanguageHints: List of languages to use for TEXT_DETECTION. In most
	// cases, an empty value
	// yields the best results since it enables automatic language
	// detection. For
	// languages based on the Latin alphabet, setting `language_hints` is
	// not
	// needed. In rare cases, when the language of the text in the image is
	// known,
	// setting a hint will help get better results (although it will be
	// a
	// significant hindrance if the hint is wrong). Text detection returns
	// an
	// error if one or more of the specified languages is not one of
	// the
	// [supported languages](/vision/docs/languages).
	LanguageHints []string `json:"languageHints,omitempty"`

	// LatLongRect: lat/long rectangle that specifies the location of the
	// image.
	LatLongRect *LatLongRect `json:"latLongRect,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CropHintsParams") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CropHintsParams") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ImageContext) MarshalJSON() ([]byte, error) {
	type noMethod ImageContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImageProperties: Stores image properties, such as dominant colors.
type ImageProperties struct {
	// DominantColors: If present, dominant colors completed successfully.
	DominantColors *DominantColorsAnnotation `json:"dominantColors,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DominantColors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DominantColors") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ImageProperties) MarshalJSON() ([]byte, error) {
	type noMethod ImageProperties
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImageSource: External image source (Google Cloud Storage image
// location).
type ImageSource struct {
	// GcsImageUri: NOTE: For new code `image_uri` below is
	// preferred.
	// Google Cloud Storage image URI, which must be in the following
	// form:
	// `gs://bucket_name/object_name` (for details, see
	// [Google Cloud Storage
	// Request
	// URIs](https://cloud.google.com/storage/docs/reference-uris)).
	//
	// NOTE: Cloud Storage object versioning is not supported.
	GcsImageUri string `json:"gcsImageUri,omitempty"`

	// ImageUri: Image URI which supports:
	// 1) Google Cloud Storage image URI, which must be in the following
	// form:
	// `gs://bucket_name/object_name` (for details, see
	// [Google Cloud Storage
	// Request
	// URIs](https://cloud.google.com/storage/docs/reference-uris)).
	//
	// NOTE: Cloud Storage object versioning is not supported.
	// 2) Publicly accessible image HTTP/HTTPS URL.
	// This is preferred over the legacy `gcs_image_uri` above. When
	// both
	// `gcs_image_uri` and `image_uri` are specified, `image_uri`
	// takes
	// precedence.
	ImageUri string `json:"imageUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "GcsImageUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "GcsImageUri") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ImageSource) MarshalJSON() ([]byte, error) {
	type noMethod ImageSource
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Landmark: A face-specific landmark (for example, a face
// feature).
// Landmark positions may fall outside the bounds of the image
// if the face is near one or more edges of the image.
// Therefore it is NOT guaranteed that `0 <= x < width` or
// `0 <= y < height`.
type Landmark struct {
	// Position: Face landmark position.
	Position *Position `json:"position,omitempty"`

	// Type: Face landmark type.
	//
	// Possible values:
	//   "UNKNOWN_LANDMARK" - Unknown face landmark detected. Should not be
	// filled.
	//   "LEFT_EYE" - Left eye.
	//   "RIGHT_EYE" - Right eye.
	//   "LEFT_OF_LEFT_EYEBROW" - Left of left eyebrow.
	//   "RIGHT_OF_LEFT_EYEBROW" - Right of left eyebrow.
	//   "LEFT_OF_RIGHT_EYEBROW" - Left of right eyebrow.
	//   "RIGHT_OF_RIGHT_EYEBROW" - Right of right eyebrow.
	//   "MIDPOINT_BETWEEN_EYES" - Midpoint between eyes.
	//   "NOSE_TIP" - Nose tip.
	//   "UPPER_LIP" - Upper lip.
	//   "LOWER_LIP" - Lower lip.
	//   "MOUTH_LEFT" - Mouth left.
	//   "MOUTH_RIGHT" - Mouth right.
	//   "MOUTH_CENTER" - Mouth center.
	//   "NOSE_BOTTOM_RIGHT" - Nose, bottom right.
	//   "NOSE_BOTTOM_LEFT" - Nose, bottom left.
	//   "NOSE_BOTTOM_CENTER" - Nose, bottom center.
	//   "LEFT_EYE_TOP_BOUNDARY" - Left eye, top boundary.
	//   "LEFT_EYE_RIGHT_CORNER" - Left eye, right corner.
	//   "LEFT_EYE_BOTTOM_BOUNDARY" - Left eye, bottom boundary.
	//   "LEFT_EYE_LEFT_CORNER" - Left eye, left corner.
	//   "RIGHT_EYE_TOP_BOUNDARY" - Right eye, top boundary.
	//   "RIGHT_EYE_RIGHT_CORNER" - Right eye, right corner.
	//   "RIGHT_EYE_BOTTOM_BOUNDARY" - Right eye, bottom boundary.
	//   "RIGHT_EYE_LEFT_CORNER" - Right eye, left corner.
	//   "LEFT_EYEBROW_UPPER_MIDPOINT" - Left eyebrow, upper midpoint.
	//   "RIGHT_EYEBROW_UPPER_MIDPOINT" - Right eyebrow, upper midpoint.
	//   "LEFT_EAR_TRAGION" - Left ear tragion.
	//   "RIGHT_EAR_TRAGION" - Right ear tragion.
	//   "LEFT_EYE_PUPIL" - Left eye pupil.
	//   "RIGHT_EYE_PUPIL" - Right eye pupil.
	//   "FOREHEAD_GLABELLA" - Forehead glabella.
	//   "CHIN_GNATHION" - Chin gnathion.
	//   "CHIN_LEFT_GONION" - Chin left gonion.
	//   "CHIN_RIGHT_GONION" - Chin right gonion.
	Type string `json:"type,omitempty"`

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

func (s *Landmark) MarshalJSON() ([]byte, error) {
	type noMethod Landmark
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LatLng: An object representing a latitude/longitude pair. This is
// expressed as a pair
// of doubles representing degrees latitude and degrees longitude.
// Unless
// specified otherwise, this must conform to the
// <a
// href="http://www.unoosa.org/pdf/icg/2012/template/WGS_84.pdf">WGS84
// st
// andard</a>. Values must be within normalized ranges.
//
// Example of normalization code in Python:
//
//     def NormalizeLongitude(longitude):
//       """Wraps decimal degrees longitude to [-180.0, 180.0]."""
//       q, r = divmod(longitude, 360.0)
//       if r > 180.0 or (r == 180.0 and q <= -1.0):
//         return r - 360.0
//       return r
//
//     def NormalizeLatLng(latitude, longitude):
//       """Wraps decimal degrees latitude and longitude to
//       [-90.0, 90.0] and [-180.0, 180.0], respectively."""
//       r = latitude % 360.0
//       if r <= 90.0:
//         return r, NormalizeLongitude(longitude)
//       elif r >= 270.0:
//         return r - 360, NormalizeLongitude(longitude)
//       else:
//         return 180 - r, NormalizeLongitude(longitude + 180.0)
//
//     assert 180.0 == NormalizeLongitude(180.0)
//     assert -180.0 == NormalizeLongitude(-180.0)
//     assert -179.0 == NormalizeLongitude(181.0)
//     assert (0.0, 0.0) == NormalizeLatLng(360.0, 0.0)
//     assert (0.0, 0.0) == NormalizeLatLng(-360.0, 0.0)
//     assert (85.0, 180.0) == NormalizeLatLng(95.0, 0.0)
//     assert (-85.0, -170.0) == NormalizeLatLng(-95.0, 10.0)
//     assert (90.0, 10.0) == NormalizeLatLng(90.0, 10.0)
//     assert (-90.0, -10.0) == NormalizeLatLng(-90.0, -10.0)
//     assert (0.0, -170.0) == NormalizeLatLng(-180.0, 10.0)
//     assert (0.0, -170.0) == NormalizeLatLng(180.0, 10.0)
//     assert (-90.0, 10.0) == NormalizeLatLng(270.0, 10.0)
//     assert (90.0, 10.0) == NormalizeLatLng(-270.0, 10.0)
type LatLng struct {
	// Latitude: The latitude in degrees. It must be in the range [-90.0,
	// +90.0].
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: The longitude in degrees. It must be in the range [-180.0,
	// +180.0].
	Longitude float64 `json:"longitude,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Latitude") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Latitude") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LatLng) MarshalJSON() ([]byte, error) {
	type noMethod LatLng
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *LatLng) UnmarshalJSON(data []byte) error {
	type noMethod LatLng
	var s1 struct {
		Latitude  gensupport.JSONFloat64 `json:"latitude"`
		Longitude gensupport.JSONFloat64 `json:"longitude"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Latitude = float64(s1.Latitude)
	s.Longitude = float64(s1.Longitude)
	return nil
}

// LatLongRect: Rectangle determined by min and max `LatLng` pairs.
type LatLongRect struct {
	// MaxLatLng: Max lat/long pair.
	MaxLatLng *LatLng `json:"maxLatLng,omitempty"`

	// MinLatLng: Min lat/long pair.
	MinLatLng *LatLng `json:"minLatLng,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxLatLng") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxLatLng") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LatLongRect) MarshalJSON() ([]byte, error) {
	type noMethod LatLongRect
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LocationInfo: Detected entity location information.
type LocationInfo struct {
	// LatLng: lat/long location coordinates.
	LatLng *LatLng `json:"latLng,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LatLng") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LatLng") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LocationInfo) MarshalJSON() ([]byte, error) {
	type noMethod LocationInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Page: Detected page from OCR.
type Page struct {
	// Blocks: List of blocks of text, images etc on this page.
	Blocks []*Block `json:"blocks,omitempty"`

	// Height: Page height in pixels.
	Height int64 `json:"height,omitempty"`

	// Property: Additional information detected on the page.
	Property *TextProperty `json:"property,omitempty"`

	// Width: Page width in pixels.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Blocks") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Blocks") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Page) MarshalJSON() ([]byte, error) {
	type noMethod Page
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Paragraph: Structural unit of text representing a number of words in
// certain order.
type Paragraph struct {
	// BoundingBox: The bounding box for the paragraph.
	// The vertices are in the order of top-left, top-right,
	// bottom-right,
	// bottom-left. When a rotation of the bounding box is detected the
	// rotation
	// is represented as around the top-left corner as defined when the text
	// is
	// read in the 'natural' orientation.
	// For example:
	//   * when the text is horizontal it might look like:
	//      0----1
	//      |    |
	//      3----2
	//   * when it's rotated 180 degrees around the top-left corner it
	// becomes:
	//      2----3
	//      |    |
	//      1----0
	//   and the vertice order will still be (0, 1, 2, 3).
	BoundingBox *BoundingPoly `json:"boundingBox,omitempty"`

	// Property: Additional information detected for the paragraph.
	Property *TextProperty `json:"property,omitempty"`

	// Words: List of words in this paragraph.
	Words []*Word `json:"words,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoundingBox") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoundingBox") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Paragraph) MarshalJSON() ([]byte, error) {
	type noMethod Paragraph
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Position: A 3D position in the image, used primarily for Face
// detection landmarks.
// A valid Position must have both x and y coordinates.
// The position coordinates are in the same scale as the original image.
type Position struct {
	// X: X coordinate.
	X float64 `json:"x,omitempty"`

	// Y: Y coordinate.
	Y float64 `json:"y,omitempty"`

	// Z: Z coordinate (or depth).
	Z float64 `json:"z,omitempty"`

	// ForceSendFields is a list of field names (e.g. "X") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "X") to include in API
	// requests with the JSON null value. By default, fields with empty
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

func (s *Position) UnmarshalJSON(data []byte) error {
	type noMethod Position
	var s1 struct {
		X gensupport.JSONFloat64 `json:"x"`
		Y gensupport.JSONFloat64 `json:"y"`
		Z gensupport.JSONFloat64 `json:"z"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.X = float64(s1.X)
	s.Y = float64(s1.Y)
	s.Z = float64(s1.Z)
	return nil
}

// Property: A `Property` consists of a user-supplied name/value pair.
type Property struct {
	// Name: Name of the property.
	Name string `json:"name,omitempty"`

	// Uint64Value: Value of numeric properties.
	Uint64Value uint64 `json:"uint64Value,omitempty,string"`

	// Value: Value of the property.
	Value string `json:"value,omitempty"`

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

func (s *Property) MarshalJSON() ([]byte, error) {
	type noMethod Property
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SafeSearchAnnotation: Set of features pertaining to the image,
// computed by computer vision
// methods over safe-search verticals (for example, adult, spoof,
// medical,
// violence).
type SafeSearchAnnotation struct {
	// Adult: Represents the adult content likelihood for the image.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	Adult string `json:"adult,omitempty"`

	// Medical: Likelihood that this is a medical image.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	Medical string `json:"medical,omitempty"`

	// Spoof: Spoof likelihood. The likelihood that an modification
	// was made to the image's canonical version to make it appear
	// funny or offensive.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	Spoof string `json:"spoof,omitempty"`

	// Violence: Violence likelihood.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - It is very unlikely that the image belongs to the
	// specified vertical.
	//   "UNLIKELY" - It is unlikely that the image belongs to the specified
	// vertical.
	//   "POSSIBLE" - It is possible that the image belongs to the specified
	// vertical.
	//   "LIKELY" - It is likely that the image belongs to the specified
	// vertical.
	//   "VERY_LIKELY" - It is very likely that the image belongs to the
	// specified vertical.
	Violence string `json:"violence,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Adult") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Adult") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SafeSearchAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod SafeSearchAnnotation
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

// Symbol: A single symbol representation.
type Symbol struct {
	// BoundingBox: The bounding box for the symbol.
	// The vertices are in the order of top-left, top-right,
	// bottom-right,
	// bottom-left. When a rotation of the bounding box is detected the
	// rotation
	// is represented as around the top-left corner as defined when the text
	// is
	// read in the 'natural' orientation.
	// For example:
	//   * when the text is horizontal it might look like:
	//      0----1
	//      |    |
	//      3----2
	//   * when it's rotated 180 degrees around the top-left corner it
	// becomes:
	//      2----3
	//      |    |
	//      1----0
	//   and the vertice order will still be (0, 1, 2, 3).
	BoundingBox *BoundingPoly `json:"boundingBox,omitempty"`

	// Property: Additional information detected for the symbol.
	Property *TextProperty `json:"property,omitempty"`

	// Text: The actual UTF-8 representation of the symbol.
	Text string `json:"text,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoundingBox") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoundingBox") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Symbol) MarshalJSON() ([]byte, error) {
	type noMethod Symbol
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TextAnnotation: TextAnnotation contains a structured representation
// of OCR extracted text.
// The hierarchy of an OCR extracted text structure is like this:
//     TextAnnotation -> Page -> Block -> Paragraph -> Word ->
// Symbol
// Each structural component, starting from Page, may further have their
// own
// properties. Properties describe detected languages, breaks etc..
// Please
// refer to the google.cloud.vision.v1.TextAnnotation.TextProperty
// message
// definition below for more detail.
type TextAnnotation struct {
	// Pages: List of pages detected by OCR.
	Pages []*Page `json:"pages,omitempty"`

	// Text: UTF-8 text detected on the pages.
	Text string `json:"text,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Pages") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Pages") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TextAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod TextAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TextProperty: Additional information detected on the structural
// component.
type TextProperty struct {
	// DetectedBreak: Detected start or end of a text segment.
	DetectedBreak *DetectedBreak `json:"detectedBreak,omitempty"`

	// DetectedLanguages: A list of detected languages together with
	// confidence.
	DetectedLanguages []*DetectedLanguage `json:"detectedLanguages,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DetectedBreak") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DetectedBreak") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TextProperty) MarshalJSON() ([]byte, error) {
	type noMethod TextProperty
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Vertex: A vertex represents a 2D point in the image.
// NOTE: the vertex coordinates are in the same scale as the original
// image.
type Vertex struct {
	// X: X coordinate.
	X int64 `json:"x,omitempty"`

	// Y: Y coordinate.
	Y int64 `json:"y,omitempty"`

	// ForceSendFields is a list of field names (e.g. "X") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "X") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Vertex) MarshalJSON() ([]byte, error) {
	type noMethod Vertex
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WebDetection: Relevant information for the image from the Internet.
type WebDetection struct {
	// FullMatchingImages: Fully matching images from the Internet.
	// Can include resized copies of the query image.
	FullMatchingImages []*WebImage `json:"fullMatchingImages,omitempty"`

	// PagesWithMatchingImages: Web pages containing the matching images
	// from the Internet.
	PagesWithMatchingImages []*WebPage `json:"pagesWithMatchingImages,omitempty"`

	// PartialMatchingImages: Partial matching images from the
	// Internet.
	// Those images are similar enough to share some key-point features.
	// For
	// example an original image will likely have partial matching for its
	// crops.
	PartialMatchingImages []*WebImage `json:"partialMatchingImages,omitempty"`

	// VisuallySimilarImages: The visually similar image results.
	VisuallySimilarImages []*WebImage `json:"visuallySimilarImages,omitempty"`

	// WebEntities: Deduced entities from similar images on the Internet.
	WebEntities []*WebEntity `json:"webEntities,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FullMatchingImages")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FullMatchingImages") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *WebDetection) MarshalJSON() ([]byte, error) {
	type noMethod WebDetection
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WebEntity: Entity deduced from similar images on the Internet.
type WebEntity struct {
	// Description: Canonical description of the entity, in English.
	Description string `json:"description,omitempty"`

	// EntityId: Opaque entity ID.
	EntityId string `json:"entityId,omitempty"`

	// Score: Overall relevancy score for the entity.
	// Not normalized and not comparable across different image queries.
	Score float64 `json:"score,omitempty"`

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

func (s *WebEntity) MarshalJSON() ([]byte, error) {
	type noMethod WebEntity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *WebEntity) UnmarshalJSON(data []byte) error {
	type noMethod WebEntity
	var s1 struct {
		Score gensupport.JSONFloat64 `json:"score"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Score = float64(s1.Score)
	return nil
}

// WebImage: Metadata for online images.
type WebImage struct {
	// Score: (Deprecated) Overall relevancy score for the image.
	Score float64 `json:"score,omitempty"`

	// Url: The result image URL.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Score") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Score") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WebImage) MarshalJSON() ([]byte, error) {
	type noMethod WebImage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *WebImage) UnmarshalJSON(data []byte) error {
	type noMethod WebImage
	var s1 struct {
		Score gensupport.JSONFloat64 `json:"score"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Score = float64(s1.Score)
	return nil
}

// WebPage: Metadata for web pages.
type WebPage struct {
	// Score: (Deprecated) Overall relevancy score for the web page.
	Score float64 `json:"score,omitempty"`

	// Url: The result web page URL.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Score") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Score") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WebPage) MarshalJSON() ([]byte, error) {
	type noMethod WebPage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *WebPage) UnmarshalJSON(data []byte) error {
	type noMethod WebPage
	var s1 struct {
		Score gensupport.JSONFloat64 `json:"score"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Score = float64(s1.Score)
	return nil
}

// Word: A word representation.
type Word struct {
	// BoundingBox: The bounding box for the word.
	// The vertices are in the order of top-left, top-right,
	// bottom-right,
	// bottom-left. When a rotation of the bounding box is detected the
	// rotation
	// is represented as around the top-left corner as defined when the text
	// is
	// read in the 'natural' orientation.
	// For example:
	//   * when the text is horizontal it might look like:
	//      0----1
	//      |    |
	//      3----2
	//   * when it's rotated 180 degrees around the top-left corner it
	// becomes:
	//      2----3
	//      |    |
	//      1----0
	//   and the vertice order will still be (0, 1, 2, 3).
	BoundingBox *BoundingPoly `json:"boundingBox,omitempty"`

	// Property: Additional information detected for the word.
	Property *TextProperty `json:"property,omitempty"`

	// Symbols: List of symbols in the word.
	// The order of the symbols follows the natural reading order.
	Symbols []*Symbol `json:"symbols,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoundingBox") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoundingBox") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Word) MarshalJSON() ([]byte, error) {
	type noMethod Word
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "vision.images.annotate":

type ImagesAnnotateCall struct {
	s                          *Service
	batchannotateimagesrequest *BatchAnnotateImagesRequest
	urlParams_                 gensupport.URLParams
	ctx_                       context.Context
	header_                    http.Header
}

// Annotate: Run image detection and annotation for a batch of images.
func (r *ImagesService) Annotate(batchannotateimagesrequest *BatchAnnotateImagesRequest) *ImagesAnnotateCall {
	c := &ImagesAnnotateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.batchannotateimagesrequest = batchannotateimagesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ImagesAnnotateCall) Fields(s ...googleapi.Field) *ImagesAnnotateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ImagesAnnotateCall) Context(ctx context.Context) *ImagesAnnotateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ImagesAnnotateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ImagesAnnotateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.batchannotateimagesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/images:annotate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "vision.images.annotate" call.
// Exactly one of *BatchAnnotateImagesResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *BatchAnnotateImagesResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ImagesAnnotateCall) Do(opts ...googleapi.CallOption) (*BatchAnnotateImagesResponse, error) {
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
	ret := &BatchAnnotateImagesResponse{
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
	//   "description": "Run image detection and annotation for a batch of images.",
	//   "flatPath": "v1/images:annotate",
	//   "httpMethod": "POST",
	//   "id": "vision.images.annotate",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/images:annotate",
	//   "request": {
	//     "$ref": "BatchAnnotateImagesRequest"
	//   },
	//   "response": {
	//     "$ref": "BatchAnnotateImagesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/cloud-vision"
	//   ]
	// }

}
