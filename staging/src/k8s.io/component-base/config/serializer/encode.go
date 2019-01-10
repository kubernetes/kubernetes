package serializer

import (
	"bytes"
	gojson "encoding/json"
	"io"
	"io/ioutil"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"sigs.k8s.io/yaml"
)

type ComponentConfigEncodingFormat string

const (
	YAML       ComponentConfigEncodingFormat = "YAML"
	JSON       ComponentConfigEncodingFormat = "JSON"
	JSONPretty ComponentConfigEncodingFormat = "JSONPretty"
)

// DecodeFileInto reads a file and decodes the it into an internal type
func DecodeFileInto(filePath string, obj runtime.Object, codecs *serializer.CodecFactory, strictErrWriter io.Writer) error {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return err
	}
	// Regardless of if the bytes are of the v1 or v1beta1 version,
	// it will be read successfully and converted into the internal version
	return DecodeInto(content, obj, codecs, strictErrWriter)
}

// DecodeInto reads a file and decodes the it into an internal type
func DecodeInto(b []byte, obj runtime.Object, codecs *serializer.CodecFactory, strictErrWriter io.Writer) error {
	if strictErrWriter != nil {
		if err := VerifyUnmarshalObjectStrict(b, obj); err != nil {
			strictErrWriter.Write([]byte(err.Error()))
		}
	}
	// Regardless of if the bytes are of the v1 or v1beta1 version,
	// it will be read successfully and converted into the internal version
	return runtime.DecodeInto(codecs.UniversalDecoder(), b, obj)
}

func Encode(format ComponentConfigEncodingFormat, gv schema.GroupVersion, obj runtime.Object, scheme *runtime.Scheme, codecs *serializer.CodecFactory) ([]byte, error) {
	// encoder is a generic-purpose encoder to the specified format for this scheme
	var encoder runtime.Encoder
	switch format {
	case YAML:
		encoder = json.NewYAMLSerializer(json.DefaultMetaFactory, scheme, scheme)
	case JSON:
		encoder = json.NewSerializer(json.DefaultMetaFactory, scheme, scheme, false)
	case JSONPretty:
		encoder = json.NewSerializer(json.DefaultMetaFactory, scheme, scheme, true)
	default:
		return []byte{}, errors.Errorf("componentconfig encoding format not supported: %s", string(format))
	}
	// versionSpecificEncoder writes out YAML bytes for exactly this v1beta1 version
	versionSpecificEncoder := codecs.EncoderForVersion(encoder, gv)
	// Encode the object to YAML for the given version
	return runtime.Encode(versionSpecificEncoder, obj)
}

// VerifyUnmarshalStrict takes a YAML byte slice and a GroupVersionKind and verifies if the YAML
// schema is known and if it unmarshals with strict mode.
func VerifyUnmarshalStrict(b []byte, gvk schema.GroupVersionKind, scheme *runtime.Scheme) error {
	obj, err := scheme.New(gvk)
	if err != nil {
		return errors.Errorf("unknown configuration %#v for scheme definitions in %q",
			gvk, scheme.Name())
	}
	return VerifyUnmarshalObjectStrict(b, obj)
}

// VerifyUnmarshalStrict takes a YAML byte slice and a GroupVersionKind and verifies if the YAML
// schema is known and if it unmarshals with strict mode.
func VerifyUnmarshalObjectStrict(b []byte, obj runtime.Object) error {
	teststrictobj := obj.DeepCopyObject()
	jsonbytes, err := yaml.YAMLToJSONStrict(b)
	if err != nil {
		return errors.Wrapf(err, "reading YAML in strict mode failed")
	}
	d := gojson.NewDecoder(bytes.NewReader(jsonbytes))
	d.DisallowUnknownFields()
	if err := d.Decode(teststrictobj); err != nil {
		return errors.Wrapf(err, "error unmarshaling configuration in strict mode")
	}
	return nil
}
