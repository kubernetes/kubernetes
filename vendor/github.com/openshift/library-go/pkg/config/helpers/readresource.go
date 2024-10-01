package helpers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	kyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"
)

// InstallFunc is the "normal" function for installing scheme
type InstallFunc func(scheme *runtime.Scheme) error

// ReadYAMLToInternal reads content of a reader and returns the runtime.Object that matches it.  It chooses the match from
// the scheme installation that you provide.  It converts to internal for you.
func ReadYAMLToInternal(reader io.Reader, schemeFns ...InstallFunc) (runtime.Object, error) {
	if reader == nil || reflect.ValueOf(reader).IsNil() {
		return nil, nil
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	jsonData, err := kyaml.ToJSON(data)
	if err != nil {
		// maybe we were already json
		jsonData = data
	}

	scheme := runtime.NewScheme()
	for _, schemeFn := range schemeFns {
		err := schemeFn(scheme)
		if err != nil {
			return nil, err
		}
	}
	codec := serializer.NewCodecFactory(scheme).LegacyCodec(scheme.PrioritizedVersionsAllGroups()...)

	obj, err := runtime.Decode(codec, jsonData)
	if err != nil {
		return nil, captureSurroundingJSONForError("error reading config: ", jsonData, err)
	}
	// make sure there are no extra fields in jsonData
	if err := strictDecodeCheck(jsonData, obj, scheme); err != nil {
		return nil, err
	}

	return obj, nil
}

// ReadYAML reads content of a reader and returns the runtime.Object that matches it.  It chooses the match from
// the scheme installation that you provide.  It does not convert and it does not default.
func ReadYAML(reader io.Reader, schemeFns ...InstallFunc) (runtime.Object, error) {
	if reader == nil || reflect.ValueOf(reader).IsNil() {
		return nil, nil
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	jsonData, err := kyaml.ToJSON(data)
	if err != nil {
		// maybe we were already json
		jsonData = data
	}

	scheme := runtime.NewScheme()
	for _, schemeFn := range schemeFns {
		err := schemeFn(scheme)
		if err != nil {
			return nil, err
		}
	}
	codec := serializer.NewCodecFactory(scheme).UniversalDeserializer()

	obj, err := runtime.Decode(codec, jsonData)
	if err != nil {
		return nil, captureSurroundingJSONForError("error reading config: ", jsonData, err)
	}
	// make sure there are no extra fields in jsonData
	if err := strictDecodeCheck(jsonData, obj, scheme); err != nil {
		return nil, err
	}

	return obj, nil
}

// TODO: we ultimately want a better decoder for JSON that allows us exact line numbers and better
// surrounding text description. This should be removed / replaced when that happens.
func captureSurroundingJSONForError(prefix string, data []byte, err error) error {
	if syntaxErr, ok := err.(*json.SyntaxError); err != nil && ok {
		offset := syntaxErr.Offset
		begin := offset - 20
		if begin < 0 {
			begin = 0
		}
		end := offset + 20
		if end > int64(len(data)) {
			end = int64(len(data))
		}
		return fmt.Errorf("%s%v (found near '%s')", prefix, err, string(data[begin:end]))
	}
	if err != nil {
		return fmt.Errorf("%s%v", prefix, err)
	}
	return err
}

// strictDecodeCheck fails decodes when jsonData contains fields not included in the external version of obj
func strictDecodeCheck(jsonData []byte, obj runtime.Object, scheme *runtime.Scheme) error {
	out, err := getExternalZeroValue(obj, scheme) // we need the external version of obj as that has the correct JSON struct tags
	if err != nil {
		klog.Errorf("Encountered config error %v in object %T, raw JSON:\n%s", err, obj, string(jsonData)) // TODO just return the error and die
		// never error for now, we need to determine a safe way to make this check fatal
		return nil
	}
	d := json.NewDecoder(bytes.NewReader(jsonData))
	d.DisallowUnknownFields()
	// note that we only care about the error, out is discarded
	if err := d.Decode(out); err != nil {
		klog.Errorf("Encountered config error %v in object %T, raw JSON:\n%s", err, obj, string(jsonData)) // TODO just return the error and die
	}
	// never error for now, we need to determine a safe way to make this check fatal
	return nil
}

// getExternalZeroValue returns the zero value of the external version of obj
func getExternalZeroValue(obj runtime.Object, scheme *runtime.Scheme) (runtime.Object, error) {
	gvks, _, err := scheme.ObjectKinds(obj)
	if err != nil {
		return nil, err
	}
	if len(gvks) == 0 { // should never happen
		return nil, fmt.Errorf("no gvks found for %#v", obj)
	}
	return scheme.New(gvks[0])
}

// WriteYAML serializes a yaml file based on the scheme functions provided
func WriteYAML(obj runtime.Object, schemeFns ...InstallFunc) ([]byte, error) {
	scheme := runtime.NewScheme()
	for _, schemeFn := range schemeFns {
		err := schemeFn(scheme)
		if err != nil {
			return nil, err
		}
	}
	codec := serializer.NewCodecFactory(scheme).LegacyCodec(scheme.PrioritizedVersionsAllGroups()...)

	json, err := runtime.Encode(codec, obj)
	if err != nil {
		return nil, err
	}

	content, err := yaml.JSONToYAML(json)
	if err != nil {
		return nil, err
	}
	return content, err
}
