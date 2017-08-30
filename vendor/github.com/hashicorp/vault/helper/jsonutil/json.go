package jsonutil

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"

	"github.com/hashicorp/vault/helper/compressutil"
)

// Encodes/Marshals the given object into JSON
func EncodeJSON(in interface{}) ([]byte, error) {
	if in == nil {
		return nil, fmt.Errorf("input for encoding is nil")
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	if err := enc.Encode(in); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// EncodeJSONAndCompress encodes the given input into JSON and compresses the
// encoded value (using Gzip format BestCompression level, by default). A
// canary byte is placed at the beginning of the returned bytes for the logic
// in decompression method to identify compressed input.
func EncodeJSONAndCompress(in interface{}, config *compressutil.CompressionConfig) ([]byte, error) {
	if in == nil {
		return nil, fmt.Errorf("input for encoding is nil")
	}

	// First JSON encode the given input
	encodedBytes, err := EncodeJSON(in)
	if err != nil {
		return nil, err
	}

	if config == nil {
		config = &compressutil.CompressionConfig{
			Type:                 compressutil.CompressionTypeGzip,
			GzipCompressionLevel: gzip.BestCompression,
		}
	}

	return compressutil.Compress(encodedBytes, config)
}

// DecodeJSON tries to decompress the given data. The call to decompress, fails
// if the content was not compressed in the first place, which is identified by
// a canary byte before the compressed data. If the data is not compressed, it
// is JSON decoded directly. Otherwise the decompressed data will be JSON
// decoded.
func DecodeJSON(data []byte, out interface{}) error {
	if data == nil || len(data) == 0 {
		return fmt.Errorf("'data' being decoded is nil")
	}
	if out == nil {
		return fmt.Errorf("output parameter 'out' is nil")
	}

	// Decompress the data if it was compressed in the first place
	decompressedBytes, uncompressed, err := compressutil.Decompress(data)
	if err != nil {
		return fmt.Errorf("failed to decompress JSON: err: %v", err)
	}
	if !uncompressed && (decompressedBytes == nil || len(decompressedBytes) == 0) {
		return fmt.Errorf("decompressed data being decoded is invalid")
	}

	// If the input supplied failed to contain the compression canary, it
	// will be notified by the compression utility. Decode the decompressed
	// input.
	if !uncompressed {
		data = decompressedBytes
	}

	return DecodeJSONFromReader(bytes.NewReader(data), out)
}

// Decodes/Unmarshals the given io.Reader pointing to a JSON, into a desired object
func DecodeJSONFromReader(r io.Reader, out interface{}) error {
	if r == nil {
		return fmt.Errorf("'io.Reader' being decoded is nil")
	}
	if out == nil {
		return fmt.Errorf("output parameter 'out' is nil")
	}

	dec := json.NewDecoder(r)

	// While decoding JSON values, intepret the integer values as `json.Number`s instead of `float64`.
	dec.UseNumber()

	// Since 'out' is an interface representing a pointer, pass it to the decoder without an '&'
	return dec.Decode(out)
}
