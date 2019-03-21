// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package runtime

const (
	// HeaderContentType represents a http content-type header, it's value is supposed to be a mime type
	HeaderContentType = "Content-Type"

	// HeaderTransferEncoding represents a http transfer-encoding header.
	HeaderTransferEncoding = "Transfer-Encoding"

	// HeaderAccept the Accept header
	HeaderAccept = "Accept"

	charsetKey = "charset"

	// DefaultMime the default fallback mime type
	DefaultMime = "application/octet-stream"
	// JSONMime the json mime type
	JSONMime = "application/json"
	// YAMLMime the yaml mime type
	YAMLMime = "application/x-yaml"
	// XMLMime the xml mime type
	XMLMime = "application/xml"
	// TextMime the text mime type
	TextMime = "text/plain"
	// HTMLMime the html mime type
	HTMLMime = "text/html"
	// MultipartFormMime the multipart form mime type
	MultipartFormMime = "multipart/form-data"
	// URLencodedFormMime the url encoded form mime type
	URLencodedFormMime = "application/x-www-form-urlencoded"
)
