package s3

import (
	"crypto/md5"
	"encoding/base64"
	"net/http"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

var errSSERequiresSSL = awserr.New("ConfigError", "cannot send SSE keys over HTTP.", nil)

func validateSSERequiresSSL(r *request.Request) {
	if r.HTTPRequest.URL.Scheme == "https" {
		return
	}

	if iface, ok := r.Params.(sseCustomerKeyGetter); ok {
		if len(iface.getSSECustomerKey()) > 0 {
			r.Error = errSSERequiresSSL
			return
		}
	}

	if iface, ok := r.Params.(copySourceSSECustomerKeyGetter); ok {
		if len(iface.getCopySourceSSECustomerKey()) > 0 {
			r.Error = errSSERequiresSSL
			return
		}
	}
}

const (
	sseKeyHeader    = "x-amz-server-side-encryption-customer-key"
	sseKeyMD5Header = sseKeyHeader + "-md5"
)

func computeSSEKeyMD5(r *request.Request) {
	var key string
	if g, ok := r.Params.(sseCustomerKeyGetter); ok {
		key = g.getSSECustomerKey()
	}

	computeKeyMD5(sseKeyHeader, sseKeyMD5Header, key, r.HTTPRequest)
}

const (
	copySrcSSEKeyHeader    = "x-amz-copy-source-server-side-encryption-customer-key"
	copySrcSSEKeyMD5Header = copySrcSSEKeyHeader + "-md5"
)

func computeCopySourceSSEKeyMD5(r *request.Request) {
	var key string
	if g, ok := r.Params.(copySourceSSECustomerKeyGetter); ok {
		key = g.getCopySourceSSECustomerKey()
	}

	computeKeyMD5(copySrcSSEKeyHeader, copySrcSSEKeyMD5Header, key, r.HTTPRequest)
}

func computeKeyMD5(keyHeader, keyMD5Header, key string, r *http.Request) {
	if len(key) == 0 {
		// Backwards compatiablity where user just set the header value instead
		// of using the API parameter, or setting the header value for an
		// operation without the parameters modeled.
		key = r.Header.Get(keyHeader)
		if len(key) == 0 {
			return
		}

		// In backwards compatible, the header's value is not base64 encoded,
		// and needs to be encoded and updated by the SDK's customizations.
		b64Key := base64.StdEncoding.EncodeToString([]byte(key))
		r.Header.Set(keyHeader, b64Key)
	}

	// Only update Key's MD5 if not already set.
	if len(r.Header.Get(keyMD5Header)) == 0 {
		sum := md5.Sum([]byte(key))
		keyMD5 := base64.StdEncoding.EncodeToString(sum[:])
		r.Header.Set(keyMD5Header, keyMD5)
	}
}
