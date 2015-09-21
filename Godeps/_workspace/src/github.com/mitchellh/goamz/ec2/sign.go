package ec2

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"github.com/mitchellh/goamz/aws"
	"sort"
	"strings"
)

// ----------------------------------------------------------------------------
// EC2 signing (http://goo.gl/fQmAN)

var b64 = base64.StdEncoding

func sign(auth aws.Auth, method, path string, params map[string]string, host string) {
	params["AWSAccessKeyId"] = auth.AccessKey
	params["SignatureVersion"] = "2"
	params["SignatureMethod"] = "HmacSHA256"
	if auth.Token != "" {
		params["SecurityToken"] = auth.Token
	}

	// AWS specifies that the parameters in a signed request must
	// be provided in the natural order of the keys. This is distinct
	// from the natural order of the encoded value of key=value.
	// Percent and equals affect the sorting order.
	var keys, sarray []string
	for k, _ := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		sarray = append(sarray, aws.Encode(k)+"="+aws.Encode(params[k]))
	}
	joined := strings.Join(sarray, "&")
	payload := method + "\n" + host + "\n" + path + "\n" + joined
	hash := hmac.New(sha256.New, []byte(auth.SecretKey))
	hash.Write([]byte(payload))
	signature := make([]byte, b64.EncodedLen(hash.Size()))
	b64.Encode(signature, hash.Sum(nil))

	params["Signature"] = string(signature)
}
