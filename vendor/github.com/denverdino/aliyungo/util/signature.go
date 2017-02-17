package util

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"net/url"
	"strings"
)

//CreateSignature creates signature for string following Aliyun rules
func CreateSignature(stringToSignature, accessKeySecret string) string {
	// Crypto by HMAC-SHA1
	hmacSha1 := hmac.New(sha1.New, []byte(accessKeySecret))
	hmacSha1.Write([]byte(stringToSignature))
	sign := hmacSha1.Sum(nil)

	// Encode to Base64
	base64Sign := base64.StdEncoding.EncodeToString(sign)

	return base64Sign
}

func percentReplace(str string) string {
	str = strings.Replace(str, "+", "%20", -1)
	str = strings.Replace(str, "*", "%2A", -1)
	str = strings.Replace(str, "%7E", "~", -1)

	return str
}

// CreateSignatureForRequest creates signature for query string values
func CreateSignatureForRequest(method string, values *url.Values, accessKeySecret string) string {

	canonicalizedQueryString := percentReplace(values.Encode())

	stringToSign := method + "&%2F&" + url.QueryEscape(canonicalizedQueryString)

	return CreateSignature(stringToSign, accessKeySecret)
}
