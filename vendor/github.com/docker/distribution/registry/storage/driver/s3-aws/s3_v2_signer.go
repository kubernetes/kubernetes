package s3

// Source: https://github.com/pivotal-golang/s3cli

// Copyright (c) 2013 Damien Le Berrigaud and Nick Wade

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
	log "github.com/sirupsen/logrus"
)

const (
	signatureVersion = "2"
	signatureMethod  = "HmacSHA1"
	timeFormat       = "2006-01-02T15:04:05Z"
)

type signer struct {
	// Values that must be populated from the request
	Request      *http.Request
	Time         time.Time
	Credentials  *credentials.Credentials
	Query        url.Values
	stringToSign string
	signature    string
}

var s3ParamsToSign = map[string]bool{
	"acl":                          true,
	"location":                     true,
	"logging":                      true,
	"notification":                 true,
	"partNumber":                   true,
	"policy":                       true,
	"requestPayment":               true,
	"torrent":                      true,
	"uploadId":                     true,
	"uploads":                      true,
	"versionId":                    true,
	"versioning":                   true,
	"versions":                     true,
	"response-content-type":        true,
	"response-content-language":    true,
	"response-expires":             true,
	"response-cache-control":       true,
	"response-content-disposition": true,
	"response-content-encoding":    true,
	"website":                      true,
	"delete":                       true,
}

// setv2Handlers will setup v2 signature signing on the S3 driver
func setv2Handlers(svc *s3.S3) {
	svc.Handlers.Build.PushBack(func(r *request.Request) {
		parsedURL, err := url.Parse(r.HTTPRequest.URL.String())
		if err != nil {
			log.Fatalf("Failed to parse URL: %v", err)
		}
		r.HTTPRequest.URL.Opaque = parsedURL.Path
	})

	svc.Handlers.Sign.Clear()
	svc.Handlers.Sign.PushBack(Sign)
	svc.Handlers.Sign.PushBackNamed(corehandlers.BuildContentLengthHandler)
}

// Sign requests with signature version 2.
//
// Will sign the requests with the service config's Credentials object
// Signing is skipped if the credentials is the credentials.AnonymousCredentials
// object.
func Sign(req *request.Request) {
	// If the request does not need to be signed ignore the signing of the
	// request if the AnonymousCredentials object is used.
	if req.Config.Credentials == credentials.AnonymousCredentials {
		return
	}

	v2 := signer{
		Request:     req.HTTPRequest,
		Time:        req.Time,
		Credentials: req.Config.Credentials,
	}
	v2.Sign()
}

func (v2 *signer) Sign() error {
	credValue, err := v2.Credentials.Get()
	if err != nil {
		return err
	}
	accessKey := credValue.AccessKeyID
	var (
		md5, ctype, date, xamz string
		xamzDate               bool
		sarray                 []string
		smap                   map[string]string
		sharray                []string
	)

	headers := v2.Request.Header
	params := v2.Request.URL.Query()
	parsedURL, err := url.Parse(v2.Request.URL.String())
	if err != nil {
		return err
	}
	host, canonicalPath := parsedURL.Host, parsedURL.Path
	v2.Request.Header["Host"] = []string{host}
	v2.Request.Header["date"] = []string{v2.Time.In(time.UTC).Format(time.RFC1123)}
	if credValue.SessionToken != "" {
		v2.Request.Header["x-amz-security-token"] = []string{credValue.SessionToken}
	}

	smap = make(map[string]string)
	for k, v := range headers {
		k = strings.ToLower(k)
		switch k {
		case "content-md5":
			md5 = v[0]
		case "content-type":
			ctype = v[0]
		case "date":
			if !xamzDate {
				date = v[0]
			}
		default:
			if strings.HasPrefix(k, "x-amz-") {
				vall := strings.Join(v, ",")
				smap[k] = k + ":" + vall
				if k == "x-amz-date" {
					xamzDate = true
					date = ""
				}
				sharray = append(sharray, k)
			}
		}
	}
	if len(sharray) > 0 {
		sort.StringSlice(sharray).Sort()
		for _, h := range sharray {
			sarray = append(sarray, smap[h])
		}
		xamz = strings.Join(sarray, "\n") + "\n"
	}

	expires := false
	if v, ok := params["Expires"]; ok {
		expires = true
		date = v[0]
		params["AWSAccessKeyId"] = []string{accessKey}
	}

	sarray = sarray[0:0]
	for k, v := range params {
		if s3ParamsToSign[k] {
			for _, vi := range v {
				if vi == "" {
					sarray = append(sarray, k)
				} else {
					sarray = append(sarray, k+"="+vi)
				}
			}
		}
	}
	if len(sarray) > 0 {
		sort.StringSlice(sarray).Sort()
		canonicalPath = canonicalPath + "?" + strings.Join(sarray, "&")
	}

	v2.stringToSign = strings.Join([]string{
		v2.Request.Method,
		md5,
		ctype,
		date,
		xamz + canonicalPath,
	}, "\n")
	hash := hmac.New(sha1.New, []byte(credValue.SecretAccessKey))
	hash.Write([]byte(v2.stringToSign))
	v2.signature = base64.StdEncoding.EncodeToString(hash.Sum(nil))

	if expires {
		params["Signature"] = []string{string(v2.signature)}
	} else {
		headers["Authorization"] = []string{"AWS " + accessKey + ":" + string(v2.signature)}
	}

	log.WithFields(log.Fields{
		"string-to-sign": v2.stringToSign,
		"signature":      v2.signature,
	}).Debugln("request signature")
	return nil
}
