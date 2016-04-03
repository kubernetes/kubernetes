package s3

import (
	"bytes"
	"io/ioutil"
	"net/http"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

func copyMultipartStatusOKUnmarhsalError(r *request.Request) {
	b, err := ioutil.ReadAll(r.HTTPResponse.Body)
	if err != nil {
		r.Error = awserr.New("SerializationError", "unable to read response body", err)
		return
	}
	body := bytes.NewReader(b)
	r.HTTPResponse.Body = aws.ReadSeekCloser(body)
	defer r.HTTPResponse.Body.(aws.ReaderSeekerCloser).Seek(0, 0)

	if body.Len() == 0 {
		// If there is no body don't attempt to parse the body.
		return
	}

	unmarshalError(r)
	if err, ok := r.Error.(awserr.Error); ok && err != nil {
		if err.Code() == "SerializationError" {
			r.Error = nil
			return
		}
		r.HTTPResponse.StatusCode = http.StatusServiceUnavailable
	}
}
