package s3crypto

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/service/s3"
)

// SaveStrategy is how the data's metadata wants to be saved
type SaveStrategy interface {
	Save(Envelope, *request.Request) error
}

// S3SaveStrategy will save the metadata to a separate instruction file in S3
type S3SaveStrategy struct {
	Client                *s3.S3
	InstructionFileSuffix string
}

// Save will save the envelope contents to s3.
func (strat S3SaveStrategy) Save(env Envelope, req *request.Request) error {
	input := req.Params.(*s3.PutObjectInput)
	b, err := json.Marshal(env)
	if err != nil {
		return err
	}

	instInput := s3.PutObjectInput{
		Bucket: input.Bucket,
		Body:   bytes.NewReader(b),
	}

	if strat.InstructionFileSuffix == "" {
		instInput.Key = aws.String(*input.Key + DefaultInstructionKeySuffix)
	} else {
		instInput.Key = aws.String(*input.Key + strat.InstructionFileSuffix)
	}

	_, err = strat.Client.PutObject(&instInput)
	return err
}

// HeaderV2SaveStrategy will save the metadata of the crypto contents to the header of
// the object.
type HeaderV2SaveStrategy struct{}

// Save will save the envelope to the request's header.
func (strat HeaderV2SaveStrategy) Save(env Envelope, req *request.Request) error {
	input := req.Params.(*s3.PutObjectInput)
	if input.Metadata == nil {
		input.Metadata = map[string]*string{}
	}

	input.Metadata[http.CanonicalHeaderKey(keyV2Header)] = &env.CipherKey
	input.Metadata[http.CanonicalHeaderKey(ivHeader)] = &env.IV
	input.Metadata[http.CanonicalHeaderKey(matDescHeader)] = &env.MatDesc
	input.Metadata[http.CanonicalHeaderKey(wrapAlgorithmHeader)] = &env.WrapAlg
	input.Metadata[http.CanonicalHeaderKey(cekAlgorithmHeader)] = &env.CEKAlg
	input.Metadata[http.CanonicalHeaderKey(unencryptedContentLengthHeader)] = &env.UnencryptedContentLen

	if len(env.TagLen) > 0 {
		input.Metadata[http.CanonicalHeaderKey(tagLengthHeader)] = &env.TagLen
	}
	return nil
}

// LoadStrategy ...
type LoadStrategy interface {
	Load(*request.Request) (Envelope, error)
}

// S3LoadStrategy will load the instruction file from s3
type S3LoadStrategy struct {
	Client                *s3.S3
	InstructionFileSuffix string
}

// Load from a given instruction file suffix
func (load S3LoadStrategy) Load(req *request.Request) (Envelope, error) {
	env := Envelope{}
	if load.InstructionFileSuffix == "" {
		load.InstructionFileSuffix = DefaultInstructionKeySuffix
	}

	input := req.Params.(*s3.GetObjectInput)
	out, err := load.Client.GetObject(&s3.GetObjectInput{
		Key:    aws.String(strings.Join([]string{*input.Key, load.InstructionFileSuffix}, "")),
		Bucket: input.Bucket,
	})
	if err != nil {
		return env, err
	}

	b, err := ioutil.ReadAll(out.Body)
	if err != nil {
		return env, err
	}
	err = json.Unmarshal(b, &env)
	return env, err
}

// HeaderV2LoadStrategy will load the envelope from the metadata
type HeaderV2LoadStrategy struct{}

// Load from a given object's header
func (load HeaderV2LoadStrategy) Load(req *request.Request) (Envelope, error) {
	env := Envelope{}
	env.CipherKey = req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, keyV2Header}, "-"))
	env.IV = req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, ivHeader}, "-"))
	env.MatDesc = req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, matDescHeader}, "-"))
	env.WrapAlg = req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, wrapAlgorithmHeader}, "-"))
	env.CEKAlg = req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, cekAlgorithmHeader}, "-"))
	env.TagLen = req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, tagLengthHeader}, "-"))
	env.UnencryptedContentLen = req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, unencryptedContentLengthHeader}, "-"))
	return env, nil
}

type defaultV2LoadStrategy struct {
	client *s3.S3
	suffix string
}

func (load defaultV2LoadStrategy) Load(req *request.Request) (Envelope, error) {
	if value := req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, keyV2Header}, "-")); value != "" {
		strat := HeaderV2LoadStrategy{}
		return strat.Load(req)
	} else if value = req.HTTPResponse.Header.Get(strings.Join([]string{metaHeader, keyV1Header}, "-")); value != "" {
		return Envelope{}, awserr.New("V1NotSupportedError", "The AWS SDK for Go does not support version 1", nil)
	}

	strat := S3LoadStrategy{
		Client:                load.client,
		InstructionFileSuffix: load.suffix,
	}
	return strat.Load(req)
}
