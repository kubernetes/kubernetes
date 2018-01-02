package test

import (
	"encoding/json"
	"errors"
	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
	"io"
	"testing"
)

func Test_skip(t *testing.T) {
	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			should := require.New(t)
			var dst typeForTest
			stdErr := json.Unmarshal([]byte(input), &dst)
			iter := jsoniter.ParseString(jsoniter.ConfigDefault, input)
			iter.Skip()
			iter.ReadNil() // trigger looking forward
			err := iter.Error
			if err == io.EOF {
				err = nil
			} else {
				err = errors.New("remaining bytes")
			}
			if stdErr == nil {
				should.Nil(err)
			} else {
				should.NotNil(err)
			}
		})
	}
}
