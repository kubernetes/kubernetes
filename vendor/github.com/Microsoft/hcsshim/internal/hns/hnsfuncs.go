//go:build windows

package hns

import (
	"encoding/json"
	"fmt"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

func hnsCallRawResponse(method, path, request string) (*hnsResponse, error) {
	var responseBuffer *uint16
	logrus.Debugf("[%s]=>[%s] Request : %s", method, path, request)

	err := _hnsCall(method, path, request, &responseBuffer)
	if err != nil {
		return nil, hcserror.New(err, "hnsCall ", "")
	}
	response := interop.ConvertAndFreeCoTaskMemString(responseBuffer)

	hnsresponse := &hnsResponse{}
	if err = json.Unmarshal([]byte(response), &hnsresponse); err != nil {
		return nil, err
	}
	return hnsresponse, nil
}

func hnsCall(method, path, request string, returnResponse interface{}) error {
	hnsresponse, err := hnsCallRawResponse(method, path, request)
	if err != nil {
		return fmt.Errorf("failed during hnsCallRawResponse: %w", err)
	}
	if !hnsresponse.Success {
		return fmt.Errorf("hns failed with error : %s", hnsresponse.Error)
	}

	if len(hnsresponse.Output) == 0 {
		return nil
	}

	logrus.Debugf("Network Response : %s", hnsresponse.Output)
	err = json.Unmarshal(hnsresponse.Output, returnResponse)
	if err != nil {
		return err
	}

	return nil
}
