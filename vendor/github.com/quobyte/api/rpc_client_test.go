package quobyte

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
)

func TestSuccesfullEncodeRequest(t *testing.T) {
	req := &CreateVolumeRequest{
		RootUserID:  "root",
		RootGroupID: "root",
		Name:        "test",
	}

	//Generate Params here
	var param map[string]interface{}
	byt, _ := json.Marshal(req)
	_ = json.Unmarshal(byt, &param)

	expectedRPCRequest := &request{
		ID:      "0",
		Method:  "createVolume",
		Version: "2.0",
		Params:  param,
	}

	res, _ := encodeRequest("createVolume", req)

	var reqResult request
	json.Unmarshal(res, &reqResult)

	if expectedRPCRequest.Version != reqResult.Version {
		t.Logf("Expected Version: %s got %s\n", expectedRPCRequest.Version, reqResult.Version)
		t.Fail()
	}

	if expectedRPCRequest.Method != reqResult.Method {
		t.Logf("Expected Method: %s got %s\n", expectedRPCRequest.Method, reqResult.Method)
		t.Fail()
	}

	if !reflect.DeepEqual(expectedRPCRequest.Params, reqResult.Params) {
		t.Logf("Expected Params: %v got %v\n", expectedRPCRequest.Params, reqResult.Params)
		t.Fail()
	}
}

func TestSuccesfullDecodeResponse(t *testing.T) {
	var byt json.RawMessage
	byt, _ = json.Marshal(map[string]interface{}{"volume_uuid": "1234"})

	expectedResult := &response{
		ID:      "0",
		Version: "2.0",
		Result:  &byt,
	}

	res, _ := json.Marshal(expectedResult)

	var resp volumeUUID
	err := decodeResponse(bytes.NewReader(res), &resp)
	if err != nil {
		t.Log(err)
		t.Fail()
	}

	if "1234" != resp.VolumeUUID {
		t.Logf("Expected Volume UUID: %v got %v\n", "1234", resp.VolumeUUID)
		t.Fail()
	}
}

func TestSuccesfullDecodeResponseWithErrorMessage(t *testing.T) {
	errorMessage := "ERROR_CODE_INVALID_REQUEST"
	var byt json.RawMessage
	byt, _ = json.Marshal(&rpcError{
		Code:    -32600,
		Message: "ERROR_CODE_INVALID_REQUEST",
	})

	expectedResult := &response{
		ID:      "0",
		Version: "2.0",
		Error:   &byt,
	}

	res, _ := json.Marshal(expectedResult)

	var resp volumeUUID
	err := decodeResponse(bytes.NewReader(res), &resp)
	if err == nil {
		t.Log("No error occured")
		t.Fail()
	}

	if errorMessage != err.Error() {
		t.Logf("Expected: %s got %s\n", errorMessage, err.Error())
		t.Fail()
	}
}

func TestSuccesfullDecodeResponseWithErrorCode(t *testing.T) {
	errorMessage := "ERROR_CODE_INVALID_REQUEST"
	var byt json.RawMessage
	byt, _ = json.Marshal(&rpcError{
		Code: -32600,
	})

	expectedResult := &response{
		ID:      "0",
		Version: "2.0",
		Error:   &byt,
	}

	res, _ := json.Marshal(expectedResult)

	var resp volumeUUID
	err := decodeResponse(bytes.NewReader(res), &resp)
	if err == nil {
		t.Log("No error occured")
		t.Fail()
	}

	if errorMessage != err.Error() {
		t.Logf("Expected: %s got %s\n", errorMessage, err.Error())
		t.Fail()
	}
}

func TestBadDecodeResponse(t *testing.T) {
	expectedResult := &response{
		ID:      "0",
		Version: "2.0",
	}

	res, _ := json.Marshal(expectedResult)

	var resp volumeUUID
	err := decodeResponse(bytes.NewReader(res), &resp)
	if err == nil {
		t.Log("No error occured")
		t.Fail()
	}

	if emptyResponse != err.Error() {
		t.Logf("Expected: %s got %s\n", emptyResponse, err.Error())
		t.Fail()
	}
}

type decodeErrorCodeTest struct {
	code     int64
	expected string
}

func TestDecodeErrorCode(t *testing.T) {
	tests := []*decodeErrorCodeTest{
		&decodeErrorCodeTest{code: -32600, expected: "ERROR_CODE_INVALID_REQUEST"},
		&decodeErrorCodeTest{code: -32603, expected: "ERROR_CODE_JSON_ENCODING_FAILED"},
		&decodeErrorCodeTest{code: -32601, expected: "ERROR_CODE_METHOD_NOT_FOUND"},
		&decodeErrorCodeTest{code: -32700, expected: "ERROR_CODE_PARSE_ERROR"},
	}

	_ = tests
	for _, decodeTest := range tests {
		err := &rpcError{
			Code: decodeTest.code,
		}

		if decodeTest.expected != err.decodeErrorCode() {
			t.Logf("Expected: %s got %s\n", decodeTest.expected, err.decodeErrorCode())
			t.Fail()
		}
	}
}
