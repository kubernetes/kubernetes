package streamformatter

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRawProgressFormatterFormatStatus(t *testing.T) {
	sf := rawProgressFormatter{}
	res := sf.formatStatus("ID", "%s%d", "a", 1)
	assert.Equal(t, "a1\r\n", string(res))
}

func TestRawProgressFormatterFormatProgress(t *testing.T) {
	sf := rawProgressFormatter{}
	jsonProgress := &jsonmessage.JSONProgress{
		Current: 15,
		Total:   30,
		Start:   1,
	}
	res := sf.formatProgress("id", "action", jsonProgress, nil)
	out := string(res)
	assert.True(t, strings.HasPrefix(out, "action [===="))
	assert.Contains(t, out, "15B/30B")
	assert.True(t, strings.HasSuffix(out, "\r"))
}

func TestFormatStatus(t *testing.T) {
	res := FormatStatus("ID", "%s%d", "a", 1)
	expected := `{"status":"a1","id":"ID"}` + streamNewline
	assert.Equal(t, expected, string(res))
}

func TestFormatError(t *testing.T) {
	res := FormatError(errors.New("Error for formatter"))
	expected := `{"errorDetail":{"message":"Error for formatter"},"error":"Error for formatter"}` + "\r\n"
	assert.Equal(t, expected, string(res))
}

func TestFormatJSONError(t *testing.T) {
	err := &jsonmessage.JSONError{Code: 50, Message: "Json error"}
	res := FormatError(err)
	expected := `{"errorDetail":{"code":50,"message":"Json error"},"error":"Json error"}` + streamNewline
	assert.Equal(t, expected, string(res))
}

func TestJsonProgressFormatterFormatProgress(t *testing.T) {
	sf := &jsonProgressFormatter{}
	jsonProgress := &jsonmessage.JSONProgress{
		Current: 15,
		Total:   30,
		Start:   1,
	}
	res := sf.formatProgress("id", "action", jsonProgress, &AuxFormatter{Writer: &bytes.Buffer{}})
	msg := &jsonmessage.JSONMessage{}

	require.NoError(t, json.Unmarshal(res, msg))
	assert.Equal(t, "id", msg.ID)
	assert.Equal(t, "action", msg.Status)

	// jsonProgress will always be in the format of:
	// [=========================>                         ]      15B/30B 412910h51m30s
	// The last entry '404933h7m11s' is the timeLeftBox.
	// However, the timeLeftBox field may change as jsonProgress.String() depends on time.Now().
	// Therefore, we have to strip the timeLeftBox from the strings to do the comparison.

	// Compare the jsonProgress strings before the timeLeftBox
	expectedProgress := "[=========================>                         ]      15B/30B"
	// if terminal column is <= 110, expectedProgressShort is expected.
	expectedProgressShort := "      15B/30B"
	if !(strings.HasPrefix(msg.ProgressMessage, expectedProgress) ||
		strings.HasPrefix(msg.ProgressMessage, expectedProgressShort)) {
		t.Fatalf("ProgressMessage without the timeLeftBox must be %s or %s, got: %s",
			expectedProgress, expectedProgressShort, msg.ProgressMessage)
	}

	assert.Equal(t, jsonProgress, msg.Progress)
}

func TestJsonProgressFormatterFormatStatus(t *testing.T) {
	sf := jsonProgressFormatter{}
	res := sf.formatStatus("ID", "%s%d", "a", 1)
	assert.Equal(t, `{"status":"a1","id":"ID"}`+streamNewline, string(res))
}

func TestNewJSONProgressOutput(t *testing.T) {
	b := bytes.Buffer{}
	b.Write(FormatStatus("id", "Downloading"))
	_ = NewJSONProgressOutput(&b, false)
	assert.Equal(t, `{"status":"Downloading","id":"id"}`+streamNewline, b.String())
}

func TestAuxFormatterEmit(t *testing.T) {
	b := bytes.Buffer{}
	aux := &AuxFormatter{Writer: &b}
	sampleAux := &struct {
		Data string
	}{"Additional data"}
	err := aux.Emit(sampleAux)
	require.NoError(t, err)
	assert.Equal(t, `{"aux":{"Data":"Additional data"}}`+streamNewline, b.String())
}
