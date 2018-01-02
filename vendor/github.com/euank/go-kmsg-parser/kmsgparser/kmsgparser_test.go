/*
Copyright 2016 Euan Kemp

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kmsgparser

import (
	"bufio"
	"bytes"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Logger that errors on warnings and errors
type warningAndErrorTestLogger struct {
	t *testing.T
}

func (warningAndErrorTestLogger) Infof(string, ...interface{}) {}
func (w warningAndErrorTestLogger) Warningf(s string, i ...interface{}) {
	w.t.Errorf(s, i)
}
func (w warningAndErrorTestLogger) Errorf(s string, i ...interface{}) {
	w.t.Errorf(s, i)
}

func TestParseMessage(t *testing.T) {
	bootTime := time.Unix(0xb100, 0x5ea1).Round(time.Microsecond)
	p := parser{
		log:      warningAndErrorTestLogger{t: t},
		bootTime: bootTime,
	}
	msg, err := p.parseMessage("6,2565,102258085667,-;docker0: port 2(vethc1bb733) entered blocking state")
	if err != nil {
		t.Fatalf("error parsing: %v", err)
	}

	assert.Equal(t, msg.Message, "docker0: port 2(vethc1bb733) entered blocking state")

	assert.Equal(t, msg.Priority, 6)
	assert.Equal(t, msg.SequenceNumber, 2565)
	assert.Equal(t, msg.Timestamp, bootTime.Add(102258085667*time.Microsecond))
}

func TestParse(t *testing.T) {
	bootTime := time.Unix(0xb100, 0x5ea1).Round(time.Microsecond)
	p := parser{
		log:      warningAndErrorTestLogger{t: t},
		bootTime: bootTime,
	}
	f, err := os.Open(filepath.Join("test_data", "sample1.kmsg"))
	if err != nil {
		t.Fatalf("could not find sample data: %v", err)
	}
	defer f.Close()

	expectedMessages := []Message{
		{
			Priority:       6,
			SequenceNumber: 1804,
			Timestamp:      bootTime.Add(47700428483 * time.Microsecond),
			Message:        "wlp4s0: associated",
		},
		{
			Priority:       6,
			SequenceNumber: 1805,
			Timestamp:      bootTime.Add(51742248189 * time.Microsecond),
			Message:        "thinkpad_acpi: EC reports that Thermal Table has changed",
		},
		{
			Priority:       6,
			SequenceNumber: 2651,
			Timestamp:      bootTime.Add(106819644585 * time.Microsecond),
			Message:        "CPU1: Package temperature/speed normal",
		},
	}

	s := bufio.NewScanner(f)
	mockKmsg, mockKmsgInput := io.Pipe()
	p.kmsgReader = mockSeeker(ioutil.NopCloser(mockKmsg))
	go func() {
		for s.Scan() {
			_, err := mockKmsgInput.Write(s.Bytes())
			if err != nil {
				panic(err)
			}
		}
		mockKmsgInput.Close()
	}()

	lines := p.Parse()

	messages := []Message{}
	for line := range lines {
		messages = append(messages, line)
	}

	assert.Equal(t, expectedMessages, messages)
}

func TestSeekEnd(t *testing.T) {
	p := parser{
		log: warningAndErrorTestLogger{t: t},
	}
	var mockKmsg bytes.Buffer
	mockkmsg := mockSeeker(ioutil.NopCloser(&mockKmsg))
	p.kmsgReader = mockkmsg

	err := p.SeekEnd()
	assert.Nil(t, err)
	assert.Equal(t, mockkmsg.seekCalls, [][2]int64{[2]int64{0, int64(os.SEEK_END)}})

}

type mseeker struct {
	io.ReadCloser
	seekCalls [][2]int64
}

func mockSeeker(rc io.ReadCloser) *mseeker {
	return &mseeker{ReadCloser: rc}
}

func (m *mseeker) Seek(x int64, y int) (int64, error) {
	m.seekCalls = append(m.seekCalls, [2]int64{x, int64(y)})
	return 0, nil
}
