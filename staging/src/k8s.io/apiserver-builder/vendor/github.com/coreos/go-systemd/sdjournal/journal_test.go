// Copyright 2015 RedHat, Inc.
// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sdjournal

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coreos/go-systemd/journal"
)

func TestJournalFollow(t *testing.T) {
	r, err := NewJournalReader(JournalReaderConfig{
		Since: time.Duration(-15) * time.Second,
		Matches: []Match{
			{
				Field: SD_JOURNAL_FIELD_SYSTEMD_UNIT,
				Value: "NetworkManager.service",
			},
		},
	})

	if err != nil {
		t.Fatalf("Error opening journal: %s", err)
	}

	if r == nil {
		t.Fatal("Got a nil reader")
	}

	defer r.Close()

	// start writing some test entries
	done := make(chan struct{}, 1)
	defer close(done)
	go func() {
		for {
			select {
			case <-done:
				return
			default:
				if err := journal.Print(journal.PriInfo, "test message %s", time.Now()); err != nil {
					t.Fatalf("Error writing to journal: %s", err)
				}

				time.Sleep(time.Second)
			}
		}
	}()

	// and follow the reader synchronously
	timeout := time.Duration(5) * time.Second
	if err = r.Follow(time.After(timeout), os.Stdout); err != ErrExpired {
		t.Fatalf("Error during follow: %s", err)
	}
}

func TestJournalGetUsage(t *testing.T) {
	j, err := NewJournal()

	if err != nil {
		t.Fatalf("Error opening journal: %s", err)
	}

	if j == nil {
		t.Fatal("Got a nil journal")
	}

	defer j.Close()

	_, err = j.GetUsage()

	if err != nil {
		t.Fatalf("Error getting journal size: %s", err)
	}
}

func TestJournalCursorGetSeekAndTest(t *testing.T) {
	j, err := NewJournal()
	if err != nil {
		t.Fatalf("Error opening journal: %s", err)
	}

	if j == nil {
		t.Fatal("Got a nil journal")
	}

	defer j.Close()

	waitAndNext := func(j *Journal) error {
		r := j.Wait(time.Duration(1) * time.Second)
		if r < 0 {
			return errors.New("Error waiting to journal")
		}

		n, err := j.Next()
		if err != nil {
			return fmt.Errorf("Error reading to journal: %s", err)
		}

		if n == 0 {
			return fmt.Errorf("Error reading to journal: %s", io.EOF)
		}

		return nil
	}

	err = journal.Print(journal.PriInfo, "test message for cursor %s", time.Now())
	if err != nil {
		t.Fatalf("Error writing to journal: %s", err)
	}

	if err = waitAndNext(j); err != nil {
		t.Fatalf(err.Error())
	}

	c, err := j.GetCursor()
	if err != nil {
		t.Fatalf("Error getting cursor from journal: %s", err)
	}

	err = j.SeekCursor(c)
	if err != nil {
		t.Fatalf("Error seeking cursor to journal: %s", err)
	}

	if err = waitAndNext(j); err != nil {
		t.Fatalf(err.Error())
	}

	err = j.TestCursor(c)
	if err != nil {
		t.Fatalf("Error testing cursor to journal: %s", err)
	}
}

func TestNewJournalFromDir(t *testing.T) {
	// test for error handling
	dir := "/ClearlyNonExistingPath/"
	j, err := NewJournalFromDir(dir)
	if err == nil {
		defer j.Close()
		t.Fatalf("Error expected when opening dummy path (%s)", dir)
	}
	// test for main code path
	dir, err = ioutil.TempDir("", "go-systemd-test")
	if err != nil {
		t.Fatalf("Error creating tempdir: %s", err)
	}
	defer os.RemoveAll(dir)
	j, err = NewJournalFromDir(dir)
	if err != nil {
		t.Fatalf("Error opening journal: %s", err)
	}
	if j == nil {
		t.Fatal("Got a nil journal")
	}
	j.Close()
}

func setupJournalRoundtrip() (*Journal, map[string]string, error) {
	j, err := NewJournal()
	if err != nil {
		return nil, nil, fmt.Errorf("Error opening journal: %s", err)
	}

	if j == nil {
		return nil, nil, fmt.Errorf("Got a nil journal")
	}

	j.FlushMatches()

	matchField := "TESTJOURNALENTRY"
	matchValue := fmt.Sprintf("%d", time.Now().UnixNano())
	m := Match{Field: matchField, Value: matchValue}
	err = j.AddMatch(m.String())
	if err != nil {
		return nil, nil, fmt.Errorf("Error adding matches to journal: %s", err)
	}

	msg := fmt.Sprintf("test journal get entry message %s", time.Now())
	data := map[string]string{matchField: matchValue}
	err = journal.Send(msg, journal.PriInfo, data)
	if err != nil {
		return nil, nil, fmt.Errorf("Error writing to journal: %s", err)
	}

	time.Sleep(time.Duration(1) * time.Second)

	n, err := j.Next()
	if err != nil {
		return nil, nil, fmt.Errorf("Error reading from journal: %s", err)
	}

	if n == 0 {
		return nil, nil, fmt.Errorf("Error reading from journal: %s", io.EOF)
	}

	data["MESSAGE"] = msg

	return j, data, nil
}

func TestJournalGetData(t *testing.T) {
	j, wantEntry, err := setupJournalRoundtrip()
	if err != nil {
		t.Fatal(err.Error())
	}

	defer j.Close()

	for k, v := range wantEntry {
		data := fmt.Sprintf("%s=%s", k, v)

		dataStr, err := j.GetData(k)
		if err != nil {
			t.Fatalf("GetData() error: %v", err)
		}

		if dataStr != data {
			t.Fatalf("Invalid data for \"%s\": got %s, want %s", k, dataStr, data)
		}

		dataBytes, err := j.GetDataBytes(k)
		if err != nil {
			t.Fatalf("GetDataBytes() error: %v", err)
		}

		if string(dataBytes) != data {
			t.Fatalf("Invalid data bytes for \"%s\": got %s, want %s", k, string(dataBytes), data)
		}

		valStr, err := j.GetDataValue(k)
		if err != nil {
			t.Fatalf("GetDataValue() error: %v", err)
		}

		if valStr != v {
			t.Fatalf("Invalid data value for \"%s\": got %s, want %s", k, valStr, v)
		}

		valBytes, err := j.GetDataValueBytes(k)
		if err != nil {
			t.Fatalf("GetDataValueBytes() error: %v", err)
		}

		if string(valBytes) != v {
			t.Fatalf("Invalid data value bytes for \"%s\": got %s, want %s", k, string(valBytes), v)
		}
	}
}

func TestJournalGetEntry(t *testing.T) {
	j, wantEntry, err := setupJournalRoundtrip()
	if err != nil {
		t.Fatal(err.Error())
	}

	defer j.Close()

	entry, err := j.GetEntry()
	if err != nil {
		t.Fatalf("Error getting the entry to journal: %s", err)
	}

	for k, wantV := range wantEntry {
		gotV := entry.Fields[k]
		if gotV != wantV {
			t.Fatalf("Bad result for entry.Fields[\"%s\"]: got %s, want %s", k, gotV, wantV)
		}
	}
}

// Check for incorrect read into small buffers,
// see https://github.com/coreos/go-systemd/issues/172
func TestJournalReaderSmallReadBuffer(t *testing.T) {
	// Write a long entry ...
	delim := "%%%%%%"
	longEntry := strings.Repeat("a", 256)
	matchField := "TESTJOURNALREADERSMALLBUF"
	matchValue := fmt.Sprintf("%d", time.Now().UnixNano())
	r, err := NewJournalReader(JournalReaderConfig{
		Since: time.Duration(-15) * time.Second,
		Matches: []Match{
			{
				Field: matchField,
				Value: matchValue,
			},
		},
	})
	if err != nil {
		t.Fatalf("Error opening journal: %s", err)
	}
	if r == nil {
		t.Fatal("Got a nil reader")
	}
	defer r.Close()

	want := fmt.Sprintf("%slongentry %s%s", delim, longEntry, delim)
	err = journal.Send(want, journal.PriInfo, map[string]string{matchField: matchValue})
	if err != nil {
		t.Fatal("Error writing to journal", err)
	}
	time.Sleep(time.Second)

	// ... and try to read it back piece by piece via a small buffer
	finalBuff := new(bytes.Buffer)
	var e error
	for c := -1; c != 0 && e == nil; {
		smallBuf := make([]byte, 5)
		c, e = r.Read(smallBuf)
		if c > len(smallBuf) {
			t.Fatalf("Got unexpected read length: %d vs %d", c, len(smallBuf))
		}
		_, _ = finalBuff.Write(smallBuf)
	}
	b := finalBuff.String()
	got := strings.Split(b, delim)
	if len(got) != 3 {
		t.Fatalf("Got unexpected entry %s", b)
	}
	if got[1] != strings.Trim(want, delim) {
		t.Fatalf("Got unexpected message %s", got[1])
	}
}

func TestJournalGetUniqueValues(t *testing.T) {
	j, err := NewJournal()
	if err != nil {
		t.Fatal(err)
	}

	defer j.Close()

	uniqueString := generateRandomField(20)
	testEntries := []string{"A", "B", "C", "D"}
	for _, v := range testEntries {
		err := journal.Send("TEST: "+uniqueString, journal.PriInfo, map[string]string{uniqueString: v})
		if err != nil {
			t.Fatal(err)
		}
	}

	// TODO: add proper `waitOnMatch` function which should wait for journal entry with filter to commit.
	time.Sleep(time.Millisecond * 500)

	values, err := j.GetUniqueValues(uniqueString)
	if err != nil {
		t.Fatal(err)
	}

	if len(values) != len(testEntries) {
		t.Fatalf("Expect %d entries. Got %d", len(testEntries), len(values))
	}

	if !contains(values, "A") || !contains(values, "B") || !contains(values, "C") || !contains(values, "D") {
		t.Fatalf("Expect 4 values for %s field: A,B,C,D. Got %s", uniqueString, values)
	}
}

func contains(s []string, v string) bool {
	for _, entry := range s {
		if entry == v {
			return true
		}
	}
	return false
}

func generateRandomField(n int) string {
	letters := []rune("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	s := make([]rune, n)
	rand.Seed(time.Now().UnixNano())
	for i := range s {
		s[i] = letters[rand.Intn(len(letters))]
	}
	return string(s)
}
