package eventstream

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"testing"
)

type testCase struct {
	Name    string
	Encoded []byte
	Decoded decodedMessage
}

type testErrorCase struct {
	Name    string
	Encoded []byte
	Err     string
}

type rawTestCase struct {
	Name             string
	Encoded, Decoded []byte
}

func readRawTestCases(root, class string) (map[string]rawTestCase, error) {
	encoded, err := readTests(filepath.Join(root, "encoded", class))
	if err != nil {
		return nil, err
	}

	decoded, err := readTests(filepath.Join(root, "decoded", class))
	if err != nil {
		return nil, err
	}

	if len(encoded) == 0 {
		return nil, fmt.Errorf("expect encoded cases, found none")
	}

	if len(encoded) != len(decoded) {
		return nil, fmt.Errorf("encoded and decoded sets different")
	}

	rawCases := map[string]rawTestCase{}
	for name, encData := range encoded {
		decData, ok := decoded[name]
		if !ok {
			return nil, fmt.Errorf("encoded %q case not found in decoded set", name)
		}

		rawCases[name] = rawTestCase{
			Name:    name,
			Encoded: encData,
			Decoded: decData,
		}
	}

	return rawCases, nil
}

func readNegativeTests(root string) ([]testErrorCase, error) {
	rawCases, err := readRawTestCases(root, "negative")
	if err != nil {
		return nil, err
	}

	cases := make([]testErrorCase, 0, len(rawCases))
	for name, rawCase := range rawCases {
		cases = append(cases, testErrorCase{
			Name:    name,
			Encoded: rawCase.Encoded,
			Err:     string(rawCase.Decoded),
		})
	}

	return cases, nil
}

func readPositiveTests(root string) ([]testCase, error) {
	rawCases, err := readRawTestCases(root, "positive")
	if err != nil {
		return nil, err
	}

	cases := make([]testCase, 0, len(rawCases))
	for name, rawCase := range rawCases {

		var dec decodedMessage
		if err := json.Unmarshal(rawCase.Decoded, &dec); err != nil {
			return nil, fmt.Errorf("failed to decode %q, %v", name, err)
		}

		cases = append(cases, testCase{
			Name:    name,
			Encoded: rawCase.Encoded,
			Decoded: dec,
		})
	}

	return cases, nil
}

func readTests(root string) (map[string][]byte, error) {
	items, err := ioutil.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("failed to read test suite %q dirs, %v", root, err)
	}

	cases := map[string][]byte{}
	for _, item := range items {
		if item.IsDir() {
			continue
		}

		filename := filepath.Join(root, item.Name())
		data, err := ioutil.ReadFile(filename)
		if err != nil {
			return nil, fmt.Errorf("failed to read test_data file %q, %v", filename, err)
		}

		cases[item.Name()] = data
	}

	return cases, nil
}

func compareLines(t *testing.T, a, b []byte) bool {
	as := bufio.NewScanner(bytes.NewBuffer(a))
	bs := bufio.NewScanner(bytes.NewBuffer(b))

	var failed bool
	for {
		if ab, bb := as.Scan(), bs.Scan(); ab != bb {
			t.Errorf("expect a & b to have same number of lines")
			return false
		} else if !ab {
			break
		}

		if v1, v2 := as.Text(), bs.Text(); v1 != v2 {
			t.Errorf("expect %q to be %q", v1, v2)
			failed = true
		}
	}

	return !failed
}
