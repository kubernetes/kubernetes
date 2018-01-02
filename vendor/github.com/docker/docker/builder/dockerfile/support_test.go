package dockerfile

import "testing"

type testCase struct {
	name       string
	args       []string
	attributes map[string]bool
	expected   []string
}

func initTestCases() []testCase {
	testCases := []testCase{}

	testCases = append(testCases, testCase{
		name:       "empty args",
		args:       []string{},
		attributes: make(map[string]bool),
		expected:   []string{},
	})

	jsonAttributes := make(map[string]bool)
	jsonAttributes["json"] = true

	testCases = append(testCases, testCase{
		name:       "json attribute with one element",
		args:       []string{"foo"},
		attributes: jsonAttributes,
		expected:   []string{"foo"},
	})

	testCases = append(testCases, testCase{
		name:       "json attribute with two elements",
		args:       []string{"foo", "bar"},
		attributes: jsonAttributes,
		expected:   []string{"foo", "bar"},
	})

	testCases = append(testCases, testCase{
		name:       "no attributes",
		args:       []string{"foo", "bar"},
		attributes: nil,
		expected:   []string{"foo bar"},
	})

	return testCases
}

func TestHandleJSONArgs(t *testing.T) {
	testCases := initTestCases()

	for _, test := range testCases {
		arguments := handleJSONArgs(test.args, test.attributes)

		if len(arguments) != len(test.expected) {
			t.Fatalf("In test \"%s\": length of returned slice is incorrect. Expected: %d, got: %d", test.name, len(test.expected), len(arguments))
		}

		for i := range test.expected {
			if arguments[i] != test.expected[i] {
				t.Fatalf("In test \"%s\": element as position %d is incorrect. Expected: %s, got: %s", test.name, i, test.expected[i], arguments[i])
			}
		}
	}
}
