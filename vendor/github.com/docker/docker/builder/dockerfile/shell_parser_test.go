package dockerfile

import (
	"bufio"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShellParser4EnvVars(t *testing.T) {
	fn := "envVarTest"
	lineCount := 0

	file, err := os.Open(fn)
	assert.NoError(t, err)
	defer file.Close()

	shlex := NewShellLex('\\')
	scanner := bufio.NewScanner(file)
	envs := []string{"PWD=/home", "SHELL=bash", "KOREAN=한국어"}
	for scanner.Scan() {
		line := scanner.Text()
		lineCount++

		// Trim comments and blank lines
		i := strings.Index(line, "#")
		if i >= 0 {
			line = line[:i]
		}
		line = strings.TrimSpace(line)

		if line == "" {
			continue
		}

		words := strings.Split(line, "|")
		assert.Len(t, words, 3)

		platform := strings.TrimSpace(words[0])
		source := strings.TrimSpace(words[1])
		expected := strings.TrimSpace(words[2])

		// Key W=Windows; A=All; U=Unix
		if platform != "W" && platform != "A" && platform != "U" {
			t.Fatalf("Invalid tag %s at line %d of %s. Must be W, A or U", platform, lineCount, fn)
		}

		if ((platform == "W" || platform == "A") && runtime.GOOS == "windows") ||
			((platform == "U" || platform == "A") && runtime.GOOS != "windows") {
			newWord, err := shlex.ProcessWord(source, envs)
			if expected == "error" {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, newWord, expected)
			}
		}
	}
}

func TestShellParser4Words(t *testing.T) {
	fn := "wordsTest"

	file, err := os.Open(fn)
	if err != nil {
		t.Fatalf("Can't open '%s': %s", err, fn)
	}
	defer file.Close()

	shlex := NewShellLex('\\')
	envs := []string{}
	scanner := bufio.NewScanner(file)
	lineNum := 0
	for scanner.Scan() {
		line := scanner.Text()
		lineNum = lineNum + 1

		if strings.HasPrefix(line, "#") {
			continue
		}

		if strings.HasPrefix(line, "ENV ") {
			line = strings.TrimLeft(line[3:], " ")
			envs = append(envs, line)
			continue
		}

		words := strings.Split(line, "|")
		if len(words) != 2 {
			t.Fatalf("Error in '%s'(line %d) - should be exactly one | in: %q", fn, lineNum, line)
		}
		test := strings.TrimSpace(words[0])
		expected := strings.Split(strings.TrimLeft(words[1], " "), ",")

		result, err := shlex.ProcessWords(test, envs)

		if err != nil {
			result = []string{"error"}
		}

		if len(result) != len(expected) {
			t.Fatalf("Error on line %d. %q was suppose to result in %q, but got %q instead", lineNum, test, expected, result)
		}
		for i, w := range expected {
			if w != result[i] {
				t.Fatalf("Error on line %d. %q was suppose to result in %q, but got %q instead", lineNum, test, expected, result)
			}
		}
	}
}

func TestGetEnv(t *testing.T) {
	sw := &shellWord{envs: nil}

	sw.envs = []string{}
	if sw.getEnv("foo") != "" {
		t.Fatal("2 - 'foo' should map to ''")
	}

	sw.envs = []string{"foo"}
	if sw.getEnv("foo") != "" {
		t.Fatal("3 - 'foo' should map to ''")
	}

	sw.envs = []string{"foo="}
	if sw.getEnv("foo") != "" {
		t.Fatal("4 - 'foo' should map to ''")
	}

	sw.envs = []string{"foo=bar"}
	if sw.getEnv("foo") != "bar" {
		t.Fatal("5 - 'foo' should map to 'bar'")
	}

	sw.envs = []string{"foo=bar", "car=hat"}
	if sw.getEnv("foo") != "bar" {
		t.Fatal("6 - 'foo' should map to 'bar'")
	}
	if sw.getEnv("car") != "hat" {
		t.Fatal("7 - 'car' should map to 'hat'")
	}

	// Make sure we grab the first 'car' in the list
	sw.envs = []string{"foo=bar", "car=hat", "car=bike"}
	if sw.getEnv("car") != "hat" {
		t.Fatal("8 - 'car' should map to 'hat'")
	}
}
