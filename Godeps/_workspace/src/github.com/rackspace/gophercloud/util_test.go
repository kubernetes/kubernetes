package gophercloud

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
)

func TestWaitFor(t *testing.T) {
	err := WaitFor(5, func() (bool, error) {
		return true, nil
	})
	th.CheckNoErr(t, err)
}

func TestNormalizeURL(t *testing.T) {
	urls := []string{
		"NoSlashAtEnd",
		"SlashAtEnd/",
	}
	expected := []string{
		"NoSlashAtEnd/",
		"SlashAtEnd/",
	}
	for i := 0; i < len(expected); i++ {
		th.CheckEquals(t, expected[i], NormalizeURL(urls[i]))
	}

}

func TestNormalizePathURL(t *testing.T) {
	baseDir, _ := os.Getwd()

	rawPath := "template.yaml"
	basePath, _ := filepath.Abs(".")
	result, _ := NormalizePathURL(basePath, rawPath)
	expected := strings.Join([]string{"file:/", filepath.ToSlash(baseDir), "template.yaml"}, "/")
	th.CheckEquals(t, expected, result)

	rawPath = "http://www.google.com"
	basePath, _ = filepath.Abs(".")
	result, _ = NormalizePathURL(basePath, rawPath)
	expected = "http://www.google.com"
	th.CheckEquals(t, expected, result)

	rawPath = "very/nested/file.yaml"
	basePath, _ = filepath.Abs(".")
	result, _ = NormalizePathURL(basePath, rawPath)
	expected = strings.Join([]string{"file:/", filepath.ToSlash(baseDir), "very/nested/file.yaml"}, "/")
	th.CheckEquals(t, expected, result)

	rawPath = "very/nested/file.yaml"
	basePath = "http://www.google.com"
	result, _ = NormalizePathURL(basePath, rawPath)
	expected = "http://www.google.com/very/nested/file.yaml"
	th.CheckEquals(t, expected, result)

	rawPath = "very/nested/file.yaml/"
	basePath = "http://www.google.com/"
	result, _ = NormalizePathURL(basePath, rawPath)
	expected = "http://www.google.com/very/nested/file.yaml"
	th.CheckEquals(t, expected, result)

	rawPath = "very/nested/file.yaml"
	basePath = "http://www.google.com/even/more"
	result, _ = NormalizePathURL(basePath, rawPath)
	expected = "http://www.google.com/even/more/very/nested/file.yaml"
	th.CheckEquals(t, expected, result)

	rawPath = "very/nested/file.yaml"
	basePath = strings.Join([]string{"file:/", filepath.ToSlash(baseDir), "only/file/even/more"}, "/")
	result, _ = NormalizePathURL(basePath, rawPath)
	expected = strings.Join([]string{"file:/", filepath.ToSlash(baseDir), "only/file/even/more/very/nested/file.yaml"}, "/")
	th.CheckEquals(t, expected, result)

	rawPath = "very/nested/file.yaml/"
	basePath = strings.Join([]string{"file:/", filepath.ToSlash(baseDir), "only/file/even/more"}, "/")
	result, _ = NormalizePathURL(basePath, rawPath)
	expected = strings.Join([]string{"file:/", filepath.ToSlash(baseDir), "only/file/even/more/very/nested/file.yaml"}, "/")
	th.CheckEquals(t, expected, result)

}
