package units

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func testParse(t *testing.T, suffix string, b int64, bi int64) {
	n, err := Parse("10" + suffix)
	require.NoError(t, err, "Parse")
	require.Equal(t, int64(10*bi), n, "Parse")

	n, err = Parse("10" + " " + strings.ToLower(suffix))
	require.NoError(t, err, "Parse")
	require.Equal(t, int64(10*bi), n, "Parse")

	n, err = Parse("10" + " " + strings.ToUpper(suffix))
	require.NoError(t, err, "Parse")
	require.Equal(t, int64(10*bi), n, "Parse")

	n, err = Parse("1  0" + suffix)
	require.Error(t, err, "Parse")

	n, err = Parse("10" + suffix + "z")
	require.Error(t, err, "Parse")

	if len(suffix) == 0 {
		return
	}

	/*
		// No support for Mega
		n, err = Parse("10" + strings.ToLower(suffix) + "b")
		require.NoError(t, err, "Parse")
		require.Equal(t, int64(10*b), n, "Parse")
		n, err = Parse("10" + strings.ToUpper(suffix) + "B")
		require.NoError(t, err, "Parse")
		require.Equal(t, int64(10*b), n, "Parse")

		n, err = Parse("10" + " " + strings.ToLower(suffix) + "b")
		require.NoError(t, err, "Parse")
		require.Equal(t, int64(10*b), n, "Parse")

		n, err = Parse("10" + " " + strings.ToUpper(suffix) + "B")
		require.NoError(t, err, "Parse")
		require.Equal(t, int64(10*b), n, "Parse")
	*/
}

func TestParse(t *testing.T) {
	testParse(t, "k", 1000, 1024)
	testParse(t, "m", 1000*1000, 1024*1024)
	testParse(t, "", 1000, 1024*1024*1024)
	testParse(t, "g", 1000*1000*1000, 1024*1024*1024)
	testParse(t, "t", 1000*1000*1000*1000, 1024*1024*1024*1024)
	testParse(t, "p", 1000*1000*1000*1000*1000, 1024*1024*1024*1024*1024)
}
