package pagination

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud/testhelper"
)

// SinglePage sample and test cases.

type SinglePageResult struct {
	SinglePageBase
}

func (r SinglePageResult) IsEmpty() (bool, error) {
	is, err := ExtractSingleInts(r)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

func ExtractSingleInts(page Page) ([]int, error) {
	var response struct {
		Ints []int `mapstructure:"ints"`
	}

	err := mapstructure.Decode(page.(SinglePageResult).Body, &response)
	if err != nil {
		return nil, err
	}

	return response.Ints, nil
}

func setupSinglePaged() Pager {
	testhelper.SetupHTTP()
	client := createClient()

	testhelper.Mux.HandleFunc("/only", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `{ "ints": [1, 2, 3] }`)
	})

	createPage := func(r PageResult) Page {
		return SinglePageResult{SinglePageBase(r)}
	}

	return NewPager(client, testhelper.Server.URL+"/only", createPage)
}

func TestEnumerateSinglePaged(t *testing.T) {
	callCount := 0
	pager := setupSinglePaged()
	defer testhelper.TeardownHTTP()

	err := pager.EachPage(func(page Page) (bool, error) {
		callCount++

		expected := []int{1, 2, 3}
		actual, err := ExtractSingleInts(page)
		testhelper.AssertNoErr(t, err)
		testhelper.CheckDeepEquals(t, expected, actual)
		return true, nil
	})
	testhelper.CheckNoErr(t, err)
	testhelper.CheckEquals(t, 1, callCount)
}

func TestAllPagesSingle(t *testing.T) {
	pager := setupSinglePaged()
	defer testhelper.TeardownHTTP()

	page, err := pager.AllPages()
	testhelper.AssertNoErr(t, err)

	expected := []int{1, 2, 3}
	actual, err := ExtractSingleInts(page)
	testhelper.AssertNoErr(t, err)
	testhelper.CheckDeepEquals(t, expected, actual)
}
