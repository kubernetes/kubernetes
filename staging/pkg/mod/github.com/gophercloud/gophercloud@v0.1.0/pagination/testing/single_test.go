package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/pagination"
	"github.com/gophercloud/gophercloud/testhelper"
)

// SinglePage sample and test cases.

type SinglePageResult struct {
	pagination.SinglePageBase
}

func (r SinglePageResult) IsEmpty() (bool, error) {
	is, err := ExtractSingleInts(r)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

func ExtractSingleInts(r pagination.Page) ([]int, error) {
	var s struct {
		Ints []int `json:"ints"`
	}
	err := (r.(SinglePageResult)).ExtractInto(&s)
	return s.Ints, err
}

func setupSinglePaged() pagination.Pager {
	testhelper.SetupHTTP()
	client := createClient()

	testhelper.Mux.HandleFunc("/only", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `{ "ints": [1, 2, 3] }`)
	})

	createPage := func(r pagination.PageResult) pagination.Page {
		return SinglePageResult{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, testhelper.Server.URL+"/only", createPage)
}

func TestEnumerateSinglePaged(t *testing.T) {
	callCount := 0
	pager := setupSinglePaged()
	defer testhelper.TeardownHTTP()

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
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
