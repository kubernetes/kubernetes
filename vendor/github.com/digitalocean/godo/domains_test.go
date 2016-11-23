package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestDomains_ListDomains(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/domains", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"domains": [{"name":"foo.com"},{"name":"bar.com"}]}`)
	})

	domains, _, err := client.Domains.List(nil)
	if err != nil {
		t.Errorf("Domains.List returned error: %v", err)
	}

	expected := []Domain{{Name: "foo.com"}, {Name: "bar.com"}}
	if !reflect.DeepEqual(domains, expected) {
		t.Errorf("Domains.List returned %+v, expected %+v", domains, expected)
	}
}

func TestDomains_ListDomainsMultiplePages(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/domains", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"domains": [{"id":1},{"id":2}], "links":{"pages":{"next":"http://example.com/v2/domains/?page=2"}}}`)
	})

	_, resp, err := client.Domains.List(nil)
	if err != nil {
		t.Fatal(err)
	}

	checkCurrentPage(t, resp, 1)
}

func TestDomains_RetrievePageByNumber(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"domains": [{"id":1},{"id":2}],
		"links":{
			"pages":{
				"next":"http://example.com/v2/domains/?page=3",
				"prev":"http://example.com/v2/domains/?page=1",
				"last":"http://example.com/v2/domains/?page=3",
				"first":"http://example.com/v2/domains/?page=1"
			}
		}
	}`

	mux.HandleFunc("/v2/domains", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	opt := &ListOptions{Page: 2}
	_, resp, err := client.Domains.List(opt)
	if err != nil {
		t.Fatal(err)
	}

	checkCurrentPage(t, resp, 2)
}

func TestDomains_GetDomain(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/domains/example.com", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"domain":{"name":"example.com"}}`)
	})

	domains, _, err := client.Domains.Get("example.com")
	if err != nil {
		t.Errorf("domain.Get returned error: %v", err)
	}

	expected := &Domain{Name: "example.com"}
	if !reflect.DeepEqual(domains, expected) {
		t.Errorf("domains.Get returned %+v, expected %+v", domains, expected)
	}
}

func TestDomains_Create(t *testing.T) {
	setup()
	defer teardown()

	createRequest := &DomainCreateRequest{
		Name:      "example.com",
		IPAddress: "127.0.0.1",
	}

	mux.HandleFunc("/v2/domains", func(w http.ResponseWriter, r *http.Request) {
		v := new(DomainCreateRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatal(err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, createRequest) {
			t.Errorf("Request body = %+v, expected %+v", v, createRequest)
		}

		fmt.Fprint(w, `{"domain":{"name":"example.com"}}`)
	})

	domain, _, err := client.Domains.Create(createRequest)
	if err != nil {
		t.Errorf("Domains.Create returned error: %v", err)
	}

	expected := &Domain{Name: "example.com"}
	if !reflect.DeepEqual(domain, expected) {
		t.Errorf("Domains.Create returned %+v, expected %+v", domain, expected)
	}
}

func TestDomains_Destroy(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/domains/example.com", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Domains.Delete("example.com")
	if err != nil {
		t.Errorf("Domains.Delete returned error: %v", err)
	}
}

func TestDomains_AllRecordsForDomainName(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/domains/example.com/records", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"domain_records":[{"id":1},{"id":2}]}`)
	})

	records, _, err := client.Domains.Records("example.com", nil)
	if err != nil {
		t.Errorf("Domains.List returned error: %v", err)
	}

	expected := []DomainRecord{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(records, expected) {
		t.Errorf("Domains.List returned %+v, expected %+v", records, expected)
	}
}

func TestDomains_AllRecordsForDomainName_PerPage(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/domains/example.com/records", func(w http.ResponseWriter, r *http.Request) {
		perPage := r.URL.Query().Get("per_page")
		if perPage != "2" {
			t.Fatalf("expected '2', got '%s'", perPage)
		}

		fmt.Fprint(w, `{"domain_records":[{"id":1},{"id":2}]}`)
	})

	dro := &ListOptions{PerPage: 2}
	records, _, err := client.Domains.Records("example.com", dro)
	if err != nil {
		t.Errorf("Domains.List returned error: %v", err)
	}

	expected := []DomainRecord{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(records, expected) {
		t.Errorf("Domains.List returned %+v, expected %+v", records, expected)
	}
}

func TestDomains_GetRecordforDomainName(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/domains/example.com/records/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"domain_record":{"id":1}}`)
	})

	record, _, err := client.Domains.Record("example.com", 1)
	if err != nil {
		t.Errorf("Domains.GetRecord returned error: %v", err)
	}

	expected := &DomainRecord{ID: 1}
	if !reflect.DeepEqual(record, expected) {
		t.Errorf("Domains.GetRecord returned %+v, expected %+v", record, expected)
	}
}

func TestDomains_DeleteRecordForDomainName(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/domains/example.com/records/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Domains.DeleteRecord("example.com", 1)
	if err != nil {
		t.Errorf("Domains.RecordDelete returned error: %v", err)
	}
}

func TestDomains_CreateRecordForDomainName(t *testing.T) {
	setup()
	defer teardown()

	createRequest := &DomainRecordEditRequest{
		Type:     "CNAME",
		Name:     "example",
		Data:     "@",
		Priority: 10,
		Port:     10,
		Weight:   10,
	}

	mux.HandleFunc("/v2/domains/example.com/records",
		func(w http.ResponseWriter, r *http.Request) {
			v := new(DomainRecordEditRequest)
			err := json.NewDecoder(r.Body).Decode(v)

			if err != nil {
				t.Fatalf("decode json: %v", err)
			}

			testMethod(t, r, "POST")
			if !reflect.DeepEqual(v, createRequest) {
				t.Errorf("Request body = %+v, expected %+v", v, createRequest)
			}

			fmt.Fprintf(w, `{"domain_record": {"id":1}}`)
		})

	record, _, err := client.Domains.CreateRecord("example.com", createRequest)
	if err != nil {
		t.Errorf("Domains.CreateRecord returned error: %v", err)
	}

	expected := &DomainRecord{ID: 1}
	if !reflect.DeepEqual(record, expected) {
		t.Errorf("Domains.CreateRecord returned %+v, expected %+v", record, expected)
	}
}

func TestDomains_EditRecordForDomainName(t *testing.T) {
	setup()
	defer teardown()

	editRequest := &DomainRecordEditRequest{
		Type:     "CNAME",
		Name:     "example",
		Data:     "@",
		Priority: 10,
		Port:     10,
		Weight:   10,
	}

	mux.HandleFunc("/v2/domains/example.com/records/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(DomainRecordEditRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "PUT")
		if !reflect.DeepEqual(v, editRequest) {
			t.Errorf("Request body = %+v, expected %+v", v, editRequest)
		}

		fmt.Fprintf(w, `{"id":1}`)
	})

	record, _, err := client.Domains.EditRecord("example.com", 1, editRequest)
	if err != nil {
		t.Errorf("Domains.EditRecord returned error: %v", err)
	}

	expected := &DomainRecord{ID: 1}
	if !reflect.DeepEqual(record, expected) {
		t.Errorf("Domains.EditRecord returned %+v, expected %+v", record, expected)
	}
}

func TestDomainRecord_String(t *testing.T) {
	record := &DomainRecord{
		ID:       1,
		Type:     "CNAME",
		Name:     "example",
		Data:     "@",
		Priority: 10,
		Port:     10,
		Weight:   10,
	}

	stringified := record.String()
	expected := `godo.DomainRecord{ID:1, Type:"CNAME", Name:"example", Data:"@", Priority:10, Port:10, Weight:10}`
	if expected != stringified {
		t.Errorf("DomainRecord.String returned %+v, expected %+v", stringified, expected)
	}
}

func TestDomainRecordEditRequest_String(t *testing.T) {
	record := &DomainRecordEditRequest{
		Type:     "CNAME",
		Name:     "example",
		Data:     "@",
		Priority: 10,
		Port:     10,
		Weight:   10,
	}

	stringified := record.String()
	expected := `godo.DomainRecordEditRequest{Type:"CNAME", Name:"example", Data:"@", Priority:10, Port:10, Weight:10}`
	if expected != stringified {
		t.Errorf("DomainRecordEditRequest.String returned %+v, expected %+v", stringified, expected)
	}
}
