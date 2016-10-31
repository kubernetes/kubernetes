package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestImages_List(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/images", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"images":[{"id":1},{"id":2}]}`)
	})

	images, _, err := client.Images.List(nil)
	if err != nil {
		t.Errorf("Images.List returned error: %v", err)
	}

	expected := []Image{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(images, expected) {
		t.Errorf("Images.List returned %+v, expected %+v", images, expected)
	}
}

func TestImages_ListDistribution(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/images", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		expected := "distribution"
		actual := r.URL.Query().Get("type")
		if actual != expected {
			t.Errorf("'type' query = %v, expected %v", actual, expected)
		}
		fmt.Fprint(w, `{"images":[{"id":1},{"id":2}]}`)
	})

	images, _, err := client.Images.ListDistribution(nil)
	if err != nil {
		t.Errorf("Images.ListDistribution returned error: %v", err)
	}

	expected := []Image{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(images, expected) {
		t.Errorf("Images.ListDistribution returned %+v, expected %+v", images, expected)
	}
}

func TestImages_ListApplication(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/images", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		expected := "application"
		actual := r.URL.Query().Get("type")
		if actual != expected {
			t.Errorf("'type' query = %v, expected %v", actual, expected)
		}
		fmt.Fprint(w, `{"images":[{"id":1},{"id":2}]}`)
	})

	images, _, err := client.Images.ListApplication(nil)
	if err != nil {
		t.Errorf("Images.ListApplication returned error: %v", err)
	}

	expected := []Image{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(images, expected) {
		t.Errorf("Images.ListApplication returned %+v, expected %+v", images, expected)
	}
}

func TestImages_ListUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/images", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		expected := "true"
		actual := r.URL.Query().Get("private")
		if actual != expected {
			t.Errorf("'private' query = %v, expected %v", actual, expected)
		}

		fmt.Fprint(w, `{"images":[{"id":1},{"id":2}]}`)
	})

	images, _, err := client.Images.ListUser(nil)
	if err != nil {
		t.Errorf("Images.ListUser returned error: %v", err)
	}

	expected := []Image{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(images, expected) {
		t.Errorf("Images.ListUser returned %+v, expected %+v", images, expected)
	}
}

func TestImages_ListImagesMultiplePages(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/images", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"images": [{"id":1},{"id":2}], "links":{"pages":{"next":"http://example.com/v2/images/?page=2"}}}`)
	})

	_, resp, err := client.Images.List(&ListOptions{Page: 2})
	if err != nil {
		t.Fatal(err)
	}
	checkCurrentPage(t, resp, 1)
}

func TestImages_RetrievePageByNumber(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"images": [{"id":1},{"id":2}],
		"links":{
			"pages":{
				"next":"http://example.com/v2/images/?page=3",
				"prev":"http://example.com/v2/images/?page=1",
				"last":"http://example.com/v2/images/?page=3",
				"first":"http://example.com/v2/images/?page=1"
			}
		}
	}`

	mux.HandleFunc("/v2/images", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	opt := &ListOptions{Page: 2}
	_, resp, err := client.Images.List(opt)
	if err != nil {
		t.Fatal(err)
	}

	checkCurrentPage(t, resp, 2)
}

func TestImages_GetImageByID(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/images/12345", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"image":{"id":12345}}`)
	})

	images, _, err := client.Images.GetByID(12345)
	if err != nil {
		t.Errorf("Image.GetByID returned error: %v", err)
	}

	expected := &Image{ID: 12345}
	if !reflect.DeepEqual(images, expected) {
		t.Errorf("Images.GetByID returned %+v, expected %+v", images, expected)
	}
}

func TestImages_GetImageBySlug(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/images/ubuntu", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"image":{"id":12345}}`)
	})

	images, _, err := client.Images.GetBySlug("ubuntu")
	if err != nil {
		t.Errorf("Image.GetBySlug returned error: %v", err)
	}

	expected := &Image{ID: 12345}
	if !reflect.DeepEqual(images, expected) {
		t.Errorf("Images.Get returned %+v, expected %+v", images, expected)
	}
}

func TestImages_Update(t *testing.T) {
	setup()
	defer teardown()

	updateRequest := &ImageUpdateRequest{
		Name: "name",
	}

	mux.HandleFunc("/v2/images/12345", func(w http.ResponseWriter, r *http.Request) {
		expected := map[string]interface{}{
			"name": "name",
		}

		var v map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		if !reflect.DeepEqual(v, expected) {
			t.Errorf("Request body = %#v, expected %#v", v, expected)
		}

		fmt.Fprintf(w, `{"image":{"id":1}}`)
	})

	image, _, err := client.Images.Update(12345, updateRequest)
	if err != nil {
		t.Errorf("Images.Update returned error: %v", err)
	} else {
		if id := image.ID; id != 1 {
			t.Errorf("expected id '%d', received '%d'", 1, id)
		}
	}
}

func TestImages_Destroy(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/images/12345", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Images.Delete(12345)
	if err != nil {
		t.Errorf("Image.Delete returned error: %v", err)
	}
}

func TestImage_String(t *testing.T) {
	image := &Image{
		ID:           1,
		Name:         "Image",
		Type:         "snapshot",
		Distribution: "Ubuntu",
		Slug:         "image",
		Public:       true,
		Regions:      []string{"one", "two"},
		MinDiskSize:  20,
		Created:      "2013-11-27T09:24:55Z",
	}

	stringified := image.String()
	expected := `godo.Image{ID:1, Name:"Image", Type:"snapshot", Distribution:"Ubuntu", Slug:"image", Public:true, Regions:["one" "two"], MinDiskSize:20, Created:"2013-11-27T09:24:55Z"}`
	if expected != stringified {
		t.Errorf("Image.String returned %+v, expected %+v", stringified, expected)
	}
}
