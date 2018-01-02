package storageos

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"github.com/storageos/go-api/types"
)

func TestTemplateList(t *testing.T) {
	templatesData := `[
    {
        "active": true,
        "auto_increment": true,
        "description": "Volume template - DEV",
        "format": "vol-{location}-{application}",
        "id": "7ada08e0-53f2-dd40-4727-1015c752836c",
        "name": "vol-loc-app",
        "object_types": [
            "volume"
        ],
        "padding": false,
        "padding_length": 0,
        "tags": [
            "dev"
        ],
        "weight": 0
    },
    {
        "active": true,
        "auto_increment": true,
        "description": "Volume template",
        "format": "vol-{location}-{application}-{environment}",
        "id": "ab7c481b-74c2-dbe0-0df9-58a3e446fa46",
        "name": "vol-loc-app-env",
        "object_types": [
            "volume"
        ],
        "padding": false,
        "padding_length": 0,
        "tags": [],
        "weight": 5
    }
]`

	var expected []types.Template
	if err := json.Unmarshal([]byte(templatesData), &expected); err != nil {
		t.Fatal(err)
	}

	client := newTestClient(&FakeRoundTripper{message: templatesData, status: http.StatusOK})
	templates, err := client.TemplateList(types.ListOptions{})
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(templates, expected) {
		t.Errorf("Templates: Wrong return value. Want %#v. Got %#v.", expected, templates)
	}
}

func TestTemplateCreate(t *testing.T) {
	message := "\"ef897b9f-0b47-08ee-b669-0a2057df981c\""
	fakeRT := &FakeRoundTripper{message: message, status: http.StatusOK}
	client := newTestClient(fakeRT)
	id, err := client.TemplateCreate(
		types.TemplateCreateOptions{
			Name:          "unit01",
			Description:   "Unit test template",
			Format:        "vol-{location}-{application}",
			AutoIncrement: true,
			Padding:       true,
			PaddingLength: 3,
			Active:        true,
			Weight:        5,
			ObjectTypes:   []string{"volume"},
			Labels: map[string]string{
				"foo": "bar",
			},
			Context: context.Background(),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(id) != 36 {
		t.Errorf("TemplateCreate: Wrong return value. Wanted 34 character UUID. Got %d. (%s)", len(id), id)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("TemplateCreate(): Wrong HTTP method. Want %s. Got %s.", expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(TemplateAPIPrefix, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("TemplateCreate(): Wrong request path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
}

func TestTemplate(t *testing.T) {
	body := `{
		"name": "unit01",
		"description": "Unit test template",
		"active": true,
		"weight": 5
	}`
	var expected types.Template
	if err := json.Unmarshal([]byte(body), &expected); err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	name := "tardis"
	template, err := client.Template(name)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(template, &expected) {
		t.Errorf("Template: Wrong return value. Want %#v. Got %#v.", expected, template)
	}
	req := fakeRT.requests[0]
	expectedMethod := "GET"
	if req.Method != expectedMethod {
		t.Errorf("InspectTemplate(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(TemplateAPIPrefix+"/"+name, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("TemplateCreate(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestTemplateDelete(t *testing.T) {
	name := "test"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	if err := client.TemplateDelete(name); err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "DELETE"
	if req.Method != expectedMethod {
		t.Errorf("TemplateDelete(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(TemplateAPIPrefix+"/"+name, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("TemplateDelete(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestTemplateDeleteNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such template", status: http.StatusNotFound})
	if err := client.TemplateDelete("test:"); err != ErrNoSuchTemplate {
		t.Errorf("TemplateDelete: wrong error. Want %#v. Got %#v.", ErrNoSuchTemplate, err)
	}
}

func TestTemplateDeleteInUse(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "template in use and cannot be removed", status: http.StatusConflict})
	if err := client.TemplateDelete("test:"); err != ErrTemplateInUse {
		t.Errorf("TemplateDelete: wrong error. Want %#v. Got %#v.", ErrVolumeInUse, err)
	}
}
