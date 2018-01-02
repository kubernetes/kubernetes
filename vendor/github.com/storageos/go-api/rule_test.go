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

func TestRuleList(t *testing.T) {
	rulesData := `[
    {
        "active": true,
        "description": "",
        "id": "17490903-4876-ece8-5b07-59c53e75315c",
        "labels": {
            "storageos.driver": "filesystem"
        },
        "name": "default driver",
        "rule_action": "add",
		"selector": "storageos.driver notin (disk, filesystem)",
        "weight": 0
    }
]`

	var expected []*types.Rule
	if err := json.Unmarshal([]byte(rulesData), &expected); err != nil {
		t.Fatal(err)
	}

	client := newTestClient(&FakeRoundTripper{message: rulesData, status: http.StatusOK})
	rules, err := client.RuleList(types.ListOptions{})
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(rules, expected) {
		t.Errorf("Rules: Wrong return value. Want %#v. Got %#v.", expected, rules)
	}
}

func TestRuleCreate(t *testing.T) {
	body := `{
				"active": true,
				"description": "",
				"id": "17490903-4876-ece8-5b07-59c53e75315c",
				"labels": {
						"storageos.driver": "filesystem"
				},
				"name": "default driver",				
				"rule_action": "add",
				"selector": "storageos.driver notin (disk, filesystem)",        
				"weight": 0
		}`
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	rule, err := client.RuleCreate(
		types.RuleCreateOptions{
			Name:        "unit01",
			Description: "Unit test rule",
			Active:      true,
			Weight:      5,
			RuleAction:  "add",
			Selector:    "storageos.driver notin (disk, filesystem)",
			Labels: map[string]string{
				"foo": "bar",
			},
			Context: context.Background(),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if rule == nil {
		t.Fatalf("RuleCreate(): Wrong return value. Wanted rule. Got %v.", rule)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("RuleCreate(): Wrong HTTP method. Want %s. Got %s.", expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(RuleAPIPrefix, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("RuleCreate(): Wrong request path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
}

func TestRule(t *testing.T) {
	body := `{
		"name": "unit01",
		"description": "Unit test rule",
		"active": true,
		"weight": 5
	}`
	var expected types.Rule
	if err := json.Unmarshal([]byte(body), &expected); err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	name := "tardis"
	namespace := "galaxy"
	rule, err := client.Rule(namespace, name)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(rule, &expected) {
		t.Errorf("Rule: Wrong return value. Want %#v. Got %#v.", expected, rule)
	}
	req := fakeRT.requests[0]
	expectedMethod := "GET"
	if req.Method != expectedMethod {
		t.Errorf("InspectRule(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	path, _ := namespacedRefPath(namespace, RuleAPIPrefix, name)
	u, _ := url.Parse(client.getAPIPath(path, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("RuleCreate(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestRuleDelete(t *testing.T) {
	name := "test"
	namespace := "projA"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	err := client.RuleDelete(
		types.DeleteOptions{
			Name:      name,
			Namespace: namespace,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "DELETE"
	if req.Method != expectedMethod {
		t.Errorf("RuleDelete(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	path, _ := namespacedRefPath(namespace, RuleAPIPrefix, name)
	u, _ := url.Parse(client.getAPIPath(path, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("RuleDelete(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestRuleDeleteNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such rule", status: http.StatusNotFound})
	err := client.RuleDelete(
		types.DeleteOptions{
			Name:      "badname",
			Namespace: "badnamespace",
		},
	)
	if err != ErrNoSuchRule {
		t.Errorf("RuleDelete(%q): wrong error. Want %#v. Got %#v.", "badname", ErrNoSuchRule, err)
	}
}

func TestRuleDeleteInUse(t *testing.T) {
	name := "test"
	namespace := "projA"
	client := newTestClient(&FakeRoundTripper{message: "rule in use and cannot be removed", status: http.StatusConflict})
	err := client.RuleDelete(
		types.DeleteOptions{
			Name:      name,
			Namespace: namespace,
		},
	)
	if err != ErrRuleInUse {
		t.Errorf("RuleDelete(%q): wrong error. Want %#v. Got %#v.", name, ErrRuleInUse, err)
	}
}

func TestRuleDeleteForce(t *testing.T) {
	name := "testdelete"
	namespace := "projA"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	if err := client.RuleDelete(types.DeleteOptions{Name: name, Namespace: namespace, Force: true}); err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	vals := req.URL.Query()
	if len(vals) == 0 {
		t.Error("RuleDelete: query string empty. Expected force=1.")
	}
	force := vals.Get("force")
	if force != "1" {
		t.Errorf("RuleDelete(%q): Force not set. Want %q. Got %q.", name, "1", force)
	}
}
