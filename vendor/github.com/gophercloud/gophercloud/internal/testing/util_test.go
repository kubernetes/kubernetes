package testing

import (
	"reflect"
	"testing"

	"github.com/gophercloud/gophercloud/internal"
)

func TestRemainingKeys(t *testing.T) {
	type User struct {
		UserID    string `json:"user_id"`
		Username  string `json:"username"`
		Location  string `json:"-"`
		CreatedAt string `json:"-"`
		Status    string
		IsAdmin   bool
	}

	userResponse := map[string]interface{}{
		"user_id":      "abcd1234",
		"username":     "jdoe",
		"location":     "Hawaii",
		"created_at":   "2017-06-08T02:49:03.000000",
		"status":       "active",
		"is_admin":     "true",
		"custom_field": "foo",
	}

	expected := map[string]interface{}{
		"created_at":   "2017-06-08T02:49:03.000000",
		"is_admin":     "true",
		"custom_field": "foo",
	}

	actual := internal.RemainingKeys(User{}, userResponse)

	isEqual := reflect.DeepEqual(expected, actual)
	if !isEqual {
		t.Fatalf("expected %s but got %s", expected, actual)
	}
}
