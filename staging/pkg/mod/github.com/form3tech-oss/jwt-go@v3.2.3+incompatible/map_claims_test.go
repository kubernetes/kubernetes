package jwt

import "testing"

func Test_mapClaims_list_aud(t *testing.T) {
	mapClaims := MapClaims{
		"aud": []string{"foo"},
	}
	want := true
	got := mapClaims.VerifyAudience("foo", true)

	if want != got {
		t.Fatalf("Failed to verify claims, wanted: %v got %v", want, got)
	}
}
func Test_mapClaims_list_interface_aud(t *testing.T) {
	mapClaims := MapClaims{
		"aud": []interface{}{"foo"},
	}
	want := true
	got := mapClaims.VerifyAudience("foo", true)

	if want != got {
		t.Fatalf("Failed to verify claims, wanted: %v got %v", want, got)
	}
}
func Test_mapClaims_string_aud(t *testing.T) {
	mapClaims := MapClaims{
		"aud": "foo",
	}
	want := true
	got := mapClaims.VerifyAudience("foo", true)

	if want != got {
		t.Fatalf("Failed to verify claims, wanted: %v got %v", want, got)
	}
}

func Test_mapClaims_list_aud_no_match(t *testing.T) {
	mapClaims := MapClaims{
		"aud": []string{"bar"},
	}
	want := false
	got := mapClaims.VerifyAudience("foo", true)

	if want != got {
		t.Fatalf("Failed to verify claims, wanted: %v got %v", want, got)
	}
}
func Test_mapClaims_string_aud_fail(t *testing.T) {
	mapClaims := MapClaims{
		"aud": "bar",
	}
	want := false
	got := mapClaims.VerifyAudience("foo", true)

	if want != got {
		t.Fatalf("Failed to verify claims, wanted: %v got %v", want, got)
	}
}

func Test_mapClaims_string_aud_no_claim(t *testing.T) {
	mapClaims := MapClaims{}
	want := false
	got := mapClaims.VerifyAudience("foo", true)

	if want != got {
		t.Fatalf("Failed to verify claims, wanted: %v got %v", want, got)
	}
}

func Test_mapClaims_string_aud_no_claim_not_required(t *testing.T) {
	mapClaims := MapClaims{}
	want := false
	got := mapClaims.VerifyAudience("foo", false)

	if want != got {
		t.Fatalf("Failed to verify claims, wanted: %v got %v", want, got)
	}
}
