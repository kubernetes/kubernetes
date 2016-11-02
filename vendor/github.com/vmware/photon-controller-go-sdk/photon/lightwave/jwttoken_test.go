package lightwave

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("JWTToken", func() {
	Describe("ParseTokenDetails", func() {
		Context("parsing token details", func() {
			It("parses tokens", func() {
				expected := &JWTToken{
					TokenId:    "CfPby7BAlaOI3Uj_TEq_UJOJmYXJiVOYuCYAXPw2l2U",
					Algorithm:  "RS256",
					Subject:    "ec-admin@esxcloud",
					Audience:   []string{"ec-admin@esxcloud", "rs_esxcloud"},
					Groups:     []string{"esxcloud\\ESXCloudAdmins", "esxcloud\\Everyone"},
					Issuer:     "https://10.146.64.238/openidconnect/esxcloud",
					IssuedAt:   1461795927,
					Expires:    1461803127,
					Scope:      "openid offline_access rs_esxcloud at_groups",
					TokenType:  "Bearer",
					TokenClass: "access_token",
					Tenant:     "esxcloud",
				}
				resp := ParseTokenDetails(
					"eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJlYy1hZG1pbkBlc3hjbG91ZCIsImF1ZCI6WyJlYy1hZG1pbkB" +
						"lc3hjbG91ZCIsInJzX2VzeGNsb3VkIl0sInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIHJzX2VzeGNsb3VkIGF0X" +
						"2dyb3VwcyIsImlzcyI6Imh0dHBzOlwvXC8xMC4xNDYuNjQuMjM4XC9vcGVuaWRjb25uZWN0XC9lc3hjbG91ZCIsImdyb3V" +
						"wcyI6WyJlc3hjbG91ZFxcRVNYQ2xvdWRBZG1pbnMiLCJlc3hjbG91ZFxcRXZlcnlvbmUiXSwidG9rZW5fY2xhc3MiOiJhY" +
						"2Nlc3NfdG9rZW4iLCJ0b2tlbl90eXBlIjoiQmVhcmVyIiwiZXhwIjoxNDYxODAzMTI3LCJpYXQiOjE0NjE3OTU5MjcsInR" +
						"lbmFudCI6ImVzeGNsb3VkIiwianRpIjoiQ2ZQYnk3QkFsYU9JM1VqX1RFcV9VSk9KbVlYSmlWT1l1Q1lBWFB3MmwyVSJ9." +
						"QOpb-8L8if1kEHPEQvsGe_Z8v_gdlPDpjWcu8LxMnAxZELQx6YBn7UM2MO83Qgo-0bqu2ysbcSpjz0mP4pf48z_DyKlMCa" +
						"B6ViStwavIx7lM1TENrt5nURpjqxlzQY0CxjyYIWxoYQIUbn7c5MXe-vt-OTXAg8bGkwphltj7xUak90mQlZGSBrHFCT_Q" +
						"PGwxRTNsRwWq45tF7LgKr49L4z5PnkLQ3LpC8jI7x1SUFBiYcJgi76pGNlD4qihpmKhGJK0WpspEAvXhtsGwBVavGxeXzL" +
						"-PBTYz7Zs1EjD4Isar-91pq-HeTVfhd_KBBqktaQq0WO48Vu0KtHHRv_Us90-Qs53gsY0CnrxHV8qyNR27LyaIMWhG24hq" +
						"TyBsZVgT-gzs9_-QdLqtkXNgr4Oiqoy9Gi8LAmARGFCgTXOS7uPqZ6_ut71WPhwwoUIuXVUG8vvuRD6_UIIGXyPjBM0sfg" +
						"X5rMeo45bYO51mNjqAysz7FBwMetkZUqKg6pxWmTmO_xnH5D55I1P2zd_VBo5be-hr7jjTqqDAGkGMU0PM8IajpnWe24wu" +
						"lPzQqRr5-HlQx50B0nwhYFJVCd_3KW6qCw-MmfGB-1aX-GVG2wa_vUKzc4gDDn65-z0rP_gYtrB9q8oNR-hPY4v18DQEdY" +
						"bsuoJoqriXk1A0zkeoX13kFXY")
				Expect(resp).To(BeEquivalentTo(expected))
			})
		})

		Context("parsing raw token details", func() {
			It("parses raw tokens", func() {
				resp, err := ParseRawTokenDetails(
					"eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJlYy1hZG1pbkBlc3hjbG91ZCIsImF1ZCI6WyJlYy1hZG1pbkB" +
						"lc3hjbG91ZCIsInJzX2VzeGNsb3VkIl0sInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIHJzX2VzeGNsb3VkIGF0X" +
						"2dyb3VwcyIsImlzcyI6Imh0dHBzOlwvXC8xMC4xNDYuNjQuMjM4XC9vcGVuaWRjb25uZWN0XC9lc3hjbG91ZCIsImdyb3V" +
						"wcyI6WyJlc3hjbG91ZFxcRVNYQ2xvdWRBZG1pbnMiLCJlc3hjbG91ZFxcRXZlcnlvbmUiXSwidG9rZW5fY2xhc3MiOiJhY" +
						"2Nlc3NfdG9rZW4iLCJ0b2tlbl90eXBlIjoiQmVhcmVyIiwiZXhwIjoxNDYxODAzMTI3LCJpYXQiOjE0NjE3OTU5MjcsInR" +
						"lbmFudCI6ImVzeGNsb3VkIiwianRpIjoiQ2ZQYnk3QkFsYU9JM1VqX1RFcV9VSk9KbVlYSmlWT1l1Q1lBWFB3MmwyVSJ9." +
						"QOpb-8L8if1kEHPEQvsGe_Z8v_gdlPDpjWcu8LxMnAxZELQx6YBn7UM2MO83Qgo-0bqu2ysbcSpjz0mP4pf48z_DyKlMCa" +
						"B6ViStwavIx7lM1TENrt5nURpjqxlzQY0CxjyYIWxoYQIUbn7c5MXe-vt-OTXAg8bGkwphltj7xUak90mQlZGSBrHFCT_Q" +
						"PGwxRTNsRwWq45tF7LgKr49L4z5PnkLQ3LpC8jI7x1SUFBiYcJgi76pGNlD4qihpmKhGJK0WpspEAvXhtsGwBVavGxeXzL" +
						"-PBTYz7Zs1EjD4Isar-91pq-HeTVfhd_KBBqktaQq0WO48Vu0KtHHRv_Us90-Qs53gsY0CnrxHV8qyNR27LyaIMWhG24hq" +
						"TyBsZVgT-gzs9_-QdLqtkXNgr4Oiqoy9Gi8LAmARGFCgTXOS7uPqZ6_ut71WPhwwoUIuXVUG8vvuRD6_UIIGXyPjBM0sfg" +
						"X5rMeo45bYO51mNjqAysz7FBwMetkZUqKg6pxWmTmO_xnH5D55I1P2zd_VBo5be-hr7jjTqqDAGkGMU0PM8IajpnWe24wu" +
						"lPzQqRr5-HlQx50B0nwhYFJVCd_3KW6qCw-MmfGB-1aX-GVG2wa_vUKzc4gDDn65-z0rP_gYtrB9q8oNR-hPY4v18DQEdY" +
						"bsuoJoqriXk1A0zkeoX13kFXY")
				Expect(err).To(BeNil())
				Expect(resp).ToNot(BeNil())
				Expect(len(resp)).To(BeNumerically(">", 0))
			})
		})
	})
})
