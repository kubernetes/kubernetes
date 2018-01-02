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
					TokenId:    "LTW1jD-LccorfMJN-SUELdAwUKO8lHTHwGL2kGNVc5g",
					Algorithm:  "RS256",
					Subject:    "administrator@photon.com",
					Audience:   []string{"administrator@photon.com", "rs_photon_platform"},
					Groups:     []string{"photon.com\\Users", "photon.com\\Administrators", "photon.com\\CAAdmins", "photon.com\\Everyone"},
					Issuer:     "https://10.118.108.208/openidconnect/photon.com",
					IssuedAt:   1488478342,
					Expires:    1488478642,
					Scope:      "rs_photon_platform at_groups openid offline_access",
					TokenType:  "Bearer",
					TokenClass: "access_token",
					Tenant:     "photon.com",
				}
				resp := ParseTokenDetails(
					"eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhZG1pbmlzdHJhdG9yQHBob3Rvbi5jb20iLCJhdWQiOlsiYWRta" +
						"W5pc3RyYXRvckBwaG90b24uY29tIiwicnNfcGhvdG9uX3BsYXRmb3JtIl0sInNjb3BlIjoicnNfcGhvdG9uX3BsYXRmb3J" +
						"tIGF0X2dyb3VwcyBvcGVuaWQgb2ZmbGluZV9hY2Nlc3MiLCJpc3MiOiJodHRwczpcL1wvMTAuMTE4LjEwOC4yMDhcL29wZ" +
						"W5pZGNvbm5lY3RcL3Bob3Rvbi5jb20iLCJncm91cHMiOlsicGhvdG9uLmNvbVxcVXNlcnMiLCJwaG90b24uY29tXFxBZG1" +
						"pbmlzdHJhdG9ycyIsInBob3Rvbi5jb21cXENBQWRtaW5zIiwicGhvdG9uLmNvbVxcRXZlcnlvbmUiXSwidG9rZW5fY2xhc" +
						"3MiOiJhY2Nlc3NfdG9rZW4iLCJ0b2tlbl90eXBlIjoiQmVhcmVyIiwiZXhwIjoxNDg4NDc4NjQyLCJpYXQiOjE0ODg0Nzg" +
						"zNDIsImp0aSI6IkxUVzFqRC1MY2NvcmZNSk4tU1VFTGRBd1VLTzhsSFRId0dMMmtHTlZjNWciLCJ0ZW5hbnQiOiJwaG90b" +
						"24uY29tIn0.UFiruuobguHiVZZHnhCxkqw8k98RS6y2A9Dh_7LOclhvXxthUfae0JZvLVN7sUmeVss-aDFkxTRWUVMmHaj" +
						"jDCERSI6oMBiWU2aFtcS0ZdJGEbOLbDNG2tOCyyIkI6IYaWmVEGCGjhn3bXGjxC5dvH4au0sYynxTjD97StqmaqoQ2OhWZ" +
						"075vdIWyybwJlSgVk8WCjszjuH_4oe87hvIn79QnF37WBXZua_dhaeiAOzm752LFGr3kRp6BYIfp_z-NHBFPTEL93d4Wx0" +
						"DOam7EUa65vOeoiRiLJjhjNsJ_nGhka_v9m5GMlhst_b1HqCUmLFmt6POFuQCf3UswNtEX7rcIfSlem5Z002TpzzrElPqP" +
						"oxGHrw3vWAUPjHwucJ7CIp9AmF1Xsh-TfybxS66THbObt3HxE6Zb3pCFEgsZegjUb7CUDzOaicWexDF6Ft5Xv_ppH4-NHH" +
						"fzdFlYvdrS0YATNtK4YjkacoAKYzdMH-F7usxDJjanS0b73BEXzBaTAzVCNPGflulyrE8j1iDcpazHWQMMq1NZ5_OBw7TF" +
						"xLv5Te854cWEVMbIDOkQShUGLDiN52TtNMfdqFP-4M2lOcrmkShG4QXKQrYnlTy-b3tsMsukoihpKsp-yaW-DPs9J1hvlD" +
						"wqbwm2H0GDj2tYC6X2EiVDofjJZ4YqpcUCoE")
				Expect(resp).To(BeEquivalentTo(expected))
			})
		})

		Context("parsing raw token details", func() {
			It("parses raw tokens", func() {
				resp, err := ParseRawTokenDetails(
					"eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhZG1pbmlzdHJhdG9yQHBob3Rvbi5jb20iLCJhdWQiOlsiYWRta" +
						"W5pc3RyYXRvckBwaG90b24uY29tIiwicnNfcGhvdG9uX3BsYXRmb3JtIl0sInNjb3BlIjoicnNfcGhvdG9uX3BsYXRmb3J" +
						"tIGF0X2dyb3VwcyBvcGVuaWQgb2ZmbGluZV9hY2Nlc3MiLCJpc3MiOiJodHRwczpcL1wvMTAuMTE4LjEwOC4yMDhcL29wZ" +
						"W5pZGNvbm5lY3RcL3Bob3Rvbi5jb20iLCJncm91cHMiOlsicGhvdG9uLmNvbVxcVXNlcnMiLCJwaG90b24uY29tXFxBZG1" +
						"pbmlzdHJhdG9ycyIsInBob3Rvbi5jb21cXENBQWRtaW5zIiwicGhvdG9uLmNvbVxcRXZlcnlvbmUiXSwidG9rZW5fY2xhc" +
						"3MiOiJhY2Nlc3NfdG9rZW4iLCJ0b2tlbl90eXBlIjoiQmVhcmVyIiwiZXhwIjoxNDg4NDc4NjQyLCJpYXQiOjE0ODg0Nzg" +
						"zNDIsImp0aSI6IkxUVzFqRC1MY2NvcmZNSk4tU1VFTGRBd1VLTzhsSFRId0dMMmtHTlZjNWciLCJ0ZW5hbnQiOiJwaG90b" +
						"24uY29tIn0.UFiruuobguHiVZZHnhCxkqw8k98RS6y2A9Dh_7LOclhvXxthUfae0JZvLVN7sUmeVss-aDFkxTRWUVMmHaj" +
						"jDCERSI6oMBiWU2aFtcS0ZdJGEbOLbDNG2tOCyyIkI6IYaWmVEGCGjhn3bXGjxC5dvH4au0sYynxTjD97StqmaqoQ2OhWZ" +
						"075vdIWyybwJlSgVk8WCjszjuH_4oe87hvIn79QnF37WBXZua_dhaeiAOzm752LFGr3kRp6BYIfp_z-NHBFPTEL93d4Wx0" +
						"DOam7EUa65vOeoiRiLJjhjNsJ_nGhka_v9m5GMlhst_b1HqCUmLFmt6POFuQCf3UswNtEX7rcIfSlem5Z002TpzzrElPqP" +
						"oxGHrw3vWAUPjHwucJ7CIp9AmF1Xsh-TfybxS66THbObt3HxE6Zb3pCFEgsZegjUb7CUDzOaicWexDF6Ft5Xv_ppH4-NHH" +
						"fzdFlYvdrS0YATNtK4YjkacoAKYzdMH-F7usxDJjanS0b73BEXzBaTAzVCNPGflulyrE8j1iDcpazHWQMMq1NZ5_OBw7TF" +
						"xLv5Te854cWEVMbIDOkQShUGLDiN52TtNMfdqFP-4M2lOcrmkShG4QXKQrYnlTy-b3tsMsukoihpKsp-yaW-DPs9J1hvlD" +
						"wqbwm2H0GDj2tYC6X2EiVDofjJZ4YqpcUCoE")
				Expect(err).To(BeNil())
				Expect(resp).ToNot(BeNil())
				Expect(len(resp)).To(BeNumerically(">", 0))
			})
		})
	})
})
