package ntlmssp

type negotiateFlags uint32

const (
	/*A*/ negotiateFlagNTLMSSPNEGOTIATEUNICODE negotiateFlags = 1 << 0
	/*B*/ negotiateFlagNTLMNEGOTIATEOEM = 1 << 1
	/*C*/ negotiateFlagNTLMSSPREQUESTTARGET = 1 << 2

	/*D*/
	negotiateFlagNTLMSSPNEGOTIATESIGN = 1 << 4
	/*E*/ negotiateFlagNTLMSSPNEGOTIATESEAL = 1 << 5
	/*F*/ negotiateFlagNTLMSSPNEGOTIATEDATAGRAM = 1 << 6
	/*G*/ negotiateFlagNTLMSSPNEGOTIATELMKEY = 1 << 7

	/*H*/
	negotiateFlagNTLMSSPNEGOTIATENTLM = 1 << 9

	/*J*/
	negotiateFlagANONYMOUS = 1 << 11
	/*K*/ negotiateFlagNTLMSSPNEGOTIATEOEMDOMAINSUPPLIED = 1 << 12
	/*L*/ negotiateFlagNTLMSSPNEGOTIATEOEMWORKSTATIONSUPPLIED = 1 << 13

	/*M*/
	negotiateFlagNTLMSSPNEGOTIATEALWAYSSIGN = 1 << 15
	/*N*/ negotiateFlagNTLMSSPTARGETTYPEDOMAIN = 1 << 16
	/*O*/ negotiateFlagNTLMSSPTARGETTYPESERVER = 1 << 17

	/*P*/
	negotiateFlagNTLMSSPNEGOTIATEEXTENDEDSESSIONSECURITY = 1 << 19
	/*Q*/ negotiateFlagNTLMSSPNEGOTIATEIDENTIFY = 1 << 20

	/*R*/
	negotiateFlagNTLMSSPREQUESTNONNTSESSIONKEY = 1 << 22
	/*S*/ negotiateFlagNTLMSSPNEGOTIATETARGETINFO = 1 << 23

	/*T*/
	negotiateFlagNTLMSSPNEGOTIATEVERSION = 1 << 25

	/*U*/
	negotiateFlagNTLMSSPNEGOTIATE128 = 1 << 29
	/*V*/ negotiateFlagNTLMSSPNEGOTIATEKEYEXCH = 1 << 30
	/*W*/ negotiateFlagNTLMSSPNEGOTIATE56 = 1 << 31
)

func (field negotiateFlags) Has(flags negotiateFlags) bool {
	return field&flags == flags
}

func (field *negotiateFlags) Unset(flags negotiateFlags) {
	*field = *field ^ (*field & flags)
}
