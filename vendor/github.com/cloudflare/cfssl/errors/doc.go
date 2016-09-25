/*
Package errors provides error types returned in CF SSL.

1. Type Error is intended for errors produced by CF SSL packages.
It formats to a json object that consists of an error message and a 4-digit code for error reasoning.

Example: {"code":1002, "message": "Failed to decode certificate"}

The index of codes are listed below:
	1XXX: CertificateError
	    1000: Unknown
	    1001: ReadFailed
	    1002: DecodeFailed
	    1003: ParseFailed
	    1100: SelfSigned
	    12XX: VerifyFailed
	        121X: CertificateInvalid
	            1210: NotAuthorizedToSign
	            1211: Expired
	            1212: CANotAuthorizedForThisName
	            1213: TooManyIntermediates
	            1214: IncompatibleUsage
	        1220: UnknownAuthority
	2XXX: PrivatekeyError
	    2000: Unknown
	    2001: ReadFailed
	    2002: DecodeFailed
	    2003: ParseFailed
	    2100: Encrypted
	    2200: NotRSA
	    2300: KeyMismatch
	    2400: GenerationFailed
	    2500: Unavailable
	3XXX: IntermediatesError
	4XXX: RootError
	5XXX: PolicyError
	    5100: NoKeyUsages
	    5200: InvalidPolicy
	    5300: InvalidRequest
	    5400: UnknownProfile
	    6XXX: DialError

2. Type HttpError is intended for CF SSL API to consume. It contains a HTTP status code that will be read and returned
by the API server.
*/
package errors
