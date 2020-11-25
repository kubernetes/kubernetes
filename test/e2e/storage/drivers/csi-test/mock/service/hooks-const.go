package service

// Predefinded constants for the JavaScript hooks, they must correspond to the
// error codes used by gRPC, see:
// https://github.com/grpc/grpc-go/blob/master/codes/codes.go
const (
	grpcJSCodes string = `OK = 0;
			CANCELED = 1;
			UNKNOWN = 2;
			INVALIDARGUMENT = 3;
			DEADLINEEXCEEDED = 4;
			NOTFOUND = 5;
			ALREADYEXISTS = 6;
			PERMISSIONDENIED = 7;
			RESOURCEEXHAUSTED = 8;
			FAILEDPRECONDITION = 9;
			ABORTED = 10;
			OUTOFRANGE = 11;
			UNIMPLEMENTED = 12;
			INTERNAL = 13;
			UNAVAILABLE = 14;
			DATALOSS = 15;
			UNAUTHENTICATED = 16`
)
