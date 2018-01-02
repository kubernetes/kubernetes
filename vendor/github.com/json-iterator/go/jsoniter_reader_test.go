package jsoniter

import (
	"github.com/stretchr/testify/require"
	"strings"
	"testing"
	"time"
)

func Test_reader_and_load_more(t *testing.T) {
	should := require.New(t)
	type TestObject struct {
		CreatedAt time.Time
	}
	reader := strings.NewReader(`
{
	"agency": null,
	"candidateId": 0,
	"candidate": "Blah Blah",
	"bookingId": 0,
	"shiftId": 1,
	"shiftTypeId": 0,
	"shift": "Standard",
	"bonus": 0,
	"bonusNI": 0,
	"days": [],
	"totalHours": 27,
	"expenses": [],
	"weekEndingDateSystem": "2016-10-09",
	"weekEndingDateClient": "2016-10-09",
	"submittedAt": null,
	"submittedById": null,
	"approvedAt": "2016-10-10T18:38:04Z",
	"approvedById": 0,
	"authorisedAt": "2016-10-10T18:38:04Z",
	"authorisedById": 0,
	"invoicedAt": "2016-10-10T20:00:00Z",
	"revokedAt": null,
	"revokedById": null,
	"revokeReason": null,
	"rejectedAt": null,
	"rejectedById": null,
	"rejectReasonCode": null,
	"rejectReason": null,
	"createdAt": "2016-10-03T00:00:00Z",
	"updatedAt": "2016-11-09T10:26:13Z",
	"updatedById": null,
	"overrides": [],
	"bookingApproverId": null,
	"bookingApprover": null,
	"status": "approved"
}
	`)
	decoder := ConfigCompatibleWithStandardLibrary.NewDecoder(reader)
	obj := TestObject{}
	should.Nil(decoder.Decode(&obj))
}
