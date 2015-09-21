package ec2

import (
	"github.com/mitchellh/goamz/aws"
	"time"
)

func Sign(auth aws.Auth, method, path string, params map[string]string, host string) {
	sign(auth, method, path, params, host)
}

func fixedTime() time.Time {
	return time.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC)
}

func FakeTime(fakeIt bool) {
	if fakeIt {
		timeNow = fixedTime
	} else {
		timeNow = time.Now
	}
}
