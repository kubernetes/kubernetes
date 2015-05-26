package units

import (
	"testing"
	"time"
)

func TestHumanDuration(t *testing.T) {
	// Useful duration abstractions
	day := 24 * time.Hour
	week := 7 * day
	month := 30 * day
	year := 365 * day

	assertEquals(t, "Less than a second", HumanDuration(450*time.Millisecond))
	assertEquals(t, "47 seconds", HumanDuration(47*time.Second))
	assertEquals(t, "About a minute", HumanDuration(1*time.Minute))
	assertEquals(t, "3 minutes", HumanDuration(3*time.Minute))
	assertEquals(t, "35 minutes", HumanDuration(35*time.Minute))
	assertEquals(t, "35 minutes", HumanDuration(35*time.Minute+40*time.Second))
	assertEquals(t, "About an hour", HumanDuration(1*time.Hour))
	assertEquals(t, "About an hour", HumanDuration(1*time.Hour+45*time.Minute))
	assertEquals(t, "3 hours", HumanDuration(3*time.Hour))
	assertEquals(t, "3 hours", HumanDuration(3*time.Hour+59*time.Minute))
	assertEquals(t, "4 hours", HumanDuration(3*time.Hour+60*time.Minute))
	assertEquals(t, "24 hours", HumanDuration(24*time.Hour))
	assertEquals(t, "36 hours", HumanDuration(1*day+12*time.Hour))
	assertEquals(t, "2 days", HumanDuration(2*day))
	assertEquals(t, "7 days", HumanDuration(7*day))
	assertEquals(t, "13 days", HumanDuration(13*day+5*time.Hour))
	assertEquals(t, "2 weeks", HumanDuration(2*week))
	assertEquals(t, "2 weeks", HumanDuration(2*week+4*day))
	assertEquals(t, "3 weeks", HumanDuration(3*week))
	assertEquals(t, "4 weeks", HumanDuration(4*week))
	assertEquals(t, "4 weeks", HumanDuration(4*week+3*day))
	assertEquals(t, "4 weeks", HumanDuration(1*month))
	assertEquals(t, "6 weeks", HumanDuration(1*month+2*week))
	assertEquals(t, "8 weeks", HumanDuration(2*month))
	assertEquals(t, "3 months", HumanDuration(3*month+1*week))
	assertEquals(t, "5 months", HumanDuration(5*month+2*week))
	assertEquals(t, "13 months", HumanDuration(13*month))
	assertEquals(t, "23 months", HumanDuration(23*month))
	assertEquals(t, "24 months", HumanDuration(24*month))
	assertEquals(t, "2 years", HumanDuration(24*month+2*week))
	assertEquals(t, "3 years", HumanDuration(3*year+2*month))
}
