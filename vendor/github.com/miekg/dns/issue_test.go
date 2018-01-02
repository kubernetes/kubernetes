package dns

// Tests that solve that an specific issue.

import "testing"

func TestTCPRtt(t *testing.T) {
	m := new(Msg)
	m.RecursionDesired = true
	m.SetQuestion("example.org.", TypeA)

	c := &Client{}
	for _, proto := range []string{"udp", "tcp"} {
		c.Net = proto
		_, rtt, err := c.Exchange(m, "8.8.4.4:53")
		if err != nil {
			t.Fatal(err)
		}
		if rtt == 0 {
			t.Fatalf("expecting non zero rtt %s, got zero", c.Net)
		}
	}
}
