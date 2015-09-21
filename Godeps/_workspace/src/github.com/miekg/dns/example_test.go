package dns_test

import (
	"errors"
	"fmt"
	"github.com/miekg/dns"
	"log"
	"net"
)

// Retrieve the MX records for miek.nl.
func ExampleMX() {
	config, _ := dns.ClientConfigFromFile("/etc/resolv.conf")
	c := new(dns.Client)
	m := new(dns.Msg)
	m.SetQuestion("miek.nl.", dns.TypeMX)
	m.RecursionDesired = true
	r, _, err := c.Exchange(m, config.Servers[0]+":"+config.Port)
	if err != nil {
		return
	}
	if r.Rcode != dns.RcodeSuccess {
		return
	}
	for _, a := range r.Answer {
		if mx, ok := a.(*dns.MX); ok {
			fmt.Printf("%s\n", mx.String())
		}
	}
}

// Retrieve the DNSKEY records of a zone and convert them
// to DS records for SHA1, SHA256 and SHA384.
func ExampleDS(zone string) {
	config, _ := dns.ClientConfigFromFile("/etc/resolv.conf")
	c := new(dns.Client)
	m := new(dns.Msg)
	if zone == "" {
		zone = "miek.nl"
	}
	m.SetQuestion(dns.Fqdn(zone), dns.TypeDNSKEY)
	m.SetEdns0(4096, true)
	r, _, err := c.Exchange(m, config.Servers[0]+":"+config.Port)
	if err != nil {
		return
	}
	if r.Rcode != dns.RcodeSuccess {
		return
	}
	for _, k := range r.Answer {
		if key, ok := k.(*dns.DNSKEY); ok {
			for _, alg := range []uint8{dns.SHA1, dns.SHA256, dns.SHA384} {
				fmt.Printf("%s; %d\n", key.ToDS(alg).String(), key.Flags)
			}
		}
	}
}

const TypeAPAIR = 0x0F99

type APAIR struct {
	addr [2]net.IP
}

func NewAPAIR() dns.PrivateRdata { return new(APAIR) }

func (rd *APAIR) String() string { return rd.addr[0].String() + " " + rd.addr[1].String() }
func (rd *APAIR) Parse(txt []string) error {
	if len(txt) != 2 {
		return errors.New("two addresses required for APAIR")
	}
	for i, s := range txt {
		ip := net.ParseIP(s)
		if ip == nil {
			return errors.New("invalid IP in APAIR text representation")
		}
		rd.addr[i] = ip
	}
	return nil
}

func (rd *APAIR) Pack(buf []byte) (int, error) {
	b := append([]byte(rd.addr[0]), []byte(rd.addr[1])...)
	n := copy(buf, b)
	if n != len(b) {
		return n, dns.ErrBuf
	}
	return n, nil
}

func (rd *APAIR) Unpack(buf []byte) (int, error) {
	ln := net.IPv4len * 2
	if len(buf) != ln {
		return 0, errors.New("invalid length of APAIR rdata")
	}
	cp := make([]byte, ln)
	copy(cp, buf) // clone bytes to use them in IPs

	rd.addr[0] = net.IP(cp[:3])
	rd.addr[1] = net.IP(cp[4:])

	return len(buf), nil
}

func (rd *APAIR) Copy(dest dns.PrivateRdata) error {
	cp := make([]byte, rd.Len())
	_, err := rd.Pack(cp)
	if err != nil {
		return err
	}

	d := dest.(*APAIR)
	d.addr[0] = net.IP(cp[:3])
	d.addr[1] = net.IP(cp[4:])
	return nil
}

func (rd *APAIR) Len() int {
	return net.IPv4len * 2
}

func ExamplePrivateHandle() {
	dns.PrivateHandle("APAIR", TypeAPAIR, NewAPAIR)
	defer dns.PrivateHandleRemove(TypeAPAIR)

	rr, err := dns.NewRR("miek.nl. APAIR (1.2.3.4    1.2.3.5)")
	if err != nil {
		log.Fatal("could not parse APAIR record: ", err)
	}
	fmt.Println(rr)
	// Output: miek.nl.	3600	IN	APAIR	1.2.3.4 1.2.3.5

	m := new(dns.Msg)
	m.Id = 12345
	m.SetQuestion("miek.nl.", TypeAPAIR)
	m.Answer = append(m.Answer, rr)

	fmt.Println(m)
	// ;; opcode: QUERY, status: NOERROR, id: 12345
	// ;; flags: rd; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 0
	//
	// ;; QUESTION SECTION:
	// ;miek.nl.	IN	 APAIR
	//
	// ;; ANSWER SECTION:
	// miek.nl.	3600	IN	APAIR	1.2.3.4 1.2.3.5
}
