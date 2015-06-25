/*
Package dns implements a full featured interface to the Domain Name System.
Server- and client-side programming is supported.
The package allows complete control over what is send out to the DNS. The package
API follows the less-is-more principle, by presenting a small, clean interface.

The package dns supports (asynchronous) querying/replying, incoming/outgoing zone transfers,
TSIG, EDNS0, dynamic updates, notifies and DNSSEC validation/signing.
Note that domain names MUST be fully qualified, before sending them, unqualified
names in a message will result in a packing failure.

Resource records are native types. They are not stored in wire format.
Basic usage pattern for creating a new resource record:

     r := new(dns.MX)
     r.Hdr = dns.RR_Header{Name: "miek.nl.", Rrtype: dns.TypeMX, Class: dns.ClassINET, Ttl: 3600}
     r.Preference = 10
     r.Mx = "mx.miek.nl."

Or directly from a string:

     mx, err := dns.NewRR("miek.nl. 3600 IN MX 10 mx.miek.nl.")

Or when the default TTL (3600) and class (IN) suit you:

     mx, err := dns.NewRR("miek.nl. MX 10 mx.miek.nl.")

Or even:

     mx, err := dns.NewRR("$ORIGIN nl.\nmiek 1H IN MX 10 mx.miek")

In the DNS messages are exchanged, these messages contain resource
records (sets).  Use pattern for creating a message:

     m := new(dns.Msg)
     m.SetQuestion("miek.nl.", dns.TypeMX)

Or when not certain if the domain name is fully qualified:

	m.SetQuestion(dns.Fqdn("miek.nl"), dns.TypeMX)

The message m is now a message with the question section set to ask
the MX records for the miek.nl. zone.

The following is slightly more verbose, but more flexible:

     m1 := new(dns.Msg)
     m1.Id = dns.Id()
     m1.RecursionDesired = true
     m1.Question = make([]dns.Question, 1)
     m1.Question[0] = dns.Question{"miek.nl.", dns.TypeMX, dns.ClassINET}

After creating a message it can be send.
Basic use pattern for synchronous querying the DNS at a
server configured on 127.0.0.1 and port 53:

     c := new(dns.Client)
     in, rtt, err := c.Exchange(m1, "127.0.0.1:53")

Suppressing
multiple outstanding queries (with the same question, type and class) is as easy as setting:

	c.SingleInflight = true

If these "advanced" features are not needed, a simple UDP query can be send,
with:

	in, err := dns.Exchange(m1, "127.0.0.1:53")

When this functions returns you will get dns message. A dns message consists
out of four sections.
The question section: in.Question, the answer section: in.Answer,
the authority section: in.Ns and the additional section: in.Extra.

Each of these sections (except the Question section) contain a []RR. Basic
use pattern for accessing the rdata of a TXT RR as the first RR in
the Answer section:

	if t, ok := in.Answer[0].(*dns.TXT); ok {
		// do something with t.Txt
	}

Domain Name and TXT Character String Representations

Both domain names and TXT character strings are converted to presentation
form both when unpacked and when converted to strings.

For TXT character strings, tabs, carriage returns and line feeds will be
converted to \t, \r and \n respectively. Back slashes and quotations marks
will be escaped. Bytes below 32 and above 127 will be converted to \DDD
form.

For domain names, in addition to the above rules brackets, periods,
spaces, semicolons and the at symbol are escaped.

DNSSEC

DNSSEC (DNS Security Extension) adds a layer of security to the DNS. It
uses public key cryptography to sign resource records. The
public keys are stored in DNSKEY records and the signatures in RRSIG records.

Requesting DNSSEC information for a zone is done by adding the DO (DNSSEC OK) bit
to an request.

     m := new(dns.Msg)
     m.SetEdns0(4096, true)

Signature generation, signature verification and key generation are all supported.

DYNAMIC UPDATES

Dynamic updates reuses the DNS message format, but renames three of
the sections. Question is Zone, Answer is Prerequisite, Authority is
Update, only the Additional is not renamed. See RFC 2136 for the gory details.

You can set a rather complex set of rules for the existence of absence of
certain resource records or names in a zone to specify if resource records
should be added or removed. The table from RFC 2136 supplemented with the Go
DNS function shows which functions exist to specify the prerequisites.

3.2.4 - Table Of Metavalues Used In Prerequisite Section

  CLASS    TYPE     RDATA    Meaning                    Function
  --------------------------------------------------------------
  ANY      ANY      empty    Name is in use             dns.NameUsed
  ANY      rrset    empty    RRset exists (value indep) dns.RRsetUsed
  NONE     ANY      empty    Name is not in use         dns.NameNotUsed
  NONE     rrset    empty    RRset does not exist       dns.RRsetNotUsed
  zone     rrset    rr       RRset exists (value dep)   dns.Used

The prerequisite section can also be left empty.
If you have decided on the prerequisites you can tell what RRs should
be added or deleted. The next table shows the options you have and
what functions to call.

3.4.2.6 - Table Of Metavalues Used In Update Section

  CLASS    TYPE     RDATA    Meaning                     Function
  ---------------------------------------------------------------
  ANY      ANY      empty    Delete all RRsets from name dns.RemoveName
  ANY      rrset    empty    Delete an RRset             dns.RemoveRRset
  NONE     rrset    rr       Delete an RR from RRset     dns.Remove
  zone     rrset    rr       Add to an RRset             dns.Insert

TRANSACTION SIGNATURE

An TSIG or transaction signature adds a HMAC TSIG record to each message sent.
The supported algorithms include: HmacMD5, HmacSHA1, HmacSHA256 and HmacSHA512.

Basic use pattern when querying with a TSIG name "axfr." (note that these key names
must be fully qualified - as they are domain names) and the base64 secret
"so6ZGir4GPAqINNh9U5c3A==":

	c := new(dns.Client)
	c.TsigSecret = map[string]string{"axfr.": "so6ZGir4GPAqINNh9U5c3A=="}
	m := new(dns.Msg)
	m.SetQuestion("miek.nl.", dns.TypeMX)
	m.SetTsig("axfr.", dns.HmacMD5, 300, time.Now().Unix())
	...
	// When sending the TSIG RR is calculated and filled in before sending

When requesting an zone transfer (almost all TSIG usage is when requesting zone transfers), with
TSIG, this is the basic use pattern. In this example we request an AXFR for
miek.nl. with TSIG key named "axfr." and secret "so6ZGir4GPAqINNh9U5c3A=="
and using the server 176.58.119.54:

	t := new(dns.Transfer)
	m := new(dns.Msg)
	t.TsigSecret = map[string]string{"axfr.": "so6ZGir4GPAqINNh9U5c3A=="}
	m.SetAxfr("miek.nl.")
	m.SetTsig("axfr.", dns.HmacMD5, 300, time.Now().Unix())
	c, err := t.In(m, "176.58.119.54:53")
	for r := range c { ... }

You can now read the records from the transfer as they come in. Each envelope is checked with TSIG.
If something is not correct an error is returned.

Basic use pattern validating and replying to a message that has TSIG set.

	server := &dns.Server{Addr: ":53", Net: "udp"}
	server.TsigSecret = map[string]string{"axfr.": "so6ZGir4GPAqINNh9U5c3A=="}
	go server.ListenAndServe()
	dns.HandleFunc(".", handleRequest)

	func handleRequest(w dns.ResponseWriter, r *dns.Msg) {
		m := new(Msg)
		m.SetReply(r)
		if r.IsTsig() {
			if w.TsigStatus() == nil {
				// *Msg r has an TSIG record and it was validated
				m.SetTsig("axfr.", dns.HmacMD5, 300, time.Now().Unix())
			} else {
				// *Msg r has an TSIG records and it was not valided
			}
		}
		w.WriteMsg(m)
	}

PRIVATE RRS

RFC 6895 sets aside a range of type codes for private use. This range
is 65,280 - 65,534 (0xFF00 - 0xFFFE). When experimenting with new Resource Records these
can be used, before requesting an official type code from IANA.

EDNS0

EDNS0 is an extension mechanism for the DNS defined in RFC 2671 and updated
by RFC 6891. It defines an new RR type, the OPT RR, which is then completely
abused.
Basic use pattern for creating an (empty) OPT RR:

	o := new(dns.OPT)
	o.Hdr.Name = "." // MUST be the root zone, per definition.
	o.Hdr.Rrtype = dns.TypeOPT

The rdata of an OPT RR consists out of a slice of EDNS0 (RFC 6891)
interfaces. Currently only a few have been standardized: EDNS0_NSID
(RFC 5001) and EDNS0_SUBNET (draft-vandergaast-edns-client-subnet-02). Note
that these options may be combined in an OPT RR.
Basic use pattern for a server to check if (and which) options are set:

	// o is a dns.OPT
	for _, s := range o.Option {
		switch e := s.(type) {
		case *dns.EDNS0_NSID:
			// do stuff with e.Nsid
		case *dns.EDNS0_SUBNET:
			// access e.Family, e.Address, etc.
		}
	}

SIG(0)

From RFC 2931:

    SIG(0) provides protection for DNS transactions and requests ....
    ... protection for glue records, DNS requests, protection for message headers
    on requests and responses, and protection of the overall integrity of a response.

It works like TSIG, except that SIG(0) uses public key cryptography, instead of the shared
secret approach in TSIG.
Supported algorithms: DSA, ECDSAP256SHA256, ECDSAP384SHA384, RSASHA1, RSASHA256 and
RSASHA512.

Signing subsequent messages in multi-message sessions is not implemented.
*/
package dns
