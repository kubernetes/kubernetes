package main

import "bytes"
import "flag"
import "fmt"
import "github.com/nalind/gss/pkg/gss"
import "github.com/nalind/gss/pkg/gss/misc"
import "net"
import "os"
import "strings"
import "encoding/asn1"

func connectOnce(host string, port int, service string, mcount int, quiet bool, user, pass *string, plain []byte, v1, spnego bool, pmech *asn1.ObjectIdentifier, delegate, seq, noreplay, nomutual, noauth, nowrap, noenc, nomic bool) {
	var ctx gss.ContextHandle
	var cred gss.CredHandle
	var mech asn1.ObjectIdentifier
	var tag byte
	var token []byte
	var major, minor uint32
	var sname, localstate, openstate string
	var flags gss.Flags

	/* Open the connection. */
	conn, err := net.Dial("tcp", fmt.Sprintf("%s:%d", host, port))
	if err != nil {
		fmt.Printf("Error connecting: %s\n", err)
		os.Exit(2)
	}
	defer conn.Close()

	/* Import the remote service's name. */
	if strings.Contains(service, "@") {
		sname = service
	} else {
		sname = service + "@" + host
	}
	major, minor, name := gss.ImportName(sname, gss.C_NT_HOSTBASED_SERVICE)
	if major != gss.S_COMPLETE {
		gss.DisplayGSSError("importing remote service name", major, minor, nil)
		return
	}
	defer gss.ReleaseName(name)

	/* If we were passed a user name and maybe a password, acquire some initiator creds. */
	if user != nil {
		var mechSet []asn1.ObjectIdentifier

		/* Parse the user name. */
		major, minor, username := gss.ImportName(*user, gss.C_NT_USER_NAME)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("importing client name", major, minor, nil)
			return
		}
		defer gss.ReleaseName(username)

		/* Set the mechanism OID for the creds that we want. */
		if pmech != nil || spnego {
			mechSet = make([]asn1.ObjectIdentifier, 1)
			if spnego {
				mechSet[0] = gss.Mech_spnego
			} else {
				mechSet[0] = *pmech
			}
		} else {
			mechSet = nil
		}

		/* Acquire the creds. */
		if pass != nil {
			buffer := bytes.NewBufferString(*pass)
			password := buffer.Bytes()
			major, minor, cred, _, _ = gss.AcquireCredWithPassword(username, password, gss.C_INDEFINITE, mechSet, gss.C_INITIATE)
		} else {
			major, minor, cred, _, _ = gss.AcquireCred(username, gss.C_INDEFINITE, mechSet, gss.C_INITIATE)
		}
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("acquiring creds", major, minor, &mechSet[0])
			return
		}
		defer gss.ReleaseCred(cred)
	}

	/* If we're doing SPNEGO, then a passed-in mechanism OID is the one we want to negotiate. */
	if spnego {
		if pmech != nil {
			mechSet := make([]asn1.ObjectIdentifier, 1)
			mechSet[0] = *pmech
			major, minor = gss.SetNegMechs(cred, mechSet)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("setting negotiate mechs", major, minor, nil)
				return
			}
		}
		mech = gss.Mech_spnego
	} else {
		if pmech != nil {
			mech = *pmech
		} else {
			mech = nil
		}
	}

	if noauth {
		misc.SendToken(conn, misc.TOKEN_NOOP, nil)
	} else {
		if !v1 {
			misc.SendToken(conn, misc.TOKEN_NOOP|misc.TOKEN_CONTEXT_NEXT, nil)
		}
		flags = gss.Flags{Deleg: delegate, Sequence: seq, Replay: !noreplay, Conf: !noenc, Integ: !nomic, Mutual: !nomutual}
		for true {
			/* Start/continue. */
			major, minor, _, token, flags, _, _, _ = gss.InitSecContext(cred, &ctx, name, mech, flags, gss.C_INDEFINITE, nil, token)
			if major != gss.S_COMPLETE && major != gss.S_CONTINUE_NEEDED {
				gss.DisplayGSSError("initializing security context", major, minor, &mech)
				gss.DeleteSecContext(ctx)
				return
			}
			/* If we have an output token, we need to send it. */
			if len(token) > 0 {
				if !quiet {
					fmt.Printf("Sending init_sec_context token (size=%d)...", len(token))
				}
				if v1 {
					tag = 0
				} else {
					tag = misc.TOKEN_CONTEXT
				}
				misc.SendToken(conn, tag, token)
			}
			if major == gss.S_CONTINUE_NEEDED {
				/* CONTINUE_NEEDED means we expect a token from the far end to be fed back in to InitSecContext(). */
				if !quiet {
					fmt.Printf("continue needed...")
				}
				tag, token = misc.RecvToken(conn)
				if !quiet {
					fmt.Printf("\n")
				}
				if len(token) == 0 {
					if !quiet {
						fmt.Printf("server closed connection.\n")
					}
					defer gss.DeleteSecContext(ctx)
					break
				}
			} else {
				/* COMPLETE means we're done, everything succeeded. */
				if !quiet {
					fmt.Printf("\n")
				}
				defer gss.DeleteSecContext(ctx)
				break
			}
		}
		if major != gss.S_COMPLETE {
			fmt.Printf("Error authenticating to server: %08x/%08x.\n", major, minor)
			return
		}
		if !quiet {
			gss.DisplayGSSFlags(flags, false, os.Stdout)
		}

		/* Describe the context. */
		major, minor, sname, tname, lifetime, mech, flags2, _, _, local, open := gss.InquireContext(ctx)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("inquiring context", major, minor, &mech)
			return
		}
		major, minor, srcname, srcnametype := gss.DisplayName(sname)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("displaying source name", major, minor, &mech)
			return
		}
		major, minor, targname, _ := gss.DisplayName(tname)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("displaying target name", major, minor, &mech)
			return
		}
		if local {
			localstate = "locally initiated"
		} else {
			localstate = "remotely initiated"
		}
		if open {
			openstate = "open"
		} else {
			openstate = "closed"
		}
		if !quiet {
			fmt.Printf("\"%s\" to \"%s\", lifetime %d, flags %x, %s, %s\n", srcname, targname, lifetime, gss.FlagsToRaw(flags2), localstate, openstate)
		}
		major, minor, oid := gss.OidToStr(srcnametype)
		if major != gss.S_COMPLETE {
			oid = srcnametype.String()
		}
		if !quiet {
			fmt.Printf("Name type of source name is %s.\n", oid)
		}
		major, minor, mechs := gss.InquireNamesForMech(mech)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("inquiring mech names", major, minor, &mech)
			return
		}
		major, minor, oid = gss.OidToStr(mech)
		if major != gss.S_COMPLETE {
			oid = mech.String()
		}
		if !quiet {
			fmt.Printf("Mechanism %s supports %d names\n", oid, len(mechs))
		}
		for i, nametype := range mechs {
			major, minor, oid := gss.OidToStr(nametype)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("converting OID to string", major, minor, &mech)
			} else {
				if !quiet {
					fmt.Printf("%3d: %s\n", i, oid)
				}
			}
		}
	}

	for i := 0; i < mcount; i++ {
		var wrapped []byte
		var major, minor uint32
		var encrypted bool

		if nowrap {
			wrapped = plain
		} else {
			major, minor, encrypted, wrapped = gss.Wrap(ctx, !noenc, gss.C_QOP_DEFAULT, plain)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("wrapping data", major, minor, &mech)
				return
			}
			if !noenc && !encrypted && !quiet {
				fmt.Printf("Warning!  Message not encrypted.\n")
			}
		}

		tag = misc.TOKEN_DATA
		if !nowrap {
			tag |= misc.TOKEN_WRAPPED
		}
		if !noenc {
			tag |= misc.TOKEN_ENCRYPTED
		}
		if !nomic {
			tag |= misc.TOKEN_SEND_MIC
		}
		if v1 {
			tag = 0
		}

		misc.SendToken(conn, tag, wrapped)
		tag, mictoken := misc.RecvToken(conn)
		if tag == 0 && len(mictoken) == 0 {
			if !quiet {
				fmt.Printf("Server closed connection unexpectedly.\n")
			}
			return
		}
		if nomic {
			if bytes.Equal(plain, mictoken) {
				if !quiet {
					fmt.Printf("Response differed.\n")
				}
				return
			}
			if !quiet {
				fmt.Printf("Response received.\n")
			}
		} else {
			major, minor, _ = gss.VerifyMIC(ctx, plain, mictoken)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("verifying signature", major, minor, &mech)
				return
			}
			if !quiet {
				fmt.Printf("Signature verified.\n")
			}
		}
	}
	if !v1 {
		misc.SendToken(conn, misc.TOKEN_NOOP, nil)
	}
}

func main() {
	port := flag.Int("port", 4444, "port")
	mechstr := flag.String("mech", "", "mechanism")
	spnego := flag.Bool("spnego", false, "use SPNEGO")
	iakerb := flag.Bool("iakerb", false, "use IAKERB")
	krb5 := flag.Bool("krb5", false, "use Kerberos 5")
	delegate := flag.Bool("d", false, "delegate")
	seq := flag.Bool("seq", false, "use sequence number checking")
	noreplay := flag.Bool("noreplay", false, "disable replay checking")
	nomutual := flag.Bool("nomutual", false, "perform one-way authentication")
	user := flag.String("user", "", "acquire creds as user")
	pass := flag.String("pass", "", "password for -user user")
	file := flag.Bool("f", false, "read message from file")
	v1 := flag.Bool("v1", false, "use version 1 protocol")
	quiet := flag.Bool("q", false, "quiet")
	ccount := flag.Int("ccount", 1, "connection count")
	mcount := flag.Int("mcount", 1, "message count")
	noauth := flag.Bool("na", false, "no authentication")
	nowrap := flag.Bool("nw", false, "no wrapping")
	noenc := flag.Bool("nx", false, "no encryption")
	nomic := flag.Bool("nm", false, "no MICs")
	var plain []byte
	var mech *asn1.ObjectIdentifier

	flag.Parse()
	host := flag.Arg(0)
	service := flag.Arg(1)
	msg := flag.Arg(2)
	if flag.NArg() < 3 {
		fmt.Printf("Usage: gss-client [options] host gss-service-name message-or-file\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if *file {
		msgfile, err := os.Open(msg)
		if err != nil {
			fmt.Printf("Error opening \"%s\": %s", msg, err)
			return
		}
		fi, err := msgfile.Stat()
		if err != nil {
			fmt.Printf("Error statting \"%s\": %s", msg, err)
			return
		}
		plain = make([]byte, fi.Size())
		n, err := msgfile.Read(plain)
		if int64(n) != fi.Size() {
			fmt.Printf("Error reading \"%s\": %s", msg, err)
			return
		}
	} else {
		buffer := bytes.NewBufferString(msg)
		plain = buffer.Bytes()
	}
	if *krb5 {
		tmpmech := misc.ParseOid("1.3.5.1.5.2")
		mech = &tmpmech
	}
	if *iakerb {
		tmpmech := misc.ParseOid("1.3.6.1.5.2.5")
		mech = &tmpmech
	}
	if len(*mechstr) > 0 {
		tmpmech := misc.ParseOid(*mechstr)
		mech = &tmpmech
	}
	if len(*user) == 0 {
		user = nil
	}
	if len(*pass) == 0 {
		pass = nil
	}
	if *noauth {
		*nowrap = true
		*noenc = true
		*nomic = true
	}

	for c := 0; c < *ccount; c++ {
		connectOnce(host, *port, service, *mcount, *quiet, user, pass, plain, *v1, *spnego, mech, *delegate, *seq, *noreplay, *nomutual, *noauth, *nowrap, *noenc, *nomic)
	}
}
