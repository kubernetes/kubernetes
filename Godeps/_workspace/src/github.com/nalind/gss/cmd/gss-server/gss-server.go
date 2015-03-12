package main

import "bytes"
import "encoding/asn1"
import "flag"
import "fmt"
import "github.com/nalind/gss/pkg/gss"
import "github.com/nalind/gss/pkg/gss/misc"
import "net"
import "io"
import "os"
import "strconv"

func dump(file io.Writer, data []byte) {
	var another bool

	for i, b := range data {
		fmt.Fprintf(file, "%02x", b)
		if i%16 == 15 {
			fmt.Fprintf(file, "\n")
			another = false
		} else {
			fmt.Fprintf(file, " ")
			another = true
		}
	}
	if another {
		fmt.Fprintf(file, "\n")
	}
}

func serve(conn net.Conn, cred gss.CredHandle, export, verbose bool, logfile io.Writer) {
	var ctx gss.ContextHandle
	var dcred gss.CredHandle
	var cname gss.InternalName
	var flags gss.Flags
	var mech asn1.ObjectIdentifier
	var client, localname string
	var major, minor uint32
	var conf bool

	defer conn.Close()

	tag, token := misc.RecvToken(conn)
	if tag == 0 && len(token) == 0 {
		fmt.Printf("EOF from client\n", tag)
		return
	}
	if (tag & misc.TOKEN_NOOP) == 0 {
		if logfile != nil {
			fmt.Fprintf(logfile, "Expected NOOP token, got %d token instead.\n", tag)
		}
		return
	}
	if (tag & misc.TOKEN_CONTEXT_NEXT) != 0 {
		for {
			/* Expect a context establishment token. */
			tag, token := misc.RecvToken(conn)
			if tag == 0 && len(token) == 0 {
				break
			}
			if verbose && logfile != nil {
				fmt.Fprintf(logfile, "Received token (%d bytes):\n", len(token))
				dump(logfile, token)
			}
			if tag&misc.TOKEN_CONTEXT == 0 {
				fmt.Printf("Expected context establishment token, got %d token instead.\n", tag)
				break
			}
			major, minor, cname, mech, flags, _, _, _, dcred, token = gss.AcceptSecContext(cred, &ctx, nil, token)
			if len(token) > 0 {
				/* If we got a new token, send it to the client. */
				if verbose && logfile != nil {
					fmt.Fprintf(logfile, "Sending accept_sec_context token (%d bytes):\n", len(token))
					dump(logfile, token)
				}
				misc.SendToken(conn, misc.TOKEN_CONTEXT, token)
			}
			/* We never use delegated creds, so if we got some, just make sure they get cleaned up. */
			if dcred != nil {
				defer gss.ReleaseCred(dcred)
				dcred = nil
			}
			if major != gss.S_COMPLETE && major != gss.S_CONTINUE_NEEDED {
				/* There was some kind of error. */
				gss.DisplayGSSError("accepting context", major, minor, &mech)
				return
			}
			if major == gss.S_COMPLETE {
				/* Okay, success. */
				if verbose && logfile != nil {
					fmt.Fprintf(logfile, "\n")
				}
				break
			}
			/* Wait for another context establishment token. */
			if verbose && logfile != nil {
				fmt.Fprintf(logfile, "continue needed...\n")
			}
		}
		/* Make sure the context is cleaned up eventually. */
		defer gss.DeleteSecContext(ctx)
		/* Make sure the client name gets cleaned up eventually. */
		defer gss.ReleaseName(cname)
		/* Dig up information about the connection. */
		gss.DisplayGSSFlags(flags, false, logfile)
		major, minor, oid := gss.OidToStr(mech)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("converting oid to string", major, minor, &mech)
		} else {
			if verbose && logfile != nil {
				fmt.Fprintf(logfile, "Accepted connection using mechanism OID %s.\n", oid)
			}
		}
		/* Figure out the client's attributes and displayable and local names. */
		major, minor, isMN, namemech, attrs := gss.InquireName(cname)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("inquiring name", major, minor, &mech)
		} else {
			if verbose && logfile != nil {
				if isMN {
					fmt.Fprintf(logfile, "Name is specific to mechanism %s.\n", namemech)
				} else {
					fmt.Fprintf(logfile, "Name is not specific to mechanism.\n")
				}
				for _, attr := range attrs {
					more := -1
					for more != 0 {
						major, minor, authenticated, complete, value, displayValue := gss.GetNameAttribute(cname, attr, &more)
						if major != gss.S_COMPLETE {
							gss.DisplayGSSError("getting name attribute", major, minor, &mech)
							break
						} else {
							fmt.Fprintf(logfile, "Attribute %s \"%s\"", attr, displayValue)
							if authenticated {
								fmt.Fprintf(logfile, ", authenticated")
							}
							if complete {
								fmt.Fprintf(logfile, ", complete")
							}
							if more != 0 {
								fmt.Fprintf(logfile, " (more)")
							}
							fmt.Fprintf(logfile, "\n")
							dump(logfile, value)
						}
					}
				}
			}
		}
		/* Exercise DuplicateName/ExportName. */
		major, minor, tmpname := gss.DuplicateName(cname)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("duplicating name", major, minor, &mech)
		} else {
			defer gss.ReleaseName(tmpname)
			major, minor, expname := gss.ExportName(tmpname)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("exporting name", major, minor, &mech)
			} else {
				fmt.Printf("exported name:\n")
				dump(logfile, expname)
			}
		}
		/* Exercise DisplayName. */
		major, minor, client, _ = gss.DisplayName(cname)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("displaying name", major, minor, &mech)
		}
		/* Exercise Localname. */
		major, minor, localname = gss.Localname(cname, nil)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("gss.Localname", major, minor, &mech)
		} else {
			fmt.Printf("localname: %s\n", localname)
		}
		/* Exercise PNameToUid. */
		major, minor, localuid := gss.PNameToUid(cname, nil)
		if major != gss.S_COMPLETE {
			gss.DisplayGSSError("gss.PNameToUid", major, minor, &mech)
		} else {
			fmt.Printf("UID: \"%s\"\n", localuid)
		}
	} else {
		if logfile != nil {
			fmt.Fprintf(logfile, "Accepted unauthenticated connection.\n")
		}
	}
	/* Optionally export/reimport the context a few times. */
	if export {
		for i := 0; i < 3; i++ {
			major, minor, contextToken := gss.ExportSecContext(ctx)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("exporting a context", major, minor, &mech)
			}
			major, minor, ctx = gss.ImportSecContext(contextToken)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("importing a context", major, minor, &mech)
			}
		}
	}
	/* Start processing message tokens from the client. */
	if ctx != nil {
		if len(client) > 0 {
			fmt.Printf("Accepted connection: \"%s\"\n", client)
		} else {
			fmt.Printf("Accepted connection.\n")
		}
	} else {
		fmt.Printf("Accepted unauthenticated connection.\n")
	}
	for {
		/* Read a request. */
		tag, token := misc.RecvToken(conn)
		if tag == 0 && len(token) == 0 {
			if verbose {
				fmt.Printf("EOF from client.\n")
			}
			return
		}
		/* Client indicates EOF with another NOOP token. */
		if tag&misc.TOKEN_NOOP != 0 {
			if logfile != nil {
				fmt.Fprintf(logfile, "NOOP token\n")
			}
			break
		}
		/* Expect data tokens. */
		if tag&misc.TOKEN_DATA == 0 {
			fmt.Printf("Expected data token, got %d token instead.\n", tag)
			break
		}
		if verbose && logfile != nil {
			fmt.Fprintf(logfile, "Message token (flags=%d):\n", tag)
			dump(logfile, token)
		}
		/* No context handle means no encryption or signing. */
		if ctx == nil && (tag&(misc.TOKEN_WRAPPED|misc.TOKEN_ENCRYPTED|misc.TOKEN_SEND_MIC)) != 0 {
			if logfile != nil {
				fmt.Fprintf(logfile, "Unauthenticated client requested authenticated services!\n")
			}
			break
		}
		/* If it's wrapped at all, unwrap it. */
		if tag&misc.TOKEN_WRAPPED != 0 {
			major, minor, conf, _, token = gss.Unwrap(ctx, token)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("unwrapping message", major, minor, &mech)
				break
			}
			/* If we were told it was encrypted, and it wasn't, warn. */
			if !conf && misc.TOKEN_ENCRYPTED != 0 {
				fmt.Printf("Warning!  Message not encrypted.\n")
			}
		}
		/* Log it. */
		if logfile != nil {
			fmt.Fprintf(logfile, "Received message: ")
			if token[0] >= 32 && token[0] < 127 && token[1] >= 32 && token[1] < 127 {
				buf := bytes.NewBuffer(token)
				fmt.Fprintf(logfile, "\"%s\"\n", buf)
			} else {
				fmt.Fprintf(logfile, "\n")
				dump(logfile, token)
			}
		}
		/* Reply. */
		if tag&misc.TOKEN_SEND_MIC != 0 {
			/* Send back a signature over the payload data. */
			major, minor, token := gss.GetMIC(ctx, gss.C_QOP_DEFAULT, token)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("signing message", major, minor, &mech)
				break
			}
			misc.SendToken(conn, misc.TOKEN_MIC, token)
		} else {
			/* Send back a minimal acknowledgement. */
			misc.SendToken(conn, misc.TOKEN_NOOP, nil)
		}
	}
}

func main() {
	port := flag.Int("port", 4444, "port")
	verbose := flag.Bool("verbose", false, "verbose")
	once := flag.Bool("once", false, "single-connection mode")
	export := flag.Bool("export", false, "export/reimport the context")
	keytab := flag.String("keytab", "", "keytab location")
	logfile := flag.String("logfile", "/dev/stdout", "log file for details")
	var log *os.File
	var err error

	flag.Parse()
	if flag.NArg() < 1 {
		fmt.Printf("Usage: gss-server [options] gss-service-name\n")
		flag.PrintDefaults()
		os.Exit(1)
	}
	service := flag.Arg(0)

	/* Open the log file. */
	if logfile != nil {
		log, err = os.OpenFile(*logfile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			fmt.Printf("Error opening log file \"%s\": %s\n", *logfile, err)
			return
		}
	}

	/* Set up the listener socket. */
	listener, err := net.Listen("tcp", ":"+strconv.Itoa(*port))
	if err != nil {
		fmt.Printf("Error listening for client connection: %s\n", err)
		return
	}
	defer listener.Close()

	/* Set up the server's name. */
	major, minor, name := gss.ImportName(service, gss.C_NT_HOSTBASED_SERVICE)
	if major != gss.S_COMPLETE {
		gss.DisplayGSSError("importing name", major, minor, nil)
		return
	}
	defer gss.ReleaseName(name)

	/* If we're told to use a particular keytab, do so. */
	if len(*keytab) > 0 {
		minor := gss.Krb5RegisterAcceptorIdentity(*keytab)
		if minor != 0 {
			gss.DisplayGSSError("registering acceptor identity", 0, minor, nil)
		}
	}

	/* Make sure we have acceptor creds. */
	major, minor, cred, _, _ := gss.AcquireCred(name, gss.C_INDEFINITE, nil, gss.C_ACCEPT)
	if major != gss.S_COMPLETE {
		gss.DisplayGSSError("acquiring credentials", major, minor, nil)
		return
	}

	/* Optionally export/reimport the acceptor cred a few times. */
	if *export {
		for i := 0; i < 3; i++ {
			major, minor, credToken := gss.ExportCred(cred)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("exporting a credential", major, minor, nil)
				return
			}
			major, minor, cred = gss.ImportCred(credToken)
			if major != gss.S_COMPLETE {
				gss.DisplayGSSError("importing a credential", major, minor, nil)
				return
			}
		}
	}
	defer gss.ReleaseCred(cred)

	fmt.Printf("starting...\n")
	if *once {
		/* Service exactly one client. */
		conn, err := listener.Accept()
		if err != nil {
			fmt.Printf("Error accepting client connection: %s\n", err)
			return
		}
		serve(conn, cred, *export, *verbose, log)
	} else {
		/* Just keep serving clients. */
		for {
			conn, err := listener.Accept()
			if err != nil {
				fmt.Printf("Error accepting client connection: %s\n", err)
				continue
			}
			go serve(conn, cred, *export, *verbose, log)
		}
	}
	return
}
