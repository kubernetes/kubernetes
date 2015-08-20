package ldap_test

import (
	"crypto/tls"
	"fmt"
	"log"

	"github.com/go-ldap/ldap"
)

// ExampleConn_Bind demonstrats how to bind a connection to an ldap user
// allowing access to restricted attrabutes that user has access to
func ExampleConn_Bind() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	err = l.Bind("cn=read-only-admin,dc=example,dc=com", "password")
	if err != nil {
		log.Fatal(err)
	}
}

// ExampleConn_Search demonstrates how to use the search interface
func ExampleConn_Search() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	searchRequest := ldap.NewSearchRequest(
		"dc=example,dc=com", // The base dn to search
		ldap.ScopeWholeSubtree, ldap.NeverDerefAliases, 0, 0, false,
		"(&(objectClass=organizationalPerson))", // The filter to apply
		[]string{"dn", "cn"},                    // A list attributes to retrieve
		nil,
	)

	sr, err := l.Search(searchRequest)
	if err != nil {
		log.Fatal(err)
	}

	for _, entry := range sr.Entries {
		fmt.Printf("%s: %v\n", entry.DN, entry.GetAttributeValue("cn"))
	}
}

// ExampleStartTLS demonstrates how to start a TLS connection
func ExampleConn_StartTLS() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	// Reconnect with TLS
	err = l.StartTLS(&tls.Config{InsecureSkipVerify: true})
	if err != nil {
		log.Fatal(err)
	}

	// Opertations via l are now encrypted
}

// ExampleConn_Compare demonstrates how to comapre an attribute with a value
func ExampleConn_Compare() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	matched, err := l.Compare("cn=user,dc=example,dc=com", "uid", "someuserid")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(matched)
}

func ExampleConn_PasswordModify_admin() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	err = l.Bind("cn=admin,dc=example,dc=com", "password")
	if err != nil {
		log.Fatal(err)
	}

	passwordModifyRequest := ldap.NewPasswordModifyRequest("cn=user,dc=example,dc=com", "", "NewPassword")
	_, err = l.PasswordModify(passwordModifyRequest)

	if err != nil {
		log.Fatalf("Password could not be changed: %s", err.Error())
	}
}

func ExampleConn_PasswordModify_generatedPassword() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	err = l.Bind("cn=user,dc=example,dc=com", "password")
	if err != nil {
		log.Fatal(err)
	}

	passwordModifyRequest := ldap.NewPasswordModifyRequest("", "OldPassword", "")
	passwordModifyResponse, err := l.PasswordModify(passwordModifyRequest)
	if err != nil {
		log.Fatalf("Password could not be changed: %s", err.Error())
	}

	generatedPassword := passwordModifyResponse.GeneratedPassword
	log.Printf("Generated password: %s\n", generatedPassword)
}

func ExampleConn_PasswordModify_setNewPassword() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	err = l.Bind("cn=user,dc=example,dc=com", "password")
	if err != nil {
		log.Fatal(err)
	}

	passwordModifyRequest := ldap.NewPasswordModifyRequest("", "OldPassword", "NewPassword")
	_, err = l.PasswordModify(passwordModifyRequest)

	if err != nil {
		log.Fatalf("Password could not be changed: %s", err.Error())
	}
}

func ExampleConn_Modify() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	// Add a description, and replace the mail attributes
	modify := ldap.NewModifyRequest("cn=user,dc=example,dc=com")
	modify.Add("description", []string{"An example user"})
	modify.Replace("mail", []string{"user@example.org"})

	err = l.Modify(modify)
	if err != nil {
		log.Fatal(err)
	}
}

// Example User Authentication shows how a typical application can verify a login attempt
func Example_userAuthentication() {
	// The username and password we want to check
	username := "someuser"
	password := "userpassword"

	bindusername := "readonly"
	bindpassword := "password"

	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	// Reconnect with TLS
	err = l.StartTLS(&tls.Config{InsecureSkipVerify: true})
	if err != nil {
		log.Fatal(err)
	}

	// First bind with a read only user
	err = l.Bind(bindusername, bindpassword)
	if err != nil {
		log.Fatal(err)
	}

	// Search for the given username
	searchRequest := ldap.NewSearchRequest(
		"dc=example,dc=com",
		ldap.ScopeWholeSubtree, ldap.NeverDerefAliases, 0, 0, false,
		fmt.Sprintf("(&(objectClass=organizationalPerson)&(uid=%s))", username),
		[]string{"dn"},
		nil,
	)

	sr, err := l.Search(searchRequest)
	if err != nil {
		log.Fatal(err)
	}

	if len(sr.Entries) != 1 {
		log.Fatal("User does not exist or too many entries returned")
	}

	userdn := sr.Entries[0].DN

	// Bind as the user to verify their password
	err = l.Bind(userdn, password)
	if err != nil {
		log.Fatal(err)
	}

	// Rebind as the read only user for any futher queries
	err = l.Bind(bindusername, bindpassword)
	if err != nil {
		log.Fatal(err)
	}
}

func Example_beherappolicy() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()

	controls := []ldap.Control{}
	controls = append(controls, ldap.NewControlBeheraPasswordPolicy())
	bindRequest := ldap.NewSimpleBindRequest("cn=admin,dc=example,dc=com", "password", controls)

	r, err := l.SimpleBind(bindRequest)
	ppolicyControl := ldap.FindControl(r.Controls, ldap.ControlTypeBeheraPasswordPolicy)

	var ppolicy *ldap.ControlBeheraPasswordPolicy
	if ppolicyControl != nil {
		ppolicy = ppolicyControl.(*ldap.ControlBeheraPasswordPolicy)
	} else {
		log.Printf("ppolicyControl response not avaliable.\n")
	}
	if err != nil {
		errStr := "ERROR: Cannot bind: " + err.Error()
		if ppolicy != nil && ppolicy.Error >= 0 {
			errStr += ":" + ppolicy.ErrorString
		}
		log.Print(errStr)
	} else {
		logStr := "Login Ok"
		if ppolicy != nil {
			if ppolicy.Expire >= 0 {
				logStr += fmt.Sprintf(". Password expires in %d seconds\n", ppolicy.Expire)
			} else if ppolicy.Grace >= 0 {
				logStr += fmt.Sprintf(". Password expired, %d grace logins remain\n", ppolicy.Grace)
			}
		}
		log.Print(logStr)
	}
}

func Example_vchuppolicy() {
	l, err := ldap.Dial("tcp", fmt.Sprintf("%s:%d", "ldap.example.com", 389))
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()
	l.Debug = true

	bindRequest := ldap.NewSimpleBindRequest("cn=admin,dc=example,dc=com", "password", nil)

	r, err := l.SimpleBind(bindRequest)

	passwordMustChangeControl := ldap.FindControl(r.Controls, ldap.ControlTypeVChuPasswordMustChange)
	var passwordMustChange *ldap.ControlVChuPasswordMustChange
	if passwordMustChangeControl != nil {
		passwordMustChange = passwordMustChangeControl.(*ldap.ControlVChuPasswordMustChange)
	}

	if passwordMustChange != nil && passwordMustChange.MustChange {
		log.Printf("Password Must be changed.\n")
	}

	passwordWarningControl := ldap.FindControl(r.Controls, ldap.ControlTypeVChuPasswordWarning)

	var passwordWarning *ldap.ControlVChuPasswordWarning
	if passwordWarningControl != nil {
		passwordWarning = passwordWarningControl.(*ldap.ControlVChuPasswordWarning)
	} else {
		log.Printf("ppolicyControl response not available.\n")
	}
	if err != nil {
		log.Print("ERROR: Cannot bind: " + err.Error())
	} else {
		logStr := "Login Ok"
		if passwordWarning != nil {
			if passwordWarning.Expire >= 0 {
				logStr += fmt.Sprintf(". Password expires in %d seconds\n", passwordWarning.Expire)
			}
		}
		log.Print(logStr)
	}
}
