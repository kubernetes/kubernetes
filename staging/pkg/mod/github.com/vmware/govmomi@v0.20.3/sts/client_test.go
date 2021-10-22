/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package sts

import (
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/pem"
	"log"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/ssoadmin"
	"github.com/vmware/govmomi/ssoadmin/types"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
)

// The following can help debug signature mismatch:
// % vi /usr/lib/vmware-sso/vmware-sts/conf/logging.properties
// # turn up logging for dsig:
// org.jcp.xml.dsig.internal.level = FINE
// com.sun.org.apache.xml.internal.security.level = FINE
// # restart the STS service:
// % service-control --stop vmware-stsd
// % service-control --start vmware-stsd
// % tail -f /var/log/vmware/sso/catalina.$(date +%Y-%m-%d).log

// solutionUserCreate ensures that solution user "govmomi-test" exists for uses with the tests that follow.
func solutionUserCreate(ctx context.Context, info *url.Userinfo, sts *Client) error {
	s, err := sts.Issue(ctx, TokenRequest{Userinfo: info})
	if err != nil {
		return err
	}

	admin, err := ssoadmin.NewClient(ctx, &vim25.Client{Client: sts.Client})
	if err != nil {
		return err
	}

	header := soap.Header{Security: s}
	if err = admin.Login(sts.WithHeader(ctx, header)); err != nil {
		return err
	}

	defer admin.Logout(ctx)

	id := types.PrincipalId{
		Name:   "govmomi-test",
		Domain: admin.Domain,
	}

	user, err := admin.FindSolutionUser(ctx, id.Name)
	if err != nil {
		return err
	}

	if user == nil {
		block, _ := pem.Decode([]byte(LocalhostCert))
		details := types.AdminSolutionDetails{
			Certificate: base64.StdEncoding.EncodeToString(block.Bytes),
			Description: "govmomi test solution user",
		}

		if err = admin.CreateSolutionUser(ctx, id.Name, details); err != nil {
			return err
		}
	}

	if _, err = admin.GrantWSTrustRole(ctx, id, types.RoleActAsUser); err != nil {
		return err
	}

	_, err = admin.SetRole(ctx, id, types.RoleAdministrator)
	return err
}

func solutionUserCert() *tls.Certificate {
	cert, err := tls.X509KeyPair(LocalhostCert, LocalhostKey)
	if err != nil {
		panic(err)
	}
	return &cert
}

func TestIssueHOK(t *testing.T) {
	ctx := context.Background()
	url := os.Getenv("GOVC_TEST_URL")
	if url == "" {
		t.SkipNow()
	}

	u, err := soap.ParseURL(url)
	if err != nil {
		t.Fatal(err)
	}

	c, err := vim25.NewClient(ctx, soap.NewClient(u, true))
	if err != nil {
		log.Fatal(err)
	}

	if !c.IsVC() {
		t.SkipNow()
	}

	sts, err := NewClient(ctx, c)
	if err != nil {
		t.Fatal(err)
	}

	if err = solutionUserCreate(ctx, u.User, sts); err != nil {
		t.Fatal(err)
	}

	req := TokenRequest{
		Certificate: solutionUserCert(),
		Delegatable: true,
	}

	s, err := sts.Issue(ctx, req)
	if err != nil {
		t.Fatal(err)
	}

	header := soap.Header{Security: s}

	err = session.NewManager(c).LoginByToken(c.WithHeader(ctx, header))
	if err != nil {
		t.Fatal(err)
	}

	now, err := methods.GetCurrentTime(ctx, c)
	if err != nil {
		t.Fatal(err)
	}

	log.Printf("current time=%s", now)
}

func TestIssueBearer(t *testing.T) {
	ctx := context.Background()
	url := os.Getenv("GOVC_TEST_URL")
	if url == "" {
		t.SkipNow()
	}

	u, err := soap.ParseURL(url)
	if err != nil {
		t.Fatal(err)
	}

	c, err := vim25.NewClient(ctx, soap.NewClient(u, true))
	if err != nil {
		log.Fatal(err)
	}

	if !c.IsVC() {
		t.SkipNow()
	}

	sts, err := NewClient(ctx, c)
	if err != nil {
		t.Fatal(err)
	}

	// Test that either Certificate or Userinfo is set.
	_, err = sts.Issue(ctx, TokenRequest{})
	if err == nil {
		t.Error("expected error")
	}

	req := TokenRequest{
		Userinfo: u.User,
	}

	s, err := sts.Issue(ctx, req)
	if err != nil {
		t.Fatal(err)
	}

	header := soap.Header{Security: s}

	err = session.NewManager(c).LoginByToken(c.WithHeader(ctx, header))
	if err != nil {
		t.Fatal(err)
	}

	now, err := methods.GetCurrentTime(ctx, c)
	if err != nil {
		t.Fatal(err)
	}

	log.Printf("current time=%s", now)
}

func TestIssueActAs(t *testing.T) {
	ctx := context.Background()
	url := os.Getenv("GOVC_TEST_URL")
	if url == "" {
		t.SkipNow()
	}

	u, err := soap.ParseURL(url)
	if err != nil {
		t.Fatal(err)
	}

	c, err := vim25.NewClient(ctx, soap.NewClient(u, true))
	if err != nil {
		log.Fatal(err)
	}

	if !c.IsVC() {
		t.SkipNow()
	}

	sts, err := NewClient(ctx, c)
	if err != nil {
		t.Fatal(err)
	}

	if err = solutionUserCreate(ctx, u.User, sts); err != nil {
		t.Fatal(err)
	}

	req := TokenRequest{
		Delegatable: true,
		Renewable:   true,
		Userinfo:    u.User,
	}

	s, err := sts.Issue(ctx, req)
	if err != nil {
		t.Fatal(err)
	}

	req = TokenRequest{
		Lifetime:    24 * time.Hour,
		Token:       s.Token,
		ActAs:       true,
		Delegatable: true,
		Renewable:   true,
		Certificate: solutionUserCert(),
	}

	s, err = sts.Issue(ctx, req)
	if err != nil {
		t.Fatal(err)
	}

	header := soap.Header{Security: s}

	err = session.NewManager(c).LoginByToken(c.WithHeader(ctx, header))
	if err != nil {
		t.Fatal(err)
	}

	now, err := methods.GetCurrentTime(ctx, c)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("current time=%s", now)

	duration := s.Lifetime.Expires.Sub(s.Lifetime.Created)
	if duration < req.Lifetime {
		req.Lifetime = 24 * time.Hour
		req.Token = s.Token
		log.Printf("extending lifetime from %s", s.Lifetime.Expires.Sub(s.Lifetime.Created))
		s, err = sts.Renew(ctx, req)
		if err != nil {
			t.Fatal(err)
		}
	} else {
		t.Errorf("duration=%s", duration)
	}

	t.Logf("expires in %s", s.Lifetime.Expires.Sub(s.Lifetime.Created))
}
