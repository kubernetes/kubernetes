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

package session

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/sts"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
)

type login struct {
	*flags.ClientFlag
	*flags.OutputFlag

	clone  bool
	issue  bool
	renew  bool
	long   bool
	ticket string
	life   time.Duration
	cookie string
	token  string
	ext    string
}

func init() {
	cli.Register("session.login", &login{})
}

func (cmd *login) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	f.BoolVar(&cmd.clone, "clone", false, "Acquire clone ticket")
	f.BoolVar(&cmd.issue, "issue", false, "Issue SAML token")
	f.BoolVar(&cmd.renew, "renew", false, "Renew SAML token")
	f.DurationVar(&cmd.life, "lifetime", time.Minute*10, "SAML token lifetime")
	f.BoolVar(&cmd.long, "l", false, "Output session cookie")
	f.StringVar(&cmd.ticket, "ticket", "", "Use clone ticket for login")
	f.StringVar(&cmd.cookie, "cookie", "", "Set HTTP cookie for an existing session")
	f.StringVar(&cmd.token, "token", "", "Use SAML token for login or as issue identity")
	f.StringVar(&cmd.ext, "extension", "", "Extension name")
}

func (cmd *login) Process(ctx context.Context) error {
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return cmd.ClientFlag.Process(ctx)
}

func (cmd *login) Description() string {
	return `Session login.

The session.login command is optional, all other govc commands will auto login when given credentials.
The session.login command can be used to:
- Persist a session without writing to disk via the '-cookie' flag
- Acquire a clone ticket
- Login using a clone ticket
- Login using a vCenter Extension certificate
- Issue a SAML token
- Renew a SAML token
- Login using a SAML token
- Avoid passing credentials to other govc commands

Examples:
  govc session.login -u root:password@host
  ticket=$(govc session.login -u root@host -clone)
  govc session.login -u root@host -ticket $ticket
  govc session.login -u host -extension com.vmware.vsan.health -cert rui.crt -key rui.key
  token=$(govc session.login -u host -cert user.crt -key user.key -issue) # HoK token
  bearer=$(govc session.login -u user:pass@host -issue) # Bearer token
  token=$(govc session.login -u host -cert user.crt -key user.key -issue -token "$bearer")
  govc session.login -u host -cert user.crt -key user.key -token "$token"
  token=$(govc session.login -u host -cert user.crt -key user.key -renew -lifetime 24h -token "$token")`
}

type ticketResult struct {
	cmd    *login
	Ticket string `json:",omitempty"`
	Token  string `json:",omitempty"`
	Cookie string `json:",omitempty"`
}

func (r *ticketResult) Write(w io.Writer) error {
	var output []string

	for _, val := range []string{r.Ticket, r.Token, r.Cookie} {
		if val != "" {
			output = append(output, val)
		}
	}

	if len(output) == 0 {
		return nil
	}

	fmt.Fprintln(w, strings.Join(output, " "))

	return nil
}

// Logout is called by cli.Run()
// We override ClientFlag's Logout impl to avoid ending a session when -persist-session=false,
// otherwise Logout would invalidate the cookie and/or ticket.
func (cmd *login) Logout(ctx context.Context) error {
	if cmd.long || cmd.clone || cmd.issue {
		return nil
	}
	return cmd.ClientFlag.Logout(ctx)
}

func (cmd *login) cloneSession(ctx context.Context, c *vim25.Client) error {
	return session.NewManager(c).CloneSession(ctx, cmd.ticket)
}

func (cmd *login) issueToken(ctx context.Context, vc *vim25.Client) (string, error) {
	c, err := sts.NewClient(ctx, vc)
	if err != nil {
		return "", err
	}

	req := sts.TokenRequest{
		Certificate: c.Certificate(),
		Userinfo:    cmd.Userinfo(),
		Renewable:   true,
		Delegatable: true,
		ActAs:       cmd.token != "",
		Token:       cmd.token,
		Lifetime:    cmd.life,
	}

	issue := c.Issue
	if cmd.renew {
		issue = c.Renew
	}

	s, err := issue(ctx, req)
	if err != nil {
		return "", err
	}

	if req.Token != "" {
		duration := s.Lifetime.Expires.Sub(s.Lifetime.Created)
		if duration < req.Lifetime {
			// The granted lifetime is that of the bearer token, which is 5min max.
			// Extend the lifetime via Renew.
			req.Token = s.Token
			if s, err = c.Renew(ctx, req); err != nil {
				return "", err
			}
		}
	}

	return s.Token, nil
}

func (cmd *login) loginByToken(ctx context.Context, c *vim25.Client) error {
	header := soap.Header{
		Security: &sts.Signer{
			Certificate: c.Certificate(),
			Token:       cmd.token,
		},
	}

	return session.NewManager(c).LoginByToken(c.WithHeader(ctx, header))
}

func (cmd *login) loginByExtension(ctx context.Context, c *vim25.Client) error {
	return session.NewManager(c).LoginExtensionByCertificate(ctx, cmd.ext)
}

func (cmd *login) setCookie(ctx context.Context, c *vim25.Client) error {
	url := c.URL()
	jar := c.Client.Jar
	cookies := jar.Cookies(url)
	add := true

	cookie := &http.Cookie{
		Name: soap.SessionCookieName,
	}

	for _, e := range cookies {
		if e.Name == cookie.Name {
			add = false
			cookie = e
			break
		}
	}

	if cmd.cookie == "" {
		// This is the cookie from Set-Cookie after a Login or CloneSession
		cmd.cookie = cookie.Value
	} else {
		// The cookie flag is set, set the HTTP header and skip Login()
		cookie.Value = cmd.cookie
		if add {
			cookies = append(cookies, cookie)
		}
		jar.SetCookies(url, cookies)

		// Check the session is still valid
		_, err := methods.GetCurrentTime(ctx, c)
		if err != nil {
			return err
		}
	}

	return nil
}

func (cmd *login) Run(ctx context.Context, f *flag.FlagSet) error {
	if cmd.renew {
		cmd.issue = true
	}
	switch {
	case cmd.ticket != "":
		cmd.Login = cmd.cloneSession
	case cmd.cookie != "":
		cmd.Login = cmd.setCookie
	case cmd.token != "":
		cmd.Login = cmd.loginByToken
	case cmd.ext != "":
		cmd.Login = cmd.loginByExtension
	case cmd.issue:
		cmd.Login = func(_ context.Context, _ *vim25.Client) error {
			return nil
		}
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m := session.NewManager(c)
	r := &ticketResult{cmd: cmd}

	switch {
	case cmd.clone:
		r.Ticket, err = m.AcquireCloneTicket(ctx)
		if err != nil {
			return err
		}
	case cmd.issue:
		r.Token, err = cmd.issueToken(ctx, c)
		if err != nil {
			return err
		}
		return cmd.WriteResult(r)
	}

	if cmd.cookie == "" {
		_ = cmd.setCookie(ctx, c)
		if cmd.cookie == "" {
			return flag.ErrHelp
		}
	}

	if cmd.long {
		r.Cookie = cmd.cookie
	}

	return cmd.WriteResult(r)
}
