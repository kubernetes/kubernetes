package ec2_test

import (
	"fmt"
	"github.com/mitchellh/goamz/aws"
	"github.com/mitchellh/goamz/ec2"
	"github.com/mitchellh/goamz/ec2/ec2test"
	"github.com/mitchellh/goamz/testutil"
	. "github.com/motain/gocheck"
	"regexp"
	"sort"
)

// LocalServer represents a local ec2test fake server.
type LocalServer struct {
	auth   aws.Auth
	region aws.Region
	srv    *ec2test.Server
}

func (s *LocalServer) SetUp(c *C) {
	srv, err := ec2test.NewServer()
	c.Assert(err, IsNil)
	c.Assert(srv, NotNil)

	s.srv = srv
	s.region = aws.Region{EC2Endpoint: srv.URL()}
}

// LocalServerSuite defines tests that will run
// against the local ec2test server. It includes
// selected tests from ClientTests;
// when the ec2test functionality is sufficient, it should
// include all of them, and ClientTests can be simply embedded.
type LocalServerSuite struct {
	srv LocalServer
	ServerTests
	clientTests ClientTests
}

var _ = Suite(&LocalServerSuite{})

func (s *LocalServerSuite) SetUpSuite(c *C) {
	s.srv.SetUp(c)
	s.ServerTests.ec2 = ec2.NewWithClient(s.srv.auth, s.srv.region, testutil.DefaultClient)
	s.clientTests.ec2 = ec2.NewWithClient(s.srv.auth, s.srv.region, testutil.DefaultClient)
}

func (s *LocalServerSuite) TestRunAndTerminate(c *C) {
	s.clientTests.TestRunAndTerminate(c)
}

func (s *LocalServerSuite) TestSecurityGroups(c *C) {
	s.clientTests.TestSecurityGroups(c)
}

// TestUserData is not defined on ServerTests because it
// requires the ec2test server to function.
func (s *LocalServerSuite) TestUserData(c *C) {
	data := make([]byte, 256)
	for i := range data {
		data[i] = byte(i)
	}
	inst, err := s.ec2.RunInstances(&ec2.RunInstances{
		ImageId:      imageId,
		InstanceType: "t1.micro",
		UserData:     data,
	})
	c.Assert(err, IsNil)
	c.Assert(inst, NotNil)
	c.Assert(inst.Instances[0].DNSName, Equals, inst.Instances[0].InstanceId+".example.com")

	id := inst.Instances[0].InstanceId

	defer s.ec2.TerminateInstances([]string{id})

	tinst := s.srv.srv.Instance(id)
	c.Assert(tinst, NotNil)
	c.Assert(tinst.UserData, DeepEquals, data)
}

// AmazonServerSuite runs the ec2test server tests against a live EC2 server.
// It will only be activated if the -all flag is specified.
type AmazonServerSuite struct {
	srv AmazonServer
	ServerTests
}

var _ = Suite(&AmazonServerSuite{})

func (s *AmazonServerSuite) SetUpSuite(c *C) {
	if !testutil.Amazon {
		c.Skip("AmazonServerSuite tests not enabled")
	}
	s.srv.SetUp(c)
	s.ServerTests.ec2 = ec2.NewWithClient(s.srv.auth, aws.USEast, testutil.DefaultClient)
}

// ServerTests defines a set of tests designed to test
// the ec2test local fake ec2 server.
// It is not used as a test suite in itself, but embedded within
// another type.
type ServerTests struct {
	ec2 *ec2.EC2
}

func terminateInstances(c *C, e *ec2.EC2, insts []*ec2.Instance) {
	var ids []string
	for _, inst := range insts {
		if inst != nil {
			ids = append(ids, inst.InstanceId)
		}
	}
	_, err := e.TerminateInstances(ids)
	c.Check(err, IsNil, Commentf("%d INSTANCES LEFT RUNNING!!!", len(ids)))
}

func (s *ServerTests) makeTestGroup(c *C, name, descr string) ec2.SecurityGroup {
	// Clean it up if a previous test left it around.
	_, err := s.ec2.DeleteSecurityGroup(ec2.SecurityGroup{Name: name})
	if err != nil && err.(*ec2.Error).Code != "InvalidGroup.NotFound" {
		c.Fatalf("delete security group: %v", err)
	}

	resp, err := s.ec2.CreateSecurityGroup(ec2.SecurityGroup{Name: name, Description: descr})
	c.Assert(err, IsNil)
	c.Assert(resp.Name, Equals, name)
	return resp.SecurityGroup
}

func (s *ServerTests) TestIPPerms(c *C) {
	g0 := s.makeTestGroup(c, "goamz-test0", "ec2test group 0")
	defer s.ec2.DeleteSecurityGroup(g0)

	g1 := s.makeTestGroup(c, "goamz-test1", "ec2test group 1")
	defer s.ec2.DeleteSecurityGroup(g1)

	resp, err := s.ec2.SecurityGroups([]ec2.SecurityGroup{g0, g1}, nil)
	c.Assert(err, IsNil)
	c.Assert(resp.Groups, HasLen, 2)
	c.Assert(resp.Groups[0].IPPerms, HasLen, 0)
	c.Assert(resp.Groups[1].IPPerms, HasLen, 0)

	ownerId := resp.Groups[0].OwnerId

	// test some invalid parameters
	// TODO more
	_, err = s.ec2.AuthorizeSecurityGroup(g0, []ec2.IPPerm{{
		Protocol:  "tcp",
		FromPort:  0,
		ToPort:    1024,
		SourceIPs: []string{"z127.0.0.1/24"},
	}})
	c.Assert(err, NotNil)
	c.Check(err.(*ec2.Error).Code, Equals, "InvalidPermission.Malformed")

	// Check that AuthorizeSecurityGroup adds the correct authorizations.
	_, err = s.ec2.AuthorizeSecurityGroup(g0, []ec2.IPPerm{{
		Protocol:  "tcp",
		FromPort:  2000,
		ToPort:    2001,
		SourceIPs: []string{"127.0.0.0/24"},
		SourceGroups: []ec2.UserSecurityGroup{{
			Name: g1.Name,
		}, {
			Id: g0.Id,
		}},
	}, {
		Protocol:  "tcp",
		FromPort:  2000,
		ToPort:    2001,
		SourceIPs: []string{"200.1.1.34/32"},
	}})
	c.Assert(err, IsNil)

	resp, err = s.ec2.SecurityGroups([]ec2.SecurityGroup{g0}, nil)
	c.Assert(err, IsNil)
	c.Assert(resp.Groups, HasLen, 1)
	c.Assert(resp.Groups[0].IPPerms, HasLen, 1)

	perm := resp.Groups[0].IPPerms[0]
	srcg := perm.SourceGroups
	c.Assert(srcg, HasLen, 2)

	// Normalize so we don't care about returned order.
	if srcg[0].Name == g1.Name {
		srcg[0], srcg[1] = srcg[1], srcg[0]
	}
	c.Check(srcg[0].Name, Equals, g0.Name)
	c.Check(srcg[0].Id, Equals, g0.Id)
	c.Check(srcg[0].OwnerId, Equals, ownerId)
	c.Check(srcg[1].Name, Equals, g1.Name)
	c.Check(srcg[1].Id, Equals, g1.Id)
	c.Check(srcg[1].OwnerId, Equals, ownerId)

	sort.Strings(perm.SourceIPs)
	c.Check(perm.SourceIPs, DeepEquals, []string{"127.0.0.0/24", "200.1.1.34/32"})

	// Check that we can't delete g1 (because g0 is using it)
	_, err = s.ec2.DeleteSecurityGroup(g1)
	c.Assert(err, NotNil)
	c.Check(err.(*ec2.Error).Code, Equals, "InvalidGroup.InUse")

	_, err = s.ec2.RevokeSecurityGroup(g0, []ec2.IPPerm{{
		Protocol:     "tcp",
		FromPort:     2000,
		ToPort:       2001,
		SourceGroups: []ec2.UserSecurityGroup{{Id: g1.Id}},
	}, {
		Protocol:  "tcp",
		FromPort:  2000,
		ToPort:    2001,
		SourceIPs: []string{"200.1.1.34/32"},
	}})
	c.Assert(err, IsNil)

	resp, err = s.ec2.SecurityGroups([]ec2.SecurityGroup{g0}, nil)
	c.Assert(err, IsNil)
	c.Assert(resp.Groups, HasLen, 1)
	c.Assert(resp.Groups[0].IPPerms, HasLen, 1)

	perm = resp.Groups[0].IPPerms[0]
	srcg = perm.SourceGroups
	c.Assert(srcg, HasLen, 1)
	c.Check(srcg[0].Name, Equals, g0.Name)
	c.Check(srcg[0].Id, Equals, g0.Id)
	c.Check(srcg[0].OwnerId, Equals, ownerId)

	c.Check(perm.SourceIPs, DeepEquals, []string{"127.0.0.0/24"})

	// We should be able to delete g1 now because we've removed its only use.
	_, err = s.ec2.DeleteSecurityGroup(g1)
	c.Assert(err, IsNil)

	_, err = s.ec2.DeleteSecurityGroup(g0)
	c.Assert(err, IsNil)

	f := ec2.NewFilter()
	f.Add("group-id", g0.Id, g1.Id)
	resp, err = s.ec2.SecurityGroups(nil, f)
	c.Assert(err, IsNil)
	c.Assert(resp.Groups, HasLen, 0)
}

func (s *ServerTests) TestDuplicateIPPerm(c *C) {
	name := "goamz-test"
	descr := "goamz security group for tests"

	// Clean it up, if a previous test left it around and avoid leaving it around.
	s.ec2.DeleteSecurityGroup(ec2.SecurityGroup{Name: name})
	defer s.ec2.DeleteSecurityGroup(ec2.SecurityGroup{Name: name})

	resp1, err := s.ec2.CreateSecurityGroup(ec2.SecurityGroup{Name: name, Description: descr})
	c.Assert(err, IsNil)
	c.Assert(resp1.Name, Equals, name)

	perms := []ec2.IPPerm{{
		Protocol:  "tcp",
		FromPort:  200,
		ToPort:    1024,
		SourceIPs: []string{"127.0.0.1/24"},
	}, {
		Protocol:  "tcp",
		FromPort:  0,
		ToPort:    100,
		SourceIPs: []string{"127.0.0.1/24"},
	}}

	_, err = s.ec2.AuthorizeSecurityGroup(ec2.SecurityGroup{Name: name}, perms[0:1])
	c.Assert(err, IsNil)

	_, err = s.ec2.AuthorizeSecurityGroup(ec2.SecurityGroup{Name: name}, perms[0:2])
	c.Assert(err, ErrorMatches, `.*\(InvalidPermission.Duplicate\)`)
}

type filterSpec struct {
	name   string
	values []string
}

func (s *ServerTests) TestInstanceFiltering(c *C) {
	groupResp, err := s.ec2.CreateSecurityGroup(ec2.SecurityGroup{Name: sessionName("testgroup1"), Description: "testgroup one description"})
	c.Assert(err, IsNil)
	group1 := groupResp.SecurityGroup
	defer s.ec2.DeleteSecurityGroup(group1)

	groupResp, err = s.ec2.CreateSecurityGroup(ec2.SecurityGroup{Name: sessionName("testgroup2"), Description: "testgroup two description"})
	c.Assert(err, IsNil)
	group2 := groupResp.SecurityGroup
	defer s.ec2.DeleteSecurityGroup(group2)

	insts := make([]*ec2.Instance, 3)
	inst, err := s.ec2.RunInstances(&ec2.RunInstances{
		MinCount:       2,
		ImageId:        imageId,
		InstanceType:   "t1.micro",
		SecurityGroups: []ec2.SecurityGroup{group1},
	})
	c.Assert(err, IsNil)
	insts[0] = &inst.Instances[0]
	insts[1] = &inst.Instances[1]
	defer terminateInstances(c, s.ec2, insts)

	imageId2 := "ami-e358958a" // Natty server, i386, EBS store
	inst, err = s.ec2.RunInstances(&ec2.RunInstances{
		ImageId:        imageId2,
		InstanceType:   "t1.micro",
		SecurityGroups: []ec2.SecurityGroup{group2},
	})
	c.Assert(err, IsNil)
	insts[2] = &inst.Instances[0]

	ids := func(indices ...int) (instIds []string) {
		for _, index := range indices {
			instIds = append(instIds, insts[index].InstanceId)
		}
		return
	}

	tests := []struct {
		about       string
		instanceIds []string     // instanceIds argument to Instances method.
		filters     []filterSpec // filters argument to Instances method.
		resultIds   []string     // set of instance ids of expected results.
		allowExtra  bool         // resultIds may be incomplete.
		err         string       // expected error.
	}{
		{
			about:      "check that Instances returns all instances",
			resultIds:  ids(0, 1, 2),
			allowExtra: true,
		}, {
			about:       "check that specifying two instance ids returns them",
			instanceIds: ids(0, 2),
			resultIds:   ids(0, 2),
		}, {
			about:       "check that specifying a non-existent instance id gives an error",
			instanceIds: append(ids(0), "i-deadbeef"),
			err:         `.*\(InvalidInstanceID\.NotFound\)`,
		}, {
			about: "check that a filter allowed both instances returns both of them",
			filters: []filterSpec{
				{"instance-id", ids(0, 2)},
			},
			resultIds: ids(0, 2),
		}, {
			about: "check that a filter allowing only one instance returns it",
			filters: []filterSpec{
				{"instance-id", ids(1)},
			},
			resultIds: ids(1),
		}, {
			about: "check that a filter allowing no instances returns none",
			filters: []filterSpec{
				{"instance-id", []string{"i-deadbeef12345"}},
			},
		}, {
			about: "check that filtering on group id works",
			filters: []filterSpec{
				{"group-id", []string{group1.Id}},
			},
			resultIds: ids(0, 1),
		}, {
			about: "check that filtering on group name works",
			filters: []filterSpec{
				{"group-name", []string{group1.Name}},
			},
			resultIds: ids(0, 1),
		}, {
			about: "check that filtering on image id works",
			filters: []filterSpec{
				{"image-id", []string{imageId}},
			},
			resultIds:  ids(0, 1),
			allowExtra: true,
		}, {
			about: "combination filters 1",
			filters: []filterSpec{
				{"image-id", []string{imageId, imageId2}},
				{"group-name", []string{group1.Name}},
			},
			resultIds: ids(0, 1),
		}, {
			about: "combination filters 2",
			filters: []filterSpec{
				{"image-id", []string{imageId2}},
				{"group-name", []string{group1.Name}},
			},
		},
	}
	for i, t := range tests {
		c.Logf("%d. %s", i, t.about)
		var f *ec2.Filter
		if t.filters != nil {
			f = ec2.NewFilter()
			for _, spec := range t.filters {
				f.Add(spec.name, spec.values...)
			}
		}
		resp, err := s.ec2.Instances(t.instanceIds, f)
		if t.err != "" {
			c.Check(err, ErrorMatches, t.err)
			continue
		}
		c.Assert(err, IsNil)
		insts := make(map[string]*ec2.Instance)
		for _, r := range resp.Reservations {
			for j := range r.Instances {
				inst := &r.Instances[j]
				c.Check(insts[inst.InstanceId], IsNil, Commentf("duplicate instance id: %q", inst.InstanceId))
				insts[inst.InstanceId] = inst
			}
		}
		if !t.allowExtra {
			c.Check(insts, HasLen, len(t.resultIds), Commentf("expected %d instances got %#v", len(t.resultIds), insts))
		}
		for j, id := range t.resultIds {
			c.Check(insts[id], NotNil, Commentf("instance id %d (%q) not found; got %#v", j, id, insts))
		}
	}
}

func idsOnly(gs []ec2.SecurityGroup) []ec2.SecurityGroup {
	for i := range gs {
		gs[i].Name = ""
	}
	return gs
}

func namesOnly(gs []ec2.SecurityGroup) []ec2.SecurityGroup {
	for i := range gs {
		gs[i].Id = ""
	}
	return gs
}

func (s *ServerTests) TestGroupFiltering(c *C) {
	g := make([]ec2.SecurityGroup, 4)
	for i := range g {
		resp, err := s.ec2.CreateSecurityGroup(ec2.SecurityGroup{Name: sessionName(fmt.Sprintf("testgroup%d", i)), Description: fmt.Sprintf("testdescription%d", i)})
		c.Assert(err, IsNil)
		g[i] = resp.SecurityGroup
		c.Logf("group %d: %v", i, g[i])
		defer s.ec2.DeleteSecurityGroup(g[i])
	}

	perms := [][]ec2.IPPerm{
		{{
			Protocol:  "tcp",
			FromPort:  100,
			ToPort:    200,
			SourceIPs: []string{"1.2.3.4/32"},
		}},
		{{
			Protocol:     "tcp",
			FromPort:     200,
			ToPort:       300,
			SourceGroups: []ec2.UserSecurityGroup{{Id: g[1].Id}},
		}},
		{{
			Protocol:     "udp",
			FromPort:     200,
			ToPort:       400,
			SourceGroups: []ec2.UserSecurityGroup{{Id: g[1].Id}},
		}},
	}
	for i, ps := range perms {
		_, err := s.ec2.AuthorizeSecurityGroup(g[i], ps)
		c.Assert(err, IsNil)
	}

	groups := func(indices ...int) (gs []ec2.SecurityGroup) {
		for _, index := range indices {
			gs = append(gs, g[index])
		}
		return
	}

	type groupTest struct {
		about      string
		groups     []ec2.SecurityGroup // groupIds argument to SecurityGroups method.
		filters    []filterSpec        // filters argument to SecurityGroups method.
		results    []ec2.SecurityGroup // set of expected result groups.
		allowExtra bool                // specified results may be incomplete.
		err        string              // expected error.
	}
	filterCheck := func(name, val string, gs []ec2.SecurityGroup) groupTest {
		return groupTest{
			about:      "filter check " + name,
			filters:    []filterSpec{{name, []string{val}}},
			results:    gs,
			allowExtra: true,
		}
	}
	tests := []groupTest{
		{
			about:      "check that SecurityGroups returns all groups",
			results:    groups(0, 1, 2, 3),
			allowExtra: true,
		}, {
			about:   "check that specifying two group ids returns them",
			groups:  idsOnly(groups(0, 2)),
			results: groups(0, 2),
		}, {
			about:   "check that specifying names only works",
			groups:  namesOnly(groups(0, 2)),
			results: groups(0, 2),
		}, {
			about:  "check that specifying a non-existent group id gives an error",
			groups: append(groups(0), ec2.SecurityGroup{Id: "sg-eeeeeeeee"}),
			err:    `.*\(InvalidGroup\.NotFound\)`,
		}, {
			about: "check that a filter allowed two groups returns both of them",
			filters: []filterSpec{
				{"group-id", []string{g[0].Id, g[2].Id}},
			},
			results: groups(0, 2),
		},
		{
			about:  "check that the previous filter works when specifying a list of ids",
			groups: groups(1, 2),
			filters: []filterSpec{
				{"group-id", []string{g[0].Id, g[2].Id}},
			},
			results: groups(2),
		}, {
			about: "check that a filter allowing no groups returns none",
			filters: []filterSpec{
				{"group-id", []string{"sg-eeeeeeeee"}},
			},
		},
		filterCheck("description", "testdescription1", groups(1)),
		filterCheck("group-name", g[2].Name, groups(2)),
		filterCheck("ip-permission.cidr", "1.2.3.4/32", groups(0)),
		filterCheck("ip-permission.group-name", g[1].Name, groups(1, 2)),
		filterCheck("ip-permission.protocol", "udp", groups(2)),
		filterCheck("ip-permission.from-port", "200", groups(1, 2)),
		filterCheck("ip-permission.to-port", "200", groups(0)),
		// TODO owner-id
	}
	for i, t := range tests {
		c.Logf("%d. %s", i, t.about)
		var f *ec2.Filter
		if t.filters != nil {
			f = ec2.NewFilter()
			for _, spec := range t.filters {
				f.Add(spec.name, spec.values...)
			}
		}
		resp, err := s.ec2.SecurityGroups(t.groups, f)
		if t.err != "" {
			c.Check(err, ErrorMatches, t.err)
			continue
		}
		c.Assert(err, IsNil)
		groups := make(map[string]*ec2.SecurityGroup)
		for j := range resp.Groups {
			group := &resp.Groups[j].SecurityGroup
			c.Check(groups[group.Id], IsNil, Commentf("duplicate group id: %q", group.Id))

			groups[group.Id] = group
		}
		// If extra groups may be returned, eliminate all groups that
		// we did not create in this session apart from the default group.
		if t.allowExtra {
			namePat := regexp.MustCompile(sessionName("testgroup[0-9]"))
			for id, g := range groups {
				if !namePat.MatchString(g.Name) {
					delete(groups, id)
				}
			}
		}
		c.Check(groups, HasLen, len(t.results))
		for j, g := range t.results {
			rg := groups[g.Id]
			c.Assert(rg, NotNil, Commentf("group %d (%v) not found; got %#v", j, g, groups))
			c.Check(rg.Name, Equals, g.Name, Commentf("group %d (%v)", j, g))
		}
	}
}
