package denco_test

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/go-openapi/runtime/middleware/denco"
)

func routes() []denco.Record {
	return []denco.Record{
		{"/", "testroute0"},
		{"/path/to/route", "testroute1"},
		{"/path/to/other", "testroute2"},
		{"/path/to/route/a", "testroute3"},
		{"/path/to/:param", "testroute4"},
		{"/gists/:param1/foo/:param2", "testroute12"},
		{"/gists/:param1/foo/bar", "testroute11"},
		{"/:param1/:param2/foo/:param3", "testroute13"},
		{"/path/to/wildcard/*routepath", "testroute5"},
		{"/path/to/:param1/:param2", "testroute6"},
		{"/path/to/:param1/sep/:param2", "testroute7"},
		{"/:year/:month/:day", "testroute8"},
		{"/user/:id", "testroute9"},
		{"/a/to/b/:param/*routepath", "testroute10"},
	}
}

var realURIs = []denco.Record{
	{"/authorizations", "/authorizations"},
	{"/authorizations/:id", "/authorizations/:id"},
	{"/applications/:client_id/tokens/:access_token", "/applications/:client_id/tokens/:access_token"},
	{"/events", "/events"},
	{"/repos/:owner/:repo/events", "/repos/:owner/:repo/events"},
	{"/networks/:owner/:repo/events", "/networks/:owner/:repo/events"},
	{"/orgs/:org/events", "/orgs/:org/events"},
	{"/users/:user/received_events", "/users/:user/received_events"},
	{"/users/:user/received_events/public", "/users/:user/received_events/public"},
	{"/users/:user/events", "/users/:user/events"},
	{"/users/:user/events/public", "/users/:user/events/public"},
	{"/users/:user/events/orgs/:org", "/users/:user/events/orgs/:org"},
	{"/feeds", "/feeds"},
	{"/notifications", "/notifications"},
	{"/repos/:owner/:repo/notifications", "/repos/:owner/:repo/notifications"},
	{"/notifications/threads/:id", "/notifications/threads/:id"},
	{"/notifications/threads/:id/subscription", "/notifications/threads/:id/subscription"},
	{"/repos/:owner/:repo/stargazers", "/repos/:owner/:repo/stargazers"},
	{"/users/:user/starred", "/users/:user/starred"},
	{"/user/starred", "/user/starred"},
	{"/user/starred/:owner/:repo", "/user/starred/:owner/:repo"},
	{"/repos/:owner/:repo/subscribers", "/repos/:owner/:repo/subscribers"},
	{"/users/:user/subscriptions", "/users/:user/subscriptions"},
	{"/user/subscriptions", "/user/subscriptions"},
	{"/repos/:owner/:repo/subscription", "/repos/:owner/:repo/subscription"},
	{"/user/subscriptions/:owner/:repo", "/user/subscriptions/:owner/:repo"},
	{"/users/:user/gists", "/users/:user/gists"},
	{"/gists", "/gists"},
	{"/gists/:id", "/gists/:id"},
	{"/gists/:id/star", "/gists/:id/star"},
	{"/repos/:owner/:repo/git/blobs/:sha", "/repos/:owner/:repo/git/blobs/:sha"},
	{"/repos/:owner/:repo/git/commits/:sha", "/repos/:owner/:repo/git/commits/:sha"},
	{"/repos/:owner/:repo/git/refs", "/repos/:owner/:repo/git/refs"},
	{"/repos/:owner/:repo/git/tags/:sha", "/repos/:owner/:repo/git/tags/:sha"},
	{"/repos/:owner/:repo/git/trees/:sha", "/repos/:owner/:repo/git/trees/:sha"},
	{"/issues", "/issues"},
	{"/user/issues", "/user/issues"},
	{"/orgs/:org/issues", "/orgs/:org/issues"},
	{"/repos/:owner/:repo/issues", "/repos/:owner/:repo/issues"},
	{"/repos/:owner/:repo/issues/:number", "/repos/:owner/:repo/issues/:number"},
	{"/repos/:owner/:repo/assignees", "/repos/:owner/:repo/assignees"},
	{"/repos/:owner/:repo/assignees/:assignee", "/repos/:owner/:repo/assignees/:assignee"},
	{"/repos/:owner/:repo/issues/:number/comments", "/repos/:owner/:repo/issues/:number/comments"},
	{"/repos/:owner/:repo/issues/:number/events", "/repos/:owner/:repo/issues/:number/events"},
	{"/repos/:owner/:repo/labels", "/repos/:owner/:repo/labels"},
	{"/repos/:owner/:repo/labels/:name", "/repos/:owner/:repo/labels/:name"},
	{"/repos/:owner/:repo/issues/:number/labels", "/repos/:owner/:repo/issues/:number/labels"},
	{"/repos/:owner/:repo/milestones/:number/labels", "/repos/:owner/:repo/milestones/:number/labels"},
	{"/repos/:owner/:repo/milestones", "/repos/:owner/:repo/milestones"},
	{"/repos/:owner/:repo/milestones/:number", "/repos/:owner/:repo/milestones/:number"},
	{"/emojis", "/emojis"},
	{"/gitignore/templates", "/gitignore/templates"},
	{"/gitignore/templates/:name", "/gitignore/templates/:name"},
	{"/meta", "/meta"},
	{"/rate_limit", "/rate_limit"},
	{"/users/:user/orgs", "/users/:user/orgs"},
	{"/user/orgs", "/user/orgs"},
	{"/orgs/:org", "/orgs/:org"},
	{"/orgs/:org/members", "/orgs/:org/members"},
	{"/orgs/:org/members/:user", "/orgs/:org/members/:user"},
	{"/orgs/:org/public_members", "/orgs/:org/public_members"},
	{"/orgs/:org/public_members/:user", "/orgs/:org/public_members/:user"},
	{"/orgs/:org/teams", "/orgs/:org/teams"},
	{"/teams/:id", "/teams/:id"},
	{"/teams/:id/members", "/teams/:id/members"},
	{"/teams/:id/members/:user", "/teams/:id/members/:user"},
	{"/teams/:id/repos", "/teams/:id/repos"},
	{"/teams/:id/repos/:owner/:repo", "/teams/:id/repos/:owner/:repo"},
	{"/user/teams", "/user/teams"},
	{"/repos/:owner/:repo/pulls", "/repos/:owner/:repo/pulls"},
	{"/repos/:owner/:repo/pulls/:number", "/repos/:owner/:repo/pulls/:number"},
	{"/repos/:owner/:repo/pulls/:number/commits", "/repos/:owner/:repo/pulls/:number/commits"},
	{"/repos/:owner/:repo/pulls/:number/files", "/repos/:owner/:repo/pulls/:number/files"},
	{"/repos/:owner/:repo/pulls/:number/merge", "/repos/:owner/:repo/pulls/:number/merge"},
	{"/repos/:owner/:repo/pulls/:number/comments", "/repos/:owner/:repo/pulls/:number/comments"},
	{"/user/repos", "/user/repos"},
	{"/users/:user/repos", "/users/:user/repos"},
	{"/orgs/:org/repos", "/orgs/:org/repos"},
	{"/repositories", "/repositories"},
	{"/repos/:owner/:repo", "/repos/:owner/:repo"},
	{"/repos/:owner/:repo/contributors", "/repos/:owner/:repo/contributors"},
	{"/repos/:owner/:repo/languages", "/repos/:owner/:repo/languages"},
	{"/repos/:owner/:repo/teams", "/repos/:owner/:repo/teams"},
	{"/repos/:owner/:repo/tags", "/repos/:owner/:repo/tags"},
	{"/repos/:owner/:repo/branches", "/repos/:owner/:repo/branches"},
	{"/repos/:owner/:repo/branches/:branch", "/repos/:owner/:repo/branches/:branch"},
	{"/repos/:owner/:repo/collaborators", "/repos/:owner/:repo/collaborators"},
	{"/repos/:owner/:repo/collaborators/:user", "/repos/:owner/:repo/collaborators/:user"},
	{"/repos/:owner/:repo/comments", "/repos/:owner/:repo/comments"},
	{"/repos/:owner/:repo/commits/:sha/comments", "/repos/:owner/:repo/commits/:sha/comments"},
	{"/repos/:owner/:repo/comments/:id", "/repos/:owner/:repo/comments/:id"},
	{"/repos/:owner/:repo/commits", "/repos/:owner/:repo/commits"},
	{"/repos/:owner/:repo/commits/:sha", "/repos/:owner/:repo/commits/:sha"},
	{"/repos/:owner/:repo/readme", "/repos/:owner/:repo/readme"},
	{"/repos/:owner/:repo/keys", "/repos/:owner/:repo/keys"},
	{"/repos/:owner/:repo/keys/:id", "/repos/:owner/:repo/keys/:id"},
	{"/repos/:owner/:repo/downloads", "/repos/:owner/:repo/downloads"},
	{"/repos/:owner/:repo/downloads/:id", "/repos/:owner/:repo/downloads/:id"},
	{"/repos/:owner/:repo/forks", "/repos/:owner/:repo/forks"},
	{"/repos/:owner/:repo/hooks", "/repos/:owner/:repo/hooks"},
	{"/repos/:owner/:repo/hooks/:id", "/repos/:owner/:repo/hooks/:id"},
	{"/repos/:owner/:repo/releases", "/repos/:owner/:repo/releases"},
	{"/repos/:owner/:repo/releases/:id", "/repos/:owner/:repo/releases/:id"},
	{"/repos/:owner/:repo/releases/:id/assets", "/repos/:owner/:repo/releases/:id/assets"},
	{"/repos/:owner/:repo/stats/contributors", "/repos/:owner/:repo/stats/contributors"},
	{"/repos/:owner/:repo/stats/commit_activity", "/repos/:owner/:repo/stats/commit_activity"},
	{"/repos/:owner/:repo/stats/code_frequency", "/repos/:owner/:repo/stats/code_frequency"},
	{"/repos/:owner/:repo/stats/participation", "/repos/:owner/:repo/stats/participation"},
	{"/repos/:owner/:repo/stats/punch_card", "/repos/:owner/:repo/stats/punch_card"},
	{"/repos/:owner/:repo/statuses/:ref", "/repos/:owner/:repo/statuses/:ref"},
	{"/search/repositories", "/search/repositories"},
	{"/search/code", "/search/code"},
	{"/search/issues", "/search/issues"},
	{"/search/users", "/search/users"},
	{"/legacy/issues/search/:owner/:repository/:state/:keyword", "/legacy/issues/search/:owner/:repository/:state/:keyword"},
	{"/legacy/repos/search/:keyword", "/legacy/repos/search/:keyword"},
	{"/legacy/user/search/:keyword", "/legacy/user/search/:keyword"},
	{"/legacy/user/email/:email", "/legacy/user/email/:email"},
	{"/users/:user", "/users/:user"},
	{"/user", "/user"},
	{"/users", "/users"},
	{"/user/emails", "/user/emails"},
	{"/users/:user/followers", "/users/:user/followers"},
	{"/user/followers", "/user/followers"},
	{"/users/:user/following", "/users/:user/following"},
	{"/user/following", "/user/following"},
	{"/user/following/:user", "/user/following/:user"},
	{"/users/:user/following/:target_user", "/users/:user/following/:target_user"},
	{"/users/:user/keys", "/users/:user/keys"},
	{"/user/keys", "/user/keys"},
	{"/user/keys/:id", "/user/keys/:id"},
	{"/people/:userId", "/people/:userId"},
	{"/people", "/people"},
	{"/activities/:activityId/people/:collection", "/activities/:activityId/people/:collection"},
	{"/people/:userId/people/:collection", "/people/:userId/people/:collection"},
	{"/people/:userId/openIdConnect", "/people/:userId/openIdConnect"},
	{"/people/:userId/activities/:collection", "/people/:userId/activities/:collection"},
	{"/activities/:activityId", "/activities/:activityId"},
	{"/activities", "/activities"},
	{"/activities/:activityId/comments", "/activities/:activityId/comments"},
	{"/comments/:commentId", "/comments/:commentId"},
	{"/people/:userId/moments/:collection", "/people/:userId/moments/:collection"},
}

type testcase struct {
	path   string
	value  interface{}
	params []denco.Param
	found  bool
}

func runLookupTest(t *testing.T, records []denco.Record, testcases []testcase) {
	r := denco.New()
	if err := r.Build(records); err != nil {
		t.Fatal(err)
	}
	for _, testcase := range testcases {
		data, params, found := r.Lookup(testcase.path)
		if !reflect.DeepEqual(data, testcase.value) || !reflect.DeepEqual(params, denco.Params(testcase.params)) || !reflect.DeepEqual(found, testcase.found) {
			t.Errorf("Router.Lookup(%q) => (%#v, %#v, %#v), want (%#v, %#v, %#v)", testcase.path, data, params, found, testcase.value, denco.Params(testcase.params), testcase.found)
		}
	}
}

func TestRouter_Lookup(t *testing.T) {
	testcases := []testcase{
		{"/", "testroute0", nil, true},
		{"/gists/1323/foo/bar", "testroute11", []denco.Param{{"param1", "1323"}}, true},
		{"/gists/1323/foo/133", "testroute12", []denco.Param{{"param1", "1323"}, {"param2", "133"}}, true},
		{"/234/1323/foo/133", "testroute13", []denco.Param{{"param1", "234"}, {"param2", "1323"}, {"param3", "133"}}, true},
		{"/path/to/route", "testroute1", nil, true},
		{"/path/to/other", "testroute2", nil, true},
		{"/path/to/route/a", "testroute3", nil, true},
		{"/path/to/hoge", "testroute4", []denco.Param{{"param", "hoge"}}, true},
		{"/path/to/wildcard/some/params", "testroute5", []denco.Param{{"routepath", "some/params"}}, true},
		{"/path/to/o1/o2", "testroute6", []denco.Param{{"param1", "o1"}, {"param2", "o2"}}, true},
		{"/path/to/p1/sep/p2", "testroute7", []denco.Param{{"param1", "p1"}, {"param2", "p2"}}, true},
		{"/2014/01/06", "testroute8", []denco.Param{{"year", "2014"}, {"month", "01"}, {"day", "06"}}, true},
		{"/user/777", "testroute9", []denco.Param{{"id", "777"}}, true},
		{"/a/to/b/p1/some/wildcard/params", "testroute10", []denco.Param{{"param", "p1"}, {"routepath", "some/wildcard/params"}}, true},
		{"/missing", nil, nil, false},
	}
	runLookupTest(t, routes(), testcases)

	records := []denco.Record{
		{"/", "testroute0"},
		{"/:b", "testroute1"},
		{"/*wildcard", "testroute2"},
	}
	testcases = []testcase{
		{"/", "testroute0", nil, true},
		{"/true", "testroute1", []denco.Param{{"b", "true"}}, true},
		{"/foo/bar", "testroute2", []denco.Param{{"wildcard", "foo/bar"}}, true},
	}
	runLookupTest(t, records, testcases)

	records = []denco.Record{
		{"/networks/:owner/:repo/events", "testroute0"},
		{"/orgs/:org/events", "testroute1"},
		{"/notifications/threads/:id", "testroute2"},
	}
	testcases = []testcase{
		{"/networks/:owner/:repo/events", "testroute0", []denco.Param{{"owner", ":owner"}, {"repo", ":repo"}}, true},
		{"/orgs/:org/events", "testroute1", []denco.Param{{"org", ":org"}}, true},
		{"/notifications/threads/:id", "testroute2", []denco.Param{{"id", ":id"}}, true},
	}
	runLookupTest(t, records, testcases)

	runLookupTest(t, []denco.Record{
		{"/", "route2"},
	}, []testcase{
		{"/user/alice", nil, nil, false},
	})

	runLookupTest(t, []denco.Record{
		{"/user/:name", "route1"},
	}, []testcase{
		{"/", nil, nil, false},
	})

	runLookupTest(t, []denco.Record{
		{"/*wildcard", "testroute0"},
		{"/a/:b", "testroute1"},
	}, []testcase{
		{"/a", "testroute0", []denco.Param{{"wildcard", "a"}}, true},
	})
}

func TestRouter_Lookup_withManyRoutes(t *testing.T) {
	n := 1000
	rand.Seed(time.Now().UnixNano())
	records := make([]denco.Record, n)
	for i := 0; i < n; i++ {
		records[i] = denco.Record{Key: "/" + randomString(rand.Intn(50)+10), Value: fmt.Sprintf("route%d", i)}
	}
	router := denco.New()
	if err := router.Build(records); err != nil {
		t.Fatal(err)
	}
	for _, r := range records {
		data, params, found := router.Lookup(r.Key)
		if !reflect.DeepEqual(data, r.Value) || len(params) != 0 || !reflect.DeepEqual(found, true) {
			t.Errorf("Router.Lookup(%q) => (%#v, %#v, %#v), want (%#v, %#v, %#v)", r.Key, data, len(params), found, r.Value, 0, true)
		}
	}
}

func TestRouter_Lookup_realURIs(t *testing.T) {
	testcases := []testcase{
		{"/authorizations", "/authorizations", nil, true},
		{"/authorizations/1", "/authorizations/:id", []denco.Param{{"id", "1"}}, true},
		{"/applications/1/tokens/zohRoo7e", "/applications/:client_id/tokens/:access_token", []denco.Param{{"client_id", "1"}, {"access_token", "zohRoo7e"}}, true},
		{"/events", "/events", nil, true},
		{"/repos/naoina/denco/events", "/repos/:owner/:repo/events", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/networks/naoina/denco/events", "/networks/:owner/:repo/events", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/orgs/something/events", "/orgs/:org/events", []denco.Param{{"org", "something"}}, true},
		{"/users/naoina/received_events", "/users/:user/received_events", []denco.Param{{"user", "naoina"}}, true},
		{"/users/naoina/received_events/public", "/users/:user/received_events/public", []denco.Param{{"user", "naoina"}}, true},
		{"/users/naoina/events", "/users/:user/events", []denco.Param{{"user", "naoina"}}, true},
		{"/users/naoina/events/public", "/users/:user/events/public", []denco.Param{{"user", "naoina"}}, true},
		{"/users/naoina/events/orgs/something", "/users/:user/events/orgs/:org", []denco.Param{{"user", "naoina"}, {"org", "something"}}, true},
		{"/feeds", "/feeds", nil, true},
		{"/notifications", "/notifications", nil, true},
		{"/repos/naoina/denco/notifications", "/repos/:owner/:repo/notifications", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/notifications/threads/1", "/notifications/threads/:id", []denco.Param{{"id", "1"}}, true},
		{"/notifications/threads/2/subscription", "/notifications/threads/:id/subscription", []denco.Param{{"id", "2"}}, true},
		{"/repos/naoina/denco/stargazers", "/repos/:owner/:repo/stargazers", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/users/naoina/starred", "/users/:user/starred", []denco.Param{{"user", "naoina"}}, true},
		{"/user/starred", "/user/starred", nil, true},
		{"/user/starred/naoina/denco", "/user/starred/:owner/:repo", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/subscribers", "/repos/:owner/:repo/subscribers", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/users/naoina/subscriptions", "/users/:user/subscriptions", []denco.Param{{"user", "naoina"}}, true},
		{"/user/subscriptions", "/user/subscriptions", nil, true},
		{"/repos/naoina/denco/subscription", "/repos/:owner/:repo/subscription", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/user/subscriptions/naoina/denco", "/user/subscriptions/:owner/:repo", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/users/naoina/gists", "/users/:user/gists", []denco.Param{{"user", "naoina"}}, true},
		{"/gists", "/gists", nil, true},
		{"/gists/1", "/gists/:id", []denco.Param{{"id", "1"}}, true},
		{"/gists/2/star", "/gists/:id/star", []denco.Param{{"id", "2"}}, true},
		{"/repos/naoina/denco/git/blobs/03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9", "/repos/:owner/:repo/git/blobs/:sha", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"sha", "03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9"}}, true},
		{"/repos/naoina/denco/git/commits/03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9", "/repos/:owner/:repo/git/commits/:sha", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"sha", "03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9"}}, true},
		{"/repos/naoina/denco/git/refs", "/repos/:owner/:repo/git/refs", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/git/tags/03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9", "/repos/:owner/:repo/git/tags/:sha", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"sha", "03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9"}}, true},
		{"/repos/naoina/denco/git/trees/03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9", "/repos/:owner/:repo/git/trees/:sha", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"sha", "03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9"}}, true},
		{"/issues", "/issues", nil, true},
		{"/user/issues", "/user/issues", nil, true},
		{"/orgs/something/issues", "/orgs/:org/issues", []denco.Param{{"org", "something"}}, true},
		{"/repos/naoina/denco/issues", "/repos/:owner/:repo/issues", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/issues/1", "/repos/:owner/:repo/issues/:number", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/assignees", "/repos/:owner/:repo/assignees", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/assignees/foo", "/repos/:owner/:repo/assignees/:assignee", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"assignee", "foo"}}, true},
		{"/repos/naoina/denco/issues/1/comments", "/repos/:owner/:repo/issues/:number/comments", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/issues/1/events", "/repos/:owner/:repo/issues/:number/events", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/labels", "/repos/:owner/:repo/labels", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/labels/bug", "/repos/:owner/:repo/labels/:name", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"name", "bug"}}, true},
		{"/repos/naoina/denco/issues/1/labels", "/repos/:owner/:repo/issues/:number/labels", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/milestones/1/labels", "/repos/:owner/:repo/milestones/:number/labels", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/milestones", "/repos/:owner/:repo/milestones", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/milestones/1", "/repos/:owner/:repo/milestones/:number", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/emojis", "/emojis", nil, true},
		{"/gitignore/templates", "/gitignore/templates", nil, true},
		{"/gitignore/templates/Go", "/gitignore/templates/:name", []denco.Param{{"name", "Go"}}, true},
		{"/meta", "/meta", nil, true},
		{"/rate_limit", "/rate_limit", nil, true},
		{"/users/naoina/orgs", "/users/:user/orgs", []denco.Param{{"user", "naoina"}}, true},
		{"/user/orgs", "/user/orgs", nil, true},
		{"/orgs/something", "/orgs/:org", []denco.Param{{"org", "something"}}, true},
		{"/orgs/something/members", "/orgs/:org/members", []denco.Param{{"org", "something"}}, true},
		{"/orgs/something/members/naoina", "/orgs/:org/members/:user", []denco.Param{{"org", "something"}, {"user", "naoina"}}, true},
		{"/orgs/something/public_members", "/orgs/:org/public_members", []denco.Param{{"org", "something"}}, true},
		{"/orgs/something/public_members/naoina", "/orgs/:org/public_members/:user", []denco.Param{{"org", "something"}, {"user", "naoina"}}, true},
		{"/orgs/something/teams", "/orgs/:org/teams", []denco.Param{{"org", "something"}}, true},
		{"/teams/1", "/teams/:id", []denco.Param{{"id", "1"}}, true},
		{"/teams/2/members", "/teams/:id/members", []denco.Param{{"id", "2"}}, true},
		{"/teams/3/members/naoina", "/teams/:id/members/:user", []denco.Param{{"id", "3"}, {"user", "naoina"}}, true},
		{"/teams/4/repos", "/teams/:id/repos", []denco.Param{{"id", "4"}}, true},
		{"/teams/5/repos/naoina/denco", "/teams/:id/repos/:owner/:repo", []denco.Param{{"id", "5"}, {"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/user/teams", "/user/teams", nil, true},
		{"/repos/naoina/denco/pulls", "/repos/:owner/:repo/pulls", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/pulls/1", "/repos/:owner/:repo/pulls/:number", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/pulls/1/commits", "/repos/:owner/:repo/pulls/:number/commits", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/pulls/1/files", "/repos/:owner/:repo/pulls/:number/files", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/pulls/1/merge", "/repos/:owner/:repo/pulls/:number/merge", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/repos/naoina/denco/pulls/1/comments", "/repos/:owner/:repo/pulls/:number/comments", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"number", "1"}}, true},
		{"/user/repos", "/user/repos", nil, true},
		{"/users/naoina/repos", "/users/:user/repos", []denco.Param{{"user", "naoina"}}, true},
		{"/orgs/something/repos", "/orgs/:org/repos", []denco.Param{{"org", "something"}}, true},
		{"/repositories", "/repositories", nil, true},
		{"/repos/naoina/denco", "/repos/:owner/:repo", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/contributors", "/repos/:owner/:repo/contributors", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/languages", "/repos/:owner/:repo/languages", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/teams", "/repos/:owner/:repo/teams", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/tags", "/repos/:owner/:repo/tags", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/branches", "/repos/:owner/:repo/branches", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/branches/master", "/repos/:owner/:repo/branches/:branch", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"branch", "master"}}, true},
		{"/repos/naoina/denco/collaborators", "/repos/:owner/:repo/collaborators", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/collaborators/something", "/repos/:owner/:repo/collaborators/:user", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"user", "something"}}, true},
		{"/repos/naoina/denco/comments", "/repos/:owner/:repo/comments", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/commits/03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9/comments", "/repos/:owner/:repo/commits/:sha/comments", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"sha", "03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9"}}, true},
		{"/repos/naoina/denco/comments/1", "/repos/:owner/:repo/comments/:id", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"id", "1"}}, true},
		{"/repos/naoina/denco/commits", "/repos/:owner/:repo/commits", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/commits/03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9", "/repos/:owner/:repo/commits/:sha", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"sha", "03c3bbc7f0d12268b9ca53d4fbfd8dc5ae5697b9"}}, true},
		{"/repos/naoina/denco/readme", "/repos/:owner/:repo/readme", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/keys", "/repos/:owner/:repo/keys", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/keys/1", "/repos/:owner/:repo/keys/:id", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"id", "1"}}, true},
		{"/repos/naoina/denco/downloads", "/repos/:owner/:repo/downloads", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/downloads/2", "/repos/:owner/:repo/downloads/:id", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"id", "2"}}, true},
		{"/repos/naoina/denco/forks", "/repos/:owner/:repo/forks", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/hooks", "/repos/:owner/:repo/hooks", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/hooks/2", "/repos/:owner/:repo/hooks/:id", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"id", "2"}}, true},
		{"/repos/naoina/denco/releases", "/repos/:owner/:repo/releases", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/releases/1", "/repos/:owner/:repo/releases/:id", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"id", "1"}}, true},
		{"/repos/naoina/denco/releases/1/assets", "/repos/:owner/:repo/releases/:id/assets", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"id", "1"}}, true},
		{"/repos/naoina/denco/stats/contributors", "/repos/:owner/:repo/stats/contributors", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/stats/commit_activity", "/repos/:owner/:repo/stats/commit_activity", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/stats/code_frequency", "/repos/:owner/:repo/stats/code_frequency", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/stats/participation", "/repos/:owner/:repo/stats/participation", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/stats/punch_card", "/repos/:owner/:repo/stats/punch_card", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}}, true},
		{"/repos/naoina/denco/statuses/master", "/repos/:owner/:repo/statuses/:ref", []denco.Param{{"owner", "naoina"}, {"repo", "denco"}, {"ref", "master"}}, true},
		{"/search/repositories", "/search/repositories", nil, true},
		{"/search/code", "/search/code", nil, true},
		{"/search/issues", "/search/issues", nil, true},
		{"/search/users", "/search/users", nil, true},
		{"/legacy/issues/search/naoina/denco/closed/test", "/legacy/issues/search/:owner/:repository/:state/:keyword", []denco.Param{{"owner", "naoina"}, {"repository", "denco"}, {"state", "closed"}, {"keyword", "test"}}, true},
		{"/legacy/repos/search/test", "/legacy/repos/search/:keyword", []denco.Param{{"keyword", "test"}}, true},
		{"/legacy/user/search/test", "/legacy/user/search/:keyword", []denco.Param{{"keyword", "test"}}, true},
		{"/legacy/user/email/naoina@kuune.org", "/legacy/user/email/:email", []denco.Param{{"email", "naoina@kuune.org"}}, true},
		{"/users/naoina", "/users/:user", []denco.Param{{"user", "naoina"}}, true},
		{"/user", "/user", nil, true},
		{"/users", "/users", nil, true},
		{"/user/emails", "/user/emails", nil, true},
		{"/users/naoina/followers", "/users/:user/followers", []denco.Param{{"user", "naoina"}}, true},
		{"/user/followers", "/user/followers", nil, true},
		{"/users/naoina/following", "/users/:user/following", []denco.Param{{"user", "naoina"}}, true},
		{"/user/following", "/user/following", nil, true},
		{"/user/following/naoina", "/user/following/:user", []denco.Param{{"user", "naoina"}}, true},
		{"/users/naoina/following/target", "/users/:user/following/:target_user", []denco.Param{{"user", "naoina"}, {"target_user", "target"}}, true},
		{"/users/naoina/keys", "/users/:user/keys", []denco.Param{{"user", "naoina"}}, true},
		{"/user/keys", "/user/keys", nil, true},
		{"/user/keys/1", "/user/keys/:id", []denco.Param{{"id", "1"}}, true},
		{"/people/me", "/people/:userId", []denco.Param{{"userId", "me"}}, true},
		{"/people", "/people", nil, true},
		{"/activities/foo/people/vault", "/activities/:activityId/people/:collection", []denco.Param{{"activityId", "foo"}, {"collection", "vault"}}, true},
		{"/people/me/people/vault", "/people/:userId/people/:collection", []denco.Param{{"userId", "me"}, {"collection", "vault"}}, true},
		{"/people/me/openIdConnect", "/people/:userId/openIdConnect", []denco.Param{{"userId", "me"}}, true},
		{"/people/me/activities/vault", "/people/:userId/activities/:collection", []denco.Param{{"userId", "me"}, {"collection", "vault"}}, true},
		{"/activities/foo", "/activities/:activityId", []denco.Param{{"activityId", "foo"}}, true},
		{"/activities", "/activities", nil, true},
		{"/activities/foo/comments", "/activities/:activityId/comments", []denco.Param{{"activityId", "foo"}}, true},
		{"/comments/hoge", "/comments/:commentId", []denco.Param{{"commentId", "hoge"}}, true},
		{"/people/me/moments/vault", "/people/:userId/moments/:collection", []denco.Param{{"userId", "me"}, {"collection", "vault"}}, true},
	}
	runLookupTest(t, realURIs, testcases)
}

func TestRouter_Build(t *testing.T) {
	// test for duplicate name of path parameters.
	func() {
		r := denco.New()
		if err := r.Build([]denco.Record{
			{"/:user/:id/:id", "testroute0"},
			{"/:user/:user/:id", "testroute0"},
		}); err == nil {
			t.Errorf("no error returned by duplicate name of path parameters")
		}
	}()
}

func TestRouter_Build_withoutSizeHint(t *testing.T) {
	for _, v := range []struct {
		keys     []string
		sizeHint int
	}{
		{[]string{"/user"}, 0},
		{[]string{"/user/:id"}, 1},
		{[]string{"/user/:id/post"}, 1},
		{[]string{"/user/:id/:group"}, 2},
		{[]string{"/user/:id/post/:cid"}, 2},
		{[]string{"/user/:id/post/:cid", "/admin/:id/post/:cid"}, 2},
		{[]string{"/user/:id", "/admin/:id/post/:cid"}, 2},
		{[]string{"/user/:id/post/:cid", "/admin/:id/post/:cid/:type"}, 3},
	} {
		r := denco.New()
		actual := r.SizeHint
		expect := -1
		if !reflect.DeepEqual(actual, expect) {
			t.Errorf(`before Build; Router.SizeHint => (%[1]T=%#[1]v); want (%[2]T=%#[2]v)`, actual, expect)
		}
		records := make([]denco.Record, len(v.keys))
		for i, k := range v.keys {
			records[i] = denco.Record{Key: k, Value: "value"}
		}
		if err := r.Build(records); err != nil {
			t.Fatal(err)
		}
		actual = r.SizeHint
		expect = v.sizeHint
		if !reflect.DeepEqual(actual, expect) {
			t.Errorf(`Router.Build(%#v); Router.SizeHint => (%[2]T=%#[2]v); want (%[3]T=%#[3]v)`, records, actual, expect)
		}
	}
}

func TestRouter_Build_withSizeHint(t *testing.T) {
	for _, v := range []struct {
		key      string
		sizeHint int
		expect   int
	}{
		{"/user", 0, 0},
		{"/user", 1, 1},
		{"/user", 2, 2},
		{"/user/:id", 3, 3},
		{"/user/:id/:group", 0, 0},
		{"/user/:id/:group", 1, 1},
	} {
		r := denco.New()
		r.SizeHint = v.sizeHint
		records := []denco.Record{
			{v.key, "value"},
		}
		if err := r.Build(records); err != nil {
			t.Fatal(err)
		}
		actual := r.SizeHint
		expect := v.expect
		if !reflect.DeepEqual(actual, expect) {
			t.Errorf(`Router.Build(%#v); Router.SizeHint => (%[2]T=%#[2]v); want (%[3]T=%#[3]v)`, records, actual, expect)
		}
	}
}

func TestParams_Get(t *testing.T) {
	params := denco.Params([]denco.Param{
		{"name1", "value1"},
		{"name2", "value2"},
		{"name3", "value3"},
		{"name1", "value4"},
	})
	for _, v := range []struct{ value, expected string }{
		{"name1", "value1"},
		{"name2", "value2"},
		{"name3", "value3"},
		{"name4", ""},
	} {
		actual := params.Get(v.value)
		expected := v.expected
		if !reflect.DeepEqual(actual, expected) {
			t.Errorf("Params.Get(%q) => %#v, want %#v", v.value, actual, expected)
		}
	}
}
