/*
 * Copyright 2012 Cloud9 IDE, Inc.
 *
 * This product includes software developed by
 * Cloud9 IDE, Inc (http://c9.io).
 *
 * Author: Mike de Boer <info@mikedeboer.nl>
 */

"use strict";

var Assert = require("assert");
var Client = require("./../../index");

describe("[repos]", function() {
    var client;
    var token = "c286e38330e15246a640c2cf32a45ea45d93b2ba";

    beforeEach(function() {
        client = new Client({
            version: "3.0.0"
        });
        client.authenticate({
            type: "oauth",
            token: token
        });
    });

    it("should successfully execute GET /user/repos (getAll)",  function(next) {
        client.repos.getAll(
            {
                type: "String",
                sort: "String",
                direction: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /users/:user/repos (getFromUser)",  function(next) {
        client.repos.getFromUser(
            {
                user: "String",
                type: "String",
                sort: "String",
                direction: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /orgs/:org/repos (getFromOrg)",  function(next) {
        client.repos.getFromOrg(
            {
                org: "String",
                type: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /user/repos (create)",  function(next) {
        client.repos.create(
            {
                name: "String",
                description: "String",
                homepage: "String",
                private: "Boolean",
                has_issues: "Boolean",
                has_wiki: "Boolean",
                has_downloads: "Boolean",
                auto_init: "Boolean",
                gitignore_template: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /orgs/:org/repos (createFromOrg)",  function(next) {
        client.repos.createFromOrg(
            {
                org: "String",
                name: "String",
                description: "String",
                homepage: "String",
                private: "Boolean",
                has_issues: "Boolean",
                has_wiki: "Boolean",
                has_downloads: "Boolean",
                auto_init: "Boolean",
                gitignore_template: "String",
                team_id: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo (get)",  function(next) {
        client.repos.get(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:user/:repo (update)",  function(next) {
        client.repos.update(
            {
                user: "String",
                repo: "String",
                name: "String",
                description: "String",
                homepage: "String",
                private: "Boolean",
                has_issues: "Boolean",
                has_wiki: "Boolean",
                has_downloads: "Boolean"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo (delete)",  function(next) {
        client.repos.delete(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/merges (merge)",  function(next) {
        client.repos.merge(
            {
                user: "String",
                repo: "String",
                base: "String",
                head: "String",
                commit_message: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/contributors (getContributors)",  function(next) {
        client.repos.getContributors(
            {
                user: "String",
                repo: "String",
                anon: "Boolean",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/languages (getLanguages)",  function(next) {
        client.repos.getLanguages(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/teams (getTeams)",  function(next) {
        client.repos.getTeams(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/tags (getTags)",  function(next) {
        client.repos.getTags(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/branches (getBranches)",  function(next) {
        client.repos.getBranches(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/branches/:branch (getBranch)",  function(next) {
        client.repos.getBranches(
            {
                user: "String",
                repo: "String",
                branch: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/collaborators (getCollaborators)",  function(next) {
        client.repos.getCollaborators(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/collaborators/:collabuser (getCollaborator)",  function(next) {
        client.repos.getCollaborator(
            {
                user: "String",
                repo: "String",
                collabuser: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PUT /repos/:user/:repo/collaborators/:collabuser (addCollaborator)",  function(next) {
        client.repos.addCollaborator(
            {
                user: "String",
                repo: "String",
                collabuser: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/collaborators/:collabuser (removeCollaborator)",  function(next) {
        client.repos.removeCollaborator(
            {
                user: "String",
                repo: "String",
                collabuser: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/commits (getCommits)",  function(next) {
        client.repos.getCommits(
            {
                user: "String",
                repo: "String",
                sha: "String",
                path: "String",
                page: "Number",
                per_page: "Number",
                author: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/commits/:sha (getCommit)",  function(next) {
        client.repos.getCommit(
            {
                user: "String",
                repo: "String",
                sha: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/comments (getAllCommitComments)",  function(next) {
        client.repos.getAllCommitComments(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/commits/:sha/comments (getCommitComments)",  function(next) {
        client.repos.getCommitComments(
            {
                user: "String",
                repo: "String",
                sha: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/commits/:sha/comments (createCommitComment)",  function(next) {
        client.repos.createCommitComment(
            {
                user: "String",
                repo: "String",
                sha: "String",
                body: "String",
                commit_id: "String",
                path: "String",
                position: "Number",
                line: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/comments/:id (getCommitComment)",  function(next) {
        client.repos.getCommitComment(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:user/:repo/comments/:id (updateCommitComment)",  function(next) {
        client.repos.updateCommitComment(
            {
                user: "String",
                repo: "String",
                id: "String",
                body: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/compare/:base...:head (compareCommits)",  function(next) {
        client.repos.compareCommits(
            {
                user: "String",
                repo: "String",
                base: "String",
                head: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/comments/:id (deleteCommitComment)",  function(next) {
        client.repos.deleteCommitComment(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/readme (getReadme)",  function(next) {
        client.repos.getReadme(
            {
                user: "String",
                repo: "String",
                ref: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/contents/:path (getContent)",  function(next) {
        client.repos.getContent(
            {
                user: "String",
                repo: "String",
                path: "String",
                ref: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });
    it("should successfully execute GET /repos/:user/:repo/contents/:path (createContent)",  function(next) {
        client.repos.getContent(
            {
                user: "String",
                repo: "String",
                path: "String",
                ref: "String",
                content:"String",
                message:"String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PUT /repos/:user/:repo/contents/:path (createFile)",  function(next) {
        client.repos.createFile(
            {
                user: "String",
                repo: "String",
                path: "String",
                message: "String",
                content: "String",
                branch: "String",
                author: "Json",
                committer: "Json"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PUT /repos/:user/:repo/contents/:path (updateFile)",  function(next) {
        client.repos.updateFile(
            {
                user: "String",
                repo: "String",
                path: "String",
                message: "String",
                content: "String",
                sha: "String",
                branch: "String",
                author: "Json",
                committer: "Json"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/contents/:path (deleteFile)",  function(next) {
        client.repos.deleteFile(
            {
                user: "String",
                repo: "String",
                path: "String",
                message: "String",
                sha: "String",
                branch: "String",
                author: "Json",
                committer: "Json"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/:archive_format/:ref (getArchiveLink)",  function(next) {
        client.repos.getArchiveLink(
            {
                user: "String",
                repo: "String",
                archive_format: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/downloads (getDownloads)",  function(next) {
        client.repos.getDownloads(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/downloads/:id (getDownload)",  function(next) {
        client.repos.getDownload(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/downloads/:id (deleteDownload)",  function(next) {
        client.repos.deleteDownload(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/forks (getForks)",  function(next) {
        client.repos.getForks(
            {
                user: "String",
                repo: "String",
                sort: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/forks (fork)",  function(next) {
        client.repos.fork(
            {
                user: "String",
                repo: "String",
                organization: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/keys (getKeys)",  function(next) {
        client.repos.getKeys(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/keys/:id (getKey)",  function(next) {
        client.repos.getKey(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/keys (createKey)",  function(next) {
        client.repos.createKey(
            {
                user: "String",
                repo: "String",
                title: "String",
                key: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PUT /repos/:user/:repo/keys/:id (updateKey)",  function(next) {
        client.repos.updateKey(
            {
                user: "String",
                repo: "String",
                id: "String",
                title: "String",
                key: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/keys/:id (deleteKey)",  function(next) {
        client.repos.deleteKey(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/stargazers (getStargazers)",  function(next) {
        client.repos.getStargazers(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /user/starred (getStarred)",  function(next) {
        client.repos.getStarred(
            {
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /users/:user/starred (getStarredFromUser)",  function(next) {
        client.repos.getStarredFromUser(
            {
                user: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /user/starred/:user/:repo (getStarring)",  function(next) {
        client.repos.getStarring(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PUT /user/starred/:user/:repo (watch)",  function(next) {
        client.repos.watch(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /user/starred/:user/:repo (unWatch)",  function(next) {
        client.repos.unWatch(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/watchers (getWatchers)",  function(next) {
        client.repos.getWatchers(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /user/watched (getWatched)",  function(next) {
        client.repos.getWatched(
            {
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /users/:user/watched (getWatchedFromUser)",  function(next) {
        client.repos.getWatchedFromUser(
            {
                user: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /user/watched/:user/:repo (getWatching)",  function(next) {
        client.repos.getWatching(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PUT /user/watched/:user/:repo (watch)",  function(next) {
        client.repos.watch(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /user/watched/:user/:repo (unWatch)",  function(next) {
        client.repos.unWatch(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/hooks (getHooks)",  function(next) {
        client.repos.getHooks(
            {
                user: "String",
                repo: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/hooks/:id (getHook)",  function(next) {
        client.repos.getHook(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/hooks (createHook)",  function(next) {
        client.repos.createHook(
            {
                user: "String",
                repo: "String",
                name: "String",
                config: "Json",
                events: "Array",
                active: "Boolean"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:user/:repo/hooks/:id (updateHook)",  function(next) {
        client.repos.updateHook(
            {
                user: "String",
                repo: "String",
                id: "String",
                name: "String",
                config: "Json",
                events: "Array",
                add_events: "Array",
                remove_events: "Array",
                active: "Boolean"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/hooks/:id/test (testHook)",  function(next) {
        client.repos.testHook(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/hooks/:id (deleteHook)",  function(next) {
        client.repos.deleteHook(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/stats/contributors (getStatsContributors)",  function(next) {
        client.repos.getStatsContributors(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/stats/commit_activity (getStatsCommitActivity)",  function(next) {
        client.repos.getStatsCommitActivity(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/stats/code_frequency (getStatsCodeFrequency)",  function(next) {
        client.repos.getStatsCodeFrequency(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/stats/participation (getStatsParticipation)",  function(next) {
        client.repos.getStatsParticipation(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/stats/punch_card (getStatsPunchCard)",  function(next) {
        client.repos.getStatsPunchCard(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });
});
